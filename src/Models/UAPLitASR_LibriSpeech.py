from typing import Callable, Iterable

import numpy as np
import lightning as pl
import os
import torch
import torch.nn.functional as F
from torch_pesq import PesqLoss
from torchmetrics.audio import SignalNoiseRatio

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import wandb

from src.train_utils import prepare_batch_signs_for_target_speaker, add_rand_offset
from src.perturbation_applier import PerturbationApplier
from torchmetrics.text import WordErrorRate

def load_speaker_embeddings(folder_path: str):
    """
    Loads speaker embedding vectors from .pt files in a given folder and returns:
    - A tensor of shape [192, n_speakers]
    - A dictionary mapping speaker_id (int) to column index in the tensor
    
    Assumes file naming pattern: spk_cent_<speaker_id>.pt
    """

    files = [f for f in os.listdir(folder_path) if f.startswith("spk_cent") and f.endswith(".pt")]
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))

    embeddings_list = []
    idx_to_spk_id = {}

    for idx, filename in enumerate(files_sorted):
        speaker_id = int(filename.split('_')[2].split('.')[0])
        file_path = os.path.join(folder_path, filename)

        embedding = torch.load(file_path, weights_only=True)

        if embedding.shape != torch.Size([192]):
            raise ValueError(f"Unexpected shape {embedding.shape} in file {filename}; expected [192].")

        embeddings_list.append(embedding.unsqueeze(1))
        idx_to_spk_id[idx] = str(speaker_id)

    embedding_matrix = torch.cat(embeddings_list, dim=1)
    return embedding_matrix, idx_to_spk_id

class LitUAPGenerator(pl.LightningModule):
    RATE: int = 16000
    
    def __init__(
        self,
        asv_model: torch.nn.Module,
        asr_model_size: str,
        perturbation_applier: PerturbationApplier,
        spk_centroids_dir: str,
        *,
        loss_weight_arr: Iterable[tuple[Callable, float | int]],
        rand_offset: bool = False,
        pert_lr: float = 0.01,
        target_speaker_id: str | None = None,
    ):
        super().__init__()
        self.asv_model = asv_model.eval()
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = f"openai/whisper-{asr_model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        ).to(device).eval()
        self.asr_processor = AutoProcessor.from_pretrained(model_id)
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        for param in self.asr_model.parameters():
            param.requires_grad = False
            
        self.pert_applier = perturbation_applier
        
        self.loss_weight_arr = loss_weight_arr
        self.target_speaker_id = target_speaker_id
                
        self.pert_lr = pert_lr
        self.rand_offset = rand_offset
                
        self.snr = SignalNoiseRatio()
        self.pesq = PesqLoss(sample_rate=self.RATE, factor=0.5)
        self.wer = WordErrorRate()
        
        self.wer_gt = WordErrorRate()
        self.computed_wer_gt = False

        self.spk_centroid_matrix, self.spk_idx_to_id = load_speaker_embeddings(spk_centroids_dir)
        print(f"Loaded {len(self.spk_centroid_matrix)} speaker embeddings into matrix")
    
    def on_train_start(self):
        self.pert_applier.to(self.device)
        for param in self.asv_model.parameters():
            param.requires_grad = False
    
    def training_step(self, batch, batch_idx):
        audio_batch, ids, transcripts, spk_emb = batch
        
        noised_batch = self.pert_applier(audio_batch, spk_emb)

        orig_emb = self.asv_model.encode_batch(audio_batch).squeeze(1)
        if self.rand_offset is not None:
            offset_noised_batch = add_rand_offset(noised_batch, self.RATE)
            noised_emb = self.asv_model.encode_batch(offset_noised_batch).squeeze(1)
        else:
            noised_emb = self.asv_model.encode_batch(noised_batch).squeeze(1)
        
        #TODO add logits calc
        loss = self.eval_loss(
            ids=ids,
            audio=audio_batch,
            noised_audio=noised_batch, 
            noised_emb=noised_emb,
            orig_emb=orig_emb, 
            perturbation=self.pert_applier.perturbation,
            logits=None,
            
            # for Whisper:
            asr_model=self.asr_model,
            asr_processor=self.asr_processor,
            transcripts=transcripts,
        )
        
        self.log("train_loss", loss)
        return loss
    
    def get_transcripts(self, audio_batch: torch.tensor):
        """
        Used only for validation, on training use pre-computed transcriptions
        """
        audio_batch = [elem.clone().detach().cpu().numpy() for elem in audio_batch]
        output = self.asr_pipe(audio_batch, batch_size=len(audio_batch), generate_kwargs={"language": "english"})
        return [element["text"] for element in output]
    
    def validation_step(self, batch, batch_idx):
        audio, ids, transcripts, spk_emb = batch
        ids = np.array(ids)
        self.snr.to(self.device)
        
        noised_audio = self.pert_applier(audio, spk_emb)
        noised_spk_embeddings = self.asv_model.encode_batch(noised_audio).squeeze(1)
        spk_idxs_after_attack = (noised_spk_embeddings @ self.spk_centroid_matrix).argmax(dim=1)
        pred_speaker_ids = np.array([self.spk_idx_to_id[idx.item()] for idx in spk_idxs_after_attack])
        fr = (pred_speaker_ids != ids).mean()
        
        snr = self.snr(noised_audio, audio).item()
        pesq = self.pesq.mos(noised_audio.to(torch.float), audio.to(torch.float)).mean().item()
        
        transcripts_true = transcripts
        transcripts_adv = self.get_transcripts(noised_audio)
        wer = self.wer(transcripts_adv, transcripts_true)    
        
        self.log("val_FR", fr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_SNR", snr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_PESQ", pesq, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_WER", wer, on_step=False, on_epoch=True, prog_bar=True)
        
        if not self.computed_wer_gt:
            transcripts_no_adv = self.get_transcripts(audio)
            wer_gt = self.wer_gt(transcripts_no_adv, transcripts_true) 
            self.log("WER-GT", wer_gt, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.computed_wer_gt = True
        
        self.logger.experiment.log(
            {
                "Audio/Original": wandb.Audio(audio[-1].cpu().squeeze(), sample_rate=self.RATE),
                "Audio/Attacked": wandb.Audio(noised_audio[-1].cpu().squeeze(), sample_rate=self.RATE),
            }
        )

        return fr
    
    def test_step(self, batch, batch_idx):
        audio, ids, transcripts, spk_embeddings = batch
        ids = np.array(ids)
        audio = audio.to(self.device)
        
        noised_audio = self.pert_applier(audio, spk_embeddings)
        
        # spk_embeddings = self.asv_model.encode_batch(audio).squeeze(1)
        noised_spk_embeddings = self.asv_model.encode_batch(noised_audio).squeeze(1)

        self.spk_centroid_matrix = self.spk_centroid_matrix.to(noised_spk_embeddings)
        spk_idxs_after_attack = (noised_spk_embeddings @ self.spk_centroid_matrix).argmax(dim=1)
        pred_speaker_ids = np.array([self.spk_idx_to_id[idx.item()] for idx in spk_idxs_after_attack])
        fr = (pred_speaker_ids != ids).mean()
        
        self.snr.to(self.device)
        
        noised_audio_for_pesq = noised_audio
        noised_audio_for_pesq[audio == 0] = 0
        
        test_loss = ((F.cosine_similarity(noised_spk_embeddings, spk_embeddings, dim=1) + 1) / 2).mean()
        snr_score = self.snr(noised_audio, audio).item()
        pesq_score = self.pesq.mos(noised_audio_for_pesq.to(torch.float), audio.to(torch.float)).mean().item()  # type: ignore

        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        self.log("test_snr", snr_score, on_epoch=True, prog_bar=True)
        self.log("test_pesq", pesq_score, on_epoch=True, prog_bar=True)
        self.log("test_fr", fr, on_epoch=True, prog_bar=True)

        # transcripts_true = transcripts
        # transcripts_adv = self.get_transcripts(noised_audio)
        # wer = self.wer(transcripts_adv, transcripts_true)
        # self.log("test_WER", wer, on_epoch=True, prog_bar=True)

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def eval_loss(self, **kwargs) -> torch.Tensor:
        loss = torch.zeros(1, device=self.device)
        
        if self.target_speaker_id is not None:
            sing_for_batch = prepare_batch_signs_for_target_speaker(kwargs['ids'], self.target_speaker_id)
            sing_for_batch = sing_for_batch.to(self.device)
            kwargs["orig_emb"] = sing_for_batch * kwargs["orig_emb"]
            
        for loss_fn, weight in self.loss_weight_arr:    
            loss_val = loss_fn(**kwargs)
            loss += weight * loss_val
            loss_name = str(loss_fn.__class__).split('.')[-1].replace("'>", "")
            # print(f"Loss: {loss_fn.__class__}: {loss_val}, weighted: {weight * loss_val}")
            self.log(f"train_loss_{loss_name}_{loss_fn.mode}", weight * loss_val, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.pert_applier.parameters(), lr=self.pert_lr)
    
    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint['perturbation'] = self.pert_applier.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.pert_applier.load_state_dict(checkpoint["perturbation"])
