from typing import Callable, Iterable

import numpy as np
import lightning as pl
import torch
import torch.nn.functional as F
from torch_pesq import PesqLoss
from torchmetrics.audio import SignalNoiseRatio

from src.train_utils import prepare_batch_signs_for_target_speaker, add_rand_offset
from src.perturbation_applier import PerturbationApplier


class LitUAPGenerator(pl.LightningModule):
    RATE: int = 16000
    
    def __init__(
        self,
        asv_model: torch.nn.Module,
        perturbation_applier: PerturbationApplier,
        *,
        loss_weight_arr: Iterable[tuple[Callable, float | int]],
        rand_offset: bool = False,
        pert_lr: float = 0.01,
        target_speaker_id: str | None = None,
    ):
        super().__init__()
        self.asv_model = asv_model.eval()
        self.pert_applier = perturbation_applier
        
        self.loss_weight_arr = loss_weight_arr
        self.target_speaker_id = target_speaker_id
                
        self.pert_lr = pert_lr
        self.rand_offset = rand_offset
                
        self.snr = SignalNoiseRatio()
        self.pesq = PesqLoss(sample_rate=self.RATE, factor=0.5)
    
    def on_train_start(self):
        self.pert_applier.to(self.device)
        for param in self.asv_model.parameters():
            param.requires_grad = False
    
    def training_step(self, batch, batch_idx):
        audio_batch, ids = batch
        
        noised_batch = self.pert_applier(audio_batch)

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
            logits=None
        )
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, ids = batch
        ids = np.array(ids)
        self.snr.to(self.device)
        
        noised_audio = self.pert_applier(audio)
        predicted_ids = self.asv_model(noised_audio)[3]
        fr = (predicted_ids != ids).mean()
        
        snr = self.snr(noised_audio, audio).item()
        pesq = self.pesq.mos(noised_audio, audio).mean().item()        
        
        self.log("val_FR", fr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_SNR", snr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_PESQ", pesq, on_step=False, on_epoch=True, prog_bar=True)
        return fr
    
    def test_step(self, batch, batch_idx):
        audio, ids = batch
        ids = np.array(ids)
        offset = torch.zeros(audio.shape[0], self.offset_size, device=self.device) #TODO: to be deleted
        audio = audio.to(self.device)
        
        noised_audio = self.pert_applier(audio)
        noised_audio = torch.cat([offset, noised_audio], dim=1) #TODO: to be deleted
        
        spk_embeddings = self.asv_model.encode_batch(audio).squeeze(1)
        noised_spk_embeddings = self.asv_model.encode_batch(noised_audio).squeeze(1)
        pred_labels = self.asv_model(noised_audio)[3]
        
        self.snr.to(self.device)
        
        test_loss = ((F.cosine_similarity(noised_spk_embeddings, spk_embeddings, dim=1) + 1) / 2).mean()
        snr_score = self.snr(noised_audio, audio).item()
        pesq_score = self.pesq.mos(noised_audio, audio).mean().item()  # type: ignore
        fr = (pred_labels != ids).mean()

        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        self.log("test_snr", snr_score, on_epoch=True, prog_bar=True)
        self.log("test_pesq", pesq_score, on_epoch=True, prog_bar=True)
        self.log("test_fr", fr, on_epoch=True, prog_bar=True)
    
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
            self.log(f"train_loss_{loss_fn.__class__}_{loss_fn.mode}", weight * loss_val)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.pert_applier.parameters(), lr=self.pert_lr)
    
    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint['perturbation'] = self.pert_applier.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.pert_applier.load_state_dict(checkpoint["perturbation"])
