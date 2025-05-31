from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch_pesq import PesqLoss
from torchmetrics.audio import SignalNoiseRatio
from lightning import LightningModule

from src.Models.StreamVC import Decoder
from src.train_utils import get_streamvc_decoder_input_shape, repeat_perturbation


class PLStreamVCModel(LightningModule):
    def __init__(self, *, embedding_projector: torch.nn.Module, spk_emb_model: torch.nn.Module, decoder_embeddings_dim: int, uap_len_sec: float, learning_rate: float):
        super().__init__()
        self.spk_emb_model = spk_emb_model.eval()
        self.embedding_projector = embedding_projector.train()
        self.decoder = Decoder(embedding_dim=decoder_embeddings_dim).train()
        
        self.learning_rate = learning_rate
        self.uap_len_sec = uap_len_sec
        self.decoder_embeddings_dim = decoder_embeddings_dim
        
        self.snr = SignalNoiseRatio()
        self.pesq = PesqLoss(sample_rate=16_000, factor=0.5)
    
    def on_fit_start(self):
        dec_inp = torch.rand((1, self.decoder_embeddings_dim, get_streamvc_decoder_input_shape(self.uap_len_sec * 16_000)), device=self.device)
        self.dec_inp = F.normalize(dec_inp, p=2, dim=-1)
    
    def forward(self, audio):
        spk_embeddings = self.spk_emb_model.encode_batch(audio).squeeze(1)
        proj_spk_embeddings = self.embedding_projector(spk_embeddings)
        uap = self.decoder(self.dec_inp, proj_spk_embeddings).squeeze(1)
        return F.normalize(uap, p=2, dim=-1)

    def training_step(self, batch, batch_idx):
        audio, ids = batch
        audio = audio.to(self.device)
        
        spk_embeddings = self.spk_emb_model.encode_batch(audio).squeeze(1)
        proj_spk_embeddings = self.embedding_projector(spk_embeddings)
        uap = self.decoder(self.dec_inp, proj_spk_embeddings).squeeze(1)
        uap = F.normalize(uap, p=2, dim=-1)

        repeated_uap = repeat_perturbation(uap, target_len=audio.size(1))
        noised_audio = audio + repeated_uap
        noised_spk_embeddings = self.spk_emb_model.encode_batch(noised_audio).squeeze(1)
        
        loss = ((F.cosine_similarity(noised_spk_embeddings, spk_embeddings, dim=1) + 1) / 2).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, ids = batch
        ids = np.array(ids)
        audio = audio.to(self.device)
        
        spk_embeddings = self.spk_emb_model.encode_batch(audio).squeeze(1)
        proj_spk_embeddings = self.embedding_projector(spk_embeddings)
        uap = self.decoder(self.dec_inp, proj_spk_embeddings).squeeze(1)
        uap = F.normalize(uap, p=2, dim=-1)

        repeated_uap = repeat_perturbation(uap, target_len=audio.size(1))
        noised_audio = audio + repeated_uap
        predicted_ids = self.spk_emb_model(noised_audio)[3]
        fr = (predicted_ids != ids).mean()
        
        self.log("val_FR", fr, on_step=False, on_epoch=True, prog_bar=True)
        return fr
    
    def test_step(self, batch, batch_idx):
        audio, ids = batch
        ids = np.array(ids)
        audio = audio.to(self.device)

        spk_embeddings = self.spk_emb_model.encode_batch(audio).squeeze(1)
        # spk_embeddings = torch.randn_like(spk_embeddings, device=self.device)
        proj_spk_embeddings = self.embedding_projector(spk_embeddings)
        uap = self.decoder(self.dec_inp, proj_spk_embeddings).squeeze(1)
        uap = F.normalize(uap, p=2, dim=-1)
        
        repeated_uap = repeat_perturbation(uap, target_len=audio.size(1))
        noised_audio = audio + repeated_uap
        
        noised_spk_embeddings = self.spk_emb_model.encode_batch(noised_audio).squeeze(1)
        pred_labels = self.spk_emb_model(noised_audio)[3]
        
        self.snr.to(self.device)
        
        test_loss = ((F.cosine_similarity(noised_spk_embeddings, spk_embeddings, dim=1) + 1) / 2).mean()
        snr_score = self.snr(noised_audio, audio).item()
        pesq_score = self.pesq.mos(noised_audio, audio).mean().item()
        fr = (pred_labels != ids).mean()

        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        self.log("test_snr", snr_score, on_epoch=True, prog_bar=True)
        self.log("test_pesq", pesq_score, on_epoch=True, prog_bar=True)
        self.log("test_fr", fr, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            chain(self.decoder.parameters(), self.embedding_projector.parameters()), 
            lr=self.learning_rate
        )
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()
        checkpoint['embedding_projector_state_dict'] = self.embedding_projector.state_dict()
        checkpoint['dec_inp'] = self.dec_inp

    def on_load_checkpoint(self, checkpoint):
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.embedding_projector.load_state_dict(checkpoint['embedding_projector_state_dict'])
        self.dec_inp = checkpoint['dec_inp']
