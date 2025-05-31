from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchmetrics.audio import SignalNoiseRatio
from torch_pesq import PesqLoss

from src.train_utils import repeat_perturbation
from src.VoxCelebDataset import VoxCelebDataset


@torch.no_grad()
def eval_uap_pesq(
    perturbation: torch.Tensor, 
    dataloader: torch.utils.data.DataLoader, 
    *,
    device, 
    progress_bar: bool = True,
    ) -> float:
    
    pesq_list = []
    perturbation = perturbation.to(device)

    pesq = PesqLoss(sample_rate=16_000, factor=0.5).to(device)
    
    sample_batch, _ = next(iter(dataloader))
    audio_len = sample_batch.size(1)
    sample_batch_size = sample_batch.size(0)
    
    batch_pert = repeat_perturbation(perturbation, audio_len, sample_batch_size)
    
    pb = tqdm(dataloader) if progress_bar else dataloader
    for batch, _ in pb:
        batch = batch.to(device)
        batch_size = batch.size(0)
        
        noised_batch = batch + batch_pert[:batch_size, :]
        pesq_list.append(pesq.mos(noised_batch, batch).mean().item())
    return np.mean(pesq_list)


@torch.no_grad()
def eval_uap_fr(
    perturbation: torch.Tensor, 
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    *, 
    device,
    progress_bar: bool = True
    ) -> float:
    
    perturbation = perturbation.to(device)
    successful_attack_rate = 0

    sample_batch, _ = next(iter(dataloader))
    sample_batch_size = sample_batch.size(0)
    audio_len = sample_batch.size(1)
    
    batch_pert = repeat_perturbation(perturbation, audio_len, sample_batch_size)
    
    pb = tqdm(dataloader) if progress_bar else dataloader
    for batch, ids in pb:
        batch, ids = batch.to(device), np.array(ids)
        batch_size = batch.size(0)
        
        noised_batch = batch + batch_pert[:batch_size, :]
        
        predicted_ids = model(noised_batch)[3]
        successful_attack_rate += (predicted_ids != ids).mean()

    return successful_attack_rate / len(dataloader)


@torch.no_grad()
def eval_uap_snr(
    perturbation: torch.Tensor, 
    dataloader: torch.utils.data.DataLoader, 
    *, 
    device, 
    progress_bar: bool = True
    ) -> float:
    
    SNR = SignalNoiseRatio().to(device)

    perturbation = perturbation.to(device)
    
    sample_batch, _ = next(iter(dataloader))
    sample_batch_size = sample_batch.size(0)
    audio_len = sample_batch.size(1)
    
    batch_pert = repeat_perturbation(perturbation, audio_len, sample_batch_size)
    
    snr_sum = 0
    
    pb = tqdm(dataloader) if progress_bar else dataloader
    for batch, _ in pb:
        batch = batch.to(device)
        batch_size = batch.size(0)
        
        noised_batch = batch + batch_pert[:batch_size, :]
        snr_sum += SNR(noised_batch, batch).item()
    return snr_sum / len(dataloader)


def uap_test_groupby_ids(asv_model: torch.nn.Module, test_dataset: VoxCelebDataset, uap: torch.Tensor) -> dict[str, dict[Literal["loss", "SNR", "PESQ", "FR"], float]]:
    asv_model.eval()
    snr = SignalNoiseRatio().to(asv_model.device)
    pesq = PesqLoss(sample_rate=16_000, factor=0.5)
    
    uap_res_dict = dict().fromkeys(test_dataset.get_speaker_ids())
    
    for id_ in tqdm(uap_res_dict):
        test_loss = 0
        snr_score = 0
        pesq_score = 0
        fr = 0
        
        for audio in test_dataset.get_speaker_audios(id_):
            repeated_uap = repeat_perturbation(uap, audio.size(0))
            noised_audio = audio + repeated_uap
            
            predicted_id = asv_model(noised_audio)[3][0]
            noised_spk_embeddings = asv_model.encode_batch(noised_audio).squeeze(1)
            spk_embeddings = asv_model.encode_batch(audio).squeeze(1)
            
            test_loss += ((F.cosine_similarity(noised_spk_embeddings, spk_embeddings, dim=1) + 1) / 2).mean()
            snr_score += snr(noised_audio, audio).item()
            pesq_score += pesq.mos(noised_audio, audio).mean().item()
            fr += int(predicted_id != id_)
        
        num_audios = len(list(test_dataset.get_speaker_audios(id_)))
        uap_res_dict[id_] = {
            "loss: ": test_loss / num_audios,
            "SNR": snr_score / num_audios,
            "PESQ": pesq_score / num_audios,
            "FR": fr / num_audios
        }
    return uap_res_dict
