import math
import random
import warnings
from typing import Union

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F

def get_streamvc_decoder_input_shape(target_output_shape: int | float) -> int:
    if target_output_shape % 320 != 0:
        raise ValueError("Target output shape must be divisible by 320")
    return int(target_output_shape // 320)


def repeat_perturbation(
    perturbation: torch.Tensor, target_len: int, target_batch_size: int = 1
) -> torch.Tensor:
    if perturbation.dim() == 1:
        squeeze_result = True
        perturbation = perturbation.unsqueeze(0)
    elif perturbation.dim() == 2:
        squeeze_result = False
        if perturbation.size(0) != 1 and target_batch_size == perturbation.size(0):
            raise ValueError(
                f"'target_batch_size'({target_batch_size}) and number of perturbation batch's({perturbation.size(0)}) must be the same"
            )
    else:
        raise ValueError(
            f"Perturbation should have 1 or 2 dims, but has {perturbation.dim()} dims with shape {perturbation.shape}"
        )

    pert_len = perturbation.size(1)

    times_to_repeat_pert = math.ceil(target_len / pert_len)
    batch_pert = perturbation.repeat(target_batch_size, times_to_repeat_pert)
    batch_pert = batch_pert[:, :target_len]
    return batch_pert.squeeze() if squeeze_result else batch_pert


@torch.inference_mode()
def eval_speaker_centroids(dataset: torch.utils.data.Dataset, model: torch.nn.Module, num_enrolls: int, device: str = "cpu"):
    if not isinstance(dataset, torch.utils.data.Dataset):
        raise TypeError("dataset must be an instance of torch.utils.data.Dataset")
    
    speaker_audios = {}
    for audio, speaker_id in dataset:
        if speaker_id not in speaker_audios:
            speaker_audios[speaker_id] = []
        speaker_audios[speaker_id].append(audio)
    
    centroids = {}
    
    for id_, audios in speaker_audios.items():
        if len(audios) < num_enrolls:
            warnings.warn(f"Speaker {id_} have less audios ({len(audios)}) then 'num_enrolls'")
        
        rand_idx = np.random.choice(range(len(audios)), num_enrolls, replace=False)
        rand_audio_subset = torch.stack([audios[i] for i in rand_idx])
        rand_audio_subset = rand_audio_subset.to(device)
        embs = model.encode_batch(rand_audio_subset).squeeze(1)
        centroids[id_] = embs.mean(dim=0)
    return centroids


def prepare_batch_signs_for_target_speaker(ids: Union[list[str], tuple[str], np.ndarray], target_speaker_id: str) -> torch.Tensor:
    if isinstance(ids, list) or isinstance(ids, tuple):
        ids = np.array(ids)

    coeff = 2 # govnocode, sorry
    sing_array = (ids == target_speaker_id).astype(int) * coeff - (coeff - 1)
    return torch.from_numpy(sing_array).reshape(-1, 1)


def add_rand_offset(batch: torch.Tensor, rate: int = 16000) -> torch.Tensor:
    offset = random.randint(0, rate)
    
    padded_batch = F.pad(
        batch,
        pad=(0, rate),
        mode='constant',
        value=0.0
    )
    return padded_batch.roll(offset, dims=1)
