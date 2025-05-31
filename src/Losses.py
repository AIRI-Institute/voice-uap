import inspect
from typing import Literal
from functools import wraps

import torch
import torchaudio
import torch.nn.functional as F
from torch_pesq import PesqLoss

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline

def expanded_repr(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        sig = inspect.signature(original_init)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        self._init_args = {k: v for k, v in bound_args.arguments.items() if k != 'self'}
        original_init(self, *args, **kwargs)
    
    def new_repr(self):
        args = ', '.join(f"{key}={repr(value)}" for key, value in self._init_args.items())
        return f"{self.__class__.__name__}({args})"

    cls.__init__ = new_init
    cls.__repr__ = new_repr
    return cls


def MFCC(audio, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40, sample_rate=16000):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels},
    ).to(audio.device)

    return mfcc_transform(audio)
@expanded_repr
class FoolingLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        mode: Literal['logits loss', 'logits loss rand', 'embedding cosine similarity'],
        threshold: float | None = None
    ):
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        
        self.mode_to_method = {
            'logits loss': self.logits_loss,
            'logits loss rand': self.logits_loss_rand,
            'embedding cosine similarity': self.embedding_cosine_similarity_loss
        }
        
        if self.mode not in self.mode_to_method:
            raise ValueError(f"`{self.mode}` isn't a valid mode")

    def forward(self, noised_emb, orig_emb, logits, ids, **kwargs):
        if self.mode == 'embedding cosine similarity':
            outputs = noised_emb
            targets = orig_emb
        else: 
            outputs = logits
            targets = ids
        loss_fn = self.mode_to_method[self.mode]
        return loss_fn(outputs, targets)

    def logits_loss(self, logits, target_label):
        ids_tensor = torch.as_tensor(target_label, device=logits.device).unsqueeze(1)
        target_logits = logits.gather(1, ids_tensor).squeeze()
        
        assert (
            target_logits.shape == torch.Size([logits.shape[0]])
        ), "target_logits doesn't match batch size"
        
        logits_masked = torch.scatter(logits, 1, ids_tensor, -torch.inf)
        
        max_non_target_logits = logits_masked.max(dim=1)[0]
        loss = -(max_non_target_logits - target_logits)

        if self.threshold is not None:
            loss.clamp_(min=-self.threshold, max=None)
            loss += self.threshold
        return loss.mean()
    
    def logits_loss_rand(self, logits, target_label):
        ids_tensor = torch.as_tensor(target_label, device=logits.device).unsqueeze(1)
        target_logits = logits.gather(1, ids_tensor).squeeze()
        
        assert (
            target_logits.shape == torch.Size([logits.shape[0]])
        ), "target_logits doesn't match batch size"
        
        batch_size, num_classes = logits.shape
        
        mask = torch.arange(num_classes, device=logits.device).repeat(batch_size, 1) != ids_tensor
        non_target_logits = logits[mask].view(batch_size, num_classes-1)
        
        rand_idx = torch.randint(0, non_target_logits.size(1), size=(non_target_logits.size(0), 1), device=logits.device)
        rand_non_target_logits = non_target_logits.gather(1, rand_idx)
        
        loss = -(rand_non_target_logits - target_logits)
        
        if self.threshold is not None:
            loss.clamp_(min=-self.threshold, max=None)
        return loss.mean()
    
    def embedding_cosine_similarity_loss(self, output_emb, target_emb):
        margin = self.threshold if self.threshold is not None else 0
        loss = F.cosine_embedding_loss(output_emb, target_emb, -torch.ones(output_emb.size(0), device=output_emb.device), margin=margin)
        return loss


@expanded_repr
class DistortionLoss(torch.nn.Module):
    def __init__(self, mode: Literal['l_p', 'mfcc_lp', 'variance_lp', 'boltzmann', 'pesq'], p: int = 2, **kwargs):
        super().__init__()
        self.mode = mode
        self.p = p
        self.kwargs = kwargs

        self.mode_to_method = {
            'l_p': self.l_p_loss,
            'mfcc_lp': self.mfcc_lp_loss,
            'variance_lp': self.variance_lp_loss,
            'boltzmann': self.boltzmann_potential_loss,
            'pesq': self.pesq_loss
        }
        
        self.pesq = PesqLoss(factor=0.5, sample_rate=16_000)
        
        if self.mode not in self.mode_to_method:
            raise ValueError(f"'{self.mode}' is't valid mode")
        
    def forward(self, perturbation, **kwargs):
        if perturbation.dim() == 2:
            pass  # perturbation is already (Batch, Length)
        elif perturbation.dim() == 1:
            perturbation = perturbation.unsqueeze(0)  # Convert to (1, Length)
        else:
            raise ValueError("perturbation tensor must be 1D or 2D")
        
        loss_fn = self.mode_to_method[self.mode]
        
        if self.mode == 'pesq':
            return loss_fn(kwargs["noised_audio"], kwargs["audio"])
        else:
            return loss_fn(perturbation)
        
    def l_p_loss(self, noise: torch.Tensor) -> torch.Tensor:
        return noise.abs().norm(p=self.p, dim=-1).mean() / noise.size(-1)
    
    def mfcc_lp_loss(self, noise: torch.Tensor) -> torch.Tensor:
        spectre = MFCC(noise, **self.kwargs)
        loss = spectre.abs().norm(dim=-2, p=self.p).mean() / spectre.size(-1)
        return loss
    
    def variance_lp_loss(self, noise: torch.Tensor) -> torch.Tensor:
        detached_noise = noise.clone().detach()
        diff_forward = (noise - detached_noise.roll(-1, dims=-1)).abs().pow(self.p)
        diff_backward = (noise - detached_noise.roll(1, dims=-1)).abs().pow(self.p)
        variance = (diff_forward + diff_backward) / 2
        return variance.mean(dim=-1).pow(1/self.p).mean()
    
    def boltzmann_potential_loss(self, noise: torch.Tensor) -> torch.Tensor:
        detached_noise = noise.clone().detach()
        forward_detached_noise_vals = detached_noise.roll(1, dims=1)
        
        mask_type_1 = (forward_detached_noise_vals.abs() > detached_noise.abs()).float()
        mask_type_2 = (forward_detached_noise_vals > 0).float() * (detached_noise < 0).float()
        mask_type_3 = (forward_detached_noise_vals < 0).float() * (detached_noise > 0).float()
        
        forward_noise_vals = noise.roll(1, dims=1)
        penalty_type_1 = forward_noise_vals.abs() - detached_noise.abs()
        penalty_type_2 = forward_noise_vals
        penalty_type_3 = forward_noise_vals.abs()
        
        return ((penalty_type_1.exp() - 1) * mask_type_1 
                + (penalty_type_2.exp() - 1) * mask_type_2 
                + (penalty_type_3.exp() - 1) * mask_type_3).mean()
    
    def pesq_loss(self, noised_audio: torch.Tensor, audio: torch.Tensor):        
        self.pesq = self.pesq.to(noised_audio.device)
        return self.pesq(audio, noised_audio).mean()


@expanded_repr
class ASRAttackLoss(torch.nn.Module):
    def __init__(self, offset=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = ""
        self.offset = offset
    def forward(self, asr_model, asr_processor, noised_audio: torch.tensor, transcripts: list, **kwargs):
        batch = [elem.detach().cpu().numpy() for elem in noised_audio]
        
        audio_features = asr_processor(batch, sampling_rate=16000, return_tensors="pt").input_features
        audio_features = audio_features.to(torch.half).to(noised_audio.device)
        
        text_tokens = asr_processor.tokenizer(transcripts, return_tensors="pt", padding=True).input_ids
        text_tokens = text_tokens.to(noised_audio.device)
        
        outputs = asr_model(audio_features, labels=torch.tensor(text_tokens))
        loss = outputs.loss
        
        return self.offset - loss
    
@expanded_repr
class ASRLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = ""

    def forward(self, asr_model, asr_processor, noised_audio: torch.tensor, transcripts: list, **kwargs):
        batch = [elem.detach().cpu().numpy() for elem in noised_audio]
        
        audio_features = asr_processor(batch, sampling_rate=16000, return_tensors="pt").input_features
        audio_features = audio_features.to(torch.half).to(noised_audio.device)
        
        text_tokens = asr_processor.tokenizer(transcripts, return_tensors="pt", padding=True).input_ids
        text_tokens = text_tokens.to(noised_audio.device)
        
        outputs = asr_model(audio_features, labels=torch.tensor(text_tokens))
        loss = outputs.loss
        
        return loss