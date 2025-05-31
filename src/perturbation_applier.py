from typing import Literal

import torch
import torch.nn as nn

from src.train_utils import repeat_perturbation

from src.Models.spk_emb_mlp import MLP

InitPertDistribution = Literal["uniform", "normal"]  # TODO: replace with enum


class PerturbationApplier(torch.nn.Module):
    def __init__(
        self,
        pert_len_sec: float,
        init_pert_amp: int | float = 1e-5,
        init_pert_distribution: InitPertDistribution = "uniform",
        uap_rand_start: bool = False,
        projection_p: int | None = None,
        clip_val: float | None = None,
        device: torch.device | str | None = None,
        rate: int = 16000,
        ckpt_path: str | None = None,
        use_spk_uap=False,
        use_timestep_uap=False,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cpu")
        self.rate = rate
        self.pert_len = int(pert_len_sec * rate)

        self.init_pert_amp = init_pert_amp
        self.init_pert_distribution = init_pert_distribution
        self.projection_p = projection_p
        self.clip_val = clip_val
        self.uap_rand_start = uap_rand_start
        
        if ckpt_path is None:
            self._initialize_perturbation()
        else:
            print(f"Loading perturbation from {ckpt_path}")
            pert = torch.load(ckpt_path, map_location="cpu", weights_only=True)['state_dict']["pert_applier.perturbation"]
            pert = torch.nn.Parameter(pert)
            self.perturbation = pert
            
        self.use_spk_uap = use_spk_uap
        if self.use_spk_uap:
            self.spk_emb_mlp = MLP().to(self.device)
        
        self.use_timestep_uap = use_timestep_uap
        if self.use_timestep_uap:
            self.proj_timestep_uap = nn.Linear(64, 3200)

    def _initialize_perturbation(self) -> None:
        match self.init_pert_distribution:
            case "uniform":
                perturbation = torch.rand(self.pert_len, device=self.device)
            case "normal":
                perturbation = torch.randn(self.pert_len, device=self.device)
            case _:
                raise ValueError(
                    f"'{self.init_pert_distribution}' unsupported type of distribution"
                )

        self.perturbation = torch.nn.Parameter(
            self.init_pert_amp * perturbation, requires_grad=True
        )

    def project(self) -> None:
        self._project_norm()
        self._clip_values()

    def _project_norm(self) -> None:
        if self.projection_p is not None and self.perturbation.norm(self.projection_p) > 1.0:
            self.perturbation.data = torch.nn.functional.normalize(
                self.perturbation.data.unsqueeze(0), p=self.projection_p
            ).squeeze()

    def _clip_values(self) -> None:
        if self.clip_val is not None:
            self.perturbation.data.clamp_(-self.clip_val, self.clip_val)

    def _fit_uap_to_batch(self, batch: torch.Tensor) -> torch.Tensor: #TODO: test this function
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        
        batch_size = batch.size(0)
        audio_len = batch.size(1)
        
        if audio_len < self.pert_len:
            raise ValueError(
                f"audio_len ({audio_len}) must be >= pert_len ({self.pert_len})"
            )
        
        if self.uap_rand_start:
            start_skip_len = torch.randint(0, self.pert_len, (batch_size,))
            repeated_pert = repeat_perturbation(self.perturbation.unsqueeze(0), self.pert_len + audio_len, batch_size)
            batch_pert = torch.stack(
                [repeated_pert[i, start_skip_len[i]:start_skip_len[i]+audio_len] for i in range(batch_size)]
            )
            return batch_pert
        else:
            batch_pert = repeat_perturbation(self.perturbation, audio_len, batch_size)
            return batch_pert
    
    # def _add_rand_offset(self, pert_batch: torch.Tensor) -> torch.Tensor:
    #     offset = random.randint(0, self.rate)
    #     # start_offset_len = torch.randint(0, self.rate, (batch_size,), device=pert_batch.device)  # max offset len is 1 second
        
    #     # Apply padding to entire batch at once
    #     padded_batch = F.pad(
    #         pert_batch,
    #         pad=(0, self.rate),  # Pad right side with max possible offset
    #         mode='constant',
    #         value=0.0
    #     )
        
    #     padded_batch = padded_batch.roll(offset, dims=1)
    #     return padded_batch    

    def forward(self, batch: torch.Tensor, spk_emb=None) -> torch.Tensor:
        if batch.size(-1) < self.pert_len:
            raise ValueError(f"audio_len ({batch.size(1)}) must be >= pert_len ({self.pert_len})")
        
        self.project()
        fitted_pert = self._fit_uap_to_batch(batch)
        
        if spk_emb is not None and self.use_spk_uap:
            spk_uap = self.spk_emb_mlp(spk_emb / 1000)
            spk_uap = spk_uap.clamp_(-self.clip_val, self.clip_val)
            # print(spk_uap.shape)
            fitted_spk_uap = self._fit_uap_to_batch(batch)
            # print(fitted_spk_uap.shape)
            fitted_pert = fitted_pert + fitted_spk_uap
            # print(fitted_spk_uap)
        
        pert_batch = batch + fitted_pert
        # TODO: maybe add some augmentations
        return pert_batch
    
    @property
    def uap(self) -> torch.Tensor:
        self.project()
        return self.perturbation.data.detach().cpu()
