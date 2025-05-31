import os
import sys
import warnings
import argparse
from argparse import Namespace
from datetime import datetime

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from speechbrain.inference.speaker import SpeakerRecognition
from torch.utils.data import DataLoader, random_split

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.Losses import DistortionLoss, FoolingLoss, ASRAttackLoss, ASRLoss

from src.Models.UAPLitASR_LibriSpeech import LitUAPGenerator

from src.perturbation_applier import PerturbationApplier
from src.LibriSpeechDatasetASR import LibriSpeechDatasetASR
from scripts._keys import WANDB_KEY

from scripts.test_uap_asr_vox_celeb import load_dataloaders, create_output_dir, setup_logger

warnings.filterwarnings("ignore")


DEVICE = [0]

cfg = Namespace(
    uap_len_sec=0.2,
    audio_len_sec=10,
    max_num_speakers=100,
    
    batch_size=128,
    epochs=1000,
    lr=1e-4,
    accumulate_grad_batches=1,
    clip_val=0.8,
    
    projection_p=2,
    target_volume=-23,
    max_num_speaker_audio=99,
    target_speaker_id=None,
    uap_rand_start=False,
    rand_offset=False,
    
    num_sanity_val_steps=1,
    check_val_every_n_epoch=5,
    asr_model_size='base',
    preload_data=True,     # False for debugging only
    
    run_name=None,
    pert_ckpt_path="/home/jovyan/karimov/anonymization/results/usual_uap/run_21-08-24_18-01-2025/uap-epoch=112-val_FR=0.76.ckpt",
    
    use_spk_uap=False,
    spk_centroids_dir="/home/jovyan/a.varlamov/voice_anonim/data/spk_centroids_ecapa_librispeech"
)

loss_weight_arr = [
    (FoolingLoss(mode='embedding cosine similarity'), 2),
    # (DistortionLoss(mode="pesq"), 1),
    # (DistortionLoss(mode='l_p'), 1000),
    # (DistortionLoss(mode='variance_lp'), 100),
    (DistortionLoss(mode='boltzmann'), 60),
    
    # (ASRAttackLoss(offset=10), 0.1),
    (ASRLoss(), 0.1)
]

def init_uap_generator(cfg) -> LitUAPGenerator:
    print(f"Loading speaker embedding model")
    spk_emb_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="./cached/ecapa_model/", 
        run_opts={"device": f"cuda:{DEVICE[0]}"}
    )
    if spk_emb_model is None:
        raise ValueError("Speaker embedding model is not loaded")

    print(f"Loading perturbation applier")
    perturbation_applier = PerturbationApplier(
        pert_len_sec=cfg.uap_len_sec,
        projection_p=cfg.projection_p,
        clip_val=cfg.clip_val,
        uap_rand_start=cfg.uap_rand_start,
        ckpt_path=cfg.pert_ckpt_path,
        use_spk_uap=cfg.use_spk_uap,
    )
    
    print(f"Loading UAP generator")
    uap_generator = LitUAPGenerator(
        asv_model=spk_emb_model,
        asr_model_size=cfg.asr_model_size,
        perturbation_applier=perturbation_applier,
        spk_centroids_dir=cfg.spk_centroids_dir,
        loss_weight_arr=loss_weight_arr,
        pert_lr=cfg.lr,
        rand_offset=cfg.rand_offset
    )
    
    return uap_generator


def main():            
    if cfg.audio_len_sec != 10:
        raise Exception("Transcriptions were prepared for 10 sec audio")
    
    uap_generator = init_uap_generator(cfg)
    
    dataset = LibriSpeechDatasetASR(
    
        transcript_path="transcriptions_librispeech.csv",
        dataset_dir="/home/jovyan/karimov/LibriSpeechTestClean/LibriSpeech/test-clean",

        audio_len_sec=cfg.audio_len_sec,
        # pad_strategy='repeat',
        pad_strategy="pad with zeros",
        
        preload=cfg.preload_data,
        
        target_volume=cfg.target_volume,
        use_parallel=True,
        spk_emb_folder=cfg.spk_centroids_dir
    )
    
    test_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    DEVICE = 0
    trainer = Trainer(devices=[DEVICE])
    trainer.test(uap_generator, test_dataloader)


if __name__ == "__main__":
    main()