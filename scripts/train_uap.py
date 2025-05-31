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

from src.Losses import DistortionLoss, FoolingLoss
from src.Models.UAPLit import LitUAPGenerator
from src.perturbation_applier import PerturbationApplier
from src.VoxCelebDataset import VoxCelebDataset
from scripts._keys import WANDB_KEY

warnings.filterwarnings("ignore")


DEVICE = [0]

cfg = Namespace(
    uap_len_sec=0.2,
    audio_len_sec=10,
    max_num_speakers=100,
    batch_size=128,
    epochs=1000,
    lr=3e-3,
    projection_p=2,
    clip_val=0.01,
    target_volume=-23,
    max_num_speaker_audio=99,
    target_speaker_id=None,
    uap_rand_start=False,
    rand_offset=False,
    
    check_val_every_n_epoch=1,
)

loss_weight_arr = [
    (FoolingLoss(mode='embedding cosine similarity'), 3),
    # (DistortionLoss(mode="pesq"), 1),
    # (DistortionLoss(mode='l_p'), 100),
    # (DistortionLoss(mode='variance_lp'), 100),
    (DistortionLoss(mode='boltzmann'), 40)
]

def load_dataloaders(cfg) -> dict[str, DataLoader]:
    dataset = VoxCelebDataset(
        dataset_dir="/home/jovyan/karimov/Voxceleb_dataset",
        audio_len_sec=cfg.audio_len_sec,
        max_num_speakers=cfg.max_num_speakers,
        max_num_speaker_audio=cfg.max_num_speaker_audio,
        pad_strategy='repeat',
        preload=True,
        target_volume=cfg.target_volume,
        use_parallel=True
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Num train audios: {train_size}")
    print(f"Num val audios: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=24)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=24)
    
    return {
        "train": train_dataloader,
        "val": val_dataloader
    }


def init_uap_generator(cfg) -> LitUAPGenerator:
    spk_emb_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="./cached/ecapa_model/", 
        run_opts={"device": f"cuda:{DEVICE[0]}"}
    )
    if spk_emb_model is None:
        raise ValueError("Speaker embedding model is not loaded")

    perturbation_applier = PerturbationApplier(
        pert_len_sec=cfg.uap_len_sec,
        projection_p=cfg.projection_p,
        clip_val=cfg.clip_val,
        uap_rand_start=cfg.uap_rand_start
    )
    
    uap_generator = LitUAPGenerator(
        asv_model=spk_emb_model,
        perturbation_applier=perturbation_applier,
        loss_weight_arr=loss_weight_arr,
        pert_lr=cfg.lr,
        rand_offset=cfg.rand_offset
    )
    return uap_generator


def create_output_dir() -> str:
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    output_dir = f"./results/usual_uap/run_{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir created: {output_dir}")
    return output_dir


def setup_logger(output_dir: str, cfg) -> WandbLogger:
    wandb.login(key=WANDB_KEY)
    wandb_logger = WandbLogger(project="UAP_gen", log_model=False)
    wandb_logger.log_hyperparams({
        "ckpt_path": output_dir,
        "loss_weight_arr": repr(loss_weight_arr)
    })
    wandb_logger.log_hyperparams(cfg)
    return wandb_logger


def main():    
    dataloaders = load_dataloaders(cfg)
    uap_generator = init_uap_generator(cfg)
    output_dir = create_output_dir()
    wandb_logger = setup_logger(output_dir, cfg)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir, 
        filename="uap-{epoch:02d}-{val_FR:.2f}", 
        save_top_k=1, 
        monitor="val_FR", 
        mode="max", 
        save_last=True
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        devices=DEVICE,
        logger=wandb_logger,
        log_every_n_steps=30,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=64,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(uap_generator, dataloaders["train"], dataloaders["val"])


if __name__ == "__main__":
    main()
