import os
import sys
import warnings
from argparse import Namespace
from datetime import datetime

import torch
from lightning import Trainer
from lightning.callbacks import ModelCheckpoint
from lightning.loggers import WandbLogger
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
    max_num_speakers=2,
    max_num_speaker_audio=None,
    batch_size=64,
    epochs=15,
    lr=1e-3,
    projection_p=2,
    clip_val=0.01,
    target_volume=-23,
    uap_rand_start=False,
    rand_offset=False
)

loss_weight_arr = [
    (FoolingLoss(mode='embedding cosine similarity'), 1),
    (DistortionLoss(mode='l_p'), 100),
    (DistortionLoss(mode='variance_lp'), 100)
]


# TODO: mb move this functions to train utils and five them proper args instead of cfg
def load_speakers_dataloaders(cfg) -> dict[str, dict[str, DataLoader]]:
    dataset = VoxCelebDataset(
        dataset_dir="/home/jovyan/karimov/Voxceleb_dataset/",
        audio_len_sec=cfg.audio_len_sec,
        max_num_speakers=cfg.max_num_speakers,
        max_num_speaker_audio=cfg.max_num_speaker_audio,
        pad_strategy='pad with zeros',
        preload=False,
        target_volume=cfg.target_volume,
        use_parallel=True
    )
    
    speakers_dataloaders = {}
    for speaker_id in dataset.get_speaker_ids():
        speaker_paths = dataset.get_speaker_paths(speaker_id)
        
        speaker_dataset = VoxCelebDataset(
            paths=speaker_paths,
            audio_len_sec=cfg.audio_len_sec,
            max_num_speakers=cfg.max_num_speakers,
            max_num_speaker_audio=cfg.max_num_speaker_audio,
            pad_strategy='pad with zeros',
            preload=True,
            target_volume=cfg.target_volume,
            use_parallel=True
        )
        
        train_size = int(0.9 * len(speaker_dataset))
        val_size = len(speaker_dataset) - train_size
        
        train_dataset, val_dataset = random_split(speaker_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=24)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=24)
        
        speakers_dataloaders[speaker_id] = {
            "train": train_dataloader,
            "val": val_dataloader
        }
    
    return speakers_dataloaders


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
        rand_offset=cfg.rand_offset,
        target_speaker_id=cfg.target_speaker_id
    )
    return uap_generator


def create_output_dir(speaker_id: str) -> str:
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    output_dir = f"./results/individually_trained_uaps/run_{timestamp}/{speaker_id}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir created: {output_dir}")
    return output_dir


def setup_logger(output_dir: str, speaker_id: str, cfg) -> WandbLogger:
    wandb.login(key=WANDB_KEY)
    wandb_logger = WandbLogger(project="UAP_gen", log_model=False)
    wandb_logger.log_hyperparams({
        "speaker_id": speaker_id,
        "ckpt_path": output_dir,
        "loss_weight_arr": repr(loss_weight_arr)
    })
    wandb_logger.log_hyperparams(cfg)
    return wandb_logger


def main():
    speakers_dataloaders = load_speakers_dataloaders(cfg)
    
    for speaker_id, speaker_dataloaders in speakers_dataloaders.items():    
        output_dir = create_output_dir(speaker_id)
        wandb_logger = setup_logger(output_dir, speaker_id, cfg)
        uap_generator = init_uap_generator(cfg)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir, 
            filename="best_val_uap", 
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
            callbacks=[checkpoint_callback]
        )

        trainer.fit(uap_generator, speaker_dataloaders["train"], speaker_dataloaders["val"])


if __name__ == "__main__":
    main()
