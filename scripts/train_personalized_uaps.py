import os
import sys
import warnings
from argparse import Namespace

import torch
import wandb
from lightning import Trainer
from lightning.callbacks import ModelCheckpoint
from lightning.loggers import WandbLogger
from speechbrain.inference.speaker import SpeakerRecognition
from torch.utils.data import DataLoader, random_split
from ulid import ULID as ulid


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.Models.UAPLit import LitUAPGenerator
from src.VoxCelebDataset import VoxCelebDataset
from src.Losses import DistortionLoss, FoolingLoss

warnings.filterwarnings("ignore")

DEVICE = [2]

cfg = Namespace(
    uap_len_sec=0.2,
    audio_len_sec=10,
    max_num_speakers=3,
    batch_size=12,
    epochs=25,
    lr=1e-3,
    projection_p=2,
    clip_val=None,
    target_volume=-23 
)

loss_weight_arr = [
    (FoolingLoss(mode='embedding cosine similarity'), 5),
    # (DistortionLoss(mode='l_p'), 250),
    (DistortionLoss(mode='variance_lp'), 100)
]


dataset = VoxCelebDataset(
    dataset_dir="/home/jovyan/karimov/Voxceleb_dataset/",
    audio_len_sec=cfg.audio_len_sec,
    max_num_speakers=cfg.max_num_speakers,
    pad_strategy='pad with zeros',
    preload=False,
    target_volume=cfg.target_volume,
    use_parallel=True
)

spk_emb_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="./cached/ecapa_model/", 
    run_opts={"device": f"cuda:{DEVICE[0]}"}
)
if spk_emb_model is None:
    raise ValueError("Speaker embedding model is not loaded")

# wandb.login(key=)

output_dir = f"./results/personalized_uap_generation/{ulid()}"

for speaker_id in dataset.get_speaker_ids():
    speaker_paths = dataset.get_speaker_paths(speaker_id)
    
    speaker_dataset = VoxCelebDataset(
        paths=speaker_paths,
        audio_len_sec=cfg.audio_len_sec,
        pad_strategy='pad with zeros',
        preload=True,
        target_volume=cfg.target_volume,
        use_parallel=True
    )

    train_size = int(0.9 * len(speaker_dataset))
    val_size = len(speaker_dataset) - train_size

    train_dataset, val_dataset = random_split(speaker_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"=={speaker_id}==: Num train audios: {train_size}")
    print(f"=={speaker_id}==: Num val audios: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=24)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=24)

    uap_generator = LitUAPGenerator(
        asv_model=spk_emb_model,
        pert_len_sec=cfg.uap_len_sec,
        loss_weight_arr=loss_weight_arr,
        pert_lr=cfg.lr,
        projection_p=cfg.projection_p,
        clip_val=cfg.clip_val,
    )

    speaker_ckpt = os.path.join(output_dir, speaker_id)
    os.makedirs(speaker_ckpt, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=speaker_ckpt, filename="uap-{epoch:02d}-{val_FR:.2f}", save_top_k=1, monitor="val_FR", mode="max")

    # wandb_logger = WandbLogger(project="UAP_gen", log_model=False)
    # wandb_logger.log_hyperparams({"ckpt_path": output_dir})
    # wandb_logger.log_hyperparams({"loss_weight_arr": repr(loss_weight_arr)})
    # wandb_logger.log_hyperparams(cfg)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        devices=DEVICE,
        # logger=wandb_logger,
        log_every_n_steps=30,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(uap_generator, train_dataloader, val_dataloader)
