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

from src.Models.PLStreamVC import PLStreamVCModel
from src.VoxCelebDataset import VoxCelebDataset
from scripts._keys import WANDB_KEY

warnings.filterwarnings("ignore")

DEVICE = [1]

cfg = Namespace(
    uap_len_sec=0.2,
    audio_len_sec=10,
    max_num_speakers=900,
    batch_size=128,
    epochs=1000,
    lr=1e-4,
    target_volume=-23,
    decoder_embedding_dim=64,
    spk_emb_model_dim=192,
    projector_hidden_state_dim=128,
)


dataset = VoxCelebDataset(
    dataset_dir="/home/jovyan/karimov/Voxceleb_dataset/",
    audio_len_sec=cfg.audio_len_sec,
    max_num_speakers=cfg.max_num_speakers,
    pad_strategy='pad with zeros',
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


spk_emb_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="./cached/ecapa_model/", 
    run_opts={"device": f"cuda:{DEVICE[0]}"}
)

embedding_projector = torch.nn.Sequential(
    torch.nn.Linear(cfg.spk_emb_model_dim, cfg.projector_hidden_state_dim),
    torch.nn.GELU(),
    torch.nn.Linear(cfg.projector_hidden_state_dim, cfg.decoder_embedding_dim)
)

model = PLStreamVCModel(
    embedding_projector=embedding_projector, 
    spk_emb_model=spk_emb_model,
    decoder_embeddings_dim=cfg.decoder_embedding_dim, 
    uap_len_sec=cfg.uap_len_sec, 
    learning_rate=cfg.lr
)

output_dir = f"./results/StreamVC/{ulid()}"
os.makedirs(output_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="streamvc-{epoch:02d}-{val_FR:.2f}", save_top_k=1, monitor="train_loss", mode="min")

wandb.login(key=WANDB_KEY)
wandb_logger = WandbLogger(project="StreamVC_decoder_training", log_model="all")
wandb_logger.log_hyperparams({"ckpt_path": output_dir})
wandb_logger.log_hyperparams(cfg)

trainer = Trainer(
    max_epochs=cfg.epochs,
    devices=DEVICE,
    logger=wandb_logger,
    log_every_n_steps=30,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, train_dataloader, val_dataloader)
