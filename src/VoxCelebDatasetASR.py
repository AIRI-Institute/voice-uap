import math
import os
import re
import warnings
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Final, Iterator, Literal, Optional

import librosa
import pandas as pd
import pyloudnorm as pyln
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_audio_tensor(file_path: str | os.PathLike) -> torch.Tensor:
    audio, sr = librosa.load(file_path, sr=None)
    if sr != 16_000:
        raise ValueError(f"Sample rate {sr} Hz is not supported. Only 16000 Hz (16kHz) is supported.")
    return torch.from_numpy(audio)


class VoxCelebDatasetASR(Dataset):
    __RATE: Final[int] = 16_000
    
    def __init__(
        self,
        transcript_path: str,
        dataset_dir: Optional[str] = None,
        paths: Optional[list[str]] = None,
        *,
        audio_len_sec: int = 3,
        pad_strategy: Literal["pad with zeros", "repeat"] | None = None,
        preload: bool = False,
        max_num_speakers: Optional[int] = None,
        max_num_speaker_audio: Optional[int] = None,
        rand_augment_len_sec: Optional[int] = None,
        target_volume: float | None = None,
        use_parallel: bool = True,
        spk_emb_folder: str | None = None,
    ):
        self.dataset_dir = dataset_dir
        self._check_path_arguments(dataset_dir, paths)
        
        self.audio_len_sec = audio_len_sec
        self.pad_strategy = pad_strategy
        self.rand_augment_len_sec = rand_augment_len_sec
        self._check_audio_len_args()
        
        if paths and (max_num_speakers or max_num_speaker_audio):
            warnings.warn("`max_num_speakers` and `max_num_speaker_audio` are ignored when `paths` is not None")        
        self.max_num_speakers = max_num_speakers
        self.max_num_speaker_audio = max_num_speaker_audio
        
        self.preload = preload
        self.use_parallel = use_parallel    
        
        self.target_volume = target_volume
        if self.target_volume is not None:
            self.meter = pyln.Meter(self.__RATE)
        
        if self.rand_augment_len_sec is not None:
            self.rand_augment_len = self.__RATE * self.rand_augment_len_sec
        
        self.paths = paths or self._parse_paths()
        self._sort_paths_by_speakers()
        
        self.transcript_df = pd.read_csv(transcript_path)
        
        if preload: 
            self.preload_audios()
            
        self.spk_emb_folder = spk_emb_folder

    def _check_path_arguments(self, dataset_dir, paths) -> None:
        if (dataset_dir is None) == (paths is None):
            raise ValueError("Either 'dataset_dir' or 'paths' must be provided, but not both.")
    
    def _parse_paths(self) -> list[str]:
        if self.dataset_dir is not None:
            speaker_folders = glob(os.path.join(self.dataset_dir, 'wav', '*'))
            speaker_folders = speaker_folders[: self.max_num_speakers] if self.max_num_speakers else speaker_folders
            
            paths = []
            for speaker_path in speaker_folders:
                speaker_audios = glob(os.path.join(speaker_path, "**/*.*"), recursive=True)
                speaker_audios = speaker_audios[:self.max_num_speaker_audio] if self.max_num_speaker_audio else speaker_audios
                paths.extend(speaker_audios)    
            return paths
        else:
            raise ValueError("'paths' must be provided when 'dataset_dir' is None")

    def _sort_paths_by_speakers(self):
        def extract_id(parts):
            for part in parts:
                if part.startswith('id') and part[2:].isdigit():
                    return part
            return None
        df = pd.DataFrame({'path': self.paths})
        df['path_parts'] = df['path'].str.split(os.sep)
        df['id'] = df['path_parts'].apply(extract_id)
        self.speakers_dict = df.groupby('id')['path'].apply(list).to_dict()
            
    def preload_audios(self):
        self.audios = []
        self.ids = []

        if self.use_parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    tqdm(
                        executor.map(self.load_and_process_audio, self.paths),
                        total=len(self.paths),
                        desc="Preloading audios",
                    )
                )
        else:
            results = [self.load_and_process_audio(path) for path in tqdm(self.paths)]
        
        for audio, audio_id in results:
            self.audios.append(audio)
            self.ids.append(audio_id)

    def load_and_process_audio(self, path: str | os.PathLike) -> tuple[torch.Tensor, str]:
        audio = load_audio_tensor(path)
        audio = self.pad_audio(audio)
        audio = self.normalize_audio(audio)
        
        match = re.search(r"id\d+", path) # type: ignore
        if match is not None:
            audio_id = match.group()
        else:
            raise RuntimeError(f"Cannot extract speaker id from path '{path}'")
        
        return audio, audio_id

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        init_len = audio.size(0)
        len_to_add = self.__RATE * self.audio_len_sec - init_len
        if len_to_add > 0:
            match self.pad_strategy:
                case 'pad with zeros':
                    padded_audio = F.pad(audio, (0, len_to_add))
                case 'repeat':
                    times_to_repeat = math.ceil(self.__RATE * self.audio_len_sec / init_len)
                    padded_audio = audio.repeat(times_to_repeat)[:self.__RATE * self.audio_len_sec]
                case None:
                    padded_audio = audio[:self.__RATE * self.audio_len_sec]
                case _:
                    raise NotImplementedError
        else:
            padded_audio = audio[: self.__RATE * self.audio_len_sec]
        
        return padded_audio
    
    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self.target_volume is not None:
            audio = audio.numpy()
            audio_volume = self.meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, audio_volume, self.target_volume)
            audio = torch.from_numpy(audio)
        return audio
    
    def rand_augment(self, audio: torch.Tensor) -> torch.Tensor: 
        init_len = audio.size(0)
        start_idx = torch.randint(0, init_len - self.rand_augment_len + 1, (1,)).item()
        augmented_audio = audio[start_idx:start_idx + self.rand_augment_len]
        return augmented_audio
    
    def _check_audio_len_args(self):
        if (self.rand_augment_len_sec is not None) and (self.rand_augment_len_sec > self.audio_len_sec):
            raise ValueError(f"Random augmentation '{self.rand_augment_len_sec}' must be less then audio len '{self.audio_len_sec}'")
    
    def __len__(self):
        return len(self.paths)
    
    def get_num_speakers(self):
        return len(self.speakers_dict) 
    
    def get_speaker_ids(self):
        return list(self.speakers_dict.keys())
    
    def get_speaker_paths(self, speaker_id: str) -> list[str]:
        return self.speakers_dict[speaker_id]
    
    def get_speaker_audios(self, speaker_id: str) -> Iterator[torch.Tensor]:
        if self.preload:
            #TODO: add preload support
            speaker_paths = self.speakers_dict[speaker_id]
            speaker_audios = (self.load_and_process_audio(path)[0] for path in speaker_paths)
            if self.rand_augment_len_sec:
                return (self.rand_augment(audio) for audio in speaker_audios)
            else:
                return speaker_audios
        else:
            speaker_paths = self.speakers_dict[speaker_id]
            speaker_audios = (self.load_and_process_audio(path)[0] for path in speaker_paths)
            if self.rand_augment_len_sec:
                return (self.rand_augment(audio) for audio in speaker_audios)
            else:
                return speaker_audios
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        if self.preload:
            audio, audio_id = self.audios[idx], self.ids[idx]
        else:
            audio, audio_id = self.load_and_process_audio(self.paths[idx])
        if self.rand_augment_len_sec:
            audio = self.rand_augment(audio)
            
        wav_rel_path = self.paths[idx].replace(self.dataset_dir + "/wav/", "")
        transcript = self.transcript_df[self.transcript_df["wav_path"] == wav_rel_path]["transcription"].item()
        
        spk_emb_path = os.path.join(self.spk_emb_folder, f"spk_cent_{audio_id}.pt")
        spk_emb = torch.load(spk_emb_path, weights_only=True, map_location="cpu")
        
        return audio, audio_id, transcript, spk_emb
