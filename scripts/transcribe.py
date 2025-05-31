import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, wav_folder, max_duration=10.0):
        self.wav_folder = wav_folder
        self.max_duration = max_duration  # Maximum duration in seconds
        self.file_list = []
        for root, _, files in tqdm(os.walk(wav_folder), desc="Dataset Init"):
            for file in files:
                if file.endswith(".wav") or file.endswith(".flac"):
                    wav_path = os.path.join(root, file)
                    rel_path = os.path.relpath(wav_path, wav_folder)
                    self.file_list.append((rel_path, wav_path))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path, wav_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Trim the waveform to 10 seconds
        max_frames = int(self.max_duration * sample_rate)
        if waveform.size(1) > max_frames:
            waveform = waveform[:, :max_frames]  # Trim to 10 seconds
        
        return rel_path, waveform.squeeze(), sample_rate

def collate_fn(batch):
    rel_paths, waveforms, sample_rates = zip(*batch)
    max_len = max(waveform.size(0) for waveform in waveforms)
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :waveform.size(0)] = waveform.squeeze(0)
        
    return rel_paths, padded_waveforms, sample_rates[0]

def transcribe_wavs(wav_folder, output_csv, batch_size=32):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

    print("Creating dataset:")
    dataset = AudioDataset(wav_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    print("Initialized dataset")

    # Write the CSV header once
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["wav_path", "transcription"])

    for rel_paths, waveforms, sample_rate in tqdm(dataloader, desc="Transcribing"):
        waveforms = [wave.cpu().numpy() for wave in waveforms]
        
        results = transcriber(waveforms, batch_size=len(waveforms), generate_kwargs={"language": "english"})  # Batched inference

        transcriptions = [[rel_path, result["text"]] for rel_path, result in zip(rel_paths, results)]
        
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(transcriptions)

if __name__ == "__main__":
    # path = "/home/jovyan/karimov/Voxceleb_dataset/wav"
    # transcribe_wavs(path, "transcriptions.csv", batch_size=64)

    path = "/home/jovyan/karimov/LibriSpeechTestClean/LibriSpeech/test-clean"
    transcribe_wavs(path, "transcriptions_librispeech.csv", batch_size=64)