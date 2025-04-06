import torch
import os
import librosa
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir, sr=16000, n_mfcc=40):
        self.clean_files = sorted([f for f in os.listdir(data_dir) if f.startswith("clean_")])
        self.noisy_files = sorted([f for f in os.listdir(data_dir) if f.startswith("noisy_")])
        self.data_dir = data_dir
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.data_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.data_dir, self.noisy_files[idx])

        clean, _ = librosa.load(clean_path, sr=self.sr)
        noisy, _ = librosa.load(noisy_path, sr=self.sr)

        clean_mfcc = librosa.feature.mfcc(y=clean, sr=self.sr, n_mfcc=self.n_mfcc)
        noisy_mfcc = librosa.feature.mfcc(y=noisy, sr=self.sr, n_mfcc=self.n_mfcc)

        # (n_mfcc, time) → (1, n_mfcc, time)
        return torch.tensor(noisy_mfcc).unsqueeze(0).float(), torch.tensor(clean_mfcc).unsqueeze(0).float()
