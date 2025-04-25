# Torch Datasets
from torch.utils.data import Dataset
import torch

# Python Builtin Libraries
import os 

# Data Wrangling Libraries
import librosa
import numpy as np


class AudioDataSet(Dataset):
    def __init__(self, dataframe, base_dir, extractor_fn, sr=22050):
        self.data = dataframe
        self.base_dir = base_dir
        self.extractor_fn = extractor_fn  # Pass function instead of extractor object
        self.sr = sr
        self.corrupted = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        folder = str(row["folder"])
        path = row["path"]
        file_path = os.path.join(self.base_dir, folder, path)

        try:
            waveform, sample_rate = librosa.load(file_path, sr=self.sr)

            # Use the passed extractor function
            features = self.extractor_fn(waveform, sample_rate)

            label = row["label"]
            return features, label

        except Exception as e:
            self.corrupted.append(file_path)
            return None

def create_balanced_subset(df, samples_per_class=100, label = "label"):
    return df.groupby(label).sample(n=samples_per_class, replace=False)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  
    if not batch:
        return None, None
    features, labels = zip(*batch)
    return torch.tensor(features), torch.tensor(labels)
