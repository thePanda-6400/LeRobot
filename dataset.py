from pathlib import Path

import pandas as pd
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

SCALE = 100.0   # scale tiny metre-labels up so the loss has a usable gradient


class GraspDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels = pd.read_csv(self.data_dir / "labels.csv")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        # --- image ---
        img_path = self.images_dir / row["filename"]
        image = imageio.imread(img_path)                 # (128, 128, 3), uint8, 0-255
        image = image.astype(np.float32) / 255.0         # -> float, 0-1
        image = torch.from_numpy(image)                  # to tensor
        image = image.permute(2, 0, 1)                   # (H,W,C) -> (C,H,W)

        # --- label ---     
        position = row[["x", "y", "z"]].to_numpy(dtype=np.float32)
        position = position * SCALE                      # scale up before tensor
        position = torch.from_numpy(position)            # shape (3,)


        return image, position

    