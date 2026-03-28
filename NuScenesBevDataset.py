import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class NuScenesBevDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_path = Path(input_dir)
        self.gt_path = Path(gt_dir)
        # Find all input files and sort them so they match the GT files
        self.input_files = sorted(list(self.input_path.glob("*_x.npy")))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x_file = self.input_files[idx]
        token = x_file.name.replace("_x.npy", "")
        y_file = self.gt_path / f"{token}_y.npy"

        # Load the numpy data
        x_np = np.load(x_file) # (200, 200, 3)
        y_np = np.load(y_file) # (200, 200)

        # IMPORTANT: PyTorch wants (Channels, Height, Width)
        # We move the 3 channels from the end to the front
        x_tensor = torch.from_numpy(x_np).permute(2, 0, 1).float()
        
        # Ground truth needs to be a Long tensor for the Loss function
        y_tensor = torch.from_numpy(y_np).long()

        return x_tensor, y_tensor