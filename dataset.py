import os
import cv2
import random
import torch
from torch.utils.data import Dataset
import numpy as np

class SpiderManDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, lr_patch_size=64, scale=2):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.lr_patch_size = lr_patch_size
        self.scale = scale
        self.hr_patch_size = lr_patch_size * scale
        
        # Get sorted list of all frame filenames
        self.frames = sorted(os.listdir(lr_dir))
        
    def __len__(self):
        # We subtract 1 because we always need a "previous" frame
        return len(self.frames) - 1

    def __getitem__(self, idx):
        # idx + 1 represents the "Current" time (t)
        # idx represents the "Previous" time (t-1)
        
        current_name = self.frames[idx + 1]
        prev_name = self.frames[idx]

        # Load Current Low-Res (LR_t)
        lr_t = cv2.imread(os.path.join(self.lr_dir, current_name))
        lr_t = cv2.cvtColor(lr_t, cv2.COLOR_BGR2RGB)

        # Load Current High-Res (HR_t) - This is the "Answer Key"
        hr_t = cv2.imread(os.path.join(self.hr_dir, current_name))
        hr_t = cv2.cvtColor(hr_t, cv2.COLOR_BGR2RGB)

        # Load Previous High-Res (HR_t-1) - This acts as the temporal memory
        hr_prev = cv2.imread(os.path.join(self.hr_dir, prev_name))
        hr_prev = cv2.cvtColor(hr_prev, cv2.COLOR_BGR2RGB)

        # --- RANDOM CROPPING FOR VRAM SAVINGS ---
        h, w, _ = lr_t.shape
        # Pick a random starting point for the Low-Res crop
        x = random.randint(0, w - self.lr_patch_size)
        y = random.randint(0, h - self.lr_patch_size)

        # Calculate the exact matching starting point for the High-Res crop (x2 scale)
        hx, hy = x * self.scale, y * self.scale

        # Crop all three images at the exact same location
        lr_t_crop = lr_t[y : y + self.lr_patch_size, x : x + self.lr_patch_size]
        hr_t_crop = hr_t[hy : hy + self.hr_patch_size, hx : hx + self.hr_patch_size]
        hr_prev_crop = hr_prev[hy : hy + self.hr_patch_size, hx : hx + self.hr_patch_size]

        # Convert to PyTorch Tensors and normalize (0 to 1)
        # PyTorch expects shape [Channels, Height, Width]
        lr_t_tensor = torch.from_numpy(lr_t_crop).permute(2, 0, 1).float() / 255.0
        hr_t_tensor = torch.from_numpy(hr_t_crop).permute(2, 0, 1).float() / 255.0
        hr_prev_tensor = torch.from_numpy(hr_prev_crop).permute(2, 0, 1).float() / 255.0

        return lr_t_tensor, hr_prev_tensor, hr_t_tensor