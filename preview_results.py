from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


HR_DIR = Path("dataset/HR")
LR_DIR = Path("dataset/LR")
CHECKPOINT = Path("checkpoints/medium_best.pt")
OUT_DIR = Path("results")
LR_PATCH_SIZE = 64
SCALE = 2
NUM_PREVIEWS = 6
SEED = 123


class TemporalPatchDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, lr_patch_size=64, scale=2):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.lr_patch_size = lr_patch_size
        self.scale = scale
        self.hr_patch_size = lr_patch_size * scale
        hr_names = {p.name for p in self.hr_dir.glob("*.png")}
        lr_names = {p.name for p in self.lr_dir.glob("*.png")}
        self.frames = sorted(hr_names & lr_names)

    def __len__(self):
        return len(self.frames) - 1

    def _read_rgb(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        prev_name = self.frames[idx]
        current_name = self.frames[idx + 1]
        lr_t = self._read_rgb(self.lr_dir / current_name)
        hr_prev = self._read_rgb(self.hr_dir / prev_name)
        hr_t = self._read_rgb(self.hr_dir / current_name)

        h, w, _ = lr_t.shape
        x = random.randint(0, w - self.lr_patch_size)
        y = random.randint(0, h - self.lr_patch_size)
        hx, hy = x * self.scale, y * self.scale

        lr_crop = lr_t[y:y + self.lr_patch_size, x:x + self.lr_patch_size]
        hr_prev_crop = hr_prev[hy:hy + self.hr_patch_size, hx:hx + self.hr_patch_size]
        hr_crop = hr_t[hy:hy + self.hr_patch_size, hx:hx + self.hr_patch_size]

        return (
            torch.from_numpy(lr_crop.copy()).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(hr_prev_crop.copy()).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(hr_crop.copy()).permute(2, 0, 1).float() / 255.0,
            current_name,
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x) * 0.1


class TemporalSRNet(nn.Module):
    def __init__(self, features=48, residual_blocks=6, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(6, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(features) for _ in range(residual_blocks)])
        self.tail = nn.Conv2d(features, 3, kernel_size=3, padding=1)

    def forward(self, lr_t, hr_prev):
        lr_up = F.interpolate(lr_t, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = torch.cat([lr_up, hr_prev], dim=1)
        residual = self.tail(self.body(self.head(x)))
        return (lr_up + residual).clamp(0.0, 1.0)


def to_image(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def save_preview(index, frame_name, lr_t, hr_prev, hr_t, pred):
    bicubic = F.interpolate(lr_t[None], scale_factor=SCALE, mode="bicubic", align_corners=False)[0].clamp(0, 1)
    error = (pred - hr_t).abs().mean(0, keepdim=True).repeat(3, 1, 1)

    panels = [
        ("Bicubic", bicubic),
        ("Model", pred),
        ("Ground Truth", hr_t),
        ("Error", error),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(to_image(image))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(frame_name)
    fig.tight_layout()
    output_path = OUT_DIR / f"preview_{index:02d}_{Path(frame_name).stem}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    OUT_DIR.mkdir(exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = TemporalSRNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = TemporalPatchDataset(HR_DIR, LR_DIR, LR_PATCH_SIZE, SCALE)
    sample_indices = random.sample(range(len(dataset)), k=min(NUM_PREVIEWS, len(dataset)))

    print(f"Device: {device}")
    print(f"Loaded: {CHECKPOINT}")
    print(f"Saving previews to: {OUT_DIR.resolve()}")

    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices, start=1):
            lr_t, hr_prev, hr_t, frame_name = dataset[sample_idx]
            pred = model(lr_t[None].to(device), hr_prev[None].to(device))[0].cpu()
            path = save_preview(i, frame_name, lr_t, hr_prev, hr_t, pred)
            print(path)


if __name__ == "__main__":
    main()
