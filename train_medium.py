from pathlib import Path
import math
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


class Config:
    hr_dir = Path("dataset/HR")
    lr_dir = Path("dataset/LR")
    checkpoint_dir = Path("checkpoints")
    lr_patch_size = 64
    scale = 2
    batch_size = 8
    num_workers = 0
    epochs = 3
    max_train_batches = 300
    learning_rate = 2e-4
    val_fraction = 0.08
    seed = 42
    features = 48
    residual_blocks = 6
    use_amp = True
    log_every = 25


cfg = Config()


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
        if len(self.frames) < 2:
            raise RuntimeError("Need at least two aligned HR/LR frames.")

        print(f"Aligned frame pairs: {len(self.frames)}", flush=True)
        print(f"LR files without HR target: {len(lr_names - hr_names)}", flush=True)
        print(f"HR files without LR input: {len(hr_names - lr_names)}", flush=True)

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


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


def psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=30):
    model.eval()
    losses = []
    model_psnrs = []
    bicubic_psnrs = []
    loss_fn = CharbonnierLoss().to(device)

    for batch_idx, (lr_t, hr_prev, hr_t) in enumerate(loader):
        lr_t = lr_t.to(device, non_blocking=True)
        hr_prev = hr_prev.to(device, non_blocking=True)
        hr_t = hr_t.to(device, non_blocking=True)

        pred = model(lr_t, hr_prev)
        bicubic = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        losses.append(loss_fn(pred, hr_t).item())
        model_psnrs.append(psnr(pred, hr_t))
        bicubic_psnrs.append(psnr(bicubic, hr_t))

        if batch_idx + 1 >= max_batches:
            break

    return {
        "loss": float(np.mean(losses)),
        "model_psnr": float(np.mean(model_psnrs)),
        "bicubic_psnr": float(np.mean(bicubic_psnrs)),
        "delta_db": float(np.mean(model_psnrs) - np.mean(bicubic_psnrs)),
    }


def fmt_eta(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def main():
    cfg.checkpoint_dir.mkdir(exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    dataset = TemporalPatchDataset(cfg.hr_dir, cfg.lr_dir, cfg.lr_patch_size, cfg.scale)
    val_size = max(1, int(len(dataset) * cfg.val_fraction))
    train_size = len(dataset) - val_size
    train_ds = Subset(dataset, range(0, train_size))
    val_ds = Subset(dataset, range(train_size, len(dataset)))
    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TemporalSRNet(cfg.features, cfg.residual_blocks, cfg.scale).to(device)
    init_checkpoint = cfg.checkpoint_dir / "best.pt"
    if init_checkpoint.exists():
        ck = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"Loaded starting weights from {init_checkpoint}", flush=True)

    loss_fn = CharbonnierLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_steps = cfg.epochs * min(len(train_loader), cfg.max_train_batches)
    completed_steps = 0
    best_val = float("inf")
    history = []
    start_time = time.time()

    print(
        f"Starting medium training: epochs={cfg.epochs}, max_train_batches={cfg.max_train_batches}, "
        f"batch_size={cfg.batch_size}, total_steps={total_steps}",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = []

        for batch_idx, (lr_t, hr_prev, hr_t) in enumerate(train_loader, start=1):
            lr_t = lr_t.to(device, non_blocking=True)
            hr_prev = hr_prev.to(device, non_blocking=True)
            hr_t = hr_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(lr_t, hr_prev)
                loss = loss_fn(pred, hr_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            completed_steps += 1

            if completed_steps % cfg.log_every == 0 or completed_steps == total_steps:
                elapsed = time.time() - start_time
                pct = completed_steps / total_steps * 100
                sec_per_step = elapsed / max(1, completed_steps)
                eta = sec_per_step * (total_steps - completed_steps)
                recent = float(np.mean(epoch_losses[-cfg.log_every:]))
                print(
                    f"Progress {pct:6.2f}% | step {completed_steps}/{total_steps} | "
                    f"epoch {epoch}/{cfg.epochs} batch {batch_idx} | loss {recent:.5f} | "
                    f"elapsed {fmt_eta(elapsed)} | ETA {fmt_eta(eta)}",
                    flush=True,
                )

            if batch_idx >= cfg.max_train_batches:
                break

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        train_loss = float(np.mean(epoch_losses))
        record = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(record)
        print(
            f"Epoch {epoch} complete | train {train_loss:.5f} | val {metrics['loss']:.5f} | "
            f"model PSNR {metrics['model_psnr']:.2f} dB | bicubic {metrics['bicubic_psnr']:.2f} dB | "
            f"delta {metrics['delta_db']:+.2f} dB",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "epochs": cfg.epochs,
                "max_train_batches": cfg.max_train_batches,
                "batch_size": cfg.batch_size,
                "lr_patch_size": cfg.lr_patch_size,
                "features": cfg.features,
                "residual_blocks": cfg.residual_blocks,
                "learning_rate": cfg.learning_rate,
            },
            "history": history,
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "medium_last.pt")
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            torch.save(checkpoint, cfg.checkpoint_dir / "medium_best.pt")
            print("Saved checkpoints/medium_best.pt", flush=True)

    total_elapsed = time.time() - start_time
    final = history[-1]
    print(
        f"Training finished in {fmt_eta(total_elapsed)} | final model PSNR {final['model_psnr']:.2f} dB | "
        f"bicubic {final['bicubic_psnr']:.2f} dB | delta {final['delta_db']:+.2f} dB",
        flush=True,
    )


if __name__ == "__main__":
    main()
