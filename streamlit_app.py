from pathlib import Path
import math
import random

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).parent
HR_DIR = ROOT / "dataset" / "HR"
LR_DIR = ROOT / "dataset" / "LR"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "medium_best.pt"
SCALE = 2


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


def psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_tensor(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0


def to_image(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def aligned_frames():
    hr_names = {p.name for p in HR_DIR.glob("*.png")}
    lr_names = {p.name for p in LR_DIR.glob("*.png")}
    return sorted(hr_names & lr_names)


@st.cache_resource(show_spinner=True)
def load_model(checkpoint_path, device_name):
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TemporalSRNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def make_patch(frame_idx, lr_patch_size, seed):
    frames = aligned_frames()
    current_name = frames[frame_idx]
    previous_name = frames[frame_idx - 1]

    lr_t = read_rgb(LR_DIR / current_name)
    hr_prev = read_rgb(HR_DIR / previous_name)
    hr_t = read_rgb(HR_DIR / current_name)

    h, w, _ = lr_t.shape
    rng = random.Random(seed)
    x = rng.randint(0, w - lr_patch_size)
    y = rng.randint(0, h - lr_patch_size)
    hx, hy = x * SCALE, y * SCALE
    hr_patch = lr_patch_size * SCALE

    lr_crop = lr_t[y:y + lr_patch_size, x:x + lr_patch_size]
    hr_prev_crop = hr_prev[hy:hy + hr_patch, hx:hx + hr_patch]
    hr_crop = hr_t[hy:hy + hr_patch, hx:hx + hr_patch]

    return {
        "frame": current_name,
        "previous": previous_name,
        "lr": to_tensor(lr_crop),
        "hr_prev": to_tensor(hr_prev_crop),
        "hr": to_tensor(hr_crop),
        "crop": (x, y, lr_patch_size, lr_patch_size),
    }


def run_inference(model, sample, device):
    lr = sample["lr"][None].to(device)
    hr_prev = sample["hr_prev"][None].to(device)
    hr = sample["hr"][None].to(device)
    with torch.no_grad():
        pred = model(lr, hr_prev)
        bicubic = F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)
    return pred[0].cpu(), bicubic[0].cpu(), hr[0].cpu()


st.set_page_config(page_title="Mini-DLSS Demo", layout="wide")

st.title("Mini-DLSS: Spatial-Temporal Upscaling Demo")
st.caption("Patch-level inference using the trained temporal CNN checkpoint.")

if not HR_DIR.exists() or not LR_DIR.exists():
    st.error("Dataset folders were not found. Expected dataset/HR and dataset/LR.")
    st.stop()

if not DEFAULT_CHECKPOINT.exists():
    st.error("Trained checkpoint not found. Expected checkpoints/medium_best.pt.")
    st.stop()

frames = aligned_frames()
if len(frames) < 2:
    st.error("Not enough aligned HR/LR frames found.")
    st.stop()

cuda_available = torch.cuda.is_available()
device_name = "cuda" if cuda_available else "cpu"

with st.sidebar:
    st.header("Demo Controls")
    st.write(f"Device: `{device_name}`")
    checkpoint_path = st.text_input("Checkpoint", str(DEFAULT_CHECKPOINT))
    lr_patch_size = st.select_slider("LR patch size", options=[48, 64, 80, 96], value=64)
    frame_idx = st.slider("Frame index", 1, len(frames) - 1, min(429, len(frames) - 1))
    seed = st.number_input("Crop seed", min_value=0, max_value=99999, value=123, step=1)
    if st.button("Random crop"):
        seed = random.randint(0, 99999)
        st.session_state["seed_override"] = seed
    if "seed_override" in st.session_state:
        seed = st.session_state["seed_override"]

model, checkpoint = load_model(checkpoint_path, device_name)
sample = make_patch(frame_idx, lr_patch_size, int(seed))
pred, bicubic, target = run_inference(model, sample, torch.device(device_name))
error = (pred - target).abs().mean(0, keepdim=True).repeat(3, 1, 1)

model_psnr = psnr(pred[None], target[None])
bicubic_psnr = psnr(bicubic[None], target[None])
delta = model_psnr - bicubic_psnr

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model PSNR", f"{model_psnr:.2f} dB")
col2.metric("Bicubic PSNR", f"{bicubic_psnr:.2f} dB")
col3.metric("Delta", f"{delta:+.2f} dB")
col4.metric("Checkpoint epoch", checkpoint.get("epoch", "n/a"))

st.divider()

st.subheader("Visual Comparison")
cols = st.columns(4)
cols[0].image(to_image(bicubic), caption="Bicubic upscaling", use_column_width=True)
cols[1].image(to_image(pred), caption="Model output", use_column_width=True)
cols[2].image(to_image(target), caption="Ground truth HR", use_column_width=True)
cols[3].image(to_image(error), caption="Absolute error map", use_column_width=True)

st.subheader("Sample Details")
details = {
    "Current frame": sample["frame"],
    "Previous temporal frame": sample["previous"],
    "LR crop x/y/size": sample["crop"],
    "Input": f"LR_t {lr_patch_size}x{lr_patch_size}, HR_(t-1) {lr_patch_size * SCALE}x{lr_patch_size * SCALE}",
    "Output": f"Predicted HR_t {lr_patch_size * SCALE}x{lr_patch_size * SCALE}",
}
st.table(details)

with st.expander("Training summary"):
    history = checkpoint.get("history", [])
    if history:
        st.dataframe(history, use_container_width=True)
    st.markdown(
        """
        This Streamlit demo is a patch-level validation/inference viewer.
        The model compares against bicubic upscaling because this is an image reconstruction task,
        not a classification task, so accuracy/F1/confusion matrices are not the right evaluation tools.
        """
    )
