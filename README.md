# Mini-DLSS: Spatial-Temporal Video Upscaling

Lightweight PyTorch project that experiments with DLSS-style 2x upscaling from 540p gameplay patches to 1080p patches using temporal context.

## Project Structure

- `extract_frames.py`: extracts HR 1080p frames and LR 540p frames from gameplay video.
- `dataset.py`: initial PyTorch dataset for temporal frame triplets.
- `train_mini_dlss.ipynb`: notebook for sanity checks, model definition, training, validation, and preview.
- `train_medium.py`: bounded training run with progress percentage, ETA, checkpointing, and bicubic PSNR comparison.
- `dl project specs.md`: project roadmap/specification.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the correct wheel from the official PyTorch selector:

https://pytorch.org/get-started/locally/

Run the medium training script:

```bash
python train_medium.py
```

Generated data, checkpoints, logs, and source video are intentionally ignored by Git.
