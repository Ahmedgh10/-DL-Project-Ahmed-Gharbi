markdown_content = """# Mini-DLSS: Spatial-Temporal Video Upscaling Project

## Project Overview
This project aims to build a custom, lightweight deep learning model that mimics NVIDIA's DLSS technology. The model upscales low-resolution (540p) gameplay video to high-resolution (1080p) while utilizing temporal data (previous frames) to maintain temporal stability and prevent flickering/ghosting.

## Hardware & Tech Stack
* **Target Hardware:** NVIDIA RTX 3050 Laptop GPU (4GB VRAM constraint)
* **Frameworks:** PyTorch, OpenCV, Python
* **Game Source:** *Marvel's Spider-Man 2* (Native 1080p, 30 FPS, TAA on, DLSS/FSR off)

## Project Phases & Roadmap

### Phase 1: Data Preparation & Extraction (Status: Complete)
* **Source Data:** ~3 minutes and 40 seconds of native 1080p gameplay footage providing a mix of high-speed motion, combat, and slow exploration.
* **Extraction Script (`extract_frames.py`):** Uses OpenCV to read the video, save native 1080p frames as Ground Truth ($HR$), and downsample copies to 540p using `INTER_AREA` interpolation to serve as inputs ($LR$). 

### Phase 2: PyTorch Dataset & Data Loading (Status: Complete)
* **Temporal Pairs:** A custom PyTorch `Dataset` class (`dataset.py`) designed to load three aligned images per training step:
    1.  Current Low-Res Frame ($LR_t$) - *Input*
    2.  Previous High-Res Frame ($HR_{t-1}$) - *Input (Temporal Memory)*
    3.  Current High-Res Frame ($HR_t$) - *Target / Ground Truth*
* **VRAM Optimization:** Implements random patch cropping (extracting 64x64 patches from $LR$ and matching 128x128 patches from $HR$) to ensure the training process fits safely within the 4GB VRAM limit.

### Phase 3: Model Architecture (Status: Pending)
* **Design:** A lightweight Convolutional Neural Network (CNN) or minimal U-Net architecture.
* **Input Layer Modification:** Adapted to accept a **6-channel input** (3 RGB channels from $LR_t$ concatenated with 3 RGB channels from $HR_{t-1}$).
* **Output Layer:** Generates the upscaled 3-channel RGB image for the current frame.

### Phase 4: Loss Function Design (Status: Pending)
* Instead of relying solely on Mean Squared Error (MSE), which causes blurriness, the project will implement a **Perceptual Loss function**.
* Will utilize a pre-trained feature extractor (e.g., VGG16) to compare the generated output and the ground truth on a feature/texture level, ensuring crisp geometry and details.

### Phase 5: Training Loop & Optimization (Status: Pending)
* Will implement small batch sizes (e.g., 4 or 8) to prevent `CUDA Out of Memory` errors.
* Will utilize PyTorch's **Automatic Mixed Precision (AMP)** (FP16 training) to halve memory consumption and accelerate training on the RTX 3050.

---
*Generated for AI Agent reference to track project state and architecture goals.*
"""

file_path = "Mini-DLSS_Project_Summary.md"
with open(file_path, "w") as f:
    f.write(markdown_content)

print(f"File generated successfully at {file_path}")