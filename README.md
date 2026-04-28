# Faster R-CNN Grasp Detection Models

A collection of trained Faster R-CNN models for grasp detection, including standard ResNet-50/ResNet-18 backbones and a structured-pruned variant for improved inference speed.

## Project Overview

This repository contains:
- **ResNet-50 FPN backbone** - High-accuracy grasp detection model
- **ResNet-18 FPN backbone** - Lightweight alternative with balanced performance
- **Structured Pruned ResNet-18** - Optimized for edge deployment with 2.2x speedup

All models are trained on grasp detection datasets and include inference scripts with GPU support (CUDA 12.1+).

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd into the directory
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (if not already in the repository)
   - Models are in: `model_resnet_50/`, `models_resnet18_v2/`, `Pruned_model_v3/` accessed in this link: https://drive.google.com/drive/folders/17vggB7ATP2DyJ13ozPqiVgVsqpdW-pKY?usp=sharing 
   - Sample images are in: `images_50/` accessed in this link: https://drive.google.com/drive/folders/1TTxl-jN5ifK_pZ3usGgey-lDCjdZKmR_?usp=sharing 

## Quick Inference Commands

Run these commands from the project root directory:

### ResNet-50 FPN (Highest Accuracy)
```bash
python inference_resnet_50.py \
  --image_dir ./images_50 \
  --checkpoint ./model_resnet_50/final_model.pth \
  --runs 30 --warmup 10 --fp16
```


### ResNet-18 FPN (Balanced Speed/Accuracy)
```bash
python inference_resnet_18.py \
  --image_dir ./images_50 \
  --checkpoint ./models_resnet18_v2/final_model.pth \
  --runs 30 --warmup 10 --fp16
```


### Pruned ResNet-18 (Fastest)
```bash
python inference_structured_pruning_resnet_18.py \
  --image_dir ./images_50 \
  --checkpoint ./Pruned_model_v3/structured_pruned.pth \
  --runs 30 --warmup 10 --fp16
```

## Model Benchmarks

All benchmarks run on **NVIDIA RTX 2000 Ada (16 GB VRAM)** with **FP16 precision** at **640x480 resolution**:

| Model | Backbone | Latency (ms) | Parameters | Speed vs ResNet-50 |
|-------|----------|--------------|------------|-------------------|
| **ResNet-50 FPN** | ResNet-50 | **31.83** | 43.8M | Baseline |
| **ResNet-18 FPN** | ResNet-18 | **19.11** | 12.2M | **1.67x faster** |
| **Pruned ResNet-18** | Pruned ResNet-18 | **14.56** | 6.1M | **2.19x faster** |

**Note:** Latency = mean inference time per image (forward pass only, no I/O)

## Model Descriptions

### ResNet-50 FPN (`inference_resnet_50.py`)
- **Backbone**: ResNet-50 + Feature Pyramid Network (FPN)
- **Use case**: Maximum accuracy for grasp detection
- **Output**: Bounding boxes with confidence scores for grasp points
- **Trained on**: Grasp detection dataset (~97% accuracy)

```bash
python inference_resnet_50.py \
  --image_dir ./images_50 \
  --checkpoint ./model_resnet_50/final_model.pth \
  --runs 30 --warmup 10 --fp16
```

### ResNet-18 FPN (`inference_resnet_18.py`)
- **Backbone**: ResNet-18 + FPN
- **Use case**: Speed-accuracy tradeoff for real-time applications
- **Speedup**: 1.67x faster than ResNet-50
- **Output**: Bounding boxes with confidence scores
- **Best for**: Edge devices with limited VRAM

```bash
python inference_resnet_18.py \
  --image_dir ./images_50 \
  --checkpoint ./models_resnet18_v2/final_model.pth \
  --runs 30 --warmup 10 --fp16
```

### Structured Pruned ResNet-18 (`inference_structured_pruning_resnet_18.py`)
- **Backbone**: Structured pruned ResNet-18 + FPN
- **Pruning**: Channel-level pruning for hardware efficiency
- **Use case**: Maximum speed for embedded systems
- **Speedup**: 2.19x faster than ResNet-50
- **Efficiency**: 50%+ parameter reduction with minimal accuracy loss

```bash
python inference_structured_pruning_resnet_18.py \
  --image_dir ./images_50 \
  --checkpoint ./Pruned_model_v3/structured_pruned.pth \
  --runs 30 --warmup 10 --fp16
```

## Usage Guide

### Basic Inference

**Single Image:**
```bash
python inference_resnet_50.py \
  --image_path ./images_50/pcd0100r.png \
  --checkpoint ./model_resnet_50/final_model.pth
```

**Image Directory:**
```bash
python inference_resnet_50.py \
  --image_dir ./images_50 \
  --checkpoint ./model_resnet_50/final_model.pth \
  --runs 20 --warmup 5
```

### Advanced Options

```bash
python inference_resnet_50.py \
  --image_dir ./images_50 \
  --checkpoint ./model_resnet_50/final_model.pth \
  --runs 30                    # Number of timing runs per image
  --warmup 10                  # Warmup runs before timing
  --fp16                       # Use FP16 (half precision) for 2x speedup
  --width 640 --height 480    # Input image size
  --conf_threshold 0.5         # Confidence threshold for box reporting
  --save_dir ./output         # Save annotated images with detections
  --device cuda               # Use 'cuda', 'cpu', or 'auto'
  --recursive                 # Recursively scan subdirectories
```

### Output Format

Example output:
```
2026-04-28 09:43:26,994 - INFO - pcd0100r.png: mean 31.83 ms / forward (n=20), boxes>=0.5: 62 (raw 100)

============================================================
INFERENCE LATENCY (model forward)
============================================================
Backbone       : ResNet-50 FPN
Checkpoint     : model_resnet_50/final_model.pth
Resize         : 640x480
FP16           : True
Images         : 51
Runs / image   : 20
Mean per image : 31.83 ms
Min / Max      : 31.36 ms / 33.10 ms
============================================================
```

## Understanding FP16 (Half Precision)

**What is FP16?**
- Uses 16-bit floating point instead of 32-bit (FP32)
- Reduces memory usage by 50% and speeds up computation by ~2x
- Minimal accuracy loss (<1%) for most object detection tasks

**When to use:**
- GPU has dedicated half-precision cores (RTX, A100, H100, etc.)
- Memory bandwidth is bottleneck
- Need faster inference

**Example speedup with FP16:**
```
FP32: 61.29 ms/image
FP16: 31.83 ms/image  ← ~1.93x faster
```

## Project Structure

```
.
├── inference_resnet_50.py              # ResNet-50 inference script
├── inference_resnet_18.py              # ResNet-18 inference script
├── inference_structured_pruning_resnet_18.py  # Pruned ResNet-18
├── train.py                             # ResNet-50 training script
├── train_resnet_18.py                  # ResNet-18 training script
├── structured_pruning_resnet_18.py     # Pruning script
├── eval_cornell.py                     # Cornell dataset evaluation
├── eval_jacquard.py                    # Jacquard dataset evaluation
├── images_50/                          # Sample test images
├── model_resnet_50/                    # ResNet-50 checkpoints
│   ├── final_model.pth
│   ├── best_model.pth
│   └── model_epoch_*.pth
├── models_resnet18_v2/                 # ResNet-18 checkpoints
│   ├── final_model.pth
│   └── model_epoch_*.pth
├── Pruned_model_v3/                    # Pruned ResNet-18 checkpoint
│   └── structured_pruned.pth
└── requirements.txt                    # Python dependencies
```

##  Performance Metrics

### Throughput (images/second)

| Model | Throughput | Batch Size |
|-------|-----------|-----------|
| ResNet-50 FPN | ~31 img/s | 1 |
| ResNet-18 FPN | ~52 img/s | 1 |
| Pruned ResNet-18 | **~69 img/s** | 1 |

### Latency Breakdown (ResNet-50 FPN with FP16)

- **Model forward pass**: 31.83 ms
- **GPU synchronization**: <1 ms
- **Warmup time** (10 runs): ~320 ms (first time only)

## Python API 

```python
import torch
from pathlib import Path
from inference_resnet_50 import get_model, load_model, preprocess_bgr_to_chw
import cv2

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(Path("./model_resnet_50/final_model.pth"), device)

# Process image
bgr = cv2.imread("image.png")
tensor = preprocess_bgr_to_chw(bgr, 640, 480, device, torch.float32)

# Inference
with torch.inference_mode():
    predictions = model([tensor])

# predictions[0] contains:
# - 'boxes': [N, 4] (x1, y1, x2, y2)
# - 'labels': [N] class labels
# - 'scores': [N] confidence scores
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
```

## Tips for Best Performance

1. **Use FP16 for faster inference** (2x speedup)
   ```bash
   python inference_resnet_50.py --image_dir ./images_50 --fp16
   ```

2. **Use pruned model for edge deployment** (2.2x faster than ResNet-50)
   ```bash
   python inference_structured_pruning_resnet_18.py --checkpoint ./Pruned_model_v3/structured_pruned.pth
   ```

3. **Batch multiple images** (when modifying scripts)
   - Load model once, run multiple images
   - Amortizes GPU initialization overhead

4. **Use smaller input size for speed** (at accuracy cost)
   ```bash
   python inference_resnet_50.py --width 480 --height 360
   ```


## Requirements

See `requirements.txt` for complete dependencies:

```
torch==2.4.0
torchvision==0.19.0
opencv-python==4.13.0.92
numpy>=2.0.0
matplotlib>=3.10.0
Pillow>=12.0.0
tqdm>=4.67.0
```

**CUDA Support:**
- CUDA 12.1+ (installed automatically with PyTorch)
- cuDNN 9.1+ (installed automatically with PyTorch)

## Troubleshooting

### ModuleNotFoundError: No module named 'train_resnet_18'
**Solution:** Ensure you're running inference scripts from the project root directory:
```bash
cd /path/to/Vince_Faster_rcnn
python inference_resnet_18.py ...
```

### CUDA out of memory
**Solution:** Reduce input size or use CPU:
```bash
python inference_resnet_50.py --device cpu --width 480 --height 360
```

### Slow inference
**Solution:** Enable FP16 and verify GPU is being used:
```bash
python inference_resnet_50.py --fp16
```

**Last Updated:** April 28, 2026  
**GPU Tested On:** NVIDIA RTX 2000 Ada (16 GB VRAM)  
**CUDA Version:** 12.1  
**PyTorch Version:** 2.4.0
