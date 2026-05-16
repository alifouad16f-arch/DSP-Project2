# Mini-ImageNet Wavelet Compression Study

A study on the effect of **Haar Wavelet compression** on image classification accuracy using **ResNet-18** and **MobileNetV2** fine-tuned on the Mini-ImageNet dataset.

---

## Project Structure

```
mini_imagenet_wavelet/
├── main.py                        # ← Entry point: runs all phases end-to-end
├── src/
│   ├── dataset.py                 # Dataset loading & reorganisation utilities
│   ├── compression.py             # Wavelet compression logic
│   ├── train.py                   # Training & validation loop
│   └── evaluate.py                # Evaluation & per-class degradation analysis
├── models/
│   └── .gitkeep                   # Saved .pth weights go here (not tracked)
├── results/
│   └── .gitkeep                   # Plots & metrics go here
├── scripts/
│   └── reorganise_dataset.py      # Standalone script to fix dataset splits
├── requirements.txt
└── README.md
```

---

## Dataset — Mini-ImageNet

Mini-ImageNet contains **100 classes** with 600 images each (96×96 px).  
The dataset is **not included** in this repository due to its size. You must obtain it separately and set it up using one of the two methods below.

### How the dataset was originally obtained

The dataset was downloaded as three `.tar` archive files:

```
train.tar
val.tar
test.tar
```

These were extracted into a local `mini_imagenet/` directory:

```bash
mkdir -p mini_imagenet
tar -xf train.tar -C mini_imagenet/
tar -xf val.tar   -C mini_imagenet/
tar -xf test.tar  -C mini_imagenet/
```

After extraction the splits were reorganised to produce exactly **50,000 training** and **10,000 test** images using the script in `scripts/reorganise_dataset.py`.

---

## Dataset Setup — Two Methods

### Method 1 — Google Drive + Google Colab (recommended for quick experiments)

1. Upload the three `.tar` files to your Google Drive (e.g. `MyDrive/datasets/mini_imagenet/`).
2. Open `notebooks/full_pipeline.ipynb` in [Google Colab](https://colab.research.google.com/) and set the runtime to **T4 GPU**.
3. Mount your Drive and extract:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess

base = '/content/drive/MyDrive/datasets/mini_imagenet'
os.makedirs('/content/mini_imagenet', exist_ok=True)

for split in ['train', 'val', 'test']:
    subprocess.run(['tar', '-xf', f'{base}/{split}.tar', '-C', '/content/mini_imagenet/'])
```

4. Run `scripts/reorganise_dataset.py` (or the inline cell in the notebook) to produce the fixed splits.

> **Tip:** After reorganisation, you can re-upload the fixed dataset back to Drive so you only do this step once.

---

### Method 2 — Local machine (no Colab required)

This method lets you run the full pipeline entirely on your own hardware with a CUDA-capable GPU.

**Prerequisites**

```bash
pip install -r requirements.txt
```

**Steps**

1. Place the `.tar` files in a folder, e.g. `~/data/mini_imagenet_raw/`.
2. Extract them:

```bash
mkdir -p ~/data/mini_imagenet
tar -xf ~/data/mini_imagenet_raw/train.tar -C ~/data/mini_imagenet/
tar -xf ~/data/mini_imagenet_raw/val.tar   -C ~/data/mini_imagenet/
tar -xf ~/data/mini_imagenet_raw/test.tar  -C ~/data/mini_imagenet/
```

3. Reorganise the splits:

```bash
python scripts/reorganise_dataset.py \
    --src ~/data/mini_imagenet \
    --dst ~/data/mini_imagenet_fixed
```

4. Update `data_dir` in `src/dataset.py` (or pass it as an argument) to point to `~/data/mini_imagenet_fixed`.

5. Run the full pipeline:

```bash
# All phases, both models (default)
python main.py --data_dir ~/data/mini_imagenet_fixed

# ResNet-18 only, 10 epochs, skip Phase 3
python main.py --data_dir ~/data/mini_imagenet_fixed --models resnet18 --epochs 10 --skip_phase3

# MobileNetV2 only
python main.py --data_dir ~/data/mini_imagenet_fixed --models mobilenet
```

Saved weights appear in `models/` and visualisations in `results/`.

---

## Experiment Overview

| Phase | Description |
|-------|-------------|
| **Phase 1** | Fine-tune ResNet-18 / MobileNetV2 on uncompressed Mini-ImageNet |
| **Phase 2** | Evaluate baseline models on Haar-wavelet-compressed images (2:1, 5:1, 10:1) |
| **Phase 3** | Re-train models on a pre-compressed training set; evaluate robustness to 10:1 compression |

---

## Requirements

See `requirements.txt`. Key dependencies:

- `torch` / `torchvision`
- `PyWavelets` (`pywt`)
- `numpy`
- `matplotlib`

---

## Notes

- All models are initialised from **ImageNet pre-trained weights** (transfer learning).
- Images are resized to **96×96** and normalised with standard ImageNet mean/std.
- The wavelet compression uses a **2D Haar DWT** with hard thresholding; no quantisation or entropy coding is applied — this is a coefficient-dropping study, not a file-size benchmark.
