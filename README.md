# Deepfake Image Detection using Vision Transformer (ViT)

![tests](https://github.com/DivyanBabbar/deepfake_detector/actions/workflows/tests.yml/badge.svg)

A deepfake image detection system that fine-tunes a pretrained **Vision Transformer (google/vit-base-patch16-224)** with transfer learning to classify facial images as **Real** or **Fake**.

---

## Overview

Instead of training a transformer from scratch, this project fine-tunes the pretrained `google/vit-base-patch16-224` model from Hugging Face Transformers. The classifier head is replaced with a 2-class head for real vs fake classification.

---

## Features

- Vision Transformer (ViT-Base Patch16-224) with transfer learning
- Binary classification (Real / Fake) with a confidence score
- GPU acceleration (CUDA) with PyTorch
- Dataset loading with `ImageFolder`
- Accuracy, precision, recall and F1 evaluation, plus confusion matrix
- Best-model checkpointing
- Automated tests and linting on every push (GitHub Actions, Linux and Windows)

---

## Project Structure

```text
deepfake_detector/
  train.py               training and evaluation loop
  predict.py             classify a single image
  common.py              preprocessing and label mapping shared by train and predict
  utils.py               evaluation metrics and confusion matrix
  tests/                 pytest suite
  .github/workflows/     CI (lint + tests)
  requirements.txt
  requirements-dev.txt   adds pytest and ruff
  pyproject.toml         pytest and ruff config
  dataset/               not included; see Dataset below
  models/                created when you train (best_model.pth)
  results/               created when you train (metrics.txt, confusion_matrix.png)
```

---

## Dataset

**140K Real and Fake Faces**

| Split | Images in dataset |
|-------|-------------------|
| Training | 100,000 |
| Testing | 20,000 |

Because of time and GPU constraints, a random subset was used for the results below:

| Split | Images used |
|-------|-------------|
| Training | 10,000 |
| Testing | 2,000 |

The subset is sampled with a fixed seed (42). Classes: Fake (label 0), Real (label 1).

Expected folder layout:

```text
dataset/
  real_vs_fake/
    real-vs-fake/
      train/
        fake/
        real/
      test/
        fake/
        real/
```

---

## Model

Pretrained backbone: `google/vit-base-patch16-224`

The classifier head is replaced for binary classification and the pretrained ImageNet weights are fine-tuned.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Image size | 224 x 224 |
| Hardware | NVIDIA RTX 4050 Laptop GPU (6 GB) |

---

## Installation

Tested with Python 3.12.

```bash
git clone https://github.com/DivyanBabbar/deepfake_detector.git
cd deepfake_detector
```

Create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**GPU note:** `pip install torch` may install a CPU-only build. Install the CUDA build of PyTorch from pytorch.org, then confirm:

```python
import torch
print(torch.cuda.is_available())
```

This should print `True` before you start training.

---

## Training

```bash
python train.py
```

The script samples the training directory, splits that sample into training and validation sets, and selects the best checkpoint using validation accuracy. It evaluates the selected checkpoint on the held-out test subset once after training. Metrics are saved separately under `results/validation/` and `results/test/` so they cannot be mistaken for one another.

**Windows note:** DataLoader uses `num_workers=0`. With worker processes, Windows raised multiprocessing errors. This is slower but stable.

---

## Prediction

After training:

```bash
python predict.py path/to/image.jpg
```

Output looks like `Prediction: FAKE (97.3% confidence)`. Use `--weights` to load a different checkpoint.

---

## Results

Evaluated on the **2,000-image test subset** (not the full 20,000-image test set):

| Metric | Score |
|--------|-------|
| Accuracy | 98.10% |
| Precision | 96.77% |
| Recall | 99.50% |
| F1 Score | 98.12% |

Precision, recall and F1 treat **Real** (label 1) as the positive class.

These numbers come from one training run on a subset, so treat them as preliminary.

---

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

The tests use a tiny randomly initialised ViT and generated images, so they need no GPU, no dataset and no model download. CI runs them on Linux and Windows for every push and pull request.

The tests check that the code runs correctly. They do not measure model accuracy.

---

## Limitations

- Trained on a 10,000-image subset for 2 epochs; not evaluated on the full test set
- Evaluated on one dataset only, with no cross-dataset testing
- Image-level detection of faces; not designed for video

---

## Future Improvements

- Train on the complete 100K training set and evaluate on the full 20K test set
- Train for more epochs
- Compare with CNN baselines such as XceptionNet and EfficientNet
- Cross-dataset evaluation
- Video-based deepfake detection
- Explainability with Grad-CAM
- Lightweight transformer deployment

---

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- FaceForensics++
- Celeb-DF
- DeepFake Detection Challenge (DFDC)
- Hugging Face Transformers documentation
- PyTorch documentation

---

## Author

**Divyan Babbar**
B.Tech ECE (Advanced Communication Technology), Jaypee Institute of Information Technology
Email: divyanbabbar453@gmail.com

---

## License

This project is developed for educational and academic purposes.
