# Deepfake Image Detection using Vision Transformer (ViT)

A deepfake image detection system built using **Vision Transformer (ViT-Base Patch16-224)** and **transfer learning**. The project fine-tunes a pretrained Vision Transformer to classify facial images as **Real** or **Fake**.

---

## Overview

Deepfake technology has advanced rapidly in recent years, making manipulated images increasingly difficult to distinguish from authentic ones. This project uses a pretrained Vision Transformer (ViT) to detect manipulated facial images through binary classification.

Instead of training a transformer from scratch, transfer learning is used with the pretrained **google/vit-base-patch16-224** model available through Hugging Face Transformers.

---

## Features

- Vision Transformer (ViT-Base Patch16-224)
- Transfer Learning
- Binary Classification (Real / Fake)
- GPU Acceleration (CUDA)
- PyTorch Implementation
- Hugging Face Transformers
- Automatic Dataset Loading using ImageFolder
- Confusion Matrix Generation
- Accuracy, Precision, Recall and F1 Score Evaluation
- Model Checkpoint Saving

---

## Project Structure

```text
deepfake_detection_vit/
│
├── dataset/
│   └── real_vs_fake/
│       └── real-vs-fake/
│           ├── train/
│           │   ├── real/
│           │   └── fake/
│           │
│           └── test/
│               ├── real/
│               └── fake/
│
├── models/
│   └── best_model.pth
│
├── results/
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Dataset

**Dataset Used**

140K Real and Fake Faces Dataset

Dataset Statistics

| Split | Images |
|--------|--------|
| Training | 100,000 |
| Testing | 20,000 |

Due to project deadline and computational constraints, a subset was used during training.

| Split | Images Used |
|--------|-------------|
| Training | 10,000 |
| Testing | 2,000 |

Classes

- Real
- Fake

---

## Model

Pretrained Backbone

```
google/vit-base-patch16-224
```

The classifier head was replaced to perform binary classification while the pretrained ImageNet weights were fine-tuned using transfer learning.

---

## Hyperparameters

| Parameter | Value |
|------------|-------|
| Epochs | 2 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |
| Image Size | 224 × 224 |
| Device | NVIDIA RTX 4050 Laptop GPU |

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- torchvision
- scikit-learn
- matplotlib
- NumPy
- tqdm

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-image-detection-vit.git

cd deepfake-image-detection-vit
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Place the dataset inside

```text
dataset/
```

Expected structure

```text
dataset
└── real_vs_fake
    └── real-vs-fake
        ├── train
        │   ├── fake
        │   └── real
        │
        └── test
            ├── fake
            └── real
```

---

## Training

Run

```bash
python train.py
```

The training script automatically

- Loads dataset
- Applies preprocessing
- Fine-tunes ViT
- Evaluates on test data
- Saves the best model
- Generates confusion matrix
- Stores evaluation metrics

---

## Results

| Metric | Score |
|---------|-------|
| Accuracy | **98.10%** |
| Precision | **96.77%** |
| Recall | **99.50%** |
| F1 Score | **98.12%** |

---

## Sample Pipeline

```text
Dataset
      │
      ▼
Image Preprocessing
      │
      ▼
Resize (224×224)
      │
      ▼
Normalization
      │
      ▼
Vision Transformer
      │
      ▼
Classification Head
      │
      ▼
Prediction
      │
      ▼
Real / Fake
```

---

## Future Improvements

- Train using the complete 140K dataset
- Increase training epochs
- Compare with CNN architectures such as XceptionNet and EfficientNet
- Cross-dataset evaluation
- Video-based deepfake detection
- Explainable AI using Grad-CAM
- Lightweight transformer deployment

---

## References

- Dosovitskiy et al. — *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*
- FaceForensics++
- Celeb-DF
- DeepFake Detection Challenge (DFDC)
- Hugging Face Transformers Documentation
- PyTorch Documentation

---

## Author

**Divyan Babbar**

B.Tech EC-ACT

Jaypee Institute of Information Technology

Email: **divyanbabbar453@gmail.com**

---

## License

This project is developed for educational and academic purposes.
