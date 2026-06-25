import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===========================
# CONFIG
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "dataset/real_vs_fake/real-vs-fake/train"
TEST_DIR = "dataset/real_vs_fake/real-vs-fake/test"

BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 2
LEARNING_RATE = 2e-5

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Using Device:", DEVICE)
# ===========================
# DATASET
# ===========================

from torch.utils.data import DataLoader, Subset
import random

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

# ---------- USE SMALLER SUBSET ----------
random.seed(42)

train_subset_size = 10000
test_subset_size = 2000

train_indices = random.sample(
    range(len(train_dataset)),
    train_subset_size
)

test_indices = random.sample(
    range(len(test_dataset)),
    test_subset_size
)

train_dataset = Subset(train_dataset, train_indices)
test_dataset = Subset(test_dataset, test_indices)

print("Classes:", ["fake", "real"])
print("Train Images:", len(train_dataset))
print("Test Images:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ===========================
# LOAD PRETRAINED ViT
# ===========================

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print("ViT loaded successfully!")
# ===========================
# TRAIN FUNCTION
# ===========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(pixel_values=images).logits

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
# ===========================
# EVALUATION
# ===========================

from utils import evaluate

def validate(model, loader):

    model.eval()

    predictions = []
    labels_list = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(pixel_values=images).logits

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    evaluate(labels_list, predictions)

    return accuracy_score(labels_list, predictions)
# ===========================
# MAIN
# ===========================
def main():best_acc = 0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    print(f"Training Loss: {loss:.4f}")

    acc = validate(
        model,
        test_loader
    )

    print(f"Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

    print("Best model saved.")

print("\nTraining Complete!")
print("Best Accuracy:", best_acc)
if __name__ == "__main__":
    main()