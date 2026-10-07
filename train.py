import os
import random

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from transformers import ViTForImageClassification

from common import build_transform
from utils import evaluate

# ==========================
# CONFIG
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "dataset/real_vs_fake/real-vs-fake/train"
TEST_DIR = "dataset/real_vs_fake/real-vs-fake/test"

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

# The full dataset has 100,000 train / 20,000 test images.
# A random subset is used because of time and GPU constraints (RTX 4050, 6 GB).
TRAIN_SUBSET_SIZE = 10000
TEST_SUBSET_SIZE = 2000
SEED = 42


# ==========================
# DATA
# ==========================

def build_loaders(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    train_subset=TRAIN_SUBSET_SIZE,
    test_subset=TEST_SUBSET_SIZE,
    batch_size=BATCH_SIZE,
    validation_fraction=0.2,
):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    transform = build_transform()
    full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    full_test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    rng = random.Random(SEED)
    train_indices = rng.sample(range(len(full_train_dataset)), min(train_subset, len(full_train_dataset)))
    test_indices = rng.sample(range(len(full_test_dataset)), min(test_subset, len(full_test_dataset)))

    # Split only the training directory; keep the official test directory untouched
    # until the selected checkpoint is evaluated after training.
    rng.shuffle(train_indices)
    validation_size = max(1, int(len(train_indices) * validation_fraction)) if len(train_indices) > 1 else 0
    validation_indices = train_indices[:validation_size]
    fit_indices = train_indices[validation_size:]
    train_dataset = Subset(full_train_dataset, fit_indices)
    validation_dataset = Subset(full_train_dataset, validation_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    print("Classes:", full_train_dataset.classes)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    # num_workers=0: on Windows, DataLoader worker processes are started with
    # "spawn", which caused multiprocessing crashes. Loading in the main
    # process avoids that, at the cost of slower data loading.
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)
    return train_loader, validation_loader, test_loader


# ==========================
# TRAIN ONE EPOCH
# ==========================

def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ==========================
# EVALUATION
# ==========================

def validate(model, loader, device=DEVICE, output_dir="results"):
    model.eval()

    predictions = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images).logits
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    evaluate(labels_list, predictions, output_dir=output_dir)

    return accuracy_score(labels_list, predictions)


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Using device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, validation_loader, test_loader = build_loaders()

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Training Loss: {loss:.4f}")

        acc = validate(model, validation_loader, output_dir="results/validation")
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/best_model.pth")
            print("Best model saved.")

    if not os.path.exists("models/best_model.pth"):
        raise RuntimeError("Training produced no checkpoint; validation set may be empty.")
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    test_acc = validate(model, test_loader, output_dir="results/test")

    print("\nTraining Complete!")
    print("Best Validation Accuracy:", best_acc)
    print("Final Test Accuracy:", test_acc)


if __name__ == "__main__":
    main()
