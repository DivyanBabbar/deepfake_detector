import argparse

import torch
from PIL import Image
from transformers import ViTForImageClassification

from common import LABELS, build_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/vit-base-patch16-224"
WEIGHTS_PATH = "models/best_model.pth"


def load_model(weights_path=WEIGHTS_PATH, device=DEVICE):
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, device=DEVICE):
    """Return (label, confidence) for one image."""
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(torch.argmax(probabilities).item())
    return LABELS[index], float(probabilities[index].item())


def main():
    parser = argparse.ArgumentParser(description="Classify a face image as real or fake.")
    parser.add_argument("image", nargs="?", help="Path to the image")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Path to model weights")
    args = parser.parse_args()

    image_path = args.image or input("Enter image path: ")

    model = load_model(args.weights)
    label, confidence = predict_image(model, image_path)
    print(f"Prediction: {label} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
