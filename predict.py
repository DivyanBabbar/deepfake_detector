import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(pixel_values=image).logits
    prediction = torch.argmax(output, dim=1).item()

if prediction == 0:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")