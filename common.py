from torchvision import transforms

IMAGE_SIZE = 224
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# ImageFolder assigns class indices alphabetically: fake -> 0, real -> 1.
LABELS = {0: "FAKE", 1: "REAL"}


def build_transform():
    """Preprocessing shared by training and prediction."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
