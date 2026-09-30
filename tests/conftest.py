import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification


@pytest.fixture
def tiny_model():
    """A small randomly initialised ViT, so tests never download the real weights."""
    torch.manual_seed(0)
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
    )
    return ViTForImageClassification(config)


@pytest.fixture
def fake_batches():
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    return [(images, labels)]
