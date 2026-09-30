import torch
from PIL import Image

import predict


def test_predict_image_returns_label_and_confidence(tiny_model, tmp_path):
    image_path = tmp_path / "face.png"
    Image.new("RGB", (64, 64), "gray").save(image_path)

    label, confidence = predict.predict_image(
        tiny_model, str(image_path), device=torch.device("cpu")
    )

    assert label in {"FAKE", "REAL"}
    assert 0.5 <= confidence <= 1.0
