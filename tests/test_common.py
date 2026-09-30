from PIL import Image

from common import IMAGE_SIZE, LABELS, build_transform


def test_transform_output_shape():
    tensor = build_transform()(Image.new("RGB", (64, 48)))
    assert tuple(tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_label_mapping():
    assert LABELS == {0: "FAKE", 1: "REAL"}
