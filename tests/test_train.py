import torch
from PIL import Image

import train

CPU = torch.device("cpu")


def make_image_folder(root, per_class=6):
    for split in ("train", "test"):
        for cls, color in (("fake", "red"), ("real", "blue")):
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(per_class):
                Image.new("RGB", (32, 32), color).save(folder / f"{i}.png")


def test_build_loaders_uses_train_validation_test_and_no_workers(tmp_path):
    make_image_folder(tmp_path)

    train_loader, validation_loader, test_loader = train.build_loaders(
        train_dir=str(tmp_path / "train"),
        test_dir=str(tmp_path / "test"),
        train_subset=10,
        test_subset=2,
        batch_size=2,
    )

    assert len(train_loader.dataset) + len(validation_loader.dataset) == 10
    assert len(validation_loader.dataset) == 2
    assert len(test_loader.dataset) == 2
    assert train_loader.num_workers == validation_loader.num_workers == test_loader.num_workers == 0
    assert train_loader.dataset.dataset.classes == ["fake", "real"]

    images, labels = next(iter(train_loader))
    assert images.shape == (2, 3, 224, 224)
    assert set(labels.tolist()) <= {0, 1}


def test_subset_larger_than_dataset_is_capped(tmp_path):
    make_image_folder(tmp_path)

    train_loader, validation_loader, test_loader = train.build_loaders(
        train_dir=str(tmp_path / "train"),
        test_dir=str(tmp_path / "test"),
        train_subset=1000,
        test_subset=1000,
        batch_size=2,
    )

    assert len(train_loader.dataset) + len(validation_loader.dataset) == 12
    assert len(test_loader.dataset) == 12


def test_train_one_epoch_returns_finite_loss(tiny_model, fake_batches):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    loss = train.train_one_epoch(tiny_model, fake_batches, optimizer, criterion, device=CPU)

    assert 0 < loss < float("inf")


def test_validate_returns_accuracy_and_writes_results(tiny_model, fake_batches, tmp_path):
    acc = train.validate(tiny_model, fake_batches, device=CPU, output_dir=str(tmp_path))

    assert 0.0 <= acc <= 1.0
    assert (tmp_path / "metrics.txt").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
