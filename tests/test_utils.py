import pytest

from utils import evaluate


def test_evaluate_metrics(tmp_path):
    metrics = evaluate([0, 1, 1, 0], [0, 1, 0, 0], output_dir=str(tmp_path))

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_evaluate_writes_outputs(tmp_path):
    evaluate([0, 1, 1, 0], [0, 1, 0, 0], output_dir=str(tmp_path))

    assert "Accuracy" in (tmp_path / "metrics.txt").read_text()
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 0


def test_evaluate_handles_no_positive_predictions(tmp_path):
    metrics = evaluate([0, 1], [0, 0], output_dir=str(tmp_path))

    assert metrics["precision"] == 0.0
