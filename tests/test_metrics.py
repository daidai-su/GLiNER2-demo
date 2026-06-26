from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.eval_utils import compute_classification_metrics  # noqa: E402


def test_accuracy_calculation():
    metrics = compute_classification_metrics(["a", "b", "c"], ["a", "x", "c"])
    assert metrics["accuracy"] == 2 / 3


def test_macro_f1_calculation():
    metrics = compute_classification_metrics(["a", "b"], ["a", "b"])
    assert metrics["macro_f1"] == 1.0


def test_empty_prediction_handling():
    metrics = compute_classification_metrics([], [])
    assert metrics == {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
