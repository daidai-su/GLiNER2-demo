"""Evaluation utilities for the GLiNER2 smoke-test baseline."""

from __future__ import annotations

from typing import Any, Sequence

from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    gold_labels: Sequence[str | None],
    predicted_labels: Sequence[str | None],
) -> dict[str, float]:
    """Compute accuracy, macro F1, and weighted F1 with explicit empty handling."""

    if len(gold_labels) != len(predicted_labels):
        raise ValueError("gold_labels and predicted_labels must have the same length.")
    if not gold_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    gold = [str(label) for label in gold_labels]
    predicted = [
        "__UNKNOWN__" if label is None or label == "" else str(label)
        for label in predicted_labels
    ]

    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(gold, predicted, average="weighted", zero_division=0)
        ),
    }


def collect_confusion_examples(
    examples: Sequence[dict[str, Any]],
    predicted_labels: Sequence[str | None],
    max_examples: int = 10,
) -> list[dict[str, Any]]:
    """Collect a compact set of incorrect predictions for inspection."""

    mistakes: list[dict[str, Any]] = []
    for example, predicted in zip(examples, predicted_labels):
        gold = example.get("label_text")
        if gold == predicted:
            continue
        mistakes.append(
            {
                "example_id": example.get("example_id"),
                "text": example.get("text"),
                "gold_label": gold,
                "predicted_label": predicted,
            }
        )
        if len(mistakes) >= max_examples:
            break
    return mistakes


def average_latency(latencies: Sequence[float]) -> float:
    """Return average latency in seconds, or 0.0 for an empty input."""

    if not latencies:
        return 0.0
    return float(sum(latencies) / len(latencies))
