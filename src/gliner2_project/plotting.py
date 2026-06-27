"""Matplotlib plotting helpers for schema wording experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _save_current(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_metric_bar(
    results_summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a method comparison bar chart for a metric."""

    frame = results_summary.sort_values(metric, ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(frame["method"], frame[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    return _save_current(Path(output_path))


def plot_schema_disagreement_histogram(
    disagreement: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a histogram of unique predicted label counts across variants."""

    plt.figure(figsize=(7, 4))
    if disagreement.empty:
        values = []
    else:
        values = disagreement["unique_prediction_count"]
    plt.hist(values, bins=[0.5, 1.5, 2.5, 3.5, 4.5], rwidth=0.85)
    plt.xlabel("Unique predicted labels across schema variants")
    plt.ylabel("Examples")
    plt.title("Schema Disagreement")
    return _save_current(Path(output_path))


def plot_top_label_improvements(
    deltas: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> Path:
    """Save the top per-label F1 improvements."""

    frame = deltas.sort_values("f1_delta", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(frame["label"], frame["f1_delta"])
    plt.gca().invert_yaxis()
    plt.xlabel("F1 improvement over plain_label")
    plt.title(f"Top {top_n} Label Improvements")
    return _save_current(Path(output_path))


def plot_confusion_matrix_for_pairs(
    rows: Sequence[dict],
    labels: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a compact confusion matrix limited to selected labels."""

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get("predicted_canonical")
        if gold in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[gold]][label_to_idx[pred]] += 1

    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    plt.colorbar()
    return _save_current(Path(output_path))


def plot_candidate_count_distribution(
    candidate_counts: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a candidate-count distribution bar chart."""

    plt.figure(figsize=(7, 4))
    if candidate_counts.empty:
        labels = []
        counts = []
    else:
        labels = candidate_counts["num_candidates"].astype(str)
        counts = candidate_counts["count"]
    plt.bar(labels, counts)
    plt.xlabel("Number of retrieved candidate labels")
    plt.ylabel("Examples")
    plt.title("Candidate Count Distribution")
    return _save_current(Path(output_path))


def plot_candidate_recall_by_label(
    candidate_rows: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 30,
) -> Path:
    """Save label-level candidate recall sorted ascending."""

    plt.figure(figsize=(10, 7))
    if candidate_rows.empty:
        frame = pd.DataFrame(columns=["gold_label", "candidate_recall"])
    else:
        frame = (
            candidate_rows.groupby("gold_label")["gold_in_candidates"]
            .mean()
            .reset_index(name="candidate_recall")
            .sort_values("candidate_recall", ascending=True)
            .head(top_n)
        )
    plt.barh(frame["gold_label"], frame["candidate_recall"])
    plt.xlabel("Candidate recall")
    plt.title(f"Lowest {top_n} Labels By Candidate Recall")
    return _save_current(Path(output_path))


def plot_overlap_matrix(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a method-by-method overlap matrix."""

    plt.figure(figsize=(9, 8))
    if matrix.empty:
        values = [[0]]
        labels = [""]
    else:
        values = matrix.to_numpy(dtype=float)
        labels = list(matrix.index)
    plt.imshow(values)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Method")
    plt.ylabel("Method")
    plt.title(title)
    plt.colorbar()
    return _save_current(Path(output_path))
