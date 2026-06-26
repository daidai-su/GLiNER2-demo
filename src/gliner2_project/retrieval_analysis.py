"""Analysis helpers for retrieval-aided candidate pruning."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .analysis import confusion_pairs_frame, method_metrics_frame, rows_to_frame


def retrieval_results_summary(
    predictions_by_method: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Compute method metrics with retrieval-specific candidate statistics."""

    summary = method_metrics_frame(predictions_by_method)
    extra_rows: list[dict[str, Any]] = []
    for method, rows in predictions_by_method.items():
        frame = rows_to_frame(rows)
        if frame.empty or "num_candidates" not in frame.columns:
            extra_rows.append(
                {
                    "method": method,
                    "average_num_candidates": None,
                    "median_num_candidates": None,
                    "candidate_recall": None,
                    "accuracy_given_gold_in_candidates": None,
                    "accuracy_given_gold_not_in_candidates": None,
                }
            )
            continue

        with_candidates = frame[frame["num_candidates"].notna()].copy()
        if with_candidates.empty:
            extra_rows.append(
                {
                    "method": method,
                    "average_num_candidates": None,
                    "median_num_candidates": None,
                    "candidate_recall": None,
                    "accuracy_given_gold_in_candidates": None,
                    "accuracy_given_gold_not_in_candidates": None,
                }
            )
            continue

        with_candidates["is_correct"] = (
            with_candidates["gold_label"] == with_candidates["predicted_canonical"]
        )
        gold_known = with_candidates[with_candidates["gold_in_candidates"].notna()]
        gold_in = gold_known[gold_known["gold_in_candidates"].astype(bool)]
        gold_out = gold_known[~gold_known["gold_in_candidates"].astype(bool)]
        extra_rows.append(
            {
                "method": method,
                "average_num_candidates": float(with_candidates["num_candidates"].mean()),
                "median_num_candidates": float(with_candidates["num_candidates"].median()),
                "candidate_recall": (
                    float(gold_known["gold_in_candidates"].astype(bool).mean())
                    if not gold_known.empty
                    else None
                ),
                "accuracy_given_gold_in_candidates": (
                    float(gold_in["is_correct"].mean()) if not gold_in.empty else None
                ),
                "accuracy_given_gold_not_in_candidates": (
                    float(gold_out["is_correct"].mean()) if not gold_out.empty else None
                ),
            }
        )

    return summary.merge(pd.DataFrame(extra_rows), on="method", how="left")


def candidate_recall_analysis(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Summarize candidate recall and candidate counts."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "num_examples": int(len(frame)),
                "candidate_recall": float(frame["gold_in_candidates"].mean()),
                "average_num_candidates": float(frame["num_candidates"].mean()),
                "median_num_candidates": float(frame["num_candidates"].median()),
                "average_expanded_k_used": float(frame["expanded_k_used"].mean()),
            }
        ]
    )


def candidate_count_distribution(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Return count distributions for candidate count and expanded k."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame()
    count_dist = (
        frame["num_candidates"]
        .value_counts()
        .sort_index()
        .rename_axis("num_candidates")
        .reset_index(name="count")
    )
    return count_dist


def gold_missing_candidate_examples(
    rows: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Return examples where retrieval did not include the gold label."""

    frame = pd.DataFrame(list(rows))
    if frame.empty or "gold_in_candidates" not in frame.columns:
        return pd.DataFrame()
    return frame[~frame["gold_in_candidates"].astype(bool)].copy()


def retrieval_confusion_pairs(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    top_n: int = 50,
) -> pd.DataFrame:
    """Return top confusion pairs for every method."""

    frames: list[pd.DataFrame] = []
    for method, rows in predictions_by_method.items():
        pairs = confusion_pairs_frame(rows, top_n=top_n)
        if pairs.empty:
            continue
        pairs["method"] = method
        frames.append(pairs)
    if not frames:
        return pd.DataFrame(
            columns=["gold_label", "predicted_canonical", "count", "method"]
        )
    return pd.concat(frames, ignore_index=True)
