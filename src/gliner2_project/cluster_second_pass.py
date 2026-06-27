"""Cluster second-pass helpers for zero-shot GLiNER2 schema experiments."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .label_clusters import get_cluster_labels, get_cluster_name


CLUSTER_SECOND_PASS_FROM_PLAIN = "cluster_second_pass_from_plain"
CLUSTER_SECOND_PASS_FROM_MEAN_CONFIDENCE = "cluster_second_pass_from_mean_confidence"


def should_run_second_pass(predicted_label: str | None) -> bool:
    """Return whether a first-pass prediction belongs to a known confusion cluster."""

    return get_cluster_name(predicted_label) is not None


def build_second_pass_plan(
    first_pass_rows: Sequence[dict[str, Any]],
    source_method: str,
) -> list[dict[str, Any]]:
    """Build per-example second-pass decisions from first-pass predictions."""

    plan: list[dict[str, Any]] = []
    for row in first_pass_rows:
        predicted = row.get("predicted_canonical")
        cluster_name = get_cluster_name(predicted)
        plan.append(
            {
                "example_id": row.get("example_id"),
                "source_method": source_method,
                "first_pass_prediction": predicted,
                "cluster_name": cluster_name,
                "cluster_labels": get_cluster_labels(cluster_name),
                "run_second_pass": cluster_name is not None,
            }
        )
    return plan


def merge_second_pass_prediction(
    first_pass_row: dict[str, Any],
    second_pass_row: dict[str, Any] | None,
    output_method: str,
    cluster_name: str | None = None,
) -> dict[str, Any]:
    """Merge a first-pass prediction with an optional second-pass prediction."""

    first_prediction = first_pass_row.get("predicted_canonical")
    if second_pass_row is None:
        return {
            **first_pass_row,
            "method": output_method,
            "first_pass_prediction": first_prediction,
            "second_pass_prediction": None,
            "predicted_canonical": first_prediction,
            "second_pass_triggered": False,
            "cluster_name": cluster_name,
            "second_pass_latency_sec": 0.0,
            "latency_sec": float(first_pass_row.get("latency_sec") or 0.0),
        }

    second_prediction = second_pass_row.get("predicted_canonical")
    first_latency = float(first_pass_row.get("latency_sec") or 0.0)
    second_latency = float(second_pass_row.get("latency_sec") or 0.0)
    return {
        **first_pass_row,
        "method": output_method,
        "first_pass_prediction": first_prediction,
        "second_pass_prediction": second_prediction,
        "predicted_canonical": second_prediction or first_prediction,
        "second_pass_triggered": True,
        "cluster_name": cluster_name,
        "second_pass_latency_sec": second_latency,
        "latency_sec": first_latency + second_latency,
        "second_pass_parse_error": second_pass_row.get("parse_error"),
    }


def second_pass_summary_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Summarize trigger rate and accuracy when second pass is triggered."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame()
    if "second_pass_triggered" not in frame.columns:
        frame["second_pass_triggered"] = False
    frame["is_correct"] = frame["gold_label"] == frame["predicted_canonical"]
    triggered = frame[frame["second_pass_triggered"].astype(bool)]
    return pd.DataFrame(
        [
            {
                "num_examples": int(len(frame)),
                "second_pass_triggered": int(len(triggered)),
                "trigger_rate": float(len(triggered) / len(frame)) if len(frame) else 0.0,
                "accuracy_overall": float(frame["is_correct"].mean()) if len(frame) else 0.0,
                "accuracy_when_triggered": (
                    float(triggered["is_correct"].mean()) if not triggered.empty else None
                ),
            }
        ]
    )
