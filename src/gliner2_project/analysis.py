"""Analysis helpers for schema wording robustness experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import pandas as pd
from sklearn.metrics import f1_score


def rows_to_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert prediction rows into a DataFrame with stable expected columns."""

    frame = pd.DataFrame(list(rows))
    for column in ["example_id", "text", "gold_label", "predicted_canonical", "method"]:
        if column not in frame.columns:
            frame[column] = None
    return frame


def method_metrics_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    effective_latency_components: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Build method-level metrics from prediction rows."""

    rows: list[dict[str, Any]] = []
    for method, predictions in predictions_by_method.items():
        frame = rows_to_frame(predictions)
        total = len(frame)
        parse_failures = int(frame["predicted_canonical"].isna().sum()) if total else 0

        if total:
            gold = frame["gold_label"].astype(str)
            pred = frame["predicted_canonical"].fillna("__UNKNOWN__").astype(str)
            accuracy = float((gold == pred).mean())
            macro_f1 = float(f1_score(gold, pred, average="macro", zero_division=0))
            weighted_f1 = float(
                f1_score(gold, pred, average="weighted", zero_division=0)
            )
            avg_latency = (
                float(frame["latency_sec"].mean())
                if "latency_sec" in frame.columns
                else None
            )
            total_runtime = (
                float(frame["latency_sec"].sum())
                if "latency_sec" in frame.columns
                else None
            )
        else:
            accuracy = 0.0
            macro_f1 = 0.0
            weighted_f1 = 0.0
            avg_latency = None
            total_runtime = None

        rows.append(
            {
                "method": method,
                "num_examples": total,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "effective_latency_sec": avg_latency,
                "average_latency_sec": avg_latency,
                "total_runtime_sec": total_runtime,
                "parse_failure_rate": float(parse_failures / total) if total else 0.0,
                "parse_failures": parse_failures,
            }
        )

    frame = pd.DataFrame(rows)
    if effective_latency_components and not frame.empty:
        average_latency_by_method = dict(
            zip(frame["method"], frame["average_latency_sec"])
        )
        for method, components in effective_latency_components.items():
            component_latencies = [
                average_latency_by_method.get(component)
                for component in components
            ]
            if all(pd.notna(value) for value in component_latencies):
                frame.loc[
                    frame["method"] == method,
                    "effective_latency_sec",
                ] = float(sum(component_latencies))

    column_order = [
        "method",
        "num_examples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "effective_latency_sec",
        "parse_failure_rate",
    ]
    return frame[[column for column in column_order if column in frame.columns]]


def per_label_metrics_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Compute per-label F1 for every method."""

    output_rows: list[dict[str, Any]] = []
    all_labels = sorted(
        {
            str(row.get("gold_label"))
            for rows in predictions_by_method.values()
            for row in rows
            if row.get("gold_label") is not None
        }
    )

    for method, predictions in predictions_by_method.items():
        frame = rows_to_frame(predictions)
        if frame.empty:
            continue
        gold = frame["gold_label"].astype(str)
        pred = frame["predicted_canonical"].fillna("__UNKNOWN__").astype(str)
        scores = f1_score(
            gold,
            pred,
            labels=all_labels,
            average=None,
            zero_division=0,
        )
        support = gold.value_counts().to_dict()
        for label, score in zip(all_labels, scores):
            output_rows.append(
                {
                    "method": method,
                    "label": label,
                    "f1": float(score),
                    "support": int(support.get(label, 0)),
                }
            )

    return pd.DataFrame(output_rows)


def schema_disagreement_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    schema_methods: Iterable[str],
) -> pd.DataFrame:
    """Summarize per-example disagreement across schema variants."""

    indexes = {
        method: {row.get("example_id"): row for row in predictions_by_method.get(method, [])}
        for method in schema_methods
    }
    if not indexes:
        return pd.DataFrame()

    first_method = next(iter(indexes))
    rows: list[dict[str, Any]] = []
    for example_id in indexes[first_method]:
        method_predictions = {
            method: indexes[method].get(example_id, {}).get("predicted_canonical")
            for method in indexes
        }
        labels = [label for label in method_predictions.values() if label is not None]
        unique_labels = sorted(set(labels))
        base = indexes[first_method].get(example_id, {})
        rows.append(
            {
                "example_id": example_id,
                "text": base.get("text"),
                "gold_label": base.get("gold_label"),
                "method_predictions": method_predictions,
                "unique_prediction_count": len(unique_labels),
                "unique_predictions": unique_labels,
                "all_agree": len(unique_labels) <= 1,
            }
        )
    return pd.DataFrame(rows)


def compare_method_examples(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    baseline_method: str,
    comparison_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return improved and degraded examples relative to a baseline method."""

    baseline = rows_to_frame(predictions_by_method.get(baseline_method, []))
    comparison = rows_to_frame(predictions_by_method.get(comparison_method, []))
    if baseline.empty or comparison.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = baseline[
        ["example_id", "text", "gold_label", "predicted_canonical"]
    ].merge(
        comparison[["example_id", "predicted_canonical"]],
        on="example_id",
        suffixes=("_baseline", "_comparison"),
    )
    baseline_correct = merged["gold_label"] == merged["predicted_canonical_baseline"]
    comparison_correct = merged["gold_label"] == merged["predicted_canonical_comparison"]
    improved = merged[(~baseline_correct) & comparison_correct].copy()
    degraded = merged[baseline_correct & (~comparison_correct)].copy()
    improved["baseline_method"] = baseline_method
    improved["comparison_method"] = comparison_method
    degraded["baseline_method"] = baseline_method
    degraded["comparison_method"] = comparison_method
    return improved, degraded


def paired_outcome_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    baseline_method: str,
    comparison_method: str,
) -> pd.DataFrame:
    """Return per-example paired correctness outcomes for two methods."""

    baseline = rows_to_frame(predictions_by_method.get(baseline_method, []))
    comparison = rows_to_frame(predictions_by_method.get(comparison_method, []))
    if baseline.empty or comparison.empty:
        return pd.DataFrame()

    merged = baseline[
        ["example_id", "text", "gold_label", "predicted_canonical"]
    ].merge(
        comparison[["example_id", "predicted_canonical"]],
        on="example_id",
        suffixes=("_baseline", "_comparison"),
    )
    merged["baseline_method"] = baseline_method
    merged["comparison_method"] = comparison_method
    merged["baseline_correct"] = (
        merged["gold_label"] == merged["predicted_canonical_baseline"]
    )
    merged["comparison_correct"] = (
        merged["gold_label"] == merged["predicted_canonical_comparison"]
    )

    def outcome(row: pd.Series) -> str:
        if (not row["baseline_correct"]) and row["comparison_correct"]:
            return "baseline_wrong_comparison_correct"
        if row["baseline_correct"] and (not row["comparison_correct"]):
            return "baseline_correct_comparison_wrong"
        if row["baseline_correct"] and row["comparison_correct"]:
            return "both_correct"
        return "both_wrong"

    merged["paired_outcome"] = merged.apply(outcome, axis=1)
    return merged


def paired_summary_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    baseline_method: str,
    comparison_methods: Sequence[str],
) -> pd.DataFrame:
    """Return paired correctness counts and net gains for comparisons."""

    rows: list[dict[str, Any]] = []
    for comparison_method in comparison_methods:
        paired = paired_outcome_frame(
            predictions_by_method,
            baseline_method,
            comparison_method,
        )
        counts = paired["paired_outcome"].value_counts().to_dict()
        improved = int(counts.get("baseline_wrong_comparison_correct", 0))
        degraded = int(counts.get("baseline_correct_comparison_wrong", 0))
        rows.append(
            {
                "baseline_method": baseline_method,
                "comparison_method": comparison_method,
                "plain_wrong_comparison_correct": improved,
                "plain_correct_comparison_wrong": degraded,
                "both_correct": int(counts.get("both_correct", 0)),
                "both_wrong": int(counts.get("both_wrong", 0)),
                "net_gain": improved - degraded,
                "num_examples": int(len(paired)),
            }
        )
    return pd.DataFrame(rows)


def confusion_pairs_frame(
    rows: Sequence[dict[str, Any]],
    top_n: int | None = None,
) -> pd.DataFrame:
    """Count incorrect gold/predicted label pairs."""

    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get("predicted_canonical")
        if gold is None or pred is None or gold == pred:
            continue
        counter[(str(gold), str(pred))] += 1

    items = counter.most_common(top_n)
    return pd.DataFrame(
        [
            {"gold_label": gold, "predicted_canonical": pred, "count": count}
            for (gold, pred), count in items
        ]
    )


def per_label_delta_frame(
    per_label_metrics: pd.DataFrame,
    baseline_method: str,
    comparison_method: str,
) -> pd.DataFrame:
    """Compute per-label F1 deltas between a comparison and baseline method."""

    if per_label_metrics.empty:
        return pd.DataFrame()
    baseline = per_label_metrics[per_label_metrics["method"] == baseline_method]
    comparison = per_label_metrics[per_label_metrics["method"] == comparison_method]
    merged = baseline[["label", "f1", "support"]].merge(
        comparison[["label", "f1"]],
        on="label",
        suffixes=("_baseline", "_comparison"),
    )
    merged["f1_delta"] = merged["f1_comparison"] - merged["f1_baseline"]
    merged["baseline_method"] = baseline_method
    merged["comparison_method"] = comparison_method
    return merged.sort_values("f1_delta", ascending=False)


def confidence_analysis_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Return correct/incorrect average confidence where confidence exists."""

    frame = rows_to_frame(rows)
    if frame.empty or "confidence" not in frame.columns:
        return pd.DataFrame()
    with_conf = frame[frame["confidence"].notna()].copy()
    if with_conf.empty:
        return pd.DataFrame()
    with_conf["is_correct"] = (
        with_conf["gold_label"] == with_conf["predicted_canonical"]
    )
    return (
        with_conf.groupby("is_correct", dropna=False)["confidence"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
