"""Comparison helpers for GLiNER2, retrieval, and classical baselines."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .analysis import rows_to_frame


METHOD_SETTINGS: dict[str, dict[str, Any]] = {
    "plain_label": {
        "setting": "pure_zero_shot",
        "uses_train_texts": False,
        "uses_train_labels": False,
        "trains_parameters": False,
        "uses_gliner2": True,
    },
    "mean_confidence_ensemble": {
        "setting": "pure_zero_shot",
        "uses_train_texts": False,
        "uses_train_labels": False,
        "trains_parameters": False,
        "uses_gliner2": True,
    },
    "tfidf_knn_majority": {
        "setting": "retrieval_only",
        "uses_train_texts": True,
        "uses_train_labels": True,
        "trains_parameters": False,
        "uses_gliner2": False,
    },
    "tfidf_weighted_knn": {
        "setting": "retrieval_only",
        "uses_train_texts": True,
        "uses_train_labels": True,
        "trains_parameters": False,
        "uses_gliner2": False,
    },
    "tfidf_logistic_regression": {
        "setting": "classical_supervised",
        "uses_train_texts": True,
        "uses_train_labels": True,
        "trains_parameters": True,
        "uses_gliner2": False,
    },
    "tfidf_linear_svm": {
        "setting": "classical_supervised",
        "uses_train_texts": True,
        "uses_train_labels": True,
        "trains_parameters": True,
        "uses_gliner2": False,
    },
    "tfidf_candidate_pruning_gliner2": {
        "setting": "retrieval_assisted_gliner",
        "uses_train_texts": True,
        "uses_train_labels": True,
        "trains_parameters": False,
        "uses_gliner2": True,
    },
    "tfidf_candidate_pruning_mean_confidence_ensemble": {
        "setting": "retrieval_assisted_gliner",
        "uses_train_texts": True,
        "uses_train_labels": True,
        "trains_parameters": False,
        "uses_gliner2": True,
    },
}


REQUIRED_SETTING_COLUMNS = [
    "method",
    "setting",
    "uses_train_texts",
    "uses_train_labels",
    "trains_parameters",
    "uses_gliner2",
]


def method_setting_table(methods: Sequence[str]) -> pd.DataFrame:
    """Return setting metadata for methods, using conservative defaults if unknown."""

    rows: list[dict[str, Any]] = []
    for method in methods:
        settings = METHOD_SETTINGS.get(
            method,
            {
                "setting": "unknown",
                "uses_train_texts": None,
                "uses_train_labels": None,
                "trains_parameters": None,
                "uses_gliner2": None,
            },
        )
        rows.append({"method": method, **settings})
    return pd.DataFrame(rows, columns=REQUIRED_SETTING_COLUMNS)


def _aligned_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = rows_to_frame(rows).copy()
    if "is_correct" not in frame.columns:
        frame["is_correct"] = frame["gold_label"] == frame["predicted_canonical"]
    return frame


def classical_results_summary(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    training_times: dict[str, float] | None = None,
    total_prediction_times: dict[str, float] | None = None,
    method_settings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a method-level comparison table with setting metadata."""

    rows: list[dict[str, Any]] = []
    training_times = training_times or {}
    total_prediction_times = total_prediction_times or {}
    for method, predictions in predictions_by_method.items():
        frame = _aligned_frame(predictions)
        total = len(frame)
        if total:
            gold = frame["gold_label"].astype(str)
            predicted = frame["predicted_canonical"].fillna("__UNKNOWN__").astype(str)
            accuracy = float((gold == predicted).mean())
            macro_f1 = float(f1_score(gold, predicted, average="macro", zero_division=0))
            weighted_f1 = float(
                f1_score(gold, predicted, average="weighted", zero_division=0)
            )
            parse_failures = int(frame["predicted_canonical"].isna().sum())
            average_latency = (
                float(frame["latency_sec"].mean())
                if "latency_sec" in frame.columns and frame["latency_sec"].notna().any()
                else None
            )
            total_prediction_time = total_prediction_times.get(method)
            if total_prediction_time is None and "latency_sec" in frame.columns:
                total_prediction_time = float(frame["latency_sec"].fillna(0.0).sum())
        else:
            accuracy = 0.0
            macro_f1 = 0.0
            weighted_f1 = 0.0
            parse_failures = 0
            average_latency = None
            total_prediction_time = total_prediction_times.get(method)

        rows.append(
            {
                "method": method,
                "num_examples": total,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "parse_failure_rate": float(parse_failures / total) if total else 0.0,
                "training_time_sec": training_times.get(method),
                "average_latency_sec": average_latency,
                "total_prediction_time_sec": total_prediction_time,
            }
        )

    summary = pd.DataFrame(rows)
    settings = (
        method_setting_table(list(predictions_by_method))
        if method_settings is None
        else method_settings
    )
    summary = settings.merge(summary, on="method", how="right")
    column_order = [
        "method",
        "setting",
        "uses_train_texts",
        "uses_train_labels",
        "trains_parameters",
        "uses_gliner2",
        "num_examples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "parse_failure_rate",
        "training_time_sec",
        "average_latency_sec",
        "total_prediction_time_sec",
    ]
    return summary[[column for column in column_order if column in summary.columns]]


def _merge_pair(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    method_a: str,
    method_b: str,
) -> pd.DataFrame:
    frame_a = _aligned_frame(predictions_by_method.get(method_a, []))
    frame_b = _aligned_frame(predictions_by_method.get(method_b, []))
    if frame_a.empty or frame_b.empty:
        return pd.DataFrame()
    merged = frame_a[
        ["example_id", "text", "gold_label", "predicted_canonical"]
    ].merge(
        frame_b[["example_id", "predicted_canonical"]],
        on="example_id",
        suffixes=("_a", "_b"),
    )
    merged["method_a"] = method_a
    merged["method_b"] = method_b
    merged["a_correct"] = merged["gold_label"] == merged["predicted_canonical_a"]
    merged["b_correct"] = merged["gold_label"] == merged["predicted_canonical_b"]
    return merged


def paired_comparison_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    pairs: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Return paired correctness counts for method pairs.

    Net gain is defined as B-correct/A-wrong minus A-correct/B-wrong.
    """

    rows: list[dict[str, Any]] = []
    for method_a, method_b in pairs:
        merged = _merge_pair(predictions_by_method, method_a, method_b)
        if merged.empty:
            continue
        gold = merged["gold_label"].astype(str)
        pred_a = merged["predicted_canonical_a"].fillna("__UNKNOWN__").astype(str)
        pred_b = merged["predicted_canonical_b"].fillna("__UNKNOWN__").astype(str)
        a_correct = merged["a_correct"].astype(bool)
        b_correct = merged["b_correct"].astype(bool)
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "num_examples": int(len(merged)),
                "method_a_correct": int(a_correct.sum()),
                "method_b_correct": int(b_correct.sum()),
                "both_correct": int((a_correct & b_correct).sum()),
                "both_wrong": int((~a_correct & ~b_correct).sum()),
                "a_correct_b_wrong": int((a_correct & ~b_correct).sum()),
                "a_wrong_b_correct": int((~a_correct & b_correct).sum()),
                "net_gain_of_b_over_a": int((~a_correct & b_correct).sum())
                - int((a_correct & ~b_correct).sum()),
                "accuracy_delta": float(b_correct.mean() - a_correct.mean()),
                "macro_f1_delta": float(
                    f1_score(gold, pred_b, average="macro", zero_division=0)
                    - f1_score(gold, pred_a, average="macro", zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_accuracy_ci(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    pairs: Sequence[tuple[str, str]],
    n_samples: int = 1000,
    seed: int = 42,
    include_macro_f1: bool = True,
) -> pd.DataFrame:
    """Compute paired bootstrap confidence intervals for metric deltas."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for method_a, method_b in pairs:
        merged = _merge_pair(predictions_by_method, method_a, method_b)
        if merged.empty:
            continue
        gold = merged["gold_label"].astype(str).to_numpy()
        pred_a = merged["predicted_canonical_a"].fillna("__UNKNOWN__").astype(str).to_numpy()
        pred_b = merged["predicted_canonical_b"].fillna("__UNKNOWN__").astype(str).to_numpy()
        a_correct = pred_a == gold
        b_correct = pred_b == gold
        n = len(merged)
        acc_deltas: list[float] = []
        macro_deltas: list[float] = []
        for _ in range(n_samples):
            sample_idx = rng.integers(0, n, size=n)
            acc_deltas.append(
                float(b_correct[sample_idx].mean() - a_correct[sample_idx].mean())
            )
            if include_macro_f1:
                macro_deltas.append(
                    float(
                        f1_score(
                            gold[sample_idx],
                            pred_b[sample_idx],
                            average="macro",
                            zero_division=0,
                        )
                        - f1_score(
                            gold[sample_idx],
                            pred_a[sample_idx],
                            average="macro",
                            zero_division=0,
                        )
                    )
                )

        row: dict[str, Any] = {
            "method_a": method_a,
            "method_b": method_b,
            "num_examples": n,
            "bootstrap_samples": n_samples,
            "accuracy_delta_mean": float(np.mean(acc_deltas)),
            "accuracy_delta_ci_low": float(np.percentile(acc_deltas, 2.5)),
            "accuracy_delta_ci_high": float(np.percentile(acc_deltas, 97.5)),
        }
        if include_macro_f1:
            row.update(
                {
                    "macro_f1_delta_mean": float(np.mean(macro_deltas)),
                    "macro_f1_delta_ci_low": float(np.percentile(macro_deltas, 2.5)),
                    "macro_f1_delta_ci_high": float(np.percentile(macro_deltas, 97.5)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_analysis_frame(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    groups: dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Return oracle upper-bound accuracy for named method groups."""

    rows: list[dict[str, Any]] = []
    for group_name, methods in groups.items():
        available = [method for method in methods if method in predictions_by_method]
        if not available:
            continue
        base = _aligned_frame(predictions_by_method[available[0]])[
            ["example_id", "gold_label"]
        ].copy()
        for method in available:
            frame = _aligned_frame(predictions_by_method[method])[
                ["example_id", "predicted_canonical"]
            ].rename(columns={"predicted_canonical": method})
            base = base.merge(frame, on="example_id", how="inner")
        if base.empty:
            continue
        prediction_columns = available
        oracle_correct = base.apply(
            lambda row: any(row[method] == row["gold_label"] for method in prediction_columns),
            axis=1,
        )
        rows.append(
            {
                "oracle_group": group_name,
                "methods": ", ".join(available),
                "num_examples": int(len(base)),
                "oracle_accuracy": float(oracle_correct.mean()),
                "oracle_correct": int(oracle_correct.sum()),
            }
        )
    return pd.DataFrame(rows)


def overlap_matrix(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    correct: bool = False,
) -> pd.DataFrame:
    """Return method-by-method overlap counts for wrong or correct examples."""

    methods = list(predictions_by_method)
    sets: dict[str, set[Any]] = {}
    for method in methods:
        frame = _aligned_frame(predictions_by_method[method])
        if frame.empty:
            sets[method] = set()
            continue
        mask = frame["gold_label"] == frame["predicted_canonical"]
        if not correct:
            mask = ~mask
        sets[method] = set(frame.loc[mask, "example_id"])

    matrix = pd.DataFrame(index=methods, columns=methods, dtype=int)
    for row_method in methods:
        for col_method in methods:
            matrix.loc[row_method, col_method] = len(
                sets[row_method].intersection(sets[col_method])
            )
    return matrix


def per_label_metrics(
    predictions_by_method: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Compute per-label precision, recall, F1, and support for every method."""

    all_labels = sorted(
        {
            str(row.get("gold_label"))
            for rows in predictions_by_method.values()
            for row in rows
            if row.get("gold_label") is not None
        }
    )
    rows: list[dict[str, Any]] = []
    for method, predictions in predictions_by_method.items():
        frame = _aligned_frame(predictions)
        if frame.empty:
            continue
        gold = frame["gold_label"].astype(str)
        predicted = frame["predicted_canonical"].fillna("__UNKNOWN__").astype(str)
        precision, recall, f1, support = precision_recall_fscore_support(
            gold,
            predicted,
            labels=all_labels,
            zero_division=0,
        )
        for idx, label in enumerate(all_labels):
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )
    return pd.DataFrame(rows)


def per_label_delta(
    metrics: pd.DataFrame,
    baseline_method: str,
    comparison_method: str,
) -> pd.DataFrame:
    """Compare per-label metrics between two methods."""

    if metrics.empty:
        return pd.DataFrame()
    baseline = metrics[metrics["method"] == baseline_method]
    comparison = metrics[metrics["method"] == comparison_method]
    merged = baseline[["label", "precision", "recall", "f1", "support"]].merge(
        comparison[["label", "precision", "recall", "f1"]],
        on="label",
        suffixes=("_baseline", "_comparison"),
    )
    for metric in ["precision", "recall", "f1"]:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_comparison"] - merged[f"{metric}_baseline"]
        )
    merged["baseline_method"] = baseline_method
    merged["comparison_method"] = comparison_method
    return merged.sort_values("f1_delta", ascending=False)


def confusion_pairs(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    top_n: int = 30,
) -> pd.DataFrame:
    """Count top gold/predicted confusion pairs for each method."""

    frames: list[pd.DataFrame] = []
    for method, predictions in predictions_by_method.items():
        counter: Counter[tuple[str, str]] = Counter()
        for row in predictions:
            gold = row.get("gold_label")
            pred = row.get("predicted_canonical")
            if gold is None or pred is None or str(gold) == str(pred):
                continue
            counter[(str(gold), str(pred))] += 1
        rows = [
            {
                "method": method,
                "gold_label": gold,
                "predicted_label": pred,
                "count": count,
            }
            for (gold, pred), count in counter.most_common(top_n)
        ]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["method", "gold_label", "predicted_label", "count"])
    return pd.concat(frames, ignore_index=True)


def improved_degraded_vs_baseline(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    baseline_method: str,
    comparison_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return examples improved and degraded by comparison_method."""

    merged = _merge_pair(predictions_by_method, baseline_method, comparison_method)
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    improved = merged[(~merged["a_correct"]) & merged["b_correct"]].copy()
    degraded = merged[merged["a_correct"] & (~merged["b_correct"])].copy()
    improved["baseline_method"] = baseline_method
    improved["comparison_method"] = comparison_method
    degraded["baseline_method"] = baseline_method
    degraded["comparison_method"] = comparison_method
    return improved, degraded
