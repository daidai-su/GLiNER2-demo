from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.analysis import (  # noqa: E402
    method_metrics_frame,
    paired_summary_frame,
)


def row(example_id, gold, pred, method, latency=0.1):
    return {
        "example_id": example_id,
        "text": f"text {example_id}",
        "gold_label": gold,
        "predicted_canonical": pred,
        "method": method,
        "latency_sec": latency,
    }


def test_paired_summary_counts():
    predictions = {
        "plain_label": [
            row(1, "a", "a", "plain_label"),
            row(2, "b", "x", "plain_label"),
            row(3, "c", "c", "plain_label"),
            row(4, "d", "x", "plain_label"),
        ],
        "mean_confidence_ensemble": [
            row(1, "a", "a", "mean_confidence_ensemble"),
            row(2, "b", "b", "mean_confidence_ensemble"),
            row(3, "c", "x", "mean_confidence_ensemble"),
            row(4, "d", "x", "mean_confidence_ensemble"),
        ],
    }

    summary = paired_summary_frame(
        predictions,
        baseline_method="plain_label",
        comparison_methods=["mean_confidence_ensemble"],
    )
    result = summary.iloc[0].to_dict()
    assert result["plain_wrong_comparison_correct"] == 1
    assert result["plain_correct_comparison_wrong"] == 1
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["net_gain"] == 0


def test_effective_latency_for_ensemble_components():
    predictions = {
        "plain_label": [row(1, "a", "a", "plain_label", latency=0.1)],
        "query_about_label": [row(1, "a", "a", "query_about_label", latency=0.2)],
        "banking_request_label": [
            row(1, "a", "a", "banking_request_label", latency=0.3)
        ],
        "customer_intent_label": [
            row(1, "a", "a", "customer_intent_label", latency=0.4)
        ],
        "vote_ensemble": [
            {
                "example_id": 1,
                "text": "text 1",
                "gold_label": "a",
                "predicted_canonical": "a",
                "method": "vote_ensemble",
            }
        ],
    }

    metrics = method_metrics_frame(
        predictions,
        effective_latency_components={
            "vote_ensemble": [
                "plain_label",
                "query_about_label",
                "banking_request_label",
                "customer_intent_label",
            ]
        },
    )
    vote = metrics[metrics["method"] == "vote_ensemble"].iloc[0]
    assert vote["effective_latency_sec"] == 1.0
