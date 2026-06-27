from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.cluster_second_pass import (  # noqa: E402
    build_second_pass_plan,
    merge_second_pass_prediction,
    second_pass_summary_frame,
    should_run_second_pass,
)


def row(example_id, gold, pred, latency=0.1):
    return {
        "example_id": example_id,
        "text": f"text {example_id}",
        "gold_label": gold,
        "predicted_canonical": pred,
        "latency_sec": latency,
    }


def test_second_pass_trigger_detection():
    assert should_run_second_pass("card_arrival")
    assert not should_run_second_pass("unknown_label")
    assert not should_run_second_pass(None)


def test_second_pass_plan_contains_cluster_labels():
    plan = build_second_pass_plan([row(1, "card_arrival", "card_arrival")], "plain_label")
    assert plan[0]["run_second_pass"]
    assert plan[0]["cluster_name"] == "card_lifecycle"
    assert "card_delivery_estimate" in plan[0]["cluster_labels"]


def test_merge_keeps_first_prediction_when_no_cluster_applies():
    first = row(1, "unknown_label", "unknown_label")
    merged = merge_second_pass_prediction(first, None, "cluster_second_pass_from_plain")
    assert merged["predicted_canonical"] == "unknown_label"
    assert merged["second_pass_triggered"] is False


def test_merge_uses_second_prediction_when_available():
    first = row(1, "card_delivery_estimate", "card_arrival", latency=0.2)
    second = row(1, "card_delivery_estimate", "card_delivery_estimate", latency=0.3)
    merged = merge_second_pass_prediction(
        first,
        second,
        "cluster_second_pass_from_plain",
        cluster_name="card_lifecycle",
    )
    assert merged["predicted_canonical"] == "card_delivery_estimate"
    assert merged["second_pass_triggered"] is True
    assert merged["latency_sec"] == 0.5


def test_second_pass_summary_frame_counts_triggered_accuracy():
    rows = [
        {
            **row(1, "a", "a"),
            "second_pass_triggered": False,
        },
        {
            **row(2, "b", "b"),
            "second_pass_triggered": True,
        },
    ]
    summary = second_pass_summary_frame(rows).iloc[0]
    assert summary["second_pass_triggered"] == 1
    assert summary["accuracy_when_triggered"] == 1.0
