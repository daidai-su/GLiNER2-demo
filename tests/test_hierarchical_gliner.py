from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.hierarchical_gliner import (  # noqa: E402
    build_coarse_group_schema,
    group_for_label,
    hierarchical_prediction_row,
    labels_for_hierarchical_second_stage,
    safety_group_union,
)


def test_coarse_group_schema_maps_candidates_to_group_names():
    candidates, mapping = build_coarse_group_schema()
    assert candidates
    assert set(mapping.values())
    assert mapping[candidates[0]] in {
        "card",
        "card_payment_refund",
        "cash_atm",
        "top_up",
        "transfer",
        "exchange_currency",
        "identity_security",
        "account_personal_details",
        "fees_charges",
        "region_support",
        "other",
    }


def test_group_for_label_is_deterministic():
    assert group_for_label("pending_transfer") == "transfer"
    assert group_for_label("card_arrival") == "card"


def test_safety_group_union_includes_first_coarse_and_schema_groups():
    groups = safety_group_union(
        first_pass_prediction="card_arrival",
        coarse_group_prediction="transfer",
        schema_variant_predictions={"plain_label": "top_up_failed"},
    )
    assert groups == ["card", "top_up", "transfer"]


def test_second_stage_labels_are_group_limited():
    labels = ["card_arrival", "pending_transfer", "top_up_failed"]
    assert labels_for_hierarchical_second_stage(labels, ["top_up"]) == ["top_up_failed"]


def test_hierarchical_prediction_row_is_stable():
    row = hierarchical_prediction_row(
        base_row={
            "example_id": 1,
            "text": "where is my card",
            "gold_label": "card_arrival",
            "predicted_canonical": "card_delivery_estimate",
            "latency_sec": 0.1,
        },
        coarse_row={"predicted_canonical": "card", "latency_sec": 0.2},
        fine_row={"predicted_canonical": "card_arrival", "latency_sec": 0.3},
        groups_used=["card"],
        candidate_labels_used=["card_arrival", "card_delivery_estimate"],
    )
    assert row["method"] == "hierarchical_gliner"
    assert row["predicted_canonical"] == "card_arrival"
    assert abs(row["latency_sec"] - 0.6) < 1e-9
