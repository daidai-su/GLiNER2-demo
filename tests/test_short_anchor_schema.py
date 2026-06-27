from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.anchor_coverage import (  # noqa: E402
    anchor_coverage_frame,
    validate_anchor_coverage,
)
from gliner2_project.short_anchor_schema import (  # noqa: E402
    MINIMAL_CONTRASTIVE_ANCHOR,
    QUERY_ABOUT_SHORT_ANCHOR,
    SHORT_ANCHOR_METHODS,
    build_short_anchor_schema,
    duplicate_candidate_strings,
    minimal_contrastive_anchor,
    short_anchor,
    transform_short_anchor_label,
)


EXPECTED_LABELS = [
    "card_arrival",
    "card_delivery_estimate",
    "pending_top_up",
    "top_up_failed",
    "verify_my_identity",
    "why_verify_identity",
    "cash_withdrawal_not_recognised",
    "transfer_not_received_by_recipient",
    "exchange_via_app",
    "exchange_rate",
    "fiat_currency_support",
]


def test_all_expected_labels_can_be_mapped_to_anchors():
    for label in EXPECTED_LABELS:
        assert short_anchor(label)
        assert minimal_contrastive_anchor(label)


def test_unknown_label_fallback_is_deterministic():
    assert short_anchor("new_unknown_label") == "new unknown label"
    assert short_anchor("new_unknown_label") == short_anchor("new_unknown_label")


def test_query_about_short_anchor_wraps_anchor_correctly():
    assert (
        transform_short_anchor_label("card_arrival", QUERY_ABOUT_SHORT_ANCHOR)
        == "question about card not arrived"
    )


def test_minimal_contrastive_anchor_overrides_generic_anchor():
    assert short_anchor("card_delivery_estimate") == "card delivery time"
    assert minimal_contrastive_anchor("card_delivery_estimate") == "when card will arrive"


def test_duplicate_candidate_labels_are_detected_before_disambiguation():
    labels = ["a_b", "a b"]
    duplicates = duplicate_candidate_strings(labels, MINIMAL_CONTRASTIVE_ANCHOR)
    assert "a b" in duplicates


def test_canonical_mapping_from_candidate_back_to_label_works():
    candidates, mapping = build_short_anchor_schema(
        ["card_arrival", "pending_top_up"],
        QUERY_ABOUT_SHORT_ANCHOR,
    )
    assert len(candidates) == 2
    assert set(mapping.values()) == {"card_arrival", "pending_top_up"}


def test_no_candidate_string_is_empty():
    for method in SHORT_ANCHOR_METHODS:
        candidates, _ = build_short_anchor_schema(EXPECTED_LABELS, method)
        assert all(candidate.strip() for candidate in candidates)


def test_anchor_coverage_accepts_expected_labels():
    validate_anchor_coverage(EXPECTED_LABELS)
    frame = anchor_coverage_frame(EXPECTED_LABELS)
    assert frame["coverage_ok"].all()
