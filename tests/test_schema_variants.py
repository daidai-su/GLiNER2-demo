from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.schema_variants import (  # noqa: E402
    BANKING_REQUEST_LABEL,
    CUSTOMER_INTENT_LABEL,
    PLAIN_LABEL,
    QUERY_ABOUT_LABEL,
    RAW_LABEL,
    build_schema_variant,
)


def test_schema_variant_strings():
    labels = ["card_arrival"]

    candidates, mapping = build_schema_variant(labels, RAW_LABEL)
    assert candidates == ["card_arrival"]
    assert mapping["card_arrival"] == "card_arrival"

    candidates, mapping = build_schema_variant(labels, PLAIN_LABEL)
    assert candidates == ["card arrival"]
    assert mapping["card arrival"] == "card_arrival"

    candidates, mapping = build_schema_variant(labels, QUERY_ABOUT_LABEL)
    assert candidates == ["question about card arrival"]
    assert mapping["question about card arrival"] == "card_arrival"

    candidates, mapping = build_schema_variant(labels, BANKING_REQUEST_LABEL)
    assert candidates == ["banking support request about card arrival"]
    assert mapping["banking support request about card arrival"] == "card_arrival"

    candidates, mapping = build_schema_variant(labels, CUSTOMER_INTENT_LABEL)
    assert candidates == ["customer intent: card arrival"]
    assert mapping["customer intent: card arrival"] == "card_arrival"


def test_duplicate_detection_resolves_safely():
    candidates, mapping = build_schema_variant(["a_b", "a b"], PLAIN_LABEL)
    assert len(candidates) == 2
    assert len(set(candidates)) == 2
    assert set(mapping.values()) == {"a_b", "a b"}
    assert all("[canonical:" in candidate for candidate in candidates)


def test_mapping_back_to_canonical_labels():
    candidates, mapping = build_schema_variant(
        ["pending_top_up", "top_up_failed"],
        QUERY_ABOUT_LABEL,
    )
    assert "question about pending top up" in candidates
    assert mapping["question about pending top up"] == "pending_top_up"
    assert mapping["question about top up failed"] == "top_up_failed"
