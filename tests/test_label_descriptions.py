from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.label_descriptions import (  # noqa: E402
    build_contrastive_description_schema,
    build_description_schema,
    contrastive_description,
    describe_label,
)


def test_rule_description_contains_label_meaning():
    description = describe_label("card_arrival")
    assert "card arrival" in description
    assert "not arrived" in description


def test_generic_description_generation():
    description = describe_label("beneficiary_not_allowed")
    assert description.startswith("beneficiary not allowed:")
    assert "banking app" in description


def test_description_schema_maps_back_to_canonical_label():
    labels = ["card_arrival", "pending_top_up"]
    candidates, mapping = build_description_schema(labels)
    assert len(candidates) == 2
    assert set(mapping.values()) == set(labels)


def test_contrastive_description_names_nearby_non_intents():
    description = contrastive_description("card_arrival", "card_lifecycle")
    assert "not about" in description
    assert "card delivery estimate" in description


def test_contrastive_schema_maps_back_to_original_label():
    candidates, mapping = build_contrastive_description_schema(
        ["pending_top_up", "top_up_failed"],
        cluster_name="top_up",
    )
    assert len(candidates) == 2
    assert mapping[candidates[0]] in {"pending_top_up", "top_up_failed"}
