"""Hierarchical zero-shot GLiNER helper functions."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .label_clusters import (
    COARSE_GROUP_DESCRIPTIONS,
    get_coarse_group,
    group_union_labels,
)


HIERARCHICAL_GLINER = "hierarchical_gliner"


def build_coarse_group_schema(
    group_descriptions: dict[str, str] = COARSE_GROUP_DESCRIPTIONS,
) -> tuple[list[str], dict[str, str]]:
    """Build coarse-group candidate strings and candidate-to-group mapping."""

    candidates: list[str] = []
    mapping: dict[str, str] = {}
    for group_name, description in group_descriptions.items():
        candidate = f"{group_name}: {description}"
        candidates.append(candidate)
        mapping[candidate] = group_name
    return candidates, mapping


def group_for_label(label: str | None) -> str:
    """Return deterministic coarse group for a canonical label."""

    return get_coarse_group(label)


def groups_from_schema_predictions(
    method_predictions: dict[str, str | None] | None,
) -> list[str]:
    """Map schema-variant predictions to coarse groups."""

    if not method_predictions:
        return []
    groups = {
        group_for_label(label)
        for label in method_predictions.values()
        if label is not None
    }
    return sorted(groups)


def safety_group_union(
    first_pass_prediction: str | None,
    coarse_group_prediction: str | None,
    schema_variant_predictions: dict[str, str | None] | None = None,
) -> list[str]:
    """Return safety union of first-pass, coarse, and schema-variant groups."""

    groups = set(groups_from_schema_predictions(schema_variant_predictions))
    if first_pass_prediction is not None:
        groups.add(group_for_label(first_pass_prediction))
    if coarse_group_prediction:
        groups.add(str(coarse_group_prediction))
    return sorted(group for group in groups if group)


def labels_for_hierarchical_second_stage(
    label_texts: Iterable[str],
    group_names: Sequence[str],
) -> list[str]:
    """Return fine labels for the selected coarse groups."""

    labels = group_union_labels(list(label_texts), group_names)
    return labels


def hierarchical_prediction_row(
    base_row: dict[str, Any],
    coarse_row: dict[str, Any] | None,
    fine_row: dict[str, Any],
    groups_used: Sequence[str],
    candidate_labels_used: Sequence[str],
    method_name: str = HIERARCHICAL_GLINER,
) -> dict[str, Any]:
    """Create a final hierarchical prediction row."""

    base_latency = float(base_row.get("latency_sec") or 0.0)
    coarse_latency = float((coarse_row or {}).get("latency_sec") or 0.0)
    fine_latency = float(fine_row.get("latency_sec") or 0.0)
    return {
        "example_id": base_row.get("example_id"),
        "text": base_row.get("text"),
        "gold_label": base_row.get("gold_label"),
        "method": method_name,
        "predicted_canonical": fine_row.get("predicted_canonical"),
        "first_pass_prediction": base_row.get("predicted_canonical"),
        "coarse_group_prediction": (coarse_row or {}).get("predicted_canonical"),
        "groups_used": list(groups_used),
        "candidate_labels_used": list(candidate_labels_used),
        "latency_sec": base_latency + coarse_latency + fine_latency,
        "parse_error": fine_row.get("parse_error"),
    }
