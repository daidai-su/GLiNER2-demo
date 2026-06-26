"""Deterministic schema wording variants for Banking77 labels."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

from .data_utils import normalize_label_text

LOGGER = logging.getLogger(__name__)

RAW_LABEL = "raw_label"
PLAIN_LABEL = "plain_label"
QUERY_ABOUT_LABEL = "query_about_label"
BANKING_REQUEST_LABEL = "banking_request_label"
CUSTOMER_INTENT_LABEL = "customer_intent_label"

SINGLE_SCHEMA_METHODS = [
    RAW_LABEL,
    PLAIN_LABEL,
    QUERY_ABOUT_LABEL,
    BANKING_REQUEST_LABEL,
    CUSTOMER_INTENT_LABEL,
]

ENSEMBLE_INPUT_METHODS = [
    PLAIN_LABEL,
    QUERY_ABOUT_LABEL,
    BANKING_REQUEST_LABEL,
    CUSTOMER_INTENT_LABEL,
]


def plain_label(label_text: str) -> str:
    """Convert canonical Banking77 label_text into the plain display label."""

    return normalize_label_text(label_text)


def transform_label(label_text: str, variant_name: str) -> str:
    """Return the deterministic candidate string for one canonical label."""

    canonical = str(label_text).strip()
    display = plain_label(canonical)

    if variant_name == RAW_LABEL:
        return canonical
    if variant_name == PLAIN_LABEL:
        return display
    if variant_name == QUERY_ABOUT_LABEL:
        return f"question about {display}"
    if variant_name == BANKING_REQUEST_LABEL:
        return f"banking support request about {display}"
    if variant_name == CUSTOMER_INTENT_LABEL:
        return f"customer intent: {display}"

    raise ValueError(f"Unknown schema variant: {variant_name}")


def _deduplicate_candidates(
    candidate_to_canonicals: dict[str, list[str]],
) -> tuple[list[str], dict[str, str]]:
    """Resolve duplicate candidate strings deterministically with canonical suffixes."""

    candidates: list[str] = []
    candidate_to_canonical: dict[str, str] = {}

    for candidate, canonicals in candidate_to_canonicals.items():
        if len(canonicals) == 1:
            candidates.append(candidate)
            candidate_to_canonical[candidate] = canonicals[0]
            continue

        LOGGER.warning(
            "Duplicate schema candidate %r for canonical labels: %s",
            candidate,
            ", ".join(canonicals),
        )
        for canonical in canonicals:
            safe_candidate = f"{candidate} [canonical: {canonical}]"
            candidates.append(safe_candidate)
            candidate_to_canonical[safe_candidate] = canonical

    return candidates, candidate_to_canonical


def build_schema_variant(
    label_texts: list[str],
    variant_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Build candidate strings and candidate-to-canonical mapping for a variant."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for label_text in label_texts:
        canonical = str(label_text).strip()
        if not canonical:
            continue
        grouped[transform_label(canonical, variant_name)].append(canonical)

    return _deduplicate_candidates(grouped)


def build_all_schema_variants(
    label_texts: Iterable[str],
    variant_names: Iterable[str] = SINGLE_SCHEMA_METHODS,
) -> dict[str, dict[str, object]]:
    """Build all requested variants in a JSON-serializable structure."""

    labels = [str(label) for label in label_texts]
    variants: dict[str, dict[str, object]] = {}
    for variant_name in variant_names:
        candidates, mapping = build_schema_variant(labels, variant_name)
        variants[variant_name] = {
            "candidate_labels": candidates,
            "candidate_to_canonical": mapping,
        }
    return variants
