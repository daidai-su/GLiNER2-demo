"""Coverage checks for short-anchor schema variants."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from .short_anchor_schema import (
    SHORT_ANCHOR_METHODS,
    build_short_anchor_schema,
    build_short_anchor_table,
    duplicate_candidate_strings,
)


def anchor_coverage_frame(
    label_texts: Iterable[str],
    variant_names: Sequence[str] = SHORT_ANCHOR_METHODS,
) -> pd.DataFrame:
    """Return per-variant anchor coverage diagnostics."""

    labels = [str(label).strip() for label in label_texts if str(label).strip()]
    rows: list[dict[str, object]] = []
    for variant in variant_names:
        candidates, mapping = build_short_anchor_schema(labels, variant)
        raw_duplicates = duplicate_candidate_strings(labels, variant)
        rows.append(
            {
                "variant": variant,
                "num_labels": len(labels),
                "num_candidates": len(candidates),
                "num_mapped_labels": len(set(mapping.values())),
                "empty_candidates": int(sum(1 for candidate in candidates if not candidate.strip())),
                "raw_duplicate_candidate_count": len(raw_duplicates),
                "raw_duplicate_candidates": raw_duplicates,
                "final_duplicate_candidate_count": len(candidates) - len(set(candidates)),
                "coverage_ok": (
                    len(candidates) == len(labels)
                    and len(set(mapping.values())) == len(labels)
                    and all(candidate.strip() for candidate in candidates)
                    and len(candidates) == len(set(candidates))
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_anchor_coverage(
    label_texts: Iterable[str],
    variant_names: Sequence[str] = SHORT_ANCHOR_METHODS,
) -> None:
    """Raise if any short-anchor variant lacks full deterministic coverage."""

    frame = anchor_coverage_frame(label_texts, variant_names)
    bad = frame[~frame["coverage_ok"].astype(bool)]
    if not bad.empty:
        raise ValueError(
            "Short-anchor coverage failed for variants: "
            + ", ".join(bad["variant"].astype(str))
        )


def anchor_table_frame(label_texts: Iterable[str]) -> pd.DataFrame:
    """Return a DataFrame with every canonical label and generated anchors."""

    return pd.DataFrame(build_short_anchor_table(label_texts))
