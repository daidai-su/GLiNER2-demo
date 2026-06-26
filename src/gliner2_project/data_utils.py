"""Dataset and label mapping helpers for Banking77 smoke evaluation."""

from __future__ import annotations

import random
import string
from collections import defaultdict
from typing import Any, Iterable, Sequence


EDGE_PUNCTUATION = string.punctuation


def normalize_label_text(label_text: Any) -> str:
    """Normalize a label for display while preserving the original canonical label elsewhere."""

    if label_text is None:
        return ""
    text = str(label_text).replace("_", " ").lower().strip()
    text = " ".join(text.split())
    return text.strip(EDGE_PUNCTUATION + " ")


def build_label_mappings(
    label_texts: Iterable[Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build canonical-to-display and normalized-display-to-canonical mappings."""

    canonical_to_display: dict[str, str] = {}
    display_to_canonical: dict[str, str] = {}

    for raw_label in label_texts:
        canonical = str(raw_label).strip()
        display = normalize_label_text(canonical)
        if not canonical or not display:
            continue

        previous = display_to_canonical.get(display)
        if previous is not None and previous != canonical:
            raise ValueError(
                f"Label normalization collision: {previous!r} and {canonical!r} "
                f"both map to {display!r}"
            )

        canonical_to_display[canonical] = display
        display_to_canonical[display] = canonical

    return canonical_to_display, display_to_canonical


def map_prediction_to_canonical(
    predicted_label: Any,
    display_to_canonical: dict[str, str],
) -> str | None:
    """Map a predicted display label back to the dataset's canonical label_text."""

    if predicted_label is None:
        return None
    normalized_prediction = normalize_label_text(predicted_label)
    if not normalized_prediction:
        return None
    return display_to_canonical.get(normalized_prediction)


def recover_label_texts_from_features(dataset_split: Any) -> list[str] | None:
    """Recover class names from Hugging Face ClassLabel features when possible."""

    features = getattr(dataset_split, "features", None)
    if features is None or "label" not in features:
        return None

    label_feature = features["label"]
    names = getattr(label_feature, "names", None)
    if names:
        return [str(name) for name in names]
    return None


def ensure_label_text_column(dataset_split: Any) -> Any:
    """Return a split with label_text present, recovering names from label features if possible."""

    columns = set(getattr(dataset_split, "column_names", []))
    if "label_text" in columns:
        return dataset_split

    names = recover_label_texts_from_features(dataset_split)
    if not names:
        raise ValueError(
            "Dataset split does not contain label_text and label names could not be "
            "recovered from features['label'].names."
        )

    def add_label_text(example: dict[str, Any]) -> dict[str, Any]:
        label_id = int(example["label"])
        example["label_text"] = names[label_id]
        return example

    return dataset_split.map(add_label_text)


def unique_label_texts(dataset_split: Any) -> list[str]:
    """Return canonical label_text values in first-seen order."""

    labels: list[str] = []
    seen: set[str] = set()
    for row in dataset_split:
        label = str(row["label_text"]).strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def stratified_subset(
    rows: Sequence[dict[str, Any]] | Any,
    per_label: int,
    seed: int = 42,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Sample up to per_label examples for each label_text from a dataset-like object."""

    if per_label <= 0:
        raise ValueError("per_label must be positive.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault("example_id", idx)
        label = str(item.get("label_text", "")).strip()
        if not label:
            raise ValueError("Every row must contain a non-empty label_text field.")
        grouped[label].append(item)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for label in sorted(grouped):
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_label])

    rng.shuffle(sampled)
    if max_examples is not None:
        sampled = sampled[:max_examples]
    return sampled
