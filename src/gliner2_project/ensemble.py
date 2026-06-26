"""Ensemble methods over deterministic GLiNER2 schema variants."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from .schema_variants import ENSEMBLE_INPUT_METHODS, PLAIN_LABEL

VOTE_ENSEMBLE = "vote_ensemble"
MEAN_CONFIDENCE_ENSEMBLE = "mean_confidence_ensemble"
CONFIDENCE_MARGIN_FALLBACK = "confidence_margin_fallback"


def _index_predictions(rows: Iterable[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row.get("example_id"): row for row in rows}


def _method_indexes(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    methods: Iterable[str] = ENSEMBLE_INPUT_METHODS,
) -> dict[str, dict[Any, dict[str, Any]]]:
    return {
        method: _index_predictions(predictions_by_method.get(method, []))
        for method in methods
    }


def _ordered_example_ids(indexes: dict[str, dict[Any, dict[str, Any]]]) -> list[Any]:
    first_method = next(iter(indexes))
    return list(indexes[first_method].keys())


def _base_row(
    example_id: Any,
    indexes: dict[str, dict[Any, dict[str, Any]]],
    base_method: str = PLAIN_LABEL,
) -> dict[str, Any]:
    if base_method in indexes and example_id in indexes[base_method]:
        return indexes[base_method][example_id]
    for method_index in indexes.values():
        if example_id in method_index:
            return method_index[example_id]
    return {"example_id": example_id}


def _prediction_details(
    example_id: Any,
    indexes: dict[str, dict[Any, dict[str, Any]]],
) -> tuple[dict[str, str | None], dict[str, float | None]]:
    method_predictions: dict[str, str | None] = {}
    confidence_values: dict[str, float | None] = {}
    for method, method_index in indexes.items():
        row = method_index.get(example_id, {})
        method_predictions[method] = row.get("predicted_canonical")
        confidence_values[method] = row.get("confidence")
    return method_predictions, confidence_values


def _average_confidence_by_label(
    method_predictions: dict[str, str | None],
    confidence_values: dict[str, float | None],
) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for method, label in method_predictions.items():
        confidence = confidence_values.get(method)
        if label is not None and confidence is not None:
            scores[label].append(float(confidence))
    return {label: mean(values) for label, values in scores.items() if values}


def majority_vote_ensemble(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    methods: Iterable[str] = ENSEMBLE_INPUT_METHODS,
    base_method: str = PLAIN_LABEL,
) -> list[dict[str, Any]]:
    """Choose the canonical label by majority vote across schema variants."""

    indexes = _method_indexes(predictions_by_method, methods)
    output_rows: list[dict[str, Any]] = []

    for example_id in _ordered_example_ids(indexes):
        base = _base_row(example_id, indexes, base_method)
        method_predictions, confidence_values = _prediction_details(example_id, indexes)
        valid_predictions = [
            label for label in method_predictions.values() if label is not None
        ]
        votes = Counter(valid_predictions)
        predicted = None
        tie_break = None

        if votes:
            max_votes = max(votes.values())
            tied_labels = sorted(
                label for label, count in votes.items() if count == max_votes
            )
            if len(tied_labels) == 1:
                predicted = tied_labels[0]
                tie_break = "majority"
            else:
                confidence_by_label = _average_confidence_by_label(
                    method_predictions,
                    confidence_values,
                )
                tied_scores = {
                    label: confidence_by_label[label]
                    for label in tied_labels
                    if label in confidence_by_label
                }
                if tied_scores:
                    best_score = max(tied_scores.values())
                    best_labels = sorted(
                        label
                        for label, score in tied_scores.items()
                        if score == best_score
                    )
                    predicted = best_labels[0]
                    tie_break = "average_confidence"
                else:
                    base_prediction = method_predictions.get(base_method)
                    predicted = (
                        base_prediction
                        if base_prediction in tied_labels
                        else tied_labels[0]
                    )
                    tie_break = base_method

        output_rows.append(
            {
                "example_id": example_id,
                "text": base.get("text"),
                "gold_label": base.get("gold_label"),
                "method": VOTE_ENSEMBLE,
                "predicted_canonical": predicted,
                "confidence": None,
                "votes": dict(votes),
                "method_predictions": method_predictions,
                "confidence_values": confidence_values,
                "tie_break": tie_break,
                "fallback_used": False,
                "parse_error": None if predicted is not None else "no valid votes",
            }
        )

    return output_rows


def confidence_coverage(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    methods: Iterable[str] = ENSEMBLE_INPUT_METHODS,
) -> float:
    """Return the fraction of method predictions with non-missing confidence."""

    total = 0
    present = 0
    for method in methods:
        for row in predictions_by_method.get(method, []):
            total += 1
            if row.get("confidence") is not None:
                present += 1
    return float(present / total) if total else 0.0


def mean_confidence_ensemble(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    methods: Iterable[str] = ENSEMBLE_INPUT_METHODS,
    base_method: str = PLAIN_LABEL,
) -> list[dict[str, Any]]:
    """Aggregate canonical labels by mean confidence across schema variants."""

    indexes = _method_indexes(predictions_by_method, methods)
    output_rows: list[dict[str, Any]] = []

    for example_id in _ordered_example_ids(indexes):
        base = _base_row(example_id, indexes, base_method)
        method_predictions, confidence_values = _prediction_details(example_id, indexes)
        confidence_by_label = _average_confidence_by_label(
            method_predictions,
            confidence_values,
        )
        predicted = None
        confidence = None
        parse_error = None

        if confidence_by_label:
            confidence = max(confidence_by_label.values())
            predicted = sorted(
                label
                for label, score in confidence_by_label.items()
                if score == confidence
            )[0]
        else:
            parse_error = "confidence unavailable"

        output_rows.append(
            {
                "example_id": example_id,
                "text": base.get("text"),
                "gold_label": base.get("gold_label"),
                "method": MEAN_CONFIDENCE_ENSEMBLE,
                "predicted_canonical": predicted,
                "confidence": confidence,
                "mean_confidence_by_label": confidence_by_label,
                "method_predictions": method_predictions,
                "confidence_values": confidence_values,
                "fallback_used": False,
                "parse_error": parse_error,
            }
        )

    return output_rows


def _top_two_margin(scores: dict[str, float]) -> tuple[str | None, float | None, float | None]:
    if not scores:
        return None, None, None
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    return top_label, top_score, float(top_score - second_score)


def confidence_margin_fallback(
    predictions_by_method: dict[str, list[dict[str, Any]]],
    margin_threshold: float,
    methods: Iterable[str] = ENSEMBLE_INPUT_METHODS,
    base_method: str = PLAIN_LABEL,
) -> list[dict[str, Any]]:
    """Use mean confidence only when its top-label margin is sufficiently large."""

    indexes = _method_indexes(predictions_by_method, methods)
    output_rows: list[dict[str, Any]] = []

    for example_id in _ordered_example_ids(indexes):
        base = _base_row(example_id, indexes, base_method)
        method_predictions, confidence_values = _prediction_details(example_id, indexes)
        confidence_by_label = _average_confidence_by_label(
            method_predictions,
            confidence_values,
        )
        top_label, top_score, margin = _top_two_margin(confidence_by_label)
        base_prediction = method_predictions.get(base_method)
        fallback_used = False
        parse_error = None

        if top_label is None or margin is None:
            predicted = base_prediction
            confidence = confidence_values.get(base_method)
            fallback_used = True
            parse_error = "confidence unavailable"
        elif margin >= margin_threshold:
            predicted = top_label
            confidence = top_score
        else:
            predicted = base_prediction
            confidence = confidence_values.get(base_method)
            fallback_used = True

        output_rows.append(
            {
                "example_id": example_id,
                "text": base.get("text"),
                "gold_label": base.get("gold_label"),
                "method": CONFIDENCE_MARGIN_FALLBACK,
                "predicted_canonical": predicted,
                "confidence": confidence,
                "confidence_margin": margin,
                "margin_threshold": margin_threshold,
                "mean_confidence_by_label": confidence_by_label,
                "method_predictions": method_predictions,
                "confidence_values": confidence_values,
                "fallback_used": fallback_used,
                "parse_error": parse_error,
            }
        )

    return output_rows
