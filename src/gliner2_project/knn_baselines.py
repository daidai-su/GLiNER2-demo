"""TF-IDF kNN baselines for Banking77 comparison experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from sklearn.metrics.pairwise import cosine_similarity


def assert_disjoint_example_ids(train_ids: Sequence[Any], test_ids: Sequence[Any]) -> None:
    """Raise if any evaluation example ID appears in the retrieval memory."""

    overlap = set(train_ids).intersection(set(test_ids))
    if overlap:
        raise ValueError(
            "Test example IDs found in retrieval memory: "
            + ", ".join(str(item) for item in sorted(overlap))
        )


def _top_k_indices(similarities: Sequence[float], k: int) -> list[int]:
    if k <= 0:
        return []
    return sorted(
        range(len(similarities)),
        key=lambda idx: (-float(similarities[idx]), idx),
    )[: min(k, len(similarities))]


def _neighbor_stats(
    neighbor_labels: Sequence[str],
    neighbor_similarities: Sequence[float],
) -> dict[str, dict[str, float | int | str]]:
    stats: dict[str, dict[str, float | int | str]] = {}
    for label, similarity in zip(neighbor_labels, neighbor_similarities):
        label = str(label)
        similarity = float(similarity)
        if label not in stats:
            stats[label] = {
                "label": label,
                "count": 0,
                "sum_similarity": 0.0,
                "max_similarity": float("-inf"),
            }
        stats[label]["count"] = int(stats[label]["count"]) + 1
        stats[label]["sum_similarity"] = float(stats[label]["sum_similarity"]) + similarity
        stats[label]["max_similarity"] = max(
            float(stats[label]["max_similarity"]),
            similarity,
        )
    return stats


def _majority_label(stats: dict[str, dict[str, float | int | str]]) -> str | None:
    if not stats:
        return None
    return min(
        stats,
        key=lambda label: (
            -int(stats[label]["count"]),
            -float(stats[label]["sum_similarity"]),
            -float(stats[label]["max_similarity"]),
            label,
        ),
    )


def _weighted_label(stats: dict[str, dict[str, float | int | str]]) -> str | None:
    if not stats:
        return None
    return min(
        stats,
        key=lambda label: (
            -float(stats[label]["sum_similarity"]),
            -float(stats[label]["max_similarity"]),
            -int(stats[label]["count"]),
            label,
        ),
    )


def _prediction_payload(
    predicted_label: str | None,
    neighbor_indices: Sequence[int],
    neighbor_labels: Sequence[str],
    neighbor_similarities: Sequence[float],
    stats: dict[str, dict[str, float | int | str]],
    train_ids: Sequence[Any] | None,
) -> dict[str, Any]:
    label_vote_counts = {
        label: int(values["count"]) for label, values in sorted(stats.items())
    }
    label_similarity_sums = {
        label: float(values["sum_similarity"]) for label, values in sorted(stats.items())
    }
    neighbor_ids = [
        train_ids[idx] if train_ids is not None else int(idx) for idx in neighbor_indices
    ]
    return {
        "predicted_canonical": predicted_label,
        "neighbor_indices": [int(idx) for idx in neighbor_indices],
        "neighbor_ids": neighbor_ids,
        "neighbor_labels": list(neighbor_labels),
        "neighbor_similarities": [float(value) for value in neighbor_similarities],
        "label_vote_counts": label_vote_counts,
        "label_similarity_sums": label_similarity_sums,
    }


def predict_tfidf_knn_majority(
    train_tfidf: Any,
    train_labels: Sequence[str],
    test_tfidf: Any,
    k: int,
    train_ids: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict by majority vote over top-k cosine-similar train examples.

    Ties are resolved by summed cosine similarity, then maximum cosine similarity,
    then alphabetical label order for deterministic output.
    """

    predictions: list[dict[str, Any]] = []
    train_labels = [str(label) for label in train_labels]
    for row_idx in range(test_tfidf.shape[0]):
        similarities = cosine_similarity(test_tfidf[row_idx : row_idx + 1], train_tfidf)[0]
        neighbor_indices = _top_k_indices(similarities, k)
        neighbor_labels = [train_labels[idx] for idx in neighbor_indices]
        neighbor_similarities = [float(similarities[idx]) for idx in neighbor_indices]
        stats = _neighbor_stats(neighbor_labels, neighbor_similarities)
        predicted = _majority_label(stats)
        predictions.append(
            _prediction_payload(
                predicted,
                neighbor_indices,
                neighbor_labels,
                neighbor_similarities,
                stats,
                train_ids,
            )
        )
    return predictions


def predict_tfidf_weighted_knn(
    train_tfidf: Any,
    train_labels: Sequence[str],
    test_tfidf: Any,
    k: int,
    train_ids: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict by similarity-weighted voting over top-k train examples."""

    predictions: list[dict[str, Any]] = []
    train_labels = [str(label) for label in train_labels]
    for row_idx in range(test_tfidf.shape[0]):
        similarities = cosine_similarity(test_tfidf[row_idx : row_idx + 1], train_tfidf)[0]
        neighbor_indices = _top_k_indices(similarities, k)
        neighbor_labels = [train_labels[idx] for idx in neighbor_indices]
        neighbor_similarities = [float(similarities[idx]) for idx in neighbor_indices]
        stats = _neighbor_stats(neighbor_labels, neighbor_similarities)
        predicted = _weighted_label(stats)
        ranked = sorted(
            stats.values(),
            key=lambda item: (
                -float(item["sum_similarity"]),
                -float(item["max_similarity"]),
                -int(item["count"]),
                str(item["label"]),
            ),
        )
        top_score = float(ranked[0]["sum_similarity"]) if ranked else 0.0
        second_score = float(ranked[1]["sum_similarity"]) if len(ranked) > 1 else 0.0
        payload = _prediction_payload(
            predicted,
            neighbor_indices,
            neighbor_labels,
            neighbor_similarities,
            stats,
            train_ids,
        )
        payload.update(
            {
                "top_label_score": top_score,
                "second_label_score": second_score,
                "retrieval_margin": top_score - second_score,
            }
        )
        predictions.append(payload)
    return predictions


def majority_vote_from_labels(labels: Sequence[str]) -> str | None:
    """Return a deterministic majority label for a plain label sequence."""

    if not labels:
        return None
    counts = Counter(str(label) for label in labels)
    grouped: dict[str, float] = defaultdict(float)
    for label, count in counts.items():
        grouped[label] = float(count)
    return min(grouped, key=lambda label: (-grouped[label], label))
