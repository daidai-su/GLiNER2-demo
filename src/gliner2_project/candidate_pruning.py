"""Candidate-set construction from retrieved Banking77 train neighbors."""

from __future__ import annotations

from typing import Any, Sequence

from .retrieval import TfidfRetriever, rank_labels_from_neighbors


def rank_candidate_labels(
    neighbors: Sequence[dict[str, Any]],
    known_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank canonical candidate labels from retrieved neighbors."""

    ranked = rank_labels_from_neighbors(neighbors)
    if known_labels is None:
        return ranked
    return [item for item in ranked if item["label"] in known_labels]


def build_retrieved_candidate_set(
    query_text: str,
    retriever: TfidfRetriever,
    train_examples: Sequence[dict[str, Any]] | None = None,
    k_values: list[int] | None = None,
    min_candidates: int = 5,
    max_candidates: int = 15,
    known_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Build a pruned canonical label set using train-only retrieval."""

    del train_examples  # The retriever owns the training memory.
    if k_values is None:
        k_values = [15, 25, 50]
    if min_candidates <= 0:
        raise ValueError("min_candidates must be positive.")
    if max_candidates < min_candidates:
        raise ValueError("max_candidates must be >= min_candidates.")

    final_neighbors: list[dict[str, Any]] = []
    final_ranked: list[dict[str, Any]] = []
    expanded_k_used = k_values[-1]

    for k in k_values:
        neighbors = retriever.retrieve(query_text, k)
        ranked = rank_candidate_labels(neighbors, known_labels=known_labels)
        final_neighbors = neighbors
        final_ranked = ranked
        expanded_k_used = k
        if len(ranked) >= min_candidates:
            break

    kept = final_ranked[:max_candidates]
    candidate_labels = [item["label"] for item in kept]
    candidate_label_scores = {
        item["label"]: {
            "frequency": int(item["frequency"]),
            "sum_similarity": float(item["sum_similarity"]),
            "max_similarity": float(item["max_similarity"]),
        }
        for item in kept
    }

    return {
        "candidate_labels": candidate_labels,
        "candidate_label_scores": candidate_label_scores,
        "neighbor_ids": [neighbor["example_id"] for neighbor in final_neighbors],
        "neighbor_labels": [neighbor["label_text"] for neighbor in final_neighbors],
        "neighbor_similarities": [
            float(neighbor["similarity"]) for neighbor in final_neighbors
        ],
        "expanded_k_used": expanded_k_used,
    }


def candidate_recall_row(
    example: dict[str, Any],
    candidate_set: dict[str, Any],
    gold_key: str = "label_text",
) -> dict[str, Any]:
    """Return candidate recall details for one evaluated example."""

    gold = example.get(gold_key)
    candidates = list(candidate_set.get("candidate_labels", []))
    return {
        "example_id": example.get("example_id"),
        "gold_label": gold,
        "num_candidates": len(candidates),
        "gold_in_candidates": gold in set(candidates),
        "expanded_k_used": candidate_set.get("expanded_k_used"),
    }
