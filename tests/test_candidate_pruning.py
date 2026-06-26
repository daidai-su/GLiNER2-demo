from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.candidate_pruning import (  # noqa: E402
    build_retrieved_candidate_set,
    candidate_recall_row,
    rank_candidate_labels,
)
from gliner2_project.retrieval import build_tfidf_retriever  # noqa: E402


def test_candidate_set_contains_labels_from_neighbors():
    train = [
        {"example_id": 1, "text": "card arrived", "label_text": "card_arrival"},
        {"example_id": 2, "text": "card estimate", "label_text": "card_delivery_estimate"},
    ]
    retriever = build_tfidf_retriever(train, max_features=None)
    candidate_set = build_retrieved_candidate_set(
        "card arrived",
        retriever,
        k_values=[2],
        min_candidates=1,
        max_candidates=5,
    )
    assert set(candidate_set["candidate_labels"]) == {
        "card_arrival",
        "card_delivery_estimate",
    }


def test_candidate_ranking_by_frequency_and_similarity():
    neighbors = [
        {"label_text": "b", "similarity": 0.9},
        {"label_text": "a", "similarity": 0.4},
        {"label_text": "a", "similarity": 0.3},
    ]
    ranked = rank_candidate_labels(neighbors)
    assert [item["label"] for item in ranked] == ["a", "b"]
    assert ranked[0]["frequency"] == 2


def test_max_candidates_is_respected():
    train = [
        {"example_id": i, "text": f"common token {i}", "label_text": f"label_{i}"}
        for i in range(10)
    ]
    retriever = build_tfidf_retriever(train, max_features=None)
    candidate_set = build_retrieved_candidate_set(
        "common token",
        retriever,
        k_values=[10],
        min_candidates=1,
        max_candidates=3,
    )
    assert len(candidate_set["candidate_labels"]) == 3


def test_min_candidates_expansion_works():
    train = [
        {"example_id": 1, "text": "alpha", "label_text": "label_a"},
        {"example_id": 2, "text": "beta", "label_text": "label_b"},
        {"example_id": 3, "text": "gamma", "label_text": "label_c"},
    ]
    retriever = build_tfidf_retriever(train, max_features=None)
    candidate_set = build_retrieved_candidate_set(
        "alpha",
        retriever,
        k_values=[1, 3],
        min_candidates=2,
        max_candidates=5,
    )
    assert len(candidate_set["candidate_labels"]) >= 2
    assert candidate_set["expanded_k_used"] == 3


def test_unknown_labels_are_handled_safely():
    neighbors = [
        {"label_text": "known", "similarity": 0.6},
        {"label_text": "unknown", "similarity": 1.0},
    ]
    ranked = rank_candidate_labels(neighbors, known_labels={"known"})
    assert [item["label"] for item in ranked] == ["known"]


def test_gold_label_is_not_used_during_candidate_selection():
    train = [
        {"example_id": 1, "text": "alpha", "label_text": "label_a"},
        {"example_id": 2, "text": "beta", "label_text": "label_b"},
    ]
    retriever = build_tfidf_retriever(train, max_features=None)
    candidate_set = build_retrieved_candidate_set(
        "alpha",
        retriever,
        k_values=[1],
        min_candidates=1,
        max_candidates=1,
    )
    example = {"example_id": 99, "label_text": "label_b"}
    recall = candidate_recall_row(example, candidate_set)
    assert recall["gold_label"] == "label_b"
    assert recall["gold_in_candidates"] is False
