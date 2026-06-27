from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.knn_baselines import (  # noqa: E402
    assert_disjoint_example_ids,
    predict_tfidf_knn_majority,
    predict_tfidf_weighted_knn,
)


def test_majority_voting_uses_frequency_then_similarity():
    train_tfidf = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    test_tfidf = np.array([[1.0, 0.0]])
    predictions = predict_tfidf_knn_majority(
        train_tfidf,
        ["b_label", "a_label", "a_label"],
        test_tfidf,
        k=3,
    )
    assert predictions[0]["predicted_canonical"] == "a_label"
    assert predictions[0]["label_vote_counts"]["a_label"] == 2


def test_similarity_weighted_voting_uses_similarity_sum():
    train_tfidf = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.8, 0.6],
        ]
    )
    test_tfidf = np.array([[1.0, 0.0]])
    predictions = predict_tfidf_weighted_knn(
        train_tfidf,
        ["b_label", "a_label", "a_label"],
        test_tfidf,
        k=3,
    )
    assert predictions[0]["predicted_canonical"] == "a_label"
    assert predictions[0]["top_label_score"] > predictions[0]["second_label_score"]


def test_tie_breaking_is_deterministic_by_label():
    train_tfidf = np.array([[1.0, 0.0], [1.0, 0.0]])
    test_tfidf = np.array([[1.0, 0.0]])
    predictions = predict_tfidf_knn_majority(
        train_tfidf,
        ["b_label", "a_label"],
        test_tfidf,
        k=2,
    )
    assert predictions[0]["predicted_canonical"] == "a_label"


def test_neighbors_are_sorted_by_similarity_descending():
    train_tfidf = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]])
    test_tfidf = np.array([[1.0, 0.0]])
    predictions = predict_tfidf_knn_majority(
        train_tfidf,
        ["c", "a", "b"],
        test_tfidf,
        k=3,
        train_ids=["tr0", "tr1", "tr2"],
    )
    similarities = predictions[0]["neighbor_similarities"]
    assert similarities == sorted(similarities, reverse=True)
    assert predictions[0]["neighbor_ids"][0] == "tr1"


def test_test_examples_are_not_used_as_train_memory():
    assert_disjoint_example_ids(["train_1", "train_2"], ["test_1"])
    try:
        assert_disjoint_example_ids(["train_1", "shared"], ["shared"])
    except ValueError as exc:
        assert "shared" in str(exc)
    else:
        raise AssertionError("Expected overlapping IDs to raise ValueError")
