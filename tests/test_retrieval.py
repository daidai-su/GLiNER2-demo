from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.retrieval import (  # noqa: E402
    assert_no_test_ids_in_retriever,
    build_tfidf_retriever,
    tfidf_knn_predict,
)


def toy_train():
    return [
        {"example_id": "tr1", "text": "card arrived late", "label_text": "card_arrival"},
        {
            "example_id": "tr2",
            "text": "card delivery estimate",
            "label_text": "card_delivery_estimate",
        },
        {"example_id": "tr3", "text": "top up failed", "label_text": "top_up_failed"},
        {"example_id": "tr4", "text": "top up failed again", "label_text": "top_up_failed"},
    ]


def test_tfidf_retrieval_returns_expected_nearest_examples():
    retriever = build_tfidf_retriever(toy_train(), max_features=None)
    neighbors = retriever.retrieve("my top up failed", k=2)
    assert neighbors[0]["label_text"] == "top_up_failed"
    assert neighbors[0]["similarity"] >= neighbors[1]["similarity"]


def test_no_test_examples_are_in_train_index():
    retriever = build_tfidf_retriever(toy_train(), max_features=None)
    assert_no_test_ids_in_retriever(retriever, ["test1", "test2"])


def test_overlap_with_test_examples_raises():
    retriever = build_tfidf_retriever(toy_train(), max_features=None)
    try:
        assert_no_test_ids_in_retriever(retriever, ["tr1"])
    except ValueError as exc:
        assert "tr1" in str(exc)
    else:
        raise AssertionError("Expected overlap to raise ValueError")


def test_knn_vote_tie_breaking_is_deterministic():
    train = [
        {"example_id": "a", "text": "alpha", "label_text": "b_label"},
        {"example_id": "b", "text": "alpha", "label_text": "a_label"},
    ]
    retriever = build_tfidf_retriever(train, max_features=None)
    result = tfidf_knn_predict("alpha", retriever, k=2)
    assert result["predicted_canonical"] == "a_label"


def test_similarity_scores_are_sorted_descending():
    retriever = build_tfidf_retriever(toy_train(), max_features=None)
    neighbors = retriever.retrieve("card delivery", k=4)
    similarities = [neighbor["similarity"] for neighbor in neighbors]
    assert similarities == sorted(similarities, reverse=True)
