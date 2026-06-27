from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.classical_baselines import (  # noqa: E402
    build_tfidf_vectorizer,
    fit_label_encoder,
    ids_to_labels,
    labels_to_ids,
    run_tfidf_linear_svm,
    run_tfidf_logistic_regression,
)


def toy_data():
    train_texts = [
        "card arrived late",
        "where is my card",
        "top up failed",
        "top up error",
    ]
    train_labels = [
        "card_arrival",
        "card_arrival",
        "top_up_failed",
        "top_up_failed",
    ]
    test_texts = ["card arrived", "top up failed"]
    gold_labels = ["card_arrival", "top_up_failed"]
    return train_texts, train_labels, test_texts, gold_labels


def test_tfidf_vectorizer_fits_toy_train_data():
    train_texts, _, _, _ = toy_data()
    vectorizer = build_tfidf_vectorizer(max_features=None)
    matrix = vectorizer.fit_transform(train_texts)
    assert matrix.shape[0] == len(train_texts)
    assert "card" in vectorizer.vocabulary_


def test_label_encoder_maps_back_to_canonical_labels():
    encoder = fit_label_encoder(["b_label", "a_label"])
    encoded = labels_to_ids(encoder, ["a_label", "b_label"])
    assert ids_to_labels(encoder, encoded) == ["a_label", "b_label"]


def test_logistic_regression_wrapper_trains_and_predicts():
    train_texts, train_labels, test_texts, gold_labels = toy_data()
    result = run_tfidf_logistic_regression(
        train_texts,
        train_labels,
        test_texts,
        gold_labels,
        max_features=None,
        max_iter=500,
        random_state=42,
    )
    assert result.method == "tfidf_logistic_regression"
    assert len(result.predictions) == len(test_texts)
    assert result.training_time_sec >= 0.0
    assert result.config["solver_used"] in {"saga", "lbfgs"}


def test_linear_svm_wrapper_trains_and_predicts_with_safe_dual_fallback():
    train_texts, train_labels, test_texts, gold_labels = toy_data()
    result = run_tfidf_linear_svm(
        train_texts,
        train_labels,
        test_texts,
        gold_labels,
        max_features=None,
        random_state=42,
    )
    assert result.method == "tfidf_linear_svm"
    assert len(result.predictions) == len(test_texts)
    assert result.config["dual_used"] in {"auto", "False", "True"}
