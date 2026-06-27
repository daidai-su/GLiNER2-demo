"""Classical TF-IDF baselines for intent classification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


@dataclass
class ClassicalRunResult:
    """Container for a fitted classical baseline evaluation."""

    method: str
    vectorizer: TfidfVectorizer
    classifier: Any
    label_encoder: LabelEncoder
    predictions: list[dict[str, Any]]
    training_time_sec: float
    total_prediction_time_sec: float
    config: dict[str, Any]


def build_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int | None = 50000,
    sublinear_tf: bool = True,
    lowercase: bool = True,
) -> TfidfVectorizer:
    """Create the project-standard TF-IDF vectorizer."""

    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
    )


def fit_label_encoder(labels: Sequence[str]) -> LabelEncoder:
    """Fit a stable canonical-label encoder."""

    encoder = LabelEncoder()
    encoder.fit([str(label) for label in labels])
    return encoder


def labels_to_ids(encoder: LabelEncoder, labels: Sequence[str]) -> np.ndarray:
    """Encode canonical labels as integer class IDs."""

    return encoder.transform([str(label) for label in labels])


def ids_to_labels(encoder: LabelEncoder, label_ids: Sequence[int]) -> list[str]:
    """Decode integer class IDs back to canonical labels."""

    return [str(label) for label in encoder.inverse_transform(list(label_ids))]


def _average_latency(total_prediction_time_sec: float, num_examples: int) -> float:
    if num_examples <= 0:
        return 0.0
    return float(total_prediction_time_sec / num_examples)


def _logreg_classifier(
    C: float,
    max_iter: int,
    solver: str,
    random_state: int,
) -> LogisticRegression:
    kwargs: dict[str, Any] = {
        "C": C,
        "max_iter": max_iter,
        "solver": solver,
        "random_state": random_state,
    }
    try:
        return LogisticRegression(n_jobs=-1, **kwargs)
    except TypeError:
        return LogisticRegression(**kwargs)


def _fit_logistic_regression_with_fallback(
    train_tfidf: Any,
    train_label_ids: Sequence[int],
    C: float,
    max_iter: int,
    solver: str,
    random_state: int,
) -> tuple[LogisticRegression, str, str | None]:
    """Fit LogisticRegression, falling back if the requested solver fails."""

    fallback_note = None
    try:
        classifier = _logreg_classifier(C, max_iter, solver, random_state)
        classifier.fit(train_tfidf, train_label_ids)
        return classifier, solver, fallback_note
    except Exception as exc:
        fallback_note = f"solver={solver!r} failed with {type(exc).__name__}: {exc}"
        fallback_solver = "lbfgs"
        classifier = _logreg_classifier(C, max_iter, fallback_solver, random_state)
        classifier.fit(train_tfidf, train_label_ids)
        return classifier, fallback_solver, fallback_note


def _linear_svc_candidates(C: float, random_state: int) -> list[tuple[LinearSVC, str]]:
    return [
        (LinearSVC(C=C, random_state=random_state, dual="auto"), "auto"),
        (LinearSVC(C=C, random_state=random_state, dual=False), "False"),
        (LinearSVC(C=C, random_state=random_state, dual=True), "True"),
    ]


def _fit_linear_svm_with_fallback(
    train_tfidf: Any,
    train_label_ids: Sequence[int],
    C: float,
    random_state: int,
) -> tuple[LinearSVC, str, str | None]:
    """Fit LinearSVC with a safe fallback for older sklearn versions."""

    errors: list[str] = []
    for classifier, dual_used in _linear_svc_candidates(C, random_state):
        try:
            classifier.fit(train_tfidf, train_label_ids)
            fallback_note = "; ".join(errors) if errors else None
            return classifier, dual_used, fallback_note
        except Exception as exc:
            errors.append(f"dual={dual_used} failed with {type(exc).__name__}: {exc}")
    raise RuntimeError("All LinearSVC dual settings failed: " + "; ".join(errors))


def _prediction_rows(
    method: str,
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
    total_prediction_time_sec: float,
    train_texts_used: bool,
    train_labels_used: bool,
    trains_parameters: bool,
    scores: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    average_latency = _average_latency(total_prediction_time_sec, len(predicted_labels))
    rows: list[dict[str, Any]] = []
    for idx, predicted in enumerate(predicted_labels):
        row = {
            "method": method,
            "gold_label": str(gold_labels[idx]),
            "predicted_canonical": str(predicted),
            "is_correct": str(gold_labels[idx]) == str(predicted),
            "latency_sec": average_latency,
            "training_used": trains_parameters,
            "train_texts_used": train_texts_used,
            "train_labels_used": train_labels_used,
        }
        if scores is not None:
            row["raw_scores"] = scores[idx]
        rows.append(row)
    return rows


def run_tfidf_logistic_regression(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    gold_labels: Sequence[str],
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int | None = 50000,
    sublinear_tf: bool = True,
    lowercase: bool = True,
    C: float = 1.0,
    max_iter: int = 3000,
    solver: str = "saga",
    random_state: int = 42,
) -> ClassicalRunResult:
    """Train and evaluate a TF-IDF Logistic Regression baseline."""

    vectorizer = build_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
    )
    label_encoder = fit_label_encoder(train_labels)
    train_label_ids = labels_to_ids(label_encoder, train_labels)

    train_start = time.perf_counter()
    train_tfidf = vectorizer.fit_transform([str(text) for text in train_texts])
    classifier, solver_used, fallback_note = _fit_logistic_regression_with_fallback(
        train_tfidf,
        train_label_ids,
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )
    training_time_sec = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    test_tfidf = vectorizer.transform([str(text) for text in test_texts])
    predicted_ids = classifier.predict(test_tfidf)
    predicted_labels = ids_to_labels(label_encoder, predicted_ids)
    raw_scores: list[Any] | None = None
    if hasattr(classifier, "predict_proba"):
        raw_scores = [
            {"top_confidence": float(max(row))} for row in classifier.predict_proba(test_tfidf)
        ]
    elif hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(test_tfidf)
        raw_scores = [
            {"top_decision_score": float(np.max(np.atleast_1d(row)))}
            for row in np.asarray(scores)
        ]
    total_prediction_time_sec = time.perf_counter() - predict_start

    config = {
        "C": C,
        "max_iter": max_iter,
        "requested_solver": solver,
        "solver_used": solver_used,
        "fallback_note": fallback_note,
    }
    return ClassicalRunResult(
        method="tfidf_logistic_regression",
        vectorizer=vectorizer,
        classifier=classifier,
        label_encoder=label_encoder,
        predictions=_prediction_rows(
            "tfidf_logistic_regression",
            gold_labels,
            predicted_labels,
            total_prediction_time_sec,
            train_texts_used=True,
            train_labels_used=True,
            trains_parameters=True,
            scores=raw_scores,
        ),
        training_time_sec=float(training_time_sec),
        total_prediction_time_sec=float(total_prediction_time_sec),
        config=config,
    )


def run_tfidf_linear_svm(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    gold_labels: Sequence[str],
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int | None = 50000,
    sublinear_tf: bool = True,
    lowercase: bool = True,
    C: float = 1.0,
    random_state: int = 42,
) -> ClassicalRunResult:
    """Train and evaluate a TF-IDF Linear SVM baseline."""

    vectorizer = build_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
    )
    label_encoder = fit_label_encoder(train_labels)
    train_label_ids = labels_to_ids(label_encoder, train_labels)

    train_start = time.perf_counter()
    train_tfidf = vectorizer.fit_transform([str(text) for text in train_texts])
    classifier, dual_used, fallback_note = _fit_linear_svm_with_fallback(
        train_tfidf,
        train_label_ids,
        C=C,
        random_state=random_state,
    )
    training_time_sec = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    test_tfidf = vectorizer.transform([str(text) for text in test_texts])
    predicted_ids = classifier.predict(test_tfidf)
    predicted_labels = ids_to_labels(label_encoder, predicted_ids)
    scores = classifier.decision_function(test_tfidf)
    raw_scores = [
        {"top_decision_score": float(np.max(np.atleast_1d(row)))}
        for row in np.asarray(scores)
    ]
    total_prediction_time_sec = time.perf_counter() - predict_start

    config = {
        "C": C,
        "dual_used": dual_used,
        "fallback_note": fallback_note,
    }
    return ClassicalRunResult(
        method="tfidf_linear_svm",
        vectorizer=vectorizer,
        classifier=classifier,
        label_encoder=label_encoder,
        predictions=_prediction_rows(
            "tfidf_linear_svm",
            gold_labels,
            predicted_labels,
            total_prediction_time_sec,
            train_texts_used=True,
            train_labels_used=True,
            trains_parameters=True,
            scores=raw_scores,
        ),
        training_time_sec=float(training_time_sec),
        total_prediction_time_sec=float(total_prediction_time_sec),
        config=config,
    )
