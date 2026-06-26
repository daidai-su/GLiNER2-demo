"""TF-IDF retrieval helpers for train-set-assisted candidate pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TfidfRetriever:
    """A simple TF-IDF retriever fit only on training examples."""

    vectorizer: TfidfVectorizer
    train_matrix: Any
    train_examples: list[dict[str, Any]]
    text_key: str = "text"
    label_key: str = "label_text"
    id_key: str = "example_id"

    def retrieve(self, query_text: str, k: int) -> list[dict[str, Any]]:
        """Return top-k train neighbors sorted by cosine similarity descending."""

        if k <= 0:
            return []
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.train_matrix)[0]
        order = sorted(
            range(len(similarities)),
            key=lambda idx: (-float(similarities[idx]), idx),
        )[: min(k, len(similarities))]

        neighbors: list[dict[str, Any]] = []
        for rank, train_index in enumerate(order, start=1):
            example = self.train_examples[train_index]
            neighbors.append(
                {
                    "rank": rank,
                    "train_index": train_index,
                    "example_id": example.get(self.id_key, train_index),
                    "text": example.get(self.text_key),
                    "label_text": example.get(self.label_key),
                    "similarity": float(similarities[train_index]),
                }
            )
        return neighbors


def prepare_examples(
    rows: Sequence[dict[str, Any]] | Any,
    text_key: str = "text",
    label_key: str = "label_text",
    id_key: str = "example_id",
) -> list[dict[str, Any]]:
    """Convert dataset-like rows into plain dictionaries with stable example IDs."""

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault(id_key, idx)
        if text_key not in item:
            raise ValueError(f"Missing text key: {text_key}")
        if label_key not in item:
            raise ValueError(f"Missing label key: {label_key}")
        examples.append(item)
    return examples


def build_tfidf_retriever(
    train_examples: Sequence[dict[str, Any]],
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int | None = 50000,
    text_key: str = "text",
    label_key: str = "label_text",
    id_key: str = "example_id",
) -> TfidfRetriever:
    """Fit a TF-IDF retriever on training texts only."""

    examples = prepare_examples(train_examples, text_key, label_key, id_key)
    train_texts = [str(example[text_key]) for example in examples]
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    return TfidfRetriever(
        vectorizer=vectorizer,
        train_matrix=train_matrix,
        train_examples=examples,
        text_key=text_key,
        label_key=label_key,
        id_key=id_key,
    )


def rank_labels_from_neighbors(
    neighbors: Sequence[dict[str, Any]],
    label_key: str = "label_text",
) -> list[dict[str, Any]]:
    """Rank neighbor labels by frequency, summed similarity, max similarity, label."""

    scores: dict[str, dict[str, Any]] = {}
    for neighbor in neighbors:
        label = neighbor.get(label_key)
        if label is None:
            label = neighbor.get("label_text")
        if label is None:
            continue
        label = str(label)
        similarity = float(neighbor.get("similarity", 0.0))
        if label not in scores:
            scores[label] = {
                "label": label,
                "frequency": 0,
                "sum_similarity": 0.0,
                "max_similarity": float("-inf"),
            }
        scores[label]["frequency"] += 1
        scores[label]["sum_similarity"] += similarity
        scores[label]["max_similarity"] = max(scores[label]["max_similarity"], similarity)

    ranked = list(scores.values())
    ranked.sort(
        key=lambda item: (
            -int(item["frequency"]),
            -float(item["sum_similarity"]),
            -float(item["max_similarity"]),
            str(item["label"]),
        )
    )
    return ranked


def tfidf_knn_predict(
    query_text: str,
    retriever: TfidfRetriever,
    k: int,
) -> dict[str, Any]:
    """Predict by majority vote over top-k retrieved train labels."""

    neighbors = retriever.retrieve(query_text, k)
    ranked_labels = rank_labels_from_neighbors(neighbors)
    predicted = ranked_labels[0]["label"] if ranked_labels else None
    return {
        "predicted_canonical": predicted,
        "neighbors": neighbors,
        "label_scores": {item["label"]: item for item in ranked_labels},
    }


def assert_no_test_ids_in_retriever(
    retriever: TfidfRetriever,
    test_example_ids: Sequence[Any],
) -> None:
    """Raise if any test example ID appears in the train retrieval memory."""

    train_ids = {example.get(retriever.id_key) for example in retriever.train_examples}
    overlap = train_ids.intersection(set(test_example_ids))
    if overlap:
        raise ValueError(
            "Test example IDs found in retrieval index: "
            + ", ".join(str(item) for item in sorted(overlap))
        )
