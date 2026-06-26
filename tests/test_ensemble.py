from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.ensemble import (  # noqa: E402
    CONFIDENCE_MARGIN_FALLBACK,
    MEAN_CONFIDENCE_ENSEMBLE,
    VOTE_ENSEMBLE,
    confidence_margin_fallback,
    mean_confidence_ensemble,
    majority_vote_ensemble,
)
from gliner2_project.schema_variants import (  # noqa: E402
    BANKING_REQUEST_LABEL,
    CUSTOMER_INTENT_LABEL,
    PLAIN_LABEL,
    QUERY_ABOUT_LABEL,
)


def row(example_id, gold, pred, conf, method):
    return {
        "example_id": example_id,
        "text": f"text {example_id}",
        "gold_label": gold,
        "predicted_canonical": pred,
        "confidence": conf,
        "latency_sec": 0.1,
        "method": method,
    }


def test_majority_vote():
    predictions = {
        PLAIN_LABEL: [row(1, "a", "a", 0.6, PLAIN_LABEL)],
        QUERY_ABOUT_LABEL: [row(1, "a", "a", 0.7, QUERY_ABOUT_LABEL)],
        BANKING_REQUEST_LABEL: [row(1, "a", "b", 0.9, BANKING_REQUEST_LABEL)],
        CUSTOMER_INTENT_LABEL: [row(1, "a", "a", 0.4, CUSTOMER_INTENT_LABEL)],
    }
    result = majority_vote_ensemble(predictions)[0]
    assert result["method"] == VOTE_ENSEMBLE
    assert result["predicted_canonical"] == "a"
    assert result["votes"] == {"a": 3, "b": 1}


def test_majority_vote_tie_break_by_average_confidence():
    predictions = {
        PLAIN_LABEL: [row(1, "a", "a", 0.1, PLAIN_LABEL)],
        QUERY_ABOUT_LABEL: [row(1, "a", "b", 0.9, QUERY_ABOUT_LABEL)],
        BANKING_REQUEST_LABEL: [row(1, "a", "a", 0.2, BANKING_REQUEST_LABEL)],
        CUSTOMER_INTENT_LABEL: [row(1, "a", "b", 0.8, CUSTOMER_INTENT_LABEL)],
    }
    result = majority_vote_ensemble(predictions)[0]
    assert result["predicted_canonical"] == "b"
    assert result["tie_break"] == "average_confidence"


def test_mean_confidence_aggregation():
    predictions = {
        PLAIN_LABEL: [row(1, "a", "a", 0.1, PLAIN_LABEL)],
        QUERY_ABOUT_LABEL: [row(1, "a", "b", 0.9, QUERY_ABOUT_LABEL)],
        BANKING_REQUEST_LABEL: [row(1, "a", "a", 0.2, BANKING_REQUEST_LABEL)],
        CUSTOMER_INTENT_LABEL: [row(1, "a", "b", 0.7, CUSTOMER_INTENT_LABEL)],
    }
    result = mean_confidence_ensemble(predictions)[0]
    assert result["method"] == MEAN_CONFIDENCE_ENSEMBLE
    assert result["predicted_canonical"] == "b"
    assert result["confidence"] == 0.8


def test_confidence_margin_fallback_behavior():
    predictions = {
        PLAIN_LABEL: [row(1, "a", "a", 0.5, PLAIN_LABEL)],
        QUERY_ABOUT_LABEL: [row(1, "a", "b", 0.54, QUERY_ABOUT_LABEL)],
        BANKING_REQUEST_LABEL: [row(1, "a", "a", 0.5, BANKING_REQUEST_LABEL)],
        CUSTOMER_INTENT_LABEL: [row(1, "a", "b", 0.52, CUSTOMER_INTENT_LABEL)],
    }
    result = confidence_margin_fallback(predictions, margin_threshold=0.05)[0]
    assert result["method"] == CONFIDENCE_MARGIN_FALLBACK
    assert result["predicted_canonical"] == "a"
    assert result["fallback_used"] is True


def test_missing_confidence_behavior():
    predictions = {
        PLAIN_LABEL: [row(1, "a", "a", None, PLAIN_LABEL)],
        QUERY_ABOUT_LABEL: [row(1, "a", "b", None, QUERY_ABOUT_LABEL)],
        BANKING_REQUEST_LABEL: [row(1, "a", "a", None, BANKING_REQUEST_LABEL)],
        CUSTOMER_INTENT_LABEL: [row(1, "a", "b", None, CUSTOMER_INTENT_LABEL)],
    }
    mean_result = mean_confidence_ensemble(predictions)[0]
    assert mean_result["predicted_canonical"] is None
    assert mean_result["parse_error"] == "confidence unavailable"

    fallback_result = confidence_margin_fallback(predictions, margin_threshold=0.05)[0]
    assert fallback_result["predicted_canonical"] == "a"
    assert fallback_result["fallback_used"] is True
