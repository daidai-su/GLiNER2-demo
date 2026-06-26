"""Robust wrappers around GLiNER2 local text classification."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ParsedPrediction:
    """A normalized view over possible GLiNER2 classify_text outputs."""

    label: str | None
    confidence: float | None
    raw_output: Any


def make_json_safe(value: Any) -> Any:
    """Convert common model outputs into JSON-serializable values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def load_gliner2_model(model_name: str, device: str | None = None) -> Any:
    """Load GLiNER2 locally without requiring the Pioneer/cloud API."""

    try:
        from gliner2 import GLiNER2
    except ImportError as exc:  # pragma: no cover - requires Colab dependency install.
        raise ImportError(
            "Could not import GLiNER2. Install dependencies from requirements-colab.txt "
            "before loading the model."
        ) from exc

    model = GLiNER2.from_pretrained(model_name)
    if device and hasattr(model, "to"):
        model = model.to(device)
    return model


def method_accepts_argument(method: Any, argument_name: str) -> bool:
    """Return whether a callable signature accepts a named argument."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return argument_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def classify_text_raw(
    model: Any,
    text: str,
    candidate_labels: Iterable[str],
    task_name: str = "intent",
    include_confidence: bool = True,
) -> Any:
    """Call classify_text using the official schema-driven interface."""

    classify_text = getattr(model, "classify_text")
    schema = {task_name: list(candidate_labels)}
    kwargs: dict[str, Any] = {}
    if include_confidence and method_accepts_argument(classify_text, "include_confidence"):
        kwargs["include_confidence"] = True
    try:
        return classify_text(text, schema, **kwargs)
    except TypeError:
        if not kwargs:
            raise
        return classify_text(text, schema)


def _find_first_label_and_confidence(value: Any) -> tuple[str | None, float | None]:
    """Recursively search an output object for a plausible label and confidence."""

    if isinstance(value, str):
        return value, None

    if isinstance(value, dict):
        label_keys = (
            "label",
            "prediction",
            "predicted_label",
            "class",
            "intent",
            "name",
            "text",
        )
        confidence_keys = ("confidence", "score", "probability", "prob")

        label = None
        confidence = None
        for key in label_keys:
            if key in value and isinstance(value[key], str):
                label = value[key]
                break
        for key in confidence_keys:
            if key in value and isinstance(value[key], (int, float)):
                confidence = float(value[key])
                break
        if label is not None:
            return label, confidence

        for nested in value.values():
            nested_label, nested_confidence = _find_first_label_and_confidence(nested)
            if nested_label is not None:
                return nested_label, nested_confidence

    if isinstance(value, (list, tuple)):
        for item in value:
            label, confidence = _find_first_label_and_confidence(item)
            if label is not None:
                return label, confidence

    return None, None


def parse_classification_output(raw_output: Any) -> ParsedPrediction:
    """Parse likely GLiNER2 output variants into a single prediction object."""

    label, confidence = _find_first_label_and_confidence(raw_output)
    return ParsedPrediction(label=label, confidence=confidence, raw_output=raw_output)


def predict_intent(
    model: Any,
    text: str,
    candidate_labels: Iterable[str],
    task_name: str = "intent",
) -> ParsedPrediction:
    """Run local intent classification and parse the result."""

    raw_output = classify_text_raw(
        model=model,
        text=text,
        candidate_labels=candidate_labels,
        task_name=task_name,
        include_confidence=True,
    )
    return parse_classification_output(raw_output)
