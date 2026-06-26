# Method

## Background

GLiNER2 provides a schema-driven information extraction and classification interface. In zero-shot intent classification, the model receives an input utterance and a schema containing candidate intent labels. The wording of those candidate labels is part of the model input, so classification behavior can depend on how the schema is phrased.

## Problem

Banking77 contains 77 fine-grained banking support intents. Many labels are semantically close, such as `pending_top_up`, `top_up_failed`, and `top_up_reverted`. A single terse label wording may not be enough to express the intended decision boundary for every label.

This project studies schema wording sensitivity: whether GLiNER2 predictions change when the same canonical intent label is represented by different deterministic natural-language schema phrases.

## Baseline

The primary baseline is `plain_label`.

For each canonical `label_text`, underscores are replaced with spaces and text is lowercased. For example:

- Canonical label: `card_arrival`
- Candidate string: `card arrival`

The model predicts one candidate string, which is mapped back to the canonical Banking77 `label_text`.

## Proposed Method

The proposed MVP is a deterministic paraphrase schema ensemble. It does not use LLM-generated paraphrases by default.

The compared single-schema methods are:

- `raw_label`: original `label_text`, such as `card_arrival`
- `plain_label`: normalized label, such as `card arrival`
- `query_about_label`: `question about {plain_label}`
- `banking_request_label`: `banking support request about {plain_label}`
- `customer_intent_label`: `customer intent: {plain_label}`

The ensemble methods use only `plain_label`, `query_about_label`, `banking_request_label`, and `customer_intent_label`. The `raw_label` method is included for analysis but is not part of the ensemble.

The implemented ensembles are:

- `vote_ensemble`: majority vote over the four schema variants. Ties are broken by average confidence when available; otherwise the `plain_label` prediction is used.
- `mean_confidence_ensemble`: aggregate predicted canonical labels by mean confidence. This method is unavailable if confidence values are missing.
- `confidence_margin_fallback`: use `mean_confidence_ensemble` only when the top-label confidence margin is at least a fixed threshold, otherwise fall back to `plain_label`. The default threshold is `0.05` and is not tuned on the test split.

## No Human Annotation

The experiment uses the existing Banking77 `label_text` field as the canonical gold label. No additional human labeling is performed.

## No Model Training

The experiment is zero-shot inference only. GLiNER2 is loaded locally from the pretrained checkpoint and is not fine-tuned. LoRA and other parameter-efficient training methods are outside the MVP.

## Evaluation Setup

The primary dataset is `mteb/banking77`, using the `test` split. Each method is evaluated on the exact same sampled subset:

- `smoke`: 10 examples total
- `small`: stratified sample with `SMALL_PER_LABEL` examples per label
- `full`: all test examples, requiring explicit confirmation

The notebook saves evaluated example IDs for reproducibility.

Metrics:

- Accuracy
- Macro F1
- Weighted F1
- Per-label F1
- Average latency per example
- Total runtime
- Parse failure rate

Additional analysis includes schema disagreement, improvement/degradation relative to `plain_label`, top confused label pairs, confidence analysis, and fallback rate.

## Limitations

- Schema wordings are deterministic templates, not optimized or learned prompts.
- The confidence score should not be interpreted as a calibrated probability.
- The full Banking77 test split is still only one dataset and one domain.
- No development set is used for threshold tuning; the default margin threshold is fixed in advance.
- Improvements may depend on the GLiNER2 version and model checkpoint.
