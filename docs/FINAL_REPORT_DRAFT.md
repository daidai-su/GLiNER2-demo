# Robust Schema Wording for GLiNER2: Paraphrase-Ensembled Intent Classification

## Background

GLiNER2 uses a schema-driven interface: the model receives text together with a schema that names the candidate labels or extraction targets. For zero-shot intent classification, this means the candidate label wording is part of the model input. If the wording changes, the model may change its prediction even when the underlying canonical intent is the same.

## Research Question

Does deterministic paraphrase ensembling over schema wordings improve GLiNER2 zero-shot intent classification robustness compared with a single plain-label schema?

## Model And Dataset

- Model: `fastino/gliner2-base-v1`
- Dataset: `mteb/banking77`
- Split: `test`
- Task: 77-way intent classification
- Gold label: `label_text`
- Evaluation examples: 3076
- Device: CUDA GPU
- Training: none
- Human annotation: none
- Paid APIs: none

## Baseline

The primary baseline is `plain_label`.

For each canonical Banking77 label, underscores are replaced with spaces:

- Canonical label: `card_arrival`
- Schema candidate: `card arrival`

GLiNER2 predicts one candidate string, which is mapped back to the canonical `label_text`.

## Proposed Schema Ensemble Methods

The experiment uses deterministic schema variants:

- `plain_label`: `card arrival`
- `query_about_label`: `question about card arrival`
- `banking_request_label`: `banking support request about card arrival`
- `customer_intent_label`: `customer intent: card arrival`

The final selected method is `mean_confidence_ensemble`. It runs the four schema variants, maps each prediction back to a canonical Banking77 label, and aggregates by mean confidence. The secondary method is `vote_ensemble`, which uses majority vote over the same four schema variants.

The `raw_label` method is included for analysis only and uses labels such as `card_arrival`.

## Experimental Setup

All methods are evaluated on the same Banking77 test examples. The full run used all 3076 test examples. No hyperparameters were tuned on the test split. The confidence margin threshold for `confidence_margin_fallback` was fixed at 0.05.

The notebook saves:

- prediction JSONL files per method
- `results_summary.csv`
- per-label metrics
- paired analysis files
- confusion pairs
- schema disagreement examples
- run manifest

## Main Results

| Method | Accuracy | Macro F1 | Weighted F1 | Parse Failure Rate |
|---|---:|---:|---:|---:|
| `mean_confidence_ensemble` | 0.689532 | 0.680569 | 0.680685 | 0.0 |
| `confidence_margin_fallback` | 0.687906 | 0.678603 | 0.678708 | 0.0 |
| `vote_ensemble` | 0.687581 | 0.678583 | 0.678686 | 0.0 |
| `plain_label` | 0.684330 | 0.675726 | 0.675849 | 0.0 |
| `customer_intent_label` | 0.678479 | 0.669596 | 0.669684 | 0.0 |
| `raw_label` | 0.676203 | 0.669552 | 0.669700 | 0.0 |
| `query_about_label` | 0.672302 | 0.660827 | 0.660988 | 0.0 |
| `banking_request_label` | 0.645319 | 0.656876 | 0.657096 | 0.0 |

Compared with `plain_label`, the selected `mean_confidence_ensemble` improved:

- Accuracy: +0.005202
- Macro F1: +0.004843
- Weighted F1: +0.004836

The secondary `vote_ensemble` also improved over `plain_label`:

- Accuracy: +0.003251
- Macro F1: +0.002857
- Weighted F1: +0.002837

## Schema Sensitivity Analysis

The four schema variants disagreed on 26.46% of the full test examples.

| Unique Predicted Labels Across Schema Variants | Example Count |
|---:|---:|
| 1 | 2262 |
| 2 | 568 |
| 3 | 198 |
| 4 | 48 |

This directly supports the claim that GLiNER2 predictions are sensitive to schema wording.

## Error Analysis

For `plain_label` versus `mean_confidence_ensemble`:

- Plain wrong / mean-confidence correct: 58
- Plain correct / mean-confidence wrong: 42
- Both correct: 2063
- Both wrong: 913
- Net gain: 16 examples

The largest F1 improvements were for labels such as `top_up_reverted`, `verify_top_up`, `card_swallowed`, `verify_my_identity`, and `top_up_failed`.

The largest F1 degradations were for labels such as `declined_cash_withdrawal`, `beneficiary_not_allowed`, `reverted_card_payment?`, `exchange_charge`, and `cash_withdrawal_not_recognised`.

Common plain-label confusions included:

- `get_physical_card` -> `change_pin`
- `verify_my_identity` -> `why_verify_identity`
- `card_arrival` -> `card_delivery_estimate`
- `top_up_reverted` -> `top_up_failed`
- `pending_top_up` -> `top_up_failed`
- `pending_transfer` -> `transfer_timing`

These errors are mostly semantically adjacent Banking77 intents.

## Discussion

The improvement is modest, but the experiment provides a clean result for a course project. The method requires no model training, no manual annotation, and no paid API. It also directly tests the schema wording sensitivity hypothesis.

Confidence analysis showed that incorrect predictions can still have high confidence. Therefore, the GLiNER2 confidence output should not be treated as a calibrated probability. However, because `mean_confidence_ensemble` was the best method, confidence appears to be useful as a ranking signal for aggregation in this setup.

## Limitations

- The performance gain is small.
- No statistical significance test is reported.
- The confidence scores are not calibrated probabilities.
- The experiment uses one dataset and one domain.
- The deterministic schema templates are simple.
- The result should not be claimed as SOTA.
- The method may not transfer to other datasets without further evaluation.

## Claims That Are Safe To Make

- GLiNER2 predictions changed under different schema wording.
- The full test run showed about 26.5% schema disagreement.
- `mean_confidence_ensemble` slightly outperformed `plain_label` on the full Banking77 test split.
- The approach requires no fine-tuning and no new human annotations.

## Claims To Avoid

- Do not claim a large performance improvement.
- Do not claim strong statistical significance.
- Do not describe confidence as calibrated probability.
- Do not claim the method necessarily works on other datasets.
- Do not claim SOTA performance.

## AI Usage Disclosure

An AI coding assistant was used to generate and revise project code, notebooks, tests, and report drafts. The human user ran the Colab experiments, reviewed the outputs, and provided the final full-run metrics used in this report draft.

## Conclusion

The project confirms that schema wording affects GLiNER2 zero-shot intent classification. A deterministic schema ensemble provides a small but consistent improvement over a plain-label baseline on Banking77 full test evaluation. The final selected method, `mean_confidence_ensemble`, achieved macro F1 0.680569 compared with 0.675726 for `plain_label`.
