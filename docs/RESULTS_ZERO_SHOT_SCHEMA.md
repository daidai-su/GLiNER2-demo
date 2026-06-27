# Zero-shot Contrastive Schema Results

## Purpose

This experiment tests whether GLiNER2 Banking77 intent classification can be improved using label schema design only.

It does not use:

- Banking77 train examples
- Banking77 train labels
- external LLMs
- GLiNER2 fine-tuning
- LoRA
- paid APIs
- GLiNER2 cloud API
- manual annotation of examples

The methods are still zero-shot with respect to Banking77 train data. The label descriptions and clusters are manually/rule-designed schemas, not example-level annotations.

## Methods

The notebook is:

`notebooks/06_gliner2_zero_shot_contrastive_schema_colab.ipynb`

Implemented method families:

- `description_all_labels`: all 77 labels are converted into natural-language label descriptions.
- `cluster_second_pass_from_plain`: if `plain_label` predicts a label in a known confusion cluster, rerun GLiNER2 over that cluster using contrastive descriptions.
- `cluster_second_pass_from_mean_confidence`: same second pass, starting from `mean_confidence_ensemble`.
- `hierarchical_gliner`: classify into coarse groups, then classify within a safety-union of possible groups.
- `oracle_zero_shot_schema_variants`: analysis-only upper bound across zero-shot schema variants.

## Evaluation Setting

- Dataset: `mteb/banking77`
- Split: test only
- Full test examples: 3076
- Small mode: stratified test subset with `SMALL_PER_LABEL = 5`
- Full mode: requires `CONFIRM_FULL_RUN = True`
- Output directory: `OUTPUT_DIR / "zero_shot_contrastive_schema"`

All methods must be compared on exactly the same evaluated example IDs.

## Result Table

Small run completed on 385 stratified Banking77 test examples.

This run used CUDA and did not load the Banking77 train split.

| Method | Num Examples | Accuracy | Macro F1 | Weighted F1 | Avg Latency | Parse Failure Rate |
|---|---:|---:|---:|---:|---:|---:|
| `query_about_label` | 385 | 0.698701 | 0.685261 | 0.685261 | 0.062909 | 0.0 |
| `mean_confidence_ensemble` | 385 | 0.693506 | 0.683315 | 0.683315 | n/a | 0.0 |
| `plain_label` | 385 | 0.688312 | 0.678945 | 0.678945 | 0.046213 | 0.0 |
| `hierarchical_gliner` | 385 | 0.688312 | 0.679149 | 0.679149 | 0.141843 | 0.0 |
| `customer_intent_label` | 385 | 0.685714 | 0.679603 | 0.679603 | 0.079615 | 0.0 |
| `cluster_second_pass_from_plain` | 385 | 0.685714 | 0.682417 | 0.682417 | 0.077703 | 0.0 |
| `cluster_second_pass_from_mean_confidence` | 385 | 0.685714 | 0.680778 | 0.680778 | n/a | 0.0 |
| `description_all_labels` | 385 | 0.672727 | 0.666934 | 0.666934 | 0.216508 | 0.0 |
| `banking_request_label` | 385 | 0.654545 | 0.671724 | 0.671724 | 0.090832 | 0.0 |

The best actual small-run method was `query_about_label`, not the new description or cluster methods. It improved over `plain_label` by:

- Accuracy: +0.010390
- Macro F1: +0.006316

The new schema-design methods did not improve accuracy over `plain_label` on this small run. `hierarchical_gliner` tied `plain_label` accuracy but had only a negligible macro-F1 gain. The raw latency for `mean_confidence_ensemble` and `cluster_second_pass_from_mean_confidence` in this run should not be interpreted because the notebook initially did not attach effective component latency to ensemble rows; the notebook has since been corrected for future runs.

## Second-pass Analysis

Small-run second-pass summary:

| Method | Examples | Triggered | Trigger Rate | Overall Accuracy | Accuracy When Triggered |
|---|---:|---:|---:|---:|---:|
| `cluster_second_pass_from_plain` | 385 | 241 | 0.625974 | 0.685714 | 0.601660 |
| `cluster_second_pass_from_mean_confidence` | 385 | 239 | 0.620779 | 0.685714 | 0.606695 |

The second pass was triggered for more than 60% of examples, but accuracy on triggered examples was only about 60%. This suggests the current contrastive cluster descriptions often change predictions without reliably selecting the correct label.

## Per-cluster And Per-label Analysis

The notebook saves:

- `tables/per_cluster_accuracy.csv`
- `tables/per_label_metrics.csv`
- `tables/per_label_delta_vs_plain.csv`
- `tables/confusion_pairs.csv`

Use these to identify whether schema descriptions help the known confusing families:

- card lifecycle
- top up
- identity
- transfer
- exchange/currency
- cash/ATM
- card payment/refund

Small-run examples of improvements:

- `pending_top_up` improved strongly under cluster second pass and hierarchical GLiNER.
- `card_swallowed`, `card_arrival`, `verify_my_identity`, and `request_refund` often improved under second-pass or hierarchical variants.

Small-run examples of degradations:

- `card_payment_wrong_exchange_rate`, `top_up_by_card_charge`, `declined_cash_withdrawal`, and `virtual_card_not_working` degraded under one or more new variants.

Because this is a small run with only five examples per label, these per-label deltas should be treated as diagnostic rather than final evidence.

## Oracle Analysis

The notebook saves:

`tables/oracle_zero_shot_schema_variants.csv`

This is analysis only. It answers whether better schema selection could theoretically improve over a single selected method. It is not a deployable classifier and should not be reported as the actual model performance.

Use this to judge whether a large gain such as +5 accuracy points appears possible from schema selection alone.

Small-run oracle result:

| Methods | Num Examples | Oracle Accuracy | Oracle Correct |
|---|---:|---:|---:|
| all available zero-shot schema variants | 385 | 0.779221 | 300 |

Compared with `plain_label` accuracy 0.688312, the oracle upper bound is +0.090909. This means that the evaluated schema variants collectively contain enough correct predictions to make a +5 point gain theoretically possible. However, none of the implemented non-oracle selection strategies achieved that gain.

### Schema Correctness Distribution

For each example, we counted how many available zero-shot schema variants predicted the gold label.

| Correct Variants | Examples |
|---:|---:|
| 0 | 85 |
| 1 | 8 |
| 2 | 16 |
| 3 | 7 |
| 4 | 6 |
| 5 | 13 |
| 6 | 16 |
| 7 | 16 |
| 8 | 218 |

Bucketed:

| Correct Variant Bucket | Examples |
|---|---:|
| 0 | 85 |
| 1 | 8 |
| 2 | 16 |
| 3 | 7 |
| 4+ | 269 |

Interpretation:

- 85 examples were missed by every evaluated schema variant.
- Only 8 examples were correct under exactly one variant, so the oracle gain is not mainly from extremely rare one-off schema wins.
- 269 examples were correct under at least four variants, suggesting that many examples are robustly solvable once the schema phrasing is favorable.

### Plain vs Query-about Paired Analysis

`query_about_label` was the best actual small-run method.

| Pair | Plain Correct | Query Correct | Both Correct | Both Wrong | Plain Wrong / Query Correct | Plain Correct / Query Wrong | Net Gain | Accuracy Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `plain_label` -> `query_about_label` | 265 | 269 | 255 | 106 | 14 | 10 | 4 | 0.010390 |

`query_about_label` fixed examples such as `card_arrival`, `age_limit`, `transfer_fee_charged`, `cancel_transfer`, `card_swallowed`, and `unable_to_verify_identity`. It also broke some examples that `plain_label` had correct, including `virtual_card_not_working`, `compromised_card`, `atm_support`, `card_acceptance`, and `balance_not_updated_after_bank_transfer`.

This means `query_about_label` is slightly better on this small subset, but it is not uniformly safer than `plain_label`.

### Oracle-correct But Selector-wrong Examples

We also checked cases where at least one schema variant predicted the gold label, but both `mean_confidence_ensemble` and a post-hoc four-schema majority vote were wrong.

| Diagnostic | Count |
|---|---:|
| Total examples | 385 |
| Oracle-correct examples | 300 |
| Oracle-correct but `mean_confidence_ensemble` wrong | 33 |
| Oracle-correct but four-schema vote wrong | 33 |
| Oracle-correct but both mean-confidence and four-schema vote wrong | 25 |

These 25 examples are especially useful for future selector design. Many involve semantically adjacent Banking77 labels such as:

- `card_arrival` vs `card_delivery_estimate`
- `pending_top_up` vs `top_up_failed`
- `verify_my_identity` vs `why_verify_identity`
- `card_payment_fee_charged` vs `extra_charge_on_statement`
- `reverted_card_payment?` vs `request_refund`
- `cancel_transfer` vs `request_refund`

In several examples, the correct schema prediction exists but is a minority prediction, so naive majority voting and mean-confidence aggregation can still choose the wrong nearby label.

## Reporting Guidance

Safe claims:

- The experiment uses no Banking77 train examples or train labels.
- Label descriptions are schema design, not example annotation.
- Cluster second pass is a zero-shot reranking/reclassification strategy over predefined label groups.
- Oracle accuracy is an upper bound for analysis only.

Claims to avoid:

- Do not claim SOTA.
- Do not claim generalization to other datasets.
- Do not claim statistical significance without a proper test.
- Do not claim that full-test schema iterations are confirmatory if schema wording was revised after seeing full-test results.
- Do not mix these zero-shot results with TF-IDF/SVM train-set-based baselines as if they were the same setting.

## Presentation Recommendation

For the main three-minute presentation, keep the original schema wording full result as the central story unless this notebook shows a clear, verified improvement.

Based on the current small run, do not replace the main presentation result with this experiment. Use it as exploratory evidence:

- Actual methods did not beat the best simple wording variant.
- Oracle accuracy suggests schema selection has potential.
- A better selection rule would be needed before running or claiming a full-test improvement.

If the zero-shot schema experiment improves meaningfully, present it as:

> A follow-up zero-shot schema design experiment using contrastive label descriptions and cluster second passes, without using Banking77 train data.

If it does not improve, present it as a useful negative result:

> More detailed schema descriptions did not reliably overcome GLiNER2's Banking77 intent confusions under the zero-shot setting.
