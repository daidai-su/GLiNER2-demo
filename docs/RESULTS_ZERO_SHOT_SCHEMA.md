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

Run the notebook and copy `tables/results_summary.csv` here.

Do not fabricate results. If full-test schema choices were revised after inspecting full-test output, mark the result as exploratory.

| Method | Num Examples | Accuracy | Macro F1 | Weighted F1 | Avg Latency | Parse Failure Rate |
|---|---:|---:|---:|---:|---:|---:|
| `plain_label` | | | | | | |
| `mean_confidence_ensemble` | | | | | | |
| `description_all_labels` | | | | | | |
| `cluster_second_pass_from_plain` | | | | | | |
| `cluster_second_pass_from_mean_confidence` | | | | | | |
| `hierarchical_gliner` | | | | | | |

## Second-pass Analysis

Copy `tables/second_pass_summary.csv` here after running.

Report:

- how often the cluster second pass is triggered
- accuracy when second pass is triggered
- examples fixed by the second pass
- examples broken by the second pass

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

## Oracle Analysis

The notebook saves:

`tables/oracle_zero_shot_schema_variants.csv`

This is analysis only. It answers whether better schema selection could theoretically improve over a single selected method. It is not a deployable classifier and should not be reported as the actual model performance.

Use this to judge whether a large gain such as +5 accuracy points appears possible from schema selection alone.

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

If the zero-shot schema experiment improves meaningfully, present it as:

> A follow-up zero-shot schema design experiment using contrastive label descriptions and cluster second passes, without using Banking77 train data.

If it does not improve, present it as a useful negative result:

> More detailed schema descriptions did not reliably overcome GLiNER2's Banking77 intent confusions under the zero-shot setting.
