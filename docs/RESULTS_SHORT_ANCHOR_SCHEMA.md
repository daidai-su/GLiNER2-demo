# Short Anchor Schema Results

## Research Question

Can short, natural-language label anchors improve GLiNER2 zero-shot Banking77 intent classification without using train examples, train labels, external LLMs, fine-tuning, LoRA, or paid APIs?

## Why Long Descriptions Failed

The previous zero-shot contrastive schema small run showed that long label descriptions, cluster second passes, and hierarchical GLiNER did not clearly improve over the best simple wording variant. The best actual small-run method was `query_about_label`, while `description_all_labels` degraded accuracy.

This suggests that GLiNER2 may prefer short, direct schema labels over long explanations for this closed-label intent classification setup.

## Short Anchor Hypothesis

Instead of using canonical labels such as `card_arrival` or long descriptions, each label is converted into a short human-readable anchor:

- `card_arrival` -> `card not arrived`
- `card_delivery_estimate` -> `card delivery time`
- `pending_top_up` -> `top-up is pending`
- `top_up_failed` -> `top-up failed`
- `verify_my_identity` -> `verify my identity`
- `cash_withdrawal_not_recognised` -> `unrecognized cash withdrawal`

The main hypothesis is that short anchors are easier for GLiNER2 to match than either raw canonical labels or long label descriptions.

## Methods

Notebook:

`notebooks/07_gliner2_short_anchor_schema_colab.ipynb`

Methods:

- `short_anchor`
- `query_about_short_anchor`
- `customer_request_short_anchor`
- `issue_short_anchor`
- `minimal_contrastive_anchor`
- `query_about_minimal_contrastive_anchor`

Baselines recomputed or cached in the same notebook:

- `plain_label`
- `query_about_label`
- `mean_confidence_ensemble`

The most important proposed method is:

`query_about_minimal_contrastive_anchor`

## Evaluation Setting

- Dataset: `mteb/banking77`
- Split: test only
- No Banking77 train split is loaded
- Small mode: stratified subset with `SMALL_PER_LABEL = 5`
- Full mode: all test examples only if `CONFIRM_FULL_RUN = True`

## Results

Run the notebook and copy `tables/short_anchor_results_summary.csv` here.

Do not fabricate results. If schema wording is revised after viewing full-test output, mark the result as exploratory.

| Method | Num Examples | Accuracy | Macro F1 | Weighted F1 | Avg Latency | Parse Failure Rate |
|---|---:|---:|---:|---:|---:|---:|
| `plain_label` | | | | | | |
| `query_about_label` | | | | | | |
| `query_about_short_anchor` | | | | | | |
| `query_about_minimal_contrastive_anchor` | | | | | | |

## Paired Comparison

The notebook saves `tables/paired_comparisons.csv`.

Key comparisons:

- `plain_label` vs `query_about_short_anchor`
- `query_about_label` vs `query_about_short_anchor`
- `plain_label` vs `query_about_minimal_contrastive_anchor`
- `query_about_label` vs `query_about_minimal_contrastive_anchor`

For each pair, report:

- both correct
- both wrong
- baseline correct / proposed wrong
- baseline wrong / proposed correct
- net gain
- accuracy delta

## Oracle Analysis

The notebook saves:

- `tables/oracle_analysis.csv`
- `tables/oracle_examples.csv`
- `tables/correct_vote_count_distribution.csv`
- `tables/max_vote_count_distribution.csv`

Oracle analysis is diagnostic only. It should not be reported as deployable performance.

Use it to answer whether +5 accuracy points is realistic from schema selection alone.

## Error Analysis

The notebook saves:

- `tables/improved_examples_vs_query_about.csv`
- `tables/degraded_examples_vs_query_about.csv`
- `tables/confusion_pairs.csv`
- `tables/per_label_delta_vs_query_about.csv`

Focus on whether the short-anchor methods fix or break the same nearby label families:

- card arrival vs card delivery estimate
- pending top-up vs top-up failed
- identity verification variants
- card payment fee vs extra charge
- transfer timing vs pending transfer

## Whether Full Run Is Worth Doing

Do not recommend a full run unless a short-anchor method beats `query_about_label` by at least +0.02 accuracy on the small run.

If a method improves substantially, run full only once with the selected method or clearly mark any additional full-run iteration as exploratory.

## Limitations

- The short anchors are manually/rule-designed schema wording.
- This remains one dataset and one model.
- The result should not be claimed as SOTA.
- The result should not be claimed to generalize without additional datasets.
- Full-test schema iteration must be reported as exploratory.

## Allowed Wording

- "This is zero-shot with respect to Banking77 train data."
- "No train examples or train labels are used."
- "Short anchor schemas improve/degrade performance in this experiment."
- "The result is exploratory."

## Wording To Avoid

- "This proves the method generalizes to all datasets."
- "This is SOTA."
- "We tuned on full test."
- "We used train data."
