# Classical And Retrieval Baseline Comparison

## Research Question

This phase compares pure zero-shot GLiNER2 classification with simple train-set-based baselines on Banking77.

The goal is not to claim that TF-IDF baselines improve GLiNER2. The goal is to provide a fair reference point for how strong retrieval-only and classical supervised methods are on this closed-domain intent classification dataset.

## Why These Baselines Are Included

Banking77 is a fixed 77-way intent classification benchmark. A model that can use the Banking77 train split has access to labeled in-domain examples. Therefore, simple TF-IDF retrieval or supervised linear classifiers may be strong even without neural fine-tuning.

These baselines help separate two questions:

- How well does GLiNER2 work in a pure zero-shot schema-driven setting?
- How strong are simple methods that use train texts and train labels?

## Important Setting Distinction

| Setting | Methods | Uses Train Texts | Uses Train Labels | Trains Parameters | Uses GLiNER2 |
|---|---|---:|---:|---:|---:|
| pure zero-shot | `plain_label`, `mean_confidence_ensemble` | no | no | no | yes |
| retrieval-only | `tfidf_knn_majority`, `tfidf_weighted_knn` | yes | yes | no | no |
| retrieval-assisted GLiNER | `tfidf_candidate_pruning_gliner2`, `tfidf_candidate_pruning_mean_confidence_ensemble` | yes | yes | no | yes |
| classical supervised | `tfidf_logistic_regression`, `tfidf_linear_svm` | yes | yes | yes | no |

The train-set-based methods should not be described as zero-shot improvements.

## Dataset And Split

- Dataset: `mteb/banking77`
- Train split: used only for TF-IDF fitting, retrieval memory, and supervised classical training
- Test split: used for evaluation
- Test examples are not inserted into the retrieval index
- All available methods are compared on the exact same evaluated example IDs

## Methods

- `plain_label`: cached GLiNER2 prediction from the schema wording notebook, if available
- `mean_confidence_ensemble`: cached GLiNER2 ensemble prediction, if available
- `tfidf_knn_majority`: top-k TF-IDF cosine neighbors, label majority vote
- `tfidf_weighted_knn`: top-k TF-IDF cosine neighbors, summed-similarity vote
- `tfidf_logistic_regression`: supervised TF-IDF Logistic Regression
- `tfidf_linear_svm`: supervised TF-IDF Linear SVM
- Optional retrieval-pruning GLiNER2 methods are loaded only if cached results already exist

## Main Result Table

Run `notebooks/05_gliner2_classical_baselines_comparison_colab.ipynb` and copy the saved `tables/classical_results_summary.csv` here.

Do not fill this table manually unless the notebook has actually been executed.

## Paired Comparison

The notebook saves `tables/paired_comparisons.csv` for key pairs such as:

- `plain_label` vs `tfidf_knn_majority`
- `plain_label` vs `tfidf_weighted_knn`
- `plain_label` vs `tfidf_logistic_regression`
- `plain_label` vs `tfidf_linear_svm`
- `tfidf_knn_majority` vs `tfidf_weighted_knn`
- `tfidf_weighted_knn` vs supervised TF-IDF baselines

Use paired counts instead of comparing methods evaluated on different examples.

## Bootstrap CI

The notebook optionally saves `tables/bootstrap_accuracy_ci.csv`.

If a confidence interval excludes zero, phrase the result cautiously. Do not claim strong statistical significance unless the design and interval clearly support it.

## Oracle Analysis

The notebook saves `tables/oracle_analysis.csv`.

Oracle accuracy is an analysis-only upper bound: an example is counted correct if any method in a group predicts the gold label. It is not a deployable method.

## Per-label And Confusion Analysis

The notebook saves:

- `tables/per_label_metrics.csv`
- `tables/per_label_delta_vs_plain.csv`
- `tables/confusion_pairs.csv`

Focus especially on Banking77 families that are naturally confusable:

- card arrival, card delivery estimate, order physical card, get physical card
- pending top up, top up failed, top up reverted, verify top up
- verify my identity, why verify identity, unable to verify identity
- pending transfer, transfer timing, failed transfer
- exchange via app, exchange rate, fiat currency support

## Discussion

If TF-IDF kNN or supervised TF-IDF methods outperform GLiNER2, this should be interpreted as evidence that simple train-set-based methods are strong baselines for Banking77.

It should not be written as "we improved GLiNER2" unless the method actually uses GLiNER2. It also should not be used to weaken the schema wording result: the main schema wording experiment answers a different, pure zero-shot question.

## Limitations

- Retrieval and supervised baselines use train labels and are not pure zero-shot.
- Hyperparameters are fixed and not tuned on a validation split.
- Results are limited to Banking77.
- TF-IDF can be strong on lexical intent datasets, but may not generalize to more open-ended domains.
- Cached GLiNER2 results are included only when evaluated example IDs match exactly.

## What Result Should Be Used In The Final Presentation

For a three-minute main presentation, use the full schema wording result as the central result:

- `plain_label` accuracy: 0.684330
- `mean_confidence_ensemble` accuracy: 0.689532
- schema disagreement rate: 0.264629

Use the classical baseline comparison only as an optional backup slide or report appendix, unless the assignment specifically asks for supervised baseline comparisons.
