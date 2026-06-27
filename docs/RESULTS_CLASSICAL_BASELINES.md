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

Full run completed on 3076 Banking77 test examples.

This run intentionally used only TF-IDF based methods. No GLiNER2 cached predictions were loaded, and no GLiNER2 inference was run.

| Method | Setting | Accuracy | Macro F1 | Weighted F1 | Training Time Sec | Prediction Time Sec |
|---|---|---:|---:|---:|---:|---:|
| `tfidf_linear_svm` | classical supervised | 0.894018 | 0.894057 | 0.894147 | 1.022053 | 0.080838 |
| `tfidf_logistic_regression` | classical supervised | 0.856957 | 0.855547 | 0.855625 | 2.999208 | 0.096104 |
| `tfidf_weighted_knn` | retrieval-only | 0.800390 | 0.799004 | 0.799061 | n/a | 31.489404 |
| `tfidf_knn_majority` | retrieval-only | 0.788362 | 0.786465 | 0.786534 | n/a | 38.554099 |

All four methods use Banking77 train texts and train labels. The supervised methods also train classifier parameters.

## Paired Comparison

Full-run paired comparisons:

| Method A | Method B | A Correct | B Correct | A Correct / B Wrong | A Wrong / B Correct | Net Gain Of B | Accuracy Delta | Macro F1 Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tfidf_knn_majority` | `tfidf_weighted_knn` | 2425 | 2462 | 7 | 44 | 37 | 0.012029 | 0.012538 |
| `tfidf_weighted_knn` | `tfidf_logistic_regression` | 2462 | 2636 | 97 | 271 | 174 | 0.056567 | 0.056543 |
| `tfidf_weighted_knn` | `tfidf_linear_svm` | 2462 | 2750 | 45 | 333 | 288 | 0.093628 | 0.095053 |

Use paired counts instead of comparing methods evaluated on different examples.

## Bootstrap CI

Bootstrap used 1000 paired resamples.

| Method A | Method B | Accuracy Delta Mean | Accuracy Delta 95% CI | Macro F1 Delta Mean | Macro F1 Delta 95% CI |
|---|---|---:|---|---:|---|
| `tfidf_knn_majority` | `tfidf_weighted_knn` | 0.011965 | [0.007477, 0.016580] | 0.012575 | [0.008117, 0.017241] |
| `tfidf_weighted_knn` | `tfidf_logistic_regression` | 0.056306 | [0.044538, 0.068596] | 0.056783 | [0.044697, 0.068770] |
| `tfidf_weighted_knn` | `tfidf_linear_svm` | 0.093550 | [0.080949, 0.105007] | 0.095881 | [0.083564, 0.107602] |

The intervals are useful as a stability check for this evaluation, but the result should still be phrased cautiously because the baselines use fixed hyperparameters and one dataset.

## Oracle Analysis

Because GLiNER2 methods were not included in this run, oracle rows whose group names mention GLiNER should not be used from this output.

The only meaningful oracle result from this run is:

| Oracle Group | Methods | Oracle Accuracy | Oracle Correct |
|---|---|---:|---:|
| kNN majority + weighted kNN | `tfidf_knn_majority`, `tfidf_weighted_knn` | 0.802666 | 2469 |

Oracle accuracy is an analysis-only upper bound: an example is counted correct if any method in a group predicts the gold label. It is not a deployable method.

## Per-label And Confusion Analysis

The saved per-label delta table in this run compares `tfidf_weighted_knn` with `tfidf_linear_svm`. The largest F1 improvements for Linear SVM over weighted kNN included:

- `contactless_not_working`: +0.258844
- `cash_withdrawal_charge`: +0.244990
- `cash_withdrawal_not_recognised`: +0.198880
- `reverted_card_payment?`: +0.183955
- `pending_card_payment`: +0.167470
- `virtual_card_not_working`: +0.167357
- `exchange_charge`: +0.155844

Top confusion pairs included:

- `virtual_card_not_working` -> `getting_virtual_card`
- `why_verify_identity` -> `verify_my_identity`
- `disposable_card_limits` -> `get_disposable_virtual_card`
- `contactless_not_working` -> `card_not_working`
- `card_payment_not_recognised` -> `direct_debit_payment_not_recognised`

Focus especially on Banking77 families that are naturally confusable:

- card arrival, card delivery estimate, order physical card, get physical card
- pending top up, top up failed, top up reverted, verify top up
- verify my identity, why verify identity, unable to verify identity
- pending transfer, transfer timing, failed transfer
- exchange via app, exchange rate, fiat currency support

## Discussion

The full run shows that simple train-set-based baselines are strong on Banking77. Similarity-weighted kNN improves over majority-vote kNN, and supervised TF-IDF Linear SVM is the strongest method among the four tested baselines.

Compared with the earlier GLiNER2 full zero-shot result, these TF-IDF methods are much higher numerically:

- GLiNER2 `plain_label` accuracy: 0.684330
- GLiNER2 `mean_confidence_ensemble` accuracy: 0.689532
- `tfidf_weighted_knn` accuracy: 0.800390
- `tfidf_linear_svm` accuracy: 0.894018

This comparison is informative but not equal-setting: the TF-IDF methods use Banking77 train labels, while the main GLiNER2 schema wording experiment does not.

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

Use the classical baseline comparison as an optional backup slide or report appendix:

- retrieval-only `tfidf_weighted_knn`: accuracy 0.800390
- supervised `tfidf_linear_svm`: accuracy 0.894018
- key caution: these methods use train labels and are not zero-shot GLiNER2 methods
