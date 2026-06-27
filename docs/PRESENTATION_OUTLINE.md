# Presentation Outline

Three-minute presentation, five slides.

## Slide 1: Problem

Title: GLiNER2 Schema Wording Sensitivity

Key points:

- GLiNER2 uses a schema-driven interface.
- In zero-shot intent classification, label wording is part of the model input.
- Question: do predictions change when the same intent is worded differently?

Speaker note:

I tested whether schema wording affects GLiNER2 on Banking77, and whether a lightweight deterministic ensemble can improve robustness.

## Slide 2: Method

Title: Deterministic Schema Variants And Ensemble

Schema variants:

- `plain_label`: `card arrival`
- `query_about_label`: `question about card arrival`
- `banking_request_label`: `banking support request about card arrival`
- `customer_intent_label`: `customer intent: card arrival`

Main selected method:

- `mean_confidence_ensemble`

Secondary method:

- `vote_ensemble`

Speaker note:

No LLM-generated paraphrases were used. The templates are deterministic and map back to the original Banking77 labels.

## Slide 3: Experiment

Title: Banking77 Full Test Evaluation

Setup:

- Dataset: `mteb/banking77`
- Test examples: 3076
- Labels: 77
- Model: `fastino/gliner2-base-v1`
- Inference: zero-shot local GLiNER2
- No fine-tuning
- No human annotation
- No paid API

Speaker note:

Every method was evaluated on the same test examples. The main baseline is `plain_label`.

## Slide 4: Results

Title: Small But Consistent Improvement

Main table:

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| `plain_label` | 0.684330 | 0.675726 |
| `vote_ensemble` | 0.687581 | 0.678583 |
| `mean_confidence_ensemble` | 0.689532 | 0.680569 |

Additional result:

- Schema disagreement rate: 26.46%
- Parse failure rate: 0.0

Speaker note:

The improvement is modest, but the schema disagreement rate shows that wording matters, and the selected ensemble slightly improves over the baseline.

## Slide 5: Discussion

Title: What This Shows And What It Does Not

Safe claims:

- GLiNER2 schema wording changes predictions.
- Full test run had about 26.5% schema disagreement.
- `mean_confidence_ensemble` slightly outperformed `plain_label`.
- The method uses no training or new annotations.

Limitations:

- Not a large improvement.
- No statistical significance test.
- Confidence is not calibrated probability.
- Only Banking77 was tested.
- Not a SOTA claim.

Speaker note:

The main takeaway is robustness and sensitivity analysis, not a large benchmark win.

## Optional Backup: Classical Baselines

Use only if there is time or during Q&A.

Key framing:

- TF-IDF kNN uses the Banking77 train split as labeled retrieval memory.
- Logistic Regression and Linear SVM are supervised TF-IDF baselines.
- These are not zero-shot GLiNER2 improvements.
- They are useful reference points for how strong simple train-set-based methods are on Banking77.

Full-run backup numbers:

| Method | Setting | Accuracy | Macro F1 |
|---|---|---:|---:|
| `tfidf_weighted_knn` | retrieval-only | 0.800390 | 0.799004 |
| `tfidf_linear_svm` | classical supervised | 0.894018 | 0.894057 |

Speaker note:

The main five-slide story should stay focused on schema wording sensitivity. The classical baseline comparison is a separate setting and should not be mixed into the main zero-shot claim.
