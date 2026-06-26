# Results Template

## Research Question

Does deterministic paraphrase ensembling over schema wordings improve GLiNER2 zero-shot intent classification robustness compared with a single plain-label schema?

## Dataset

- Dataset:
- Split:
- Number of labels:
- Evaluation mode:
- Number of evaluated examples:
- Label coverage:

## Model

- Model:
- Device:
- GLiNER2 version:
- Transformers version:
- Datasets version:
- Torch version:

## Baseline

Primary baseline: `plain_label`

Baseline result:

| Method | Accuracy | Macro F1 | Weighted F1 | Parse Failure Rate | Avg Latency |
|---|---:|---:|---:|---:|---:|
| plain_label | | | | | |

## Proposed Method

Compared schema variants:

- `query_about_label`
- `banking_request_label`
- `customer_intent_label`

Compared ensembles:

- `vote_ensemble`
- `mean_confidence_ensemble`
- `confidence_margin_fallback`

## Metrics

Report:

- Accuracy
- Macro F1
- Weighted F1
- Per-label F1
- Average latency per example
- Total runtime
- Parse failure rate

## Main Result Table

| Method | Accuracy | Macro F1 | Weighted F1 | Parse Failure Rate | Avg Latency |
|---|---:|---:|---:|---:|---:|
| raw_label | | | | | |
| plain_label | | | | | |
| query_about_label | | | | | |
| banking_request_label | | | | | |
| customer_intent_label | | | | | |
| vote_ensemble | | | | | |
| mean_confidence_ensemble | | | | | |
| confidence_margin_fallback | | | | | |

## Error Analysis

Schema disagreement rate:

Examples where `plain_label` was wrong but the ensemble was correct:

Examples where `plain_label` was correct but the ensemble was wrong:

Top confused label pairs:

| Gold Label | Predicted Label | Count |
|---|---|---:|
| | | |

Labels with largest F1 improvement:

Labels with largest F1 degradation:

## Discussion

Summarize whether schema wording changed predictions, whether ensemble predictions improved robustness, and which label groups remained difficult.

## Limitations

- No fine-tuning
- No human annotation
- Fixed templates only
- Fixed confidence margin threshold
- One dataset/domain
- Confidence values are not calibrated probabilities

## AI Usage

Describe how the AI coding assistant was used, which files were generated, and what human verification was performed.
