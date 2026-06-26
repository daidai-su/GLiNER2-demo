# GLiNER2 Smoke Test Report

Status: Phase 1 smoke and small evaluation completed successfully.

This report summarizes the observed Google Colab run for the GLiNER2 local-inference Banking77 baseline. The later schema wording / paraphrase method has not been implemented yet.

## Run Configuration

- Project: `gliner2_schema_wording`
- Model: `fastino/gliner2-base-v1`
- Dataset: `mteb/banking77`
- Mode: `small_eval`
- Small evaluation sampling: `SMALL_EVAL_PER_LABEL = 2`
- Force rerun: `FORCE_RERUN = True`
- Baseline method: `single_schema_plain_label`
- Candidate labels: all 77 Banking77 labels, normalized from `label_text`
- Inference mode: local GLiNER2 `classify_text`
- Cloud API usage: none
- Fine-tuning: none
- Manual annotations: none

## Run Summary

- Installation succeeded: yes
- Model loading succeeded: yes
- Device used: `cuda`
- Dataset loaded: yes
- Number of evaluated examples: 154
- Baseline accuracy: 0.701299
- Baseline macro F1: 0.679489
- Baseline weighted F1: 0.679489
- Correct predictions: 108 / 154
- Average latency: 0.068192 sec/example
- GPU memory: 14.56 GB total, 0.79 GB allocated, 1.66 GB reserved
- Observed API output format: `dict: {'intent': {'label': 'card arrival', 'confidence': 1.0}}`
- Failure points: none
- Safe to continue: yes

Because the small evaluation uses 2 examples per label, the subset is label-balanced. This explains why macro F1 and weighted F1 are identical in this run.

## Error Pattern Summary

The observed mistakes are concentrated in semantically close Banking77 intents rather than obvious parsing or label-mapping failures.

Common confusion groups:

- Top-up intents: `pending_top_up`, `top_up_reverted`, and `topping_up_by_card` were often predicted as `top_up_failed`.
- Exchange and currency intents: `exchange_charge`, `exchange_via_app`, and `fiat_currency_support` were often predicted as `exchange_rate` or another nearby currency-related label.
- Card lifecycle intents: `card_arrival`, `card_delivery_estimate`, `order_physical_card`, and `get_physical_card` showed boundary ambiguity.
- Identity verification intents: `verify_my_identity`, `why_verify_identity`, and `unable_to_verify_identity` were confused with one another.
- Transfer intents: `pending_transfer`, `transfer_timing`, `failed_transfer`, and transfer-related fee labels were close enough to create errors.

Representative mistakes from the run:

| example_id | gold_label | predicted_canonical_label | confidence |
|---:|---|---|---:|
| 663 | `pending_top_up` | `top_up_failed` | 0.911163 |
| 2787 | `exchange_charge` | `exchange_rate` | 0.808899 |
| 1340 | `topping_up_by_card` | `top_up_failed` | 0.998343 |
| 1011 | `top_up_reverted` | `top_up_failed` | 0.989822 |
| 406 | `exchange_via_app` | `exchange_rate` | 0.999998 |
| 2996 | `verify_my_identity` | `why_verify_identity` | 0.999672 |
| 1875 | `pending_transfer` | `transfer_timing` | 1.000000 |

## Interpretation

This run confirms that the notebook, local GLiNER2 model loading, Banking77 data loading, label normalization, prediction parsing, metric calculation, and JSONL output pipeline work end to end in Google Colab.

The baseline result is a useful sanity check, not a final project result. A 154-example balanced subset is still small, but it is large enough to show a meaningful pattern: plain label-only schemas struggle most when Banking77 labels are semantically adjacent. That supports the next research direction of testing richer or paraphrased schema wording.

The confidence values should not be treated as calibrated probabilities. Many incorrect predictions also have very high confidence, so future comparisons should prioritize accuracy, macro F1, per-label performance, and confusion patterns.

## Phase 1 Conclusion

Phase 1 is complete. The project is safe to continue to the next stage: schema wording and paraphrased label-description experiments for zero-shot intent classification robustness.
