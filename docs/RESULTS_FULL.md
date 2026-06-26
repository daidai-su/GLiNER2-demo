# Full Experiment Results

## Research Question

Does deterministic paraphrase ensembling over schema wordings improve GLiNER2 zero-shot intent classification robustness compared with a single plain-label schema?

## Dataset And Run

- Dataset: `mteb/banking77`
- Split: `test`
- Model: `fastino/gliner2-base-v1`
- Mode: `full`
- Number of evaluated examples: 3076
- Device: `cuda`
- Seed: 42
- Confidence coverage across ensemble input methods: 1.0
- Margin threshold for `confidence_margin_fallback`: 0.05
- Run start: 2026-06-26T08:21:37.328730+00:00
- Run end: 2026-06-26T08:42:22.927966+00:00
- Methods evaluated: `raw_label`, `plain_label`, `query_about_label`, `banking_request_label`, `customer_intent_label`, `vote_ensemble`, `mean_confidence_ensemble`, `confidence_margin_fallback`

## Main Result

The primary baseline is `plain_label`. For final reporting, the main selected method is `mean_confidence_ensemble`. The secondary method is `vote_ensemble`.

Compared with `plain_label`, the main selected `mean_confidence_ensemble` improved:

- Accuracy: 0.684330 -> 0.689532, delta +0.005202
- Macro F1: 0.675726 -> 0.680569, delta +0.004843
- Weighted F1: 0.675849 -> 0.680685, delta +0.004836

Compared with `plain_label`, the secondary `vote_ensemble` also improved:

- Accuracy: 0.684330 -> 0.687581, delta +0.003251
- Macro F1: 0.675726 -> 0.678583, delta +0.002857
- Weighted F1: 0.675849 -> 0.678686, delta +0.002837

| Method | Examples | Accuracy | Macro F1 | Weighted F1 | Avg Latency Sec | Total Runtime Sec | Parse Failure Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mean_confidence_ensemble` | 3076 | 0.689532 | 0.680569 | 0.680685 | N/A | N/A | 0.0 |
| `confidence_margin_fallback` | 3076 | 0.687906 | 0.678603 | 0.678708 | N/A | N/A | 0.0 |
| `vote_ensemble` | 3076 | 0.687581 | 0.678583 | 0.678686 | N/A | N/A | 0.0 |
| `plain_label` | 3076 | 0.684330 | 0.675726 | 0.675849 | 0.048863 | 150.302381 | 0.0 |
| `customer_intent_label` | 3076 | 0.678479 | 0.669596 | 0.669684 | 0.088113 | 271.036207 | 0.0 |
| `raw_label` | 3076 | 0.676203 | 0.669552 | 0.669700 | 0.081747 | 251.455157 | 0.0 |
| `query_about_label` | 3076 | 0.672302 | 0.660827 | 0.660988 | 0.072001 | 221.475663 | 0.0 |
| `banking_request_label` | 3076 | 0.645319 | 0.656876 | 0.657096 | 0.105448 | 324.356938 | 0.0 |

Ensemble methods do not report direct latency in the current table because they aggregate cached single-schema predictions. The model-call cost is reflected by the single-schema methods.

## Schema Sensitivity

The four ensemble input variants were:

- `plain_label`
- `query_about_label`
- `banking_request_label`
- `customer_intent_label`

They disagreed on 26.46% of full-test examples.

| Unique Predicted Labels Across Variants | Example Count |
|---:|---:|
| 1 | 2262 |
| 2 | 568 |
| 3 | 198 |
| 4 | 48 |

This confirms the main premise of the project: GLiNER2 predictions are meaningfully sensitive to schema wording.

## Improvement And Degradation

The saved improvement/degradation artifacts compare `plain_label` with the best observed ensemble for analysis, which was `mean_confidence_ensemble`.

Relative to `plain_label`, the main selected `mean_confidence_ensemble` produced:

- Improved examples: 58
- Degraded examples: 42
- Both correct: 2063
- Both wrong: 913
- Net gain: 16 examples

For the secondary `vote_ensemble`, the pasted compact output did not include the improved/degraded split. From the accuracy delta, the net gain over `plain_label` is approximately 10 examples. The notebook has been updated to save a separate `paired_analysis_vote_ensemble.csv` and `paired_analysis_summary.csv` so future output contains the exact paired counts.

The strongest improved gold-label groups included:

| Gold Label | Improved Count |
|---|---:|
| `top_up_reverted` | 9 |
| `verify_top_up` | 5 |
| `card_swallowed` | 4 |
| `verify_my_identity` | 4 |
| `top_up_by_cash_or_cheque` | 3 |
| `transfer_not_received_by_recipient` | 3 |
| `pending_top_up` | 2 |
| `compromised_card` | 2 |
| `card_arrival` | 2 |
| `pending_transfer` | 2 |
| `age_limit` | 2 |
| `fiat_currency_support` | 2 |

The strongest degraded gold-label groups included:

| Gold Label | Degraded Count |
|---|---:|
| `reverted_card_payment?` | 3 |
| `declined_cash_withdrawal` | 3 |
| `card_linking` | 2 |
| `fiat_currency_support` | 2 |
| `country_support` | 2 |
| `beneficiary_not_allowed` | 2 |
| `top_up_by_card_charge` | 2 |
| `cash_withdrawal_not_recognised` | 2 |
| `visa_or_mastercard` | 2 |
| `disposable_card_limits` | 2 |
| `card_acceptance` | 2 |
| `pin_blocked` | 2 |

## Per-Label Changes

The largest F1 improvements for the best observed ensemble were:

| Label | Support | Baseline F1 | Ensemble F1 | Delta |
|---|---:|---:|---:|---:|
| `top_up_reverted` | 40 | 0.626866 | 0.769231 | +0.142365 |
| `verify_top_up` | 40 | 0.760563 | 0.842105 | +0.081542 |
| `card_swallowed` | 40 | 0.407407 | 0.482759 | +0.075351 |
| `top_up_failed` | 40 | 0.611570 | 0.666667 | +0.055096 |
| `verify_my_identity` | 40 | 0.542373 | 0.597015 | +0.054642 |
| `why_verify_identity` | 40 | 0.504348 | 0.557692 | +0.053344 |
| `transfer_timing` | 40 | 0.586466 | 0.629032 | +0.042566 |
| `pending_transfer` | 39 | 0.514286 | 0.555556 | +0.041270 |
| `age_limit` | 40 | 0.948718 | 0.987342 | +0.038624 |
| `getting_spare_card` | 40 | 0.729730 | 0.767123 | +0.037394 |

The largest F1 degradations were:

| Label | Support | Baseline F1 | Ensemble F1 | Delta |
|---|---:|---:|---:|---:|
| `declined_cash_withdrawal` | 40 | 0.698113 | 0.608696 | -0.089418 |
| `beneficiary_not_allowed` | 40 | 0.431373 | 0.360000 | -0.071373 |
| `reverted_card_payment?` | 40 | 0.507463 | 0.437500 | -0.069963 |
| `exchange_charge` | 40 | 0.769231 | 0.716049 | -0.053181 |
| `cash_withdrawal_not_recognised` | 40 | 0.677419 | 0.633333 | -0.044086 |
| `visa_or_mastercard` | 40 | 0.935065 | 0.894737 | -0.040328 |
| `card_linking` | 40 | 0.738462 | 0.698413 | -0.040049 |
| `card_acceptance` | 40 | 0.758621 | 0.729412 | -0.029209 |
| `exchange_via_app` | 39 | 0.466667 | 0.437500 | -0.029167 |
| `get_disposable_virtual_card` | 40 | 0.789474 | 0.763158 | -0.026316 |

## Top Plain-Label Confusions

The strongest plain-label confusions remained semantically adjacent intent pairs.

| Gold Label | Predicted Label | Count |
|---|---|---:|
| `get_physical_card` | `change_pin` | 28 |
| `verify_my_identity` | `why_verify_identity` | 23 |
| `card_arrival` | `card_delivery_estimate` | 18 |
| `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | 17 |
| `top_up_reverted` | `top_up_failed` | 17 |
| `order_physical_card` | `get_physical_card` | 17 |
| `card_delivery_estimate` | `transfer_timing` | 15 |
| `exchange_via_app` | `fiat_currency_support` | 15 |
| `pending_top_up` | `top_up_failed` | 15 |
| `beneficiary_not_allowed` | `failed_transfer` | 15 |
| `receiving_money` | `fiat_currency_support` | 14 |
| `pending_transfer` | `transfer_timing` | 13 |
| `get_physical_card` | `pin_blocked` | 12 |
| `card_swallowed` | `atm_support` | 12 |
| `pin_blocked` | `change_pin` | 11 |
| `topping_up_by_card` | `transfer_into_account` | 11 |
| `reverted_card_payment?` | `declined_card_payment` | 11 |
| `cancel_transfer` | `request_refund` | 10 |
| `why_verify_identity` | `unable_to_verify_identity` | 10 |
| `unable_to_verify_identity` | `why_verify_identity` | 10 |
| `Refund_not_showing_up` | `request_refund` | 10 |

## Confidence And Fallback

- Confidence coverage: 1.0
- `confidence_margin_fallback` fallback rate: 0.033810
- Confidence margin mean: 0.790801
- Confidence margin median: 0.983169

Plain-label confidence was higher for correct predictions on average, but incorrect predictions were also often highly confident:

| Correct | Count | Mean Confidence | Median Confidence |
|---|---:|---:|---:|
| False | 971 | 0.848758 | 0.988829 |
| True | 2105 | 0.976322 | 1.000000 |

These values should not be treated as calibrated probabilities.

## Interpretation

The full run supports the central claim that deterministic schema wording changes GLiNER2 predictions and that ensembling can provide a modest robustness gain.

The gain is small but consistent: all three ensemble methods beat `plain_label` on macro F1, and the selected `mean_confidence_ensemble` was the strongest method in the full run. The secondary `vote_ensemble` also beat the baseline.

The error profile remains dominated by fine-grained neighboring intents such as identity verification variants, transfer timing versus pending transfer, card delivery versus card arrival, and top-up failure/reversal states.

## Artifact Note

The compact review output labeled some improvement/degradation counts as `vote_ensemble`, but the saved notebook artifacts compare `plain_label` against `best_observed_ensemble_for_analysis`. In this full run that method was `mean_confidence_ensemble`. The notebook has been updated so future `confusion_pairs.csv` files include top confusion pairs for every method rather than only `plain_label` and the best observed ensemble.

## Conclusion

Phase 2 full evaluation is complete. The main result is a modest but positive improvement over the plain-label baseline:

- Main selected method: `mean_confidence_ensemble`, macro F1 +0.004843 over `plain_label`
- Secondary method: `vote_ensemble`, macro F1 +0.002857 over `plain_label`

This is a defensible course-project result because the experiment is zero-shot, deterministic, locally run, and uses no additional annotations or paid APIs.
