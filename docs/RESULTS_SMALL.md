# Small Experiment Results

## Research Question

Does deterministic paraphrase ensembling over schema wordings improve GLiNER2 zero-shot intent classification robustness compared with a single plain-label schema?

## Dataset And Run

- Dataset: `mteb/banking77`
- Split: `test`
- Model: `fastino/gliner2-base-v1`
- Mode: `small`
- Sampling: `SMALL_PER_LABEL = 5`
- Number of evaluated examples: 385
- Number of labels: 77
- Device: `cuda`
- Confidence coverage across ensemble input methods: 1.0
- Margin threshold for `confidence_margin_fallback`: 0.05
- Methods evaluated: `raw_label`, `plain_label`, `query_about_label`, `banking_request_label`, `customer_intent_label`, `vote_ensemble`, `mean_confidence_ensemble`, `confidence_margin_fallback`
- Best observed ensemble for analysis: `vote_ensemble`

## Main Result

The best method in the small run was `vote_ensemble`.

Compared with the primary baseline `plain_label`, `vote_ensemble` improved:

- Accuracy: 0.688312 -> 0.701299
- Macro F1: 0.678945 -> 0.692128
- Weighted F1: 0.678945 -> 0.692128

The absolute macro F1 gain is 0.013183.

| Method | Examples | Accuracy | Macro F1 | Weighted F1 | Avg Latency Sec | Total Runtime Sec | Parse Failure Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vote_ensemble` | 385 | 0.701299 | 0.692128 | 0.692128 | N/A | N/A | 0.0 |
| `query_about_label` | 385 | 0.698701 | 0.685261 | 0.685261 | 0.069299 | 26.680158 | 0.0 |
| `mean_confidence_ensemble` | 385 | 0.693506 | 0.683315 | 0.683315 | N/A | N/A | 0.0 |
| `confidence_margin_fallback` | 385 | 0.690909 | 0.680389 | 0.680389 | N/A | N/A | 0.0 |
| `customer_intent_label` | 385 | 0.685714 | 0.679603 | 0.679603 | 0.087081 | 33.526369 | 0.0 |
| `plain_label` | 385 | 0.688312 | 0.678945 | 0.678945 | 0.044200 | 17.017023 | 0.0 |
| `banking_request_label` | 385 | 0.654545 | 0.671724 | 0.671724 | 0.107756 | 41.486086 | 0.0 |
| `raw_label` | 385 | 0.659740 | 0.648376 | 0.648376 | 0.068219 | 26.264342 | 0.0 |

Ensemble methods do not report direct latency in the current table because their rows are computed from cached single-schema predictions. The underlying model-call latency is represented by the single-schema input methods.

## Schema Sensitivity

The four ensemble input schema variants were:

- `plain_label`
- `query_about_label`
- `banking_request_label`
- `customer_intent_label`

They disagreed on 25.19% of evaluated examples.

| Unique Predicted Labels Across Variants | Example Count |
|---:|---:|
| 1 | 288 |
| 2 | 69 |
| 3 | 22 |
| 4 | 6 |

This supports the core premise that GLiNER2 predictions are sensitive to schema wording. Most examples were stable across wording variants, but a non-trivial minority changed predictions.

## Improvement And Degradation

Relative to `plain_label`, the best ensemble `vote_ensemble` produced:

- Improved examples: 8
- Degraded examples: 3
- Net gain: 5 examples

Improved examples included:

| example_id | Gold Label | plain_label Prediction | vote_ensemble Prediction |
|---:|---|---|---|
| 2465 | `top_up_by_cash_or_cheque` | `supported_cards_and_currencies` | `top_up_by_cash_or_cheque` |
| 509 | `age_limit` | `transfer_into_account` | `age_limit` |
| 14 | `card_arrival` | `card_delivery_estimate` | `card_arrival` |
| 1947 | `card_swallowed` | `atm_support` | `card_swallowed` |
| 484 | `age_limit` | `transfer_into_account` | `age_limit` |
| 2447 | `top_up_by_cash_or_cheque` | `balance_not_updated_after_cheque_or_cash_deposit` | `top_up_by_cash_or_cheque` |
| 851 | `transfer_not_received_by_recipient` | `failed_transfer` | `transfer_not_received_by_recipient` |
| 1212 | `unable_to_verify_identity` | `why_verify_identity` | `unable_to_verify_identity` |

Degraded examples included:

| example_id | Gold Label | plain_label Prediction | vote_ensemble Prediction |
|---:|---|---|---|
| 2554 | `virtual_card_not_working` | `virtual_card_not_working` | `getting_virtual_card` |
| 1421 | `compromised_card` | `compromised_card` | `beneficiary_not_allowed` |
| 1654 | `lost_or_stolen_phone` | `lost_or_stolen_phone` | `terminate_account` |

## Top Plain-Label Confusions

The strongest recurring plain-label confusions were between semantically adjacent intents.

| Gold Label | Predicted Label | Count |
|---|---|---:|
| `card_arrival` | `card_delivery_estimate` | 3 |
| `pending_transfer` | `transfer_timing` | 3 |
| `topping_up_by_card` | `top_up_failed` | 3 |
| `verify_my_identity` | `why_verify_identity` | 3 |
| `get_physical_card` | `pin_blocked` | 3 |
| `receiving_money` | `fiat_currency_support` | 3 |
| `card_payment_fee_charged` | `extra_charge_on_statement` | 3 |
| `order_physical_card` | `get_physical_card` | 3 |
| `balance_not_updated_after_bank_transfer` | `transfer_timing` | 2 |
| `Refund_not_showing_up` | `request_refund` | 2 |
| `top_up_reverted` | `top_up_failed` | 2 |
| `pending_top_up` | `top_up_failed` | 2 |
| `cancel_transfer` | `terminate_account` | 2 |
| `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | 2 |
| `exchange_via_app` | `exchange_rate` | 2 |
| `exchange_charge` | `exchange_rate` | 2 |
| `balance_not_updated_after_bank_transfer` | `failed_transfer` | 2 |
| `card_delivery_estimate` | `transfer_timing` | 2 |
| `age_limit` | `transfer_into_account` | 2 |
| `supported_cards_and_currencies` | `topping_up_by_card` | 2 |
| `get_disposable_virtual_card` | `disposable_card_limits` | 2 |
| `card_swallowed` | `atm_support` | 2 |
| `get_physical_card` | `change_pin` | 2 |
| `top_up_by_bank_transfer_charge` | `receiving_money` | 2 |

## Interpretation

The small experiment supports the project hypothesis in two ways.

First, schema wording changes predictions: 25.19% of examples had disagreement across the four wording variants used by the ensemble.

Second, deterministic ensembling can modestly improve robustness: `vote_ensemble` outperformed the primary `plain_label` baseline on accuracy, macro F1, and weighted F1, with 8 improvements and 3 degradations relative to `plain_label`.

The result is not a large gain, but it is a clean MVP signal because the methods are fixed in advance, use no fine-tuning, use no paid API, and do not rely on human annotations.

## Limitations

- The run uses a stratified subset of 385 examples, not the full test set.
- The best ensemble was identified after observing this run; future reporting should avoid selecting a method post hoc as the proposed method without clearly stating this.
- The confidence values are not calibrated probabilities.
- The deterministic templates are simple and may not capture all useful label descriptions.
- The results may vary with GLiNER2 version, checkpoint, runtime, and dataset sample.

## Recommended Next Step

Run the same fixed comparison on `MODE = "full"` with `CONFIRM_FULL_RUN = True`, then report `vote_ensemble` as the pre-specified main ensemble only if this small-run analysis is explicitly treated as pilot evidence.
