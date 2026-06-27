"""Rule-designed label descriptions for zero-shot GLiNER2 schemas."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .data_utils import normalize_label_text
from .label_clusters import get_cluster_labels, get_cluster_name


DESCRIPTION_ALL_LABELS = "description_all_labels"
CONTRASTIVE_CLUSTER_DESCRIPTIONS = "contrastive_cluster_descriptions"


DESCRIPTION_OVERRIDES: dict[str, str] = {
    "card_arrival": "card arrival: the customer says their ordered card has not arrived yet",
    "card_delivery_estimate": "card delivery estimate: the customer asks when an ordered card should arrive",
    "order_physical_card": "order physical card: the customer wants to order a new physical card",
    "get_physical_card": "get physical card: the customer asks how to obtain a physical card",
    "pending_top_up": "pending top up: the customer asks about a top-up that is still pending",
    "top_up_failed": "top up failed: the customer says an attempted top-up did not succeed",
    "top_up_reverted": "top up reverted: the customer says a top-up was returned or reversed",
    "verify_top_up": "verify top up: the customer needs to verify or authenticate a top-up",
    "cash_withdrawal_not_recognised": "cash withdrawal not recognised: the customer does not recognize a cash withdrawal",
    "verify_my_identity": "verify my identity: the customer needs to complete identity verification",
    "why_verify_identity": "why verify identity: the customer asks why identity verification is required",
    "unable_to_verify_identity": "unable to verify identity: the customer cannot complete identity verification",
    "pending_transfer": "pending transfer: the customer asks about a transfer that is still pending",
    "transfer_timing": "transfer timing: the customer asks how long a transfer should take",
    "failed_transfer": "failed transfer: the customer says a transfer attempt failed",
    "declined_transfer": "declined transfer: the customer says a transfer was declined",
    "exchange_rate": "exchange rate: the customer asks about the rate used for currency exchange",
    "exchange_charge": "exchange charge: the customer asks about fees charged for currency exchange",
    "exchange_via_app": "exchange via app: the customer asks how to exchange money inside the app",
    "fiat_currency_support": "fiat currency support: the customer asks which currencies are supported",
    "declined_cash_withdrawal": "declined cash withdrawal: the customer says an ATM cash withdrawal was declined",
    "cash_withdrawal_charge": "cash withdrawal charge: the customer asks about a fee for taking cash from an ATM",
    "wrong_amount_of_cash_received": "wrong amount of cash received: the customer says the ATM gave an incorrect cash amount",
    "card_swallowed": "card swallowed: the customer says an ATM kept or swallowed their card",
    "atm_support": "atm support: the customer asks which ATMs can be used",
    "pending_cash_withdrawal": "pending cash withdrawal: the customer asks about a cash withdrawal that is still pending",
    "declined_card_payment": "declined card payment: the customer says a card payment was declined",
    "reverted_card_payment?": "reverted card payment: the customer says a card payment was reversed or returned",
    "card_payment_not_recognised": "card payment not recognised: the customer does not recognize a card payment",
    "card_payment_fee_charged": "card payment fee charged: the customer asks about a fee on a card payment",
    "request_refund": "request refund: the customer wants to ask for a refund",
    "Refund_not_showing_up": "refund not showing up: the customer expected a refund but cannot see it",
    "extra_charge_on_statement": "extra charge on statement: the customer sees an unexpected extra charge on their statement",
    "direct_debit_payment_not_recognised": "direct debit payment not recognised: the customer does not recognize a direct debit payment",
}


def label_display(label_text: str) -> str:
    """Return a readable label phrase from a canonical Banking77 label."""

    return normalize_label_text(label_text)


def _generic_description(label_text: str) -> str:
    display = label_display(label_text)
    return f"{display}: the customer asks about {display} in a banking app"


def describe_label(label_text: str) -> str:
    """Return a natural-language description for one canonical label."""

    canonical = str(label_text).strip()
    return DESCRIPTION_OVERRIDES.get(canonical, _generic_description(canonical))


def contrastive_description(label_text: str, cluster_name: str | None = None) -> str:
    """Return a cluster-aware contrastive description for one label."""

    canonical = str(label_text).strip()
    base = describe_label(canonical)
    cluster = cluster_name or get_cluster_name(canonical)
    peers = [
        label_display(label)
        for label in get_cluster_labels(cluster)
        if label != canonical
    ]
    if not peers:
        return base
    not_clause = ", ".join(peers[:6])
    if len(peers) > 6:
        not_clause += ", or other nearby intents"
    return f"{base}. This intent is not about {not_clause}."


def _deduplicate_candidates(
    candidate_to_canonicals: dict[str, list[str]],
) -> tuple[list[str], dict[str, str]]:
    candidates: list[str] = []
    candidate_to_canonical: dict[str, str] = {}
    for candidate, canonicals in candidate_to_canonicals.items():
        if len(canonicals) == 1:
            candidates.append(candidate)
            candidate_to_canonical[candidate] = canonicals[0]
            continue
        for canonical in canonicals:
            safe_candidate = f"{candidate} [canonical: {canonical}]"
            candidates.append(safe_candidate)
            candidate_to_canonical[safe_candidate] = canonical
    return candidates, candidate_to_canonical


def build_description_schema(
    label_texts: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Build description candidates for all labels."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for label in label_texts:
        canonical = str(label).strip()
        if canonical:
            grouped[describe_label(canonical)].append(canonical)
    return _deduplicate_candidates(grouped)


def build_contrastive_description_schema(
    label_texts: Iterable[str],
    cluster_name: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build contrastive description candidates for a label subset."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for label in label_texts:
        canonical = str(label).strip()
        if canonical:
            grouped[contrastive_description(canonical, cluster_name)].append(canonical)
    return _deduplicate_candidates(grouped)


def build_mixed_description_schema(
    label_texts: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Use contrastive descriptions for clustered labels and generic descriptions otherwise."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for label in label_texts:
        canonical = str(label).strip()
        if not canonical:
            continue
        cluster_name = get_cluster_name(canonical)
        candidate = (
            contrastive_description(canonical, cluster_name)
            if cluster_name is not None
            else describe_label(canonical)
        )
        grouped[candidate].append(canonical)
    return _deduplicate_candidates(grouped)
