"""Zero-shot label cluster definitions for Banking77 schema experiments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


CONFUSION_CLUSTERS: dict[str, list[str]] = {
    "card_lifecycle": [
        "card_arrival",
        "card_delivery_estimate",
        "order_physical_card",
        "get_physical_card",
        "getting_spare_card",
    ],
    "top_up": [
        "pending_top_up",
        "top_up_failed",
        "top_up_reverted",
        "verify_top_up",
        "topping_up_by_card",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_limits",
        "automatic_top_up",
    ],
    "identity": [
        "verify_my_identity",
        "why_verify_identity",
        "unable_to_verify_identity",
    ],
    "transfer": [
        "pending_transfer",
        "transfer_timing",
        "failed_transfer",
        "declined_transfer",
        "beneficiary_not_allowed",
        "transfer_not_received_by_recipient",
        "transfer_fee_charged",
        "transfer_into_account",
        "cancel_transfer",
    ],
    "exchange_currency": [
        "exchange_via_app",
        "exchange_rate",
        "exchange_charge",
        "fiat_currency_support",
        "supported_cards_and_currencies",
        "card_payment_wrong_exchange_rate",
    ],
    "cash_atm": [
        "declined_cash_withdrawal",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "wrong_amount_of_cash_received",
        "card_swallowed",
        "atm_support",
        "pending_cash_withdrawal",
    ],
    "card_payment_refund": [
        "declined_card_payment",
        "reverted_card_payment?",
        "card_payment_not_recognised",
        "card_payment_fee_charged",
        "request_refund",
        "Refund_not_showing_up",
        "extra_charge_on_statement",
        "direct_debit_payment_not_recognised",
    ],
}


CLUSTER_TO_COARSE_GROUP = {
    "card_lifecycle": "card",
    "top_up": "top_up",
    "identity": "identity_security",
    "transfer": "transfer",
    "exchange_currency": "exchange_currency",
    "cash_atm": "cash_atm",
    "card_payment_refund": "card_payment_refund",
}


COARSE_GROUP_DESCRIPTIONS: dict[str, str] = {
    "card": "questions about ordering, receiving, activating, replacing, or using a bank card",
    "card_payment_refund": "questions about card payments, refunds, chargebacks, recognised payments, or card payment fees",
    "cash_atm": "questions about ATM support, cash withdrawals, swallowed cards, or cash withdrawal issues",
    "top_up": "questions about adding money, top-up status, top-up failures, top-up limits, or top-up charges",
    "transfer": "questions about bank transfers, transfer timing, failed transfers, beneficiaries, or transfer fees",
    "exchange_currency": "questions about exchange rates, currency exchange, supported currencies, or wrong exchange rates",
    "identity_security": "questions about identity verification, security checks, passcodes, or compromised access",
    "account_personal_details": "questions about account details, personal details, contact details, or account closure",
    "fees_charges": "questions about fees, charges, pricing, or unexpected costs",
    "region_support": "questions about country support, regional availability, or card network support",
    "other": "other banking support requests that do not clearly match the listed groups",
}


def validate_primary_clusters(clusters: dict[str, list[str]] = CONFUSION_CLUSTERS) -> None:
    """Raise if a canonical label appears in more than one primary cluster."""

    counts = Counter(label for labels in clusters.values() for label in labels)
    duplicates = sorted(label for label, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "Labels appear in multiple primary clusters: " + ", ".join(duplicates)
        )


def label_to_cluster_map(
    clusters: dict[str, list[str]] = CONFUSION_CLUSTERS,
) -> dict[str, str]:
    """Return canonical label -> primary confusion cluster."""

    validate_primary_clusters(clusters)
    return {
        label: cluster_name
        for cluster_name, labels in clusters.items()
        for label in labels
    }


def get_cluster_name(label: str | None) -> str | None:
    """Return the primary confusion cluster for a canonical label, if known."""

    if label is None:
        return None
    return label_to_cluster_map().get(str(label))


def get_cluster_labels(cluster_name: str | None) -> list[str]:
    """Return labels in a primary confusion cluster."""

    if cluster_name is None:
        return []
    return list(CONFUSION_CLUSTERS.get(str(cluster_name), []))


def labels_in_clusters() -> list[str]:
    """Return all labels covered by primary confusion clusters."""

    return [label for labels in CONFUSION_CLUSTERS.values() for label in labels]


def _heuristic_coarse_group(label: str) -> str:
    lowered = label.lower()
    if "top_up" in lowered or "topping_up" in lowered:
        return "top_up"
    if "transfer" in lowered or "beneficiary" in lowered:
        return "transfer"
    if "exchange" in lowered or "currency" in lowered:
        return "exchange_currency"
    if "cash_withdrawal" in lowered or "atm" in lowered or "card_swallowed" in lowered:
        return "cash_atm"
    if "payment" in lowered or "refund" in lowered or "direct_debit" in lowered:
        return "card_payment_refund"
    if "verify" in lowered or "identity" in lowered or "passcode" in lowered or "compromised" in lowered:
        return "identity_security"
    if "country" in lowered or "visa" in lowered or "mastercard" in lowered or "support" in lowered:
        return "region_support"
    if "fee" in lowered or "charge" in lowered or "cost" in lowered:
        return "fees_charges"
    if "account" in lowered or "cash_or_cheque" in lowered or "personal" in lowered:
        return "account_personal_details"
    if "card" in lowered:
        return "card"
    return "other"


def get_coarse_group(label: str | None) -> str:
    """Return a deterministic coarse group for a canonical Banking77 label."""

    if label is None:
        return "other"
    canonical = str(label)
    cluster_name = get_cluster_name(canonical)
    if cluster_name is not None:
        return CLUSTER_TO_COARSE_GROUP[cluster_name]
    return _heuristic_coarse_group(canonical)


def labels_for_coarse_group(
    label_texts: Iterable[str],
    group_name: str,
) -> list[str]:
    """Return canonical labels assigned to a coarse group."""

    return [label for label in label_texts if get_coarse_group(label) == group_name]


def group_union_labels(
    label_texts: Iterable[str],
    group_names: Iterable[str],
) -> list[str]:
    """Return labels whose coarse group is in group_names, preserving label_texts order."""

    groups = {str(group) for group in group_names if group}
    return [label for label in label_texts if get_coarse_group(label) in groups]
