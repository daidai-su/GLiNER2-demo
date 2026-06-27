"""Short anchor schema variants for zero-shot GLiNER2 intent classification."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .data_utils import normalize_label_text


SHORT_ANCHOR = "short_anchor"
QUERY_ABOUT_SHORT_ANCHOR = "query_about_short_anchor"
CUSTOMER_REQUEST_SHORT_ANCHOR = "customer_request_short_anchor"
ISSUE_SHORT_ANCHOR = "issue_short_anchor"
MINIMAL_CONTRASTIVE_ANCHOR = "minimal_contrastive_anchor"
QUERY_ABOUT_MINIMAL_CONTRASTIVE_ANCHOR = "query_about_minimal_contrastive_anchor"

SHORT_ANCHOR_METHODS = [
    SHORT_ANCHOR,
    QUERY_ABOUT_SHORT_ANCHOR,
    CUSTOMER_REQUEST_SHORT_ANCHOR,
    ISSUE_SHORT_ANCHOR,
    MINIMAL_CONTRASTIVE_ANCHOR,
    QUERY_ABOUT_MINIMAL_CONTRASTIVE_ANCHOR,
]


SHORT_ANCHOR_OVERRIDES: dict[str, str] = {
    "card_arrival": "card not arrived",
    "card_delivery_estimate": "card delivery time",
    "order_physical_card": "order a physical card",
    "get_physical_card": "get a physical card",
    "getting_spare_card": "get a spare card",
    "cash_withdrawal_card": "cash withdrawal card",
    "pending_top_up": "top-up is pending",
    "top_up_failed": "top-up failed",
    "top_up_reverted": "top-up was reversed",
    "verify_top_up": "verify a top-up",
    "topping_up_by_card": "add money by card",
    "top_up_by_bank_transfer_charge": "bank transfer top-up fee",
    "top_up_by_card_charge": "card top-up fee",
    "top_up_limits": "top-up limit",
    "automatic_top_up": "automatic top-up",
    "verify_my_identity": "verify my identity",
    "why_verify_identity": "why identity verification is needed",
    "unable_to_verify_identity": "cannot verify identity",
    "pending_transfer": "transfer is pending",
    "transfer_timing": "transfer arrival time",
    "failed_transfer": "transfer failed",
    "declined_transfer": "transfer declined",
    "beneficiary_not_allowed": "recipient not allowed",
    "transfer_not_received_by_recipient": "recipient did not receive transfer",
    "transfer_fee_charged": "transfer fee",
    "transfer_into_account": "transfer money into account",
    "cancel_transfer": "cancel transfer",
    "exchange_via_app": "exchange money in the app",
    "exchange_rate": "currency exchange rate",
    "exchange_charge": "currency exchange fee",
    "fiat_currency_support": "supported fiat currencies",
    "supported_cards_and_currencies": "supported cards and currencies",
    "card_payment_wrong_exchange_rate": "wrong card payment exchange rate",
    "declined_cash_withdrawal": "cash withdrawal declined",
    "cash_withdrawal_charge": "cash withdrawal fee",
    "cash_withdrawal_not_recognised": "unrecognized cash withdrawal",
    "wrong_amount_of_cash_received": "wrong cash amount received",
    "card_swallowed": "card swallowed by ATM",
    "atm_support": "ATM support",
    "pending_cash_withdrawal": "cash withdrawal pending",
    "declined_card_payment": "card payment declined",
    "reverted_card_payment?": "card payment reversed",
    "card_payment_not_recognised": "unrecognized card payment",
    "card_payment_fee_charged": "card payment fee",
    "request_refund": "request a refund",
    "Refund_not_showing_up": "refund not received",
    "extra_charge_on_statement": "unexpected extra charge on statement",
    "direct_debit_payment_not_recognised": "unrecognized direct debit",
}


MINIMAL_CONTRASTIVE_ANCHOR_OVERRIDES: dict[str, str] = {
    **SHORT_ANCHOR_OVERRIDES,
    "card_delivery_estimate": "when card will arrive",
}


def fallback_short_anchor(label_text: str) -> str:
    """Return a deterministic fallback anchor from a canonical label."""

    return normalize_label_text(label_text).replace(" not recognised", " unrecognized")


def short_anchor(label_text: str) -> str:
    """Return a short natural-language anchor for a canonical label."""

    canonical = str(label_text).strip()
    return SHORT_ANCHOR_OVERRIDES.get(canonical, fallback_short_anchor(canonical))


def minimal_contrastive_anchor(label_text: str) -> str:
    """Return a short anchor with cluster-specific contrastive overrides."""

    canonical = str(label_text).strip()
    return MINIMAL_CONTRASTIVE_ANCHOR_OVERRIDES.get(
        canonical,
        short_anchor(canonical),
    )


def transform_short_anchor_label(label_text: str, variant_name: str) -> str:
    """Return the candidate string for one short-anchor schema variant."""

    canonical = str(label_text).strip()
    anchor = short_anchor(canonical)
    contrastive = minimal_contrastive_anchor(canonical)

    if variant_name == SHORT_ANCHOR:
        return anchor
    if variant_name == QUERY_ABOUT_SHORT_ANCHOR:
        return f"question about {anchor}"
    if variant_name == CUSTOMER_REQUEST_SHORT_ANCHOR:
        return f"customer request about {anchor}"
    if variant_name == ISSUE_SHORT_ANCHOR:
        return f"{anchor} issue"
    if variant_name == MINIMAL_CONTRASTIVE_ANCHOR:
        return contrastive
    if variant_name == QUERY_ABOUT_MINIMAL_CONTRASTIVE_ANCHOR:
        return f"question about {contrastive}"

    raise ValueError(f"Unknown short-anchor schema variant: {variant_name}")


def duplicate_candidate_strings(
    label_texts: Iterable[str],
    variant_name: str,
) -> dict[str, list[str]]:
    """Return raw duplicate candidate strings before disambiguation."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for label in label_texts:
        canonical = str(label).strip()
        if canonical:
            grouped[transform_short_anchor_label(canonical, variant_name)].append(canonical)
    return {
        candidate: labels
        for candidate, labels in grouped.items()
        if len(labels) > 1
    }


def _disambiguate_candidate(candidate: str, canonical: str) -> str:
    display = normalize_label_text(canonical)
    return f"{candidate} ({display})"


def build_short_anchor_schema(
    label_texts: Iterable[str],
    variant_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Build candidate strings and candidate-to-canonical mapping."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for label in label_texts:
        canonical = str(label).strip()
        if canonical:
            grouped[transform_short_anchor_label(canonical, variant_name)].append(canonical)

    candidates: list[str] = []
    candidate_to_canonical: dict[str, str] = {}
    for candidate, canonicals in grouped.items():
        if len(canonicals) == 1:
            final_candidate = candidate
            candidates.append(final_candidate)
            candidate_to_canonical[final_candidate] = canonicals[0]
            continue
        for canonical in canonicals:
            final_candidate = _disambiguate_candidate(candidate, canonical)
            candidates.append(final_candidate)
            candidate_to_canonical[final_candidate] = canonical

    counts = Counter(candidates)
    duplicates = [candidate for candidate, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            "Duplicate candidate strings after disambiguation: "
            + ", ".join(sorted(duplicates))
        )
    return candidates, candidate_to_canonical


def build_short_anchor_table(label_texts: Iterable[str]) -> list[dict[str, str]]:
    """Return one row per canonical label with all short-anchor variants."""

    rows: list[dict[str, str]] = []
    for label in label_texts:
        canonical = str(label).strip()
        if not canonical:
            continue
        row = {"canonical_label": canonical}
        row["short_anchor"] = short_anchor(canonical)
        row["minimal_contrastive_anchor"] = minimal_contrastive_anchor(canonical)
        for method in SHORT_ANCHOR_METHODS:
            row[method] = transform_short_anchor_label(canonical, method)
        rows.append(row)
    return rows
