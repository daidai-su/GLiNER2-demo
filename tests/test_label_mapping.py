from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.data_utils import (  # noqa: E402
    build_label_mappings,
    map_prediction_to_canonical,
    normalize_label_text,
)


def test_underscore_to_space_normalization():
    assert normalize_label_text("cash_withdrawal") == "cash withdrawal"
    assert normalize_label_text("  Card_Arrival  ") == "card arrival"


def test_display_label_maps_back_to_canonical():
    _, display_to_canonical = build_label_mappings(
        ["card_arrival", "passcode_forgotten"]
    )
    assert (
        map_prediction_to_canonical("Card Arrival", display_to_canonical)
        == "card_arrival"
    )
    assert (
        map_prediction_to_canonical("passcode forgotten", display_to_canonical)
        == "passcode_forgotten"
    )


def test_unknown_prediction_handling():
    _, display_to_canonical = build_label_mappings(["card_arrival"])
    assert map_prediction_to_canonical("not a banking label", display_to_canonical) is None
    assert map_prediction_to_canonical(None, display_to_canonical) is None
