from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.label_clusters import (  # noqa: E402
    CONFUSION_CLUSTERS,
    get_cluster_labels,
    get_cluster_name,
    get_coarse_group,
    label_to_cluster_map,
    validate_primary_clusters,
)


def test_cluster_membership_lookup():
    assert get_cluster_name("card_arrival") == "card_lifecycle"
    assert "card_delivery_estimate" in get_cluster_labels("card_lifecycle")


def test_every_label_belongs_to_at_most_one_primary_cluster():
    validate_primary_clusters(CONFUSION_CLUSTERS)
    mapping = label_to_cluster_map()
    assert mapping["pending_transfer"] == "transfer"


def test_duplicate_cluster_membership_raises():
    clusters = {"a": ["x"], "b": ["x"]}
    try:
        validate_primary_clusters(clusters)
    except ValueError as exc:
        assert "x" in str(exc)
    else:
        raise AssertionError("Expected duplicate cluster membership to raise.")


def test_coarse_group_mapping_is_deterministic():
    assert get_coarse_group("verify_my_identity") == "identity_security"
    assert get_coarse_group("top_up_failed") == "top_up"
    assert get_coarse_group("some_unknown_label") == "other"
