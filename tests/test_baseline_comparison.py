from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliner2_project.baseline_comparison import (  # noqa: E402
    bootstrap_accuracy_ci,
    classical_results_summary,
    method_setting_table,
    oracle_analysis_frame,
    overlap_matrix,
    paired_comparison_frame,
)


def row(example_id, gold, pred, method):
    return {
        "example_id": example_id,
        "text": f"text {example_id}",
        "gold_label": gold,
        "predicted_canonical": pred,
        "method": method,
        "latency_sec": 0.01,
    }


def toy_predictions():
    return {
        "plain_label": [
            row(1, "a", "a", "plain_label"),
            row(2, "b", "x", "plain_label"),
            row(3, "c", "c", "plain_label"),
            row(4, "d", "x", "plain_label"),
        ],
        "tfidf_weighted_knn": [
            row(1, "a", "a", "tfidf_weighted_knn"),
            row(2, "b", "b", "tfidf_weighted_knn"),
            row(3, "c", "x", "tfidf_weighted_knn"),
            row(4, "d", "d", "tfidf_weighted_knn"),
        ],
    }


def test_accuracy_and_macro_f1_summary():
    summary = classical_results_summary(toy_predictions())
    plain = summary[summary["method"] == "plain_label"].iloc[0]
    weighted = summary[summary["method"] == "tfidf_weighted_knn"].iloc[0]
    assert plain["accuracy"] == 0.5
    assert weighted["accuracy"] == 0.75
    assert "macro_f1" in summary.columns


def test_paired_comparison_counts():
    paired = paired_comparison_frame(
        toy_predictions(),
        [("plain_label", "tfidf_weighted_knn")],
    ).iloc[0]
    assert paired["both_correct"] == 1
    assert paired["both_wrong"] == 0
    assert paired["a_correct_b_wrong"] == 1
    assert paired["a_wrong_b_correct"] == 2
    assert paired["net_gain_of_b_over_a"] == 1


def test_oracle_accuracy_calculation():
    oracle = oracle_analysis_frame(
        toy_predictions(),
        {"plain_plus_weighted": ["plain_label", "tfidf_weighted_knn"]},
    ).iloc[0]
    assert oracle["oracle_correct"] == 4
    assert oracle["oracle_accuracy"] == 1.0


def test_bootstrap_function_returns_valid_ci_shape():
    ci = bootstrap_accuracy_ci(
        toy_predictions(),
        [("plain_label", "tfidf_weighted_knn")],
        n_samples=20,
        seed=42,
    )
    assert len(ci) == 1
    assert ci.iloc[0]["accuracy_delta_ci_low"] <= ci.iloc[0]["accuracy_delta_ci_high"]


def test_method_setting_table_has_required_columns():
    settings = method_setting_table(["plain_label", "tfidf_linear_svm"])
    for column in [
        "method",
        "setting",
        "uses_train_texts",
        "uses_train_labels",
        "trains_parameters",
        "uses_gliner2",
    ]:
        assert column in settings.columns
    assert settings.loc[settings["method"] == "plain_label", "setting"].iloc[0] == "pure_zero_shot"


def test_error_overlap_matrix_counts_shared_errors():
    matrix = overlap_matrix(toy_predictions(), correct=False)
    assert matrix.loc["plain_label", "plain_label"] == 2
    assert matrix.loc["plain_label", "tfidf_weighted_knn"] == 0
