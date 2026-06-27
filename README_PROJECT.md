# GLiNER2 Schema Wording Smoke Test

This project scaffolds a first-phase Google Colab smoke test for:

**GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface**

The research direction is robust schema wording for zero-shot intent classification, but this phase does not implement paraphrased schemas or any proposed method. It only checks that local GLiNER2 inference and a small Banking77 evaluation can run.

## Files

- `notebooks/01_gliner2_smoke_test_colab.ipynb`: Colab notebook for installation, environment checks, quick inference, dataset loading, smoke evaluation, and report writing.
- `notebooks/02_gliner2_schema_wording_experiment_colab.ipynb`: Main Colab experiment for deterministic schema wording variants and paraphrase ensembles.
- `notebooks/03_gliner2_retrieval_candidate_pruning_colab.ipynb`: Optional retrieval-aided candidate pruning extension.
- `notebooks/05_gliner2_classical_baselines_comparison_colab.ipynb`: Classical and retrieval baseline comparison notebook.
- `notebooks/06_gliner2_zero_shot_contrastive_schema_colab.ipynb`: Zero-shot contrastive schema and hierarchical GLiNER follow-up notebook.
- `requirements-colab.txt`: Minimal Colab dependencies.
- `src/gliner2_project/`: Small helper modules for environment reporting, label mapping, GLiNER2 wrappers, and metrics.
- `tests/`: CPU-only unit tests that do not download GLiNER2 or Banking77.
- `SMOKE_TEST_REPORT.md`: Initial run report template. The notebook also writes an updated report after execution.
- `docs/RESULTS_SMALL.md`: Recorded Phase 2 small-run result summary.
- `docs/RESULTS_FULL.md`: Recorded Phase 2 full-test result summary.
- `docs/RESULTS_CLASSICAL_BASELINES.md`: Template and guidance for the classical baseline comparison phase.
- `docs/RESULTS_ZERO_SHOT_SCHEMA.md`: Template and guidance for the zero-shot contrastive schema phase.
- `docs/FINAL_REPORT_DRAFT.md`: Draft text for the final course report.
- `docs/PRESENTATION_OUTLINE.md`: Five-slide, three-minute presentation outline.
- `AI_USAGE.md`: Disclosure template for AI-assisted scaffolding.

## How to Run in Colab

1. Upload this project folder to Google Drive or clone/copy it into a Colab runtime.
2. Open `notebooks/01_gliner2_smoke_test_colab.ipynb` in Google Colab.
3. Select `Runtime > Change runtime type > T4 GPU` when available.
4. Run the notebook from top to bottom.

If the notebook is opened directly from GitHub and the project files are not present in the Colab runtime, it tries to clone `https://github.com/daidai-su/GLiNER2-demo.git`. For a private repository, Colab may be able to open the notebook UI but the runtime may still be unable to clone the full repository without GitHub credentials; in that case, upload or clone the full repository into the runtime first.

The notebook installs dependencies from `requirements-colab.txt`. It uses local inference with `fastino/gliner2-base-v1`; it does not call the Pioneer / GLiNER2 cloud API and does not require paid API keys.

## Main Experiment

After the smoke test succeeds, open `notebooks/02_gliner2_schema_wording_experiment_colab.ipynb`.

The main experiment compares:

- `raw_label`
- `plain_label`
- `query_about_label`
- `banking_request_label`
- `customer_intent_label`
- `vote_ensemble`
- `mean_confidence_ensemble`
- `confidence_margin_fallback`

The default `MODE = "small"` evaluates a stratified subset with `SMALL_PER_LABEL = 5`, or up to 385 examples for Banking77. Full mode requires `CONFIRM_FULL_RUN = True` so a long run is not started accidentally.

The current recorded small-run result is summarized in `docs/RESULTS_SMALL.md`. In that run, `vote_ensemble` was the best observed method, improving macro F1 over `plain_label` from 0.678945 to 0.692128 on 385 examples.

The current recorded full-test result is summarized in `docs/RESULTS_FULL.md`. In that run, the pilot-selected `vote_ensemble` improved macro F1 over `plain_label` from 0.675726 to 0.678583, while the best observed full-run method was `mean_confidence_ensemble` with macro F1 0.680569.

## Optional Retrieval-aided Candidate Pruning Extension

The optional retrieval extension is in `notebooks/03_gliner2_retrieval_candidate_pruning_colab.ipynb`.

It builds a TF-IDF retrieval index over the Banking77 train split, retrieves similar train examples for each evaluated test example, and prunes the GLiNER2 candidate label set before inference. This is not pure zero-shot because it uses the train split as retrieval memory. It still uses no fine-tuning, no LoRA, no manual annotation, and no paid API.

Suggested run order:

- `MODE = "smoke"` for a quick sanity check
- `MODE = "small"` for a 77-label stratified pilot
- `MODE = "full"` with `CONFIRM_FULL_RUN = True` for the full test split

Outputs are saved under `OUTPUT_DIR / "retrieval_pruning"` so previous schema wording results are not overwritten.

## Classical and Retrieval Baseline Comparison

The baseline comparison notebook is `notebooks/05_gliner2_classical_baselines_comparison_colab.ipynb`.

It compares:

- `plain_label`, if cached from the schema wording notebook
- `mean_confidence_ensemble`, if cached from the schema wording notebook
- `tfidf_knn_majority`
- `tfidf_weighted_knn`
- `tfidf_logistic_regression`
- `tfidf_linear_svm`
- optional cached retrieval-pruning GLiNER2 methods

These methods are not all under the same setting:

- `plain_label` and `mean_confidence_ensemble` are pure zero-shot GLiNER2 methods when loaded from the main schema wording experiment.
- TF-IDF kNN methods use the Banking77 train split as labeled retrieval memory.
- Logistic Regression and Linear SVM are supervised classical baselines trained on Banking77 train labels.
- Retrieval-pruning GLiNER2 methods use train examples to prune candidate labels, so they are retrieval-assisted rather than pure zero-shot.

Suggested run order:

- `MODE = "smoke"` for a quick sanity check
- `MODE = "small"` for a stratified 77-label pilot with `SMALL_PER_LABEL = 5`
- `MODE = "full"` only after setting `CONFIRM_FULL_RUN = True`

Outputs are saved under `OUTPUT_DIR / "classical_baselines_comparison"`:

- `predictions/{method_name}.jsonl`
- `tables/classical_results_summary.csv`
- `tables/method_setting_table.csv`
- `tables/paired_comparisons.csv`
- `tables/bootstrap_accuracy_ci.csv`
- `tables/oracle_analysis.csv`
- `tables/error_overlap_matrix.csv`
- `tables/correct_overlap_matrix.csv`
- `tables/per_label_metrics.csv`
- `tables/per_label_delta_vs_plain.csv`
- `tables/confusion_pairs.csv`
- `figures/*.png`
- `classical_run_manifest.json`

Do not describe the TF-IDF kNN or supervised TF-IDF results as zero-shot improvements. They are comparison baselines showing how strong simple train-set-based methods are on Banking77.

## Zero-shot Contrastive Schema Follow-up

The zero-shot schema follow-up notebook is `notebooks/06_gliner2_zero_shot_contrastive_schema_colab.ipynb`.

It tests schema-only methods:

- `description_all_labels`
- `cluster_second_pass_from_plain`
- `cluster_second_pass_from_mean_confidence`
- `hierarchical_gliner`
- oracle analysis across zero-shot schema variants

This notebook intentionally does not load the Banking77 train split. It uses the Banking77 test split for evaluation and manually/rule-designed label schema descriptions for prediction. It uses no train examples, no train labels, no external LLM, no fine-tuning, no LoRA, no paid API, and no GLiNER2 cloud API.

Suggested run order:

- `MODE = "smoke"` for a quick pipeline check
- `MODE = "small"` for a stratified pilot
- `MODE = "full"` only after setting `CONFIRM_FULL_RUN = True`

Outputs are saved under `OUTPUT_DIR / "zero_shot_contrastive_schema"`:

- `predictions/{method_name}.jsonl`
- `tables/results_summary.csv`
- `tables/second_pass_summary.csv`
- `tables/per_cluster_accuracy.csv`
- `tables/per_label_metrics.csv`
- `tables/per_label_delta_vs_plain.csv`
- `tables/confusion_pairs.csv`
- `tables/oracle_zero_shot_schema_variants.csv`
- `tables/zero_shot_schema_disagreement.csv`
- `figures/*.png`
- `zero_shot_schema_run_manifest.json`

If schema wording is revised after inspecting full-test output, report the result as exploratory rather than confirmatory.

The main schema wording notebook writes these files under `OUTPUT_DIR`:

- `predictions/{method_name}.jsonl`
- `run_manifest.json`
- `schema_variants.json`
- `evaluated_example_ids.json`
- `tables/results_summary.csv`
- `tables/per_label_metrics.csv`
- `tables/schema_disagreement_examples.csv`
- `tables/improved_examples.csv`
- `tables/degraded_examples.csv`
- `tables/confusion_pairs.csv`
- `figures/*.png`

## Notebook Configuration

The main variables are defined in the notebook config cell:

- `MODE = "smoke"` runs at most `SMOKE_N_EXAMPLES`.
- `MODE = "small_eval"` evaluates the stratified subset with `SMALL_EVAL_PER_LABEL` examples per label.
- `SMOKE_N_EXAMPLES` controls the maximum number of smoke examples.
- `SMALL_EVAL_PER_LABEL` controls the per-label sample size before the smoke cap.
- `OUTPUT_DIR` controls where predictions and reports are written in Colab.

If a GPU is unavailable, the notebook warns clearly and caps smoke evaluation to 5 examples. On a Colab GPU, smoke mode evaluates up to the configured `SMOKE_N_EXAMPLES` value.

For the main experiment notebook, also review:

- `MODE = "smoke" | "small" | "full"`
- `SMALL_PER_LABEL`
- `MARGIN_THRESHOLD`
- `FORCE_RERUN`
- `CONFIRM_FULL_RUN`

## Scope

The project performs no fine-tuning, no manual annotation, and no external paid API calls. The smoke notebook is only a pipeline check. The main experiment notebook is the first course-project experiment and should be interpreted together with its saved manifest, tables, and error analysis.
