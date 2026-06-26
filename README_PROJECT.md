# GLiNER2 Schema Wording Smoke Test

This project scaffolds a first-phase Google Colab smoke test for:

**GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface**

The research direction is robust schema wording for zero-shot intent classification, but this phase does not implement paraphrased schemas or any proposed method. It only checks that local GLiNER2 inference and a small Banking77 evaluation can run.

## Files

- `notebooks/01_gliner2_smoke_test_colab.ipynb`: Colab notebook for installation, environment checks, quick inference, dataset loading, smoke evaluation, and report writing.
- `notebooks/02_gliner2_schema_wording_experiment_colab.ipynb`: Main Colab experiment for deterministic schema wording variants and paraphrase ensembles.
- `requirements-colab.txt`: Minimal Colab dependencies.
- `src/gliner2_project/`: Small helper modules for environment reporting, label mapping, GLiNER2 wrappers, and metrics.
- `tests/`: CPU-only unit tests that do not download GLiNER2 or Banking77.
- `SMOKE_TEST_REPORT.md`: Initial run report template. The notebook also writes an updated report after execution.
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

Outputs are written under `OUTPUT_DIR`:

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
