# AI Usage

An AI coding assistant was used to generate project code, notebooks, tests, and documentation scaffolding for this GLiNER2 schema wording robustness project.

## Generated or Assisted Files

- `notebooks/01_gliner2_smoke_test_colab.ipynb`
- `notebooks/02_gliner2_schema_wording_experiment_colab.ipynb`
- `src/gliner2_project/`
- `tests/`
- `docs/METHOD.md`
- `docs/RESULTS_TEMPLATE.md`
- `README_PROJECT.md`
- `SMOKE_TEST_REPORT.md`

## Prompt Summary

The assistant was asked to build a graduate NLP programming project around GLiNER2 schema wording robustness on Banking77. The implemented experiment uses deterministic schema wording variants and zero-shot local GLiNER2 inference, with no fine-tuning, no paid APIs, and no manual annotations.

An additional optional retrieval-aided candidate pruning phase was scaffolded. This phase uses Banking77 train examples as a TF-IDF retrieval memory to prune candidate labels at inference time. It is not pure zero-shot, but it still uses no model training, no LoRA, no GLiNER2 cloud API, no paid API, and no manual annotations.

An additional classical baseline comparison phase was scaffolded. This phase compares cached GLiNER2 zero-shot outputs with TF-IDF kNN retrieval baselines and supervised TF-IDF Logistic Regression / Linear SVM baselines. The train-set-based methods are included as comparison baselines, not as zero-shot improvements to GLiNER2. The human user must run the notebook and verify any reported metrics.

An additional zero-shot contrastive schema phase was scaffolded. This phase uses manually/rule-designed label descriptions, predefined confusion clusters, cluster second passes, and hierarchical coarse-to-fine GLiNER inference. It does not use Banking77 train examples or train labels, does not fine-tune GLiNER2, does not use LoRA, does not call paid APIs, and does not use an external LLM to generate labels or predictions.

An additional zero-shot short-anchor schema phase was scaffolded. This phase uses manually/rule-designed short natural-language anchors for canonical Banking77 labels. It does not use Banking77 train examples or train labels, external LLM calls, GLiNER2 fine-tuning, LoRA, paid APIs, or GLiNER2 cloud APIs.

## Human Verification Steps

- Colab smoke-test execution by human user
- Colab small-evaluation execution by human user
- Review of generated metrics, saved predictions, and error examples
- Manual confirmation that no GPU results are claimed unless actually run
- Retrieval extension Colab execution by human user, if used
- Verification that retrieval results are not reported until actually run
- Classical baseline comparison Colab execution by human user, if used
- Verification that classical baseline results are not reported until actually run
- Confirmation that no external paid API or external LLM was used for the baseline comparison
- Zero-shot contrastive schema Colab execution by human user, if used
- Verification that zero-shot schema results are not reported until actually run
- Confirmation that any full-test schema iteration is reported as exploratory if schema wording was revised after seeing full-test output
- Short-anchor schema Colab execution by human user, if used
- Verification that short-anchor schema results are not reported until actually run
- Confirmation that full run is not recommended unless the small run beats `query_about_label` by the predeclared threshold

## Human-Completed Fields

- AI assistant model/name:
- Date:
- Human reviewer:
- Colab execution date:
- Manual edits after generation:
- Notes on verification:
