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

## Human Verification Steps

- Colab smoke-test execution by human user
- Colab small-evaluation execution by human user
- Review of generated metrics, saved predictions, and error examples
- Manual confirmation that no GPU results are claimed unless actually run
- Retrieval extension Colab execution by human user, if used
- Verification that retrieval results are not reported until actually run

## Human-Completed Fields

- AI assistant model/name:
- Date:
- Human reviewer:
- Colab execution date:
- Manual edits after generation:
- Notes on verification:
