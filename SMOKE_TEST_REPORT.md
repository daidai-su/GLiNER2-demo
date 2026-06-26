# GLiNER2 Smoke Test Report

Status: not yet run in Google Colab.

This repository contains the notebook and helper code needed to run the smoke test. Metrics below must be filled by running `notebooks/01_gliner2_smoke_test_colab.ipynb` from top to bottom. The notebook writes an updated report to `OUTPUT_DIR/SMOKE_TEST_REPORT.md`.

## Run Summary

- Installation succeeded: not run yet
- Model loading succeeded: not run yet
- Device used: not run yet
- Dataset loaded: not run yet
- Number of evaluated examples: not run yet
- Baseline accuracy: not run yet
- Baseline macro F1: not run yet
- Baseline weighted F1: not run yet
- Average latency: not run yet
- GPU memory: not run yet
- Observed API output format: not run yet
- Failure points: none observed locally because the Colab smoke test has not been run
- Safe to continue: pending successful Colab execution

## Notes

- The target dataset is `mteb/banking77`.
- If `mteb/banking77` fails to load, the notebook stops and reports the failure. It does not silently switch to another dataset.
- The baseline method name is `single_schema_plain_label`.
