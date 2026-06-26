# Retrieval-aided Candidate Pruning Results Template

## Research Question

Does TF-IDF retrieval over the Banking77 train split improve GLiNER2 intent classification by pruning the candidate label set before inference?

## Method

This method builds a TF-IDF retrieval index over Banking77 train texts only. For each test example, it retrieves similar train examples, ranks their canonical labels by frequency and cosine similarity, keeps up to `MAX_CANDIDATES` labels, and runs GLiNER2 on that pruned candidate set.

## Difference From Previous Schema Wording Ensemble

The previous schema wording ensemble is zero-shot with respect to Banking77 train examples. This retrieval extension is not pure zero-shot because it uses the Banking77 train split as a retrieval memory. It still does not update model parameters.

## Dataset And Split

- Dataset:
- Train examples:
- Test/evaluation examples:
- Labels:
- Mode:

## Retrieval Setup

- Vectorizer:
- N-gram range:
- Min DF:
- Max features:
- K neighbors:
- Expanded K values:
- Min candidates:
- Max candidates:

## Candidate Pruning Setup

Candidate ranking:

1. Frequency among retrieved neighbors
2. Sum of cosine similarities
3. Max cosine similarity
4. Label string tie-break

## Main Result Table

| Method | Accuracy | Macro F1 | Weighted F1 | Parse Failure Rate | Avg Latency |
|---|---:|---:|---:|---:|---:|
| plain_label | | | | | |
| mean_confidence_ensemble | | | | | |
| tfidf_knn_baseline | | | | | |
| tfidf_candidate_pruning_gliner2 | | | | | |
| tfidf_candidate_pruning_mean_confidence_ensemble | | | | | |

## Candidate Recall Analysis

- Candidate recall@MAX_CANDIDATES:
- Average number of candidates:
- Median number of candidates:
- Accuracy when gold is in candidates:
- Accuracy when gold is not in candidates:
- Most common expanded K:

## Error Analysis

Include:

- Plain-label wrong but candidate-pruned GLiNER2 correct
- Plain-label correct but candidate-pruned GLiNER2 wrong
- Mean-confidence ensemble wrong but candidate-pruned mean-confidence ensemble correct
- Candidate pruning failures where gold label is missing from candidates
- Candidate pruning failures where gold is present but GLiNER2 selects another label
- Top confusion pairs for each method

## Discussion

Discuss whether pruning improves over:

- `plain_label`
- `mean_confidence_ensemble`

Also discuss whether errors are mainly caused by retrieval recall failures or GLiNER2 selection failures after the gold label is already in the candidate set.

## Limitations

- This is not pure zero-shot.
- Results depend on the Banking77 train split.
- TF-IDF retrieval is lexical and may miss semantic matches.
- Hyperparameters are fixed and should not be tuned on the test split.
- No statistical significance test is included by default.

## Whether To Include This In Final Presentation

Include this extension only if it clearly improves or provides a useful diagnostic story. If it does not improve, present it as an optional analysis showing that retrieval recall or candidate pruning is a bottleneck.
