# Evaluation report (rules only)

| Set | Wording | Passed | Accuracy | Mean seconds |
|---|---|---|---|---|
| documents | plain | 19/19 | 100% | 0.0 |
| documents | paraphrase | 2/3 | 67% | 0.0 |

Retrieval (plain, embedder lsa): hit@1 71%, hit@3 100%, MRR 0.83

Retrieval (paraphrase, embedder lsa): hit@1 33%, hit@3 67%, MRR 0.44

## Failures

- **documents / paraphrase-selling** (lsa): answering passage at rank >10; top: POL-CMP-04: Sample distribution policy > Limits
