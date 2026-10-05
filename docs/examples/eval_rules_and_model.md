# Evaluation report (rules + Ollama (qwen3.5:4b))

| Set | Wording | Passed | Accuracy | Mean seconds |
|---|---|---|---|---|
| goals | plain | 18/18 | 100% | 3.1 |
| goals | needs a model | 5/5 | 100% | 26.6 |
| questions | plain | 9/9 | 100% | 0.0 |
| questions | needs a model | 1/3 | 33% | 24.8 |

## Failures

- **questions / synonym-sum-filter** (rules): no answer: This data can't answer that: The data has no columns that answer this question.
- **questions / synonym-average-filter** (rules): no answer: The language model's reading didn't fit the data: The column 'Customer 0261' is not a column of this table.
