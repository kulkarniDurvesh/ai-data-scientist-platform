# Evaluation report (rules only)

| Set | Wording | Passed | Accuracy | Mean seconds |
|---|---|---|---|---|
| goals | plain | 18/18 | 100% | 0.1 |
| goals | needs a model | 1/5 | 20% | 0.0 |
| questions | plain | 9/9 | 100% | 0.1 |
| questions | needs a model | 0/3 | 0% | 0.0 |

## Failures

- **goals / retail-synonym-forecast** (rules): group: expected 'Region', got None
- **goals / retail-synonym-why** (rules): task: expected 'why', got 'ask'; measure: expected 'Profit', got None
- **goals / retail-synonym-classify** (rules): task: expected ['classify', 'rank'], got None; target: expected 'Returned', got None; plan not ready: Your goal could mean several things. Which one?
- **goals / sales-synonym-plan** (rules): task: expected 'recommend', got None; user: expected 'RepCode', got None; item: expected 'AccountKey', got None; plan not ready: What would you like to do with this data?
- **questions / synonym-sum-filter** (rules): no answer: Unable to determine query intent from: 'how much did we make in the south'
- **questions / synonym-average-filter** (rules): no answer: Unable to determine query intent from: 'what is the typical basket value for northern buyers'
- **questions / synonym-count** (rules): no answer: Operation 'COUNT' requires a dataset column.
