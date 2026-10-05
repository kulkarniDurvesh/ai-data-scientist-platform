# Target EDA report: NextMonthOrder

Dataset: Pharma_AI_SFA_ML_Training_Dataset.xlsx (ML_TrainingData)

## Summary

- Rows: 10,500 (10,500 with a target)
- Positive class: 1
- Positive rate: 7.52%
- Panel data: 500 × Doctor Id over 21 periods of Month

## Warnings and checks

- **WARNING: Class imbalance.** Only 7.5% of labelled rows are in the minority class. Use a stratified split, class weights or resampling, and judge models by recall, precision and PR-AUC rather than accuracy (always predicting the majority class would already score 92.5%).

## Train / test split

Split by time: train on Month up to 2025-07-01, test on the later periods. Each Doctor Id appears in 21 periods, so a random split would put about 99% of them in both train and test and overstate accuracy. Within the training periods, keep the class mix with a stratified validation split.

| Part | From | To | Rows | Target |
| --- | --- | --- | --- | --- |
| Train | 2024-03-01 | 2025-07-01 | 8500 | 0.0748 |
| Test | 2025-08-01 | 2025-11-01 | 2000 | 0.077 |


## Segments (strongest first)

### Territory (effect 0.086, p = 4.41e-13)

| Territory | Rows | Positives | Rate | Lift |
| --- | --- | --- | --- | --- |
| Solapur | 1050 | 124 | 0.1181 | 1.57 |
| Mumbai Central | 1050 | 107 | 0.1019 | 1.35 |
| Pune North | 1050 | 106 | 0.101 | 1.34 |
| Aurangabad | 1050 | 89 | 0.0848 | 1.13 |
| Nashik | 1050 | 64 | 0.061 | 0.81 |
| Nagpur | 1050 | 62 | 0.059 | 0.78 |
| Kolhapur | 1050 | 62 | 0.059 | 0.78 |
| Ahmednagar | 1050 | 60 | 0.0571 | 0.76 |
| Pune South | 1050 | 58 | 0.0552 | 0.73 |
| Satara | 1050 | 58 | 0.0552 | 0.73 |

### Specialty (effect 0.052, p = 0.000171)

| Specialty | Rows | Positives | Rate | Lift |
| --- | --- | --- | --- | --- |
| Orthopedics | 1302 | 125 | 0.096 | 1.28 |
| Neurology | 1302 | 114 | 0.0876 | 1.16 |
| Pulmonology | 1302 | 111 | 0.0853 | 1.13 |
| General Medicine | 1323 | 105 | 0.0794 | 1.06 |
| Dermatology | 1323 | 97 | 0.0733 | 0.97 |
| Diabetology | 1323 | 92 | 0.0695 | 0.92 |
| Pediatrics | 1302 | 76 | 0.0584 | 0.78 |
| Cardiology | 1323 | 70 | 0.0529 | 0.7 |

## Numeric features (strongest first)

| Feature | AUC | Direction | Mean (positive) | Mean (negative) |
| --- | --- | --- | --- | --- |
| OrderRate | 0.894 | higher → more | 0.32 | 0.036 |
| OrdersPlaced | 0.887 | higher → more | 6.891 | 0.798 |
| NotInterested | 0.114 | higher → less | 0.335 | 9.114 |
| InterestRate | 0.831 | higher → more | 0.473 | 0.188 |
| InterestedVisits | 0.801 | higher → more | 10.47 | 4.092 |
| FollowUpRate | 0.25 | higher → less | 0.131 | 0.299 |
| FollowUpRequired | 0.288 | higher → less | 3.016 | 6.447 |
| CompetitorDiscussions | 0.423 | higher → less | 9.252 | 10.915 |
| DaysSinceLastVisit | 0.527 | higher → more | -10.161 | -11.838 |
| UniqueProducts | 0.474 | higher → less | 3.563 | 3.615 |
| VisitsThisMonth | 0.488 | higher → less | 1.642 | 1.671 |
| CompletedVisits | 0.508 | higher → more | 20.713 | 20.452 |
| TotalVisits | 0.507 | higher → more | 21.97 | 21.755 |
| Potential | 0.498 | higher → less | 62900.316 | 60324.717 |
| CancelledVisits | 0.502 | higher → more | 1.257 | 1.304 |

## Not used as features

- DoctorId: ID / key: identifies rows, carries no general signal
- LastVisitDate: date: use for the split, or derive features such as days since
- Month: date: use for the split, or derive features such as days since
