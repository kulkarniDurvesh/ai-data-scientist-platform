# User guide

How to use each part of the dashboard and how to read its output.
Start it with `python app.py` (or `python app.py --file data.xlsx`) and open <http://127.0.0.1:8050>.

- [Loading data](#loading-data)
- [Overview tab](#overview-tab)
- [Auto insights tab](#auto-insights-tab)
- [Chart builder tab](#chart-builder-tab)
- [Ask tab](#ask-tab)
- [Target tab](#target-tab)
- [Build model tab](#build-model-tab)
- [My board tab](#my-board-tab)
- [Troubleshooting](#troubleshooting)

---

## Loading data

| Format | Notes |
|---|---|
| CSV / TXT / TSV | Delimiter is detected (`,` `;` `|` tab) |
| Excel (`.xlsx`, `.xls`) | Every sheet is available; the widest sheet opens first. Switch with the **Sheet** selector in the header |
| JSON, Parquet | Tabular records |

On load, text columns holding numbers or dates are converted automatically. ID-like codes such as `ST001` or `MR001` stay text.

The header shows the file name, rows × columns and the current sheet (e.g. "sheet ML_TrainingData (12 of 12)").

**Deep links:** add `?tab=overview`, `auto`, `builder`, `ask`, `target`, `model` or `board` to the URL to open a tab directly.

## Overview tab

| Section | What it shows |
|---|---|
| **Tiles** | Rows, columns (with role counts), missing cells, duplicate rows, quality issues, the time axis and its range |
| **Data quality** | Findings by severity — *Critical* (fix before analysis), *Warning*, *Info* — e.g. impossible negative values, outliers, skew, imbalance, missing dates |
| **Columns** | How each column was understood: role (Measure, Category, Yes/No, Date/time, ID / key, Free text, Constant), detail, unique values, missing %, examples |
| **Data preview** | First 200 rows |

Roles drive everything else: which charts are recommended, what the chart builder offers, how questions are interpreted and which columns can be a model target.

## Auto insights tab

- **Recommended charts:** chosen and ranked automatically by how much each reveals (group differences, trends, correlations), with near-duplicates removed. Each card explains why it was chosen.
- **Key findings:** patterns detected by the insight engine (correlations, group differences, trends), ranked by an interestingness score, each with evidence and a plain-language explanation.

Every chart has **Pin to board** and **View data** (the numbers behind the chart).

## Chart builder tab

Pick a chart type (or *Auto*), X and Y columns, aggregation, colour/group column, time grain, sort, limit and a filter. Defaults are filled in from the column roles; invalid combinations are explained instead of failing.

## Ask tab

Type a question and press **Ask** (or Enter). Suggested questions under the box are generated from the current dataset and are always answerable.

### What you can ask

| Kind | Examples |
|---|---|
| Counts | `How many doctors are there?` · `How many doctors are there in Cardiology?` |
| Aggregates | `What is the average InterestRate?` · `total Sales by Region` |
| Rankings | `Which are the top 5 specialties by InterestRate?` · `bottom 3 regions by Profit` |
| Which-is-highest | `Which region has the highest Sales?` |
| Lists | `which doctors are present in Pune North territory` · `list customers with Sales more than 500` |
| Dates | `in March` · `March 2025` · `Q1 2025` · `in 2024` |
| Charts | `pie chart of doctors by specialty` · `bar chart of visits by competitor name` · `line chart of total visits by month` · `territory-wise bar chart of doctors` |

### How questions are interpreted

- **Columns and values come from your data.** "doctors" matches the `DoctorId` column; "Pune North" matches a value in `Territory`.
- **The date column is chosen from your words:** "doctors *visited* in March" uses `LastVisitDate`/`VisitDate`; "customers who *ordered*" uses `order_date`. Without a hint, the main time axis is used.
- **No year given?** If the month appears in several years, all are included and the answer says so — add a year to narrow it.
- **Forgiving input:** `MR01` finds `MR001`; `speciality` finds `Specialty`; `doctorsId` and `NorthTerritory` are split into words.
- **A value in several columns** (e.g. a doctor's and an MR's territory) filters the column your question points at ("*doctors* in Pune North").

### Multi-sheet workbooks

- Sheets are linked automatically through shared key columns. Details from linked sheets (names, specialty, territory…) can be used in questions and appear in list answers.
- If the selected sheet can't answer, the best sheet is used automatically: *"Answered from the 'Visits' sheet, with details from the 'Doctors' sheet."*
- Columns with the same name from two linked sheets are labelled with the sheet: `Doctors Territory`, `MRs Territory`.

### Reading the answer

Each answer card shows the question, a sentence answer, notes (year ambiguity, which sheets were used), a table for lists and rankings, and a chart where useful (chart answers keep their numbers under **View data**). Charts can be pinned.

### When it can't answer

| Message | Meaning / what to do |
|---|---|
| *'XY99' isn't in this sheet… It appears in sheet 'Visits'* | The value exists elsewhere; switch sheet with the selector |
| *I couldn't match any column or value* | Name a column or a value as it appears in the data |
| *This sheet has no date column* | Date words can't be applied to this sheet |
| *Tell me what to split the chart by* | Add "by <column>" to a chart request |

## Target tab

Analyses a **training table** against the column a model should predict.

1. **Target column:** a likely target is suggested (two-valued, outcome-like name such as `NextMonthOrder`, `will_churn`, `label`); any two-valued or numeric column can be chosen.
2. **Tiles:** rows with a target, positive rate (or average for numeric targets), balance, and panel structure (e.g. *500 × 21 — Doctor Id × Month periods*).
3. **Warnings and checks:**
   - *Possible leakage* — a column predicts the target almost perfectly on its own (AUC ≥ 0.98, correlation ≥ 0.98 or Cramér's V ≥ 0.9). Check it is known before the outcome; usually drop it.
   - *Copy of the target* — matches the target on almost every row.
   - *Name suggests future information* — e.g. "next", "after", "outcome".
   - *Dates after the snapshot period* — a date column falls after the end of the row's period.
   - *Class imbalance* — the minority class is under 10% (warning) or 2% (critical).
   - *Rows without a target*, *latest period has no positives*, *target rate changes over time*.
4. **Train / test split:** for dated data, a split by time (train on earlier periods, test on later ones). For panel data it states how many entities a random split would leak into both sets.
5. **Target rate by segment:** each category column with its effect size (Cramér's V), p-value, highest and lowest group, and charts for the strongest ones.
6. **Target over time:** watch for drift.
7. **Numeric features:** univariate AUC (0.5 = no signal; further from 0.5 = stronger; *Suspicious* near 0 or 1), direction and class means. For numeric targets: Spearman correlation.
8. **Not used as features:** IDs, dates, constants and free text, each with the reason.
9. **Download report:** the full analysis as Markdown ([example](examples/target_report_NextMonthOrder.md)).

## Build model tab

Builds, compares and explains models for a goal you choose.

1. **What do you want to build?**
   - *Rank / prioritise* — order entities by how likely the outcome is (e.g. which doctors to visit first). Judged by how much of the outcome the top of the list captures.
   - *Predict a yes/no outcome* — a probability per row.
   - *Predict a number* — a value per row.
   Only columns that fit the goal are offered; the most likely target is marked *(suggested)*.
2. **Proposed setup** — task, features, split method, the column that names rows, the models that will be compared, and the columns **left out** with the reason (IDs, dates, possible leakage, too many categories).
3. **Train models** — runs in the background (a few seconds on ~10,000 rows); progress steps are shown.
4. **Results**
   - *Tiles*: chosen model, test PR-AUC vs baseline, ROC-AUC, top-20% capture and lift (or MAE, improvement and R² for numbers).
   - *Model comparison*: every candidate's validation and test scores. The model is **chosen on the validation window**; the test period is used once, for reporting — so a model can win even if another scores slightly higher on test.
   - *Cumulative gains*: share of all positives found as you go down the ranked list, against a random list.
   - *What drives the model*: permutation importance — how much worse the model gets when a column is shuffled.
   - *Lift by group*: the test rows in ten equal groups by predicted probability.
   - *Predictions*: rows that need a prediction (rows without a known target, else the latest period), ranked, with **reasons** — the most important columns where the row is unusual, e.g. *high Order Rate (0.81, top 1%)*. Percentiles are mid-rank, so a value shared by many rows (such as 0) is not shown as extreme.
5. **Download all predictions (CSV)** and **Save model** (stored in `models/<name>/` with its goal, metrics and a data fingerprint; listed under *Saved models*). The last trained model stays visible after a page refresh.

### How to read the metrics

| Metric | Meaning | Good |
|---|---|---|
| PR-AUC | Ranking quality focused on the rare positive class | Well above the baseline (= positive rate) |
| ROC-AUC | Chance a random positive ranks above a random negative | 0.5 = random, 1 = perfect |
| Top 20% capture | Share of all positives found in the top 20% of the list | Far above 20% |
| Lift | How many times the average rate the top of the list achieves | > 1 |
| MAE / RMSE | Average / root-mean-square error for numbers | Lower than the baseline |
| R² | Share of variation explained | Closer to 1 |

## My board tab

Every chart pinned from any tab, including charts answered from another sheet. Use **Remove** on a card to unpin it.

## Troubleshooting

| Problem | Fix |
|---|---|
| Changes to the code don't show | Restart `python app.py` (the server loads code at start) |
| First question on a workbook is slow | Linking sheets takes a few seconds; it runs in the background right after loading |
| A column has the wrong role | Check the Overview → Columns table; roles come from values and names (e.g. names ending in *Id*, *Code*, *Key*, *No* are keys) |
| An answer used an unexpected date column | Mention the event in the question ("visited", "ordered", "shipped") or add a year |
