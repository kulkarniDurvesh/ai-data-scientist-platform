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
- [Recommend tab](#recommend-tab)
- [Forecast tab](#forecast-tab)
- [Segments tab](#segments-tab)
- [Investigate tab](#investigate-tab)
- [KPIs tab](#kpis-tab)
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

**Deep links:** add `?tab=overview`, `auto`, `builder`, `ask`, `target`, `model`, `recommend`, `forecast`, `segments`, `why`, `kpi` or `board` to the URL to open a tab directly.

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

## Recommend tab

Plans who should contact what, and when, from an interaction history (e.g. MR visits to doctors).

1. **Interaction table** — sheets that look like interactions (two repeated keys, a date and an outcome) are listed; the largest is chosen, even if another sheet is selected in the header.
2. **Roles** (all editable):
   - *Who acts (user)* and *Contacted (item)* — the two repeated keys; the one with more distinct values is proposed as the item.
   - *When* — the event date (filled on almost every row, not a monthly snapshot or a follow-up date).
   - *Outcome* and *Counts as success* — values with positive words are pre-selected; values with negations (*Not Interested*, *Cancelled*) are not.
   - *Plan within* — a group both sides share, e.g. *Territory (user's and item's)* from the MRs and Doctors sheets, or a column that is constant per user and per item. *No grouping* plans across everything.
3. **Settings**
   - *Plan for* — next 5 or 10 working days, **next calendar month** (every working day of the month after the last recorded contact) or a custom number of days. *Start from* (optional) moves the plan: with *next calendar month* it plans the whole month of that date.
   - *Max contacts per user per day* and *Min days between contacts* — leave blank to use typical values from the history.
   - *Keep each item with one user* (on by default) — every item (e.g. doctor) is assigned to one user of its group (e.g. one MR of the territory): the one who contacted it most, balanced so each user gets a similar share. Off: items are dealt out across all users of the group each day.
4. **Build plan** — runs in the background (a few seconds).

### Results

| Part | Meaning |
|---|---|
| Tiles | Success model and its test scores, backtest lift and success rate, contacts planned, the rules applied |
| Backtest | In each held-out week, the top 20% of that week's real contacts by each strategy, and how often they actually succeeded. *Random* = no prioritisation. Only contacts that happened can be judged, so this measures prioritisation, not the full plan |
| What drives the model | Permutation importance of the history features and item attributes |
| Plan by group / by user | Contacts, distinct items and average predicted success per group and per user (days with contacts per user) |
| Day-wise plan per user | Choose a user (e.g. an MR) to see every working day of the period: slot, item and name, predicted success, days since last contact, reasons |
| Download | The full plan for every user as CSV (filter by user in Excel) |
| Note on filled slots | When fewer contacts are planned than capacity allows, the number of items per user is the limit: each item can be contacted only about once per minimum gap. Contacts are spread evenly over the days instead of crowding the first ones |

### How the plan is built
Items are first given an owner (when *Keep each item with one user* is on). A daily limit per user spreads the contacts the period allows evenly over its days (never above the maximum per day). Then, for each working day, every item is scored as of that day; items contacted (or already planned) within the minimum gap are skipped; each user gets their best eligible items up to the daily limit. Users only get items of their own group.

## Forecast tab

Forecasts a measure over time, in total and per group.

1. **Table** — any sheet with a date column and at least 20 rows (the selected sheet first).
2. **Date** — the column that places each row in time.
3. **Measure** — a numeric column, or *Number of rows* to forecast counts (e.g. visits). Measures recorded on each row come first; columns fixed per item (looked up from another sheet, or constant per key) are listed last.
4. **Combine values by** — sum, average or distinct count (averages are proposed for 0–1 ratios).
5. **Split by** — optional category (up to 30 groups); each group gets its own series and forecast, plus the total.
6. **Period** and **Periods ahead** — proposed from the data (snapshot tables keep their period; 18+ months of events → months, 4+ months → weeks, else days).

Press **Forecast** (a few seconds).

### Results

| Part | Meaning |
|---|---|
| Chosen model | Best non-baseline model on the rolling backtest |
| MASE | Error relative to repeating last season's values; **below 1 beats seasonal naive** |
| sMAPE | Average percentage error (0% perfect); large when values are small or volatile |
| Forecast chart | History (grey), forecast (blue dotted) and shaded 80% / 95% intervals; switch the series to see each group |
| Model comparison | Every model's MAE, sMAPE and MASE averaged over cut-off points and series; baselines marked |
| Trend / seasonality | Strength 0–1 from an STL decomposition, shown only with at least three full seasons (e.g. 36 months) |
| Forecast table | Every series and period with intervals; download as CSV |
| Notes | E.g. an incomplete last period that was left out, or no model beating the baselines |

### Good to know
- The backtest never uses the future: each cut-off fits only on earlier periods.
- Intervals are approximate: they come from the chosen model's backtest errors and widen with the horizon.
- A horizon longer than half the history is refused.

## Segments tab

Groups similar units, flags unusual ones and shows how features relate.

1. **Table** and **Segment** — *Each row*, or *Each <key>* to build one unit per value of a repeated key (e.g. each doctor from doctor × month rows: numbers averaged, categories by most common value, plus the number of rows when it varies). Panel data defaults to one unit per key.
2. **Method**
   - *K-means, best number of segments* — tries 2–8 and keeps the best silhouette.
   - *K-means, fixed number of segments* — uses the number you set.
   - *Low / medium / high bands of one measure* — thirds of the chosen measure; simple and easy to explain.
3. **Features** — measures, 0/1 columns and categories with up to 15 levels are proposed; remove any you don't want to segment on (e.g. the outcome you plan to predict).
4. **Flag as unusual (%)** — share of units to flag (default 1%).

### Results

| Part | Meaning |
|---|---|
| Segment profiles | Name, size, share and *what sets it apart*: features at least 0.5 standard deviations from the average, and categories over-represented by 20+ points |
| Segment map | Units projected to two dimensions (PCA), coloured by segment |
| Profile heatmap | Each segment's difference from the average per feature (blue above, red below) |
| How many segments? | Silhouette for 2–8 segments (−1 to 1; higher = better separated); low values mean overlapping, broad segments |
| Unusual units | Highest Isolation Forest scores with reasons: features far from typical (robust z ≥ 3.5, with the typical range) and rare categories |
| Correlations | Spearman matrix; redundant pairs (|ρ| ≥ 0.9) with a keep-one suggestion; VIF (above 10 = largely explained by other columns; ∞ = an exact combination, e.g. a total of parts); Cramér's V between categories |
| Download | Every unit with its segment, anomaly score and reasons |

## Investigate tab

Explains why a number changed between two periods.

1. **Table, Date, Measure, Combine values by, Period** — as in the Forecast tab (number of rows counts records).
2. **Compare with** — the previous period, or the same period one year earlier. The current period is the latest complete one (a partly covered last period of event data is left out; monthly snapshot columns are always complete).
3. **Explain by** — categories and keys with up to 100 values. Columns that map one-to-one onto another (a key and its name or e-mail) appear once; times of day are not offered.
4. **Rank for attention** — an entity (e.g. MR or doctor) to rank; keys that are unique on almost every row are not offered. **Direction**: *higher is better* (a fall is bad) or *lower is better* (a rise is bad).

### Results

| Part | Meaning |
|---|---|
| Tiles | Current and comparison values, the change, and the entity with the highest attention score |
| Explanation | Numbered steps. Each step names the group that accounts for most of the movement and follows it into the next dimension (up to three levels). It stops, and says so, when no group accounts for at least 25% of all movement |
| Over time | The measure per period, with the two compared periods highlighted |
| Waterfall | From the comparison period to the current one, group by group (blue up, red down) |
| By <dimension> | Previous, current, change and share of the net change per group. Shares above 100% mean other groups moved the other way. For averages, **Mix effect** (group weights changed) and **Rate effect** (values within groups changed) add up to the change |
| Attention ranking | Score from the change vs the comparison period, how unusual the current period is for the group, the recent trend and the group's size; *Why* lists the factors. Download as CSV |
| Unusual this period | Groups whose current value is far from their own history (robust z ≥ 2.5) |

You can also type **"why did … change / drop / increase?"** in the Ask tab: the measure and date are taken from your words (or the main date column) and the answer is the explanation chain with the top contributions.

## KPIs tab

Business KPIs defined once in a **domain file** and computed the same way every time.

1. **Domain file** — files in the `domains/` folder that fit the dataset, with how many of their KPIs can be computed. If none fits, **Download a starter file** builds one from the detected columns; edit it and save it in `domains/`.
2. **Period** — week, month or quarter. The current period is the latest one the data fully covers; it is compared with the one before.
3. **Break down by** — any role in the file (e.g. rep, territory, specialty).

### Results

| Part | Meaning |
|---|---|
| Tiles | Each KPI's current value and change vs the previous period (*pts* for rates), with *better* / *worse* from the KPI's direction |
| Over time | The selected KPI per period, with its definition |
| Definitions and values | Current, previous, change, direction, the formula, rows used and description; download as CSV |
| By <role> | Every KPI per group for the current period |
| Not available here | KPIs whose columns or roles are missing for this dataset or breakdown, with the reason |

### Writing a domain file

```yaml
name: Sales field force
roles:
  rep: RepCode                 # same column in every sheet
  region:                      # or one column per sheet
    Contacts: Rep Region
    Accounts: Region
  date:
    Contacts: ContactedOn
  result:
    Contacts: Result
kpis:
  - id: win_rate
    label: Win rate
    table: Contacts
    numerator:   {count: rows, where: {result: Won}}
    denominator: {count: rows, where: {result: [Won, Lost]}}
    format: percent            # percent | number | currency
    direction: up              # up = higher is better, down = lower is better
  - id: coverage
    table: Contacts
    numerator:   {distinct: account}
    denominator: {count: rows, table: Accounts, where: {Active: true}}
```

Aggregations: `count` (rows or non-empty values of a column), `sum`, `mean`, `min`, `max`, `distinct`. Filters: a value, a list of values, or `{gt|ge|lt|le|ne|eq: value}`. A KPI without a denominator uses `value:`. Nothing in the file is run as code.

## My board tab

Every chart pinned from any tab, including charts answered from another sheet. Use **Remove** on a card to unpin it.

## Troubleshooting

| Problem | Fix |
|---|---|
| Changes to the code don't show | Restart `python app.py` (the server loads code at start) |
| First question on a workbook is slow | Linking sheets takes a few seconds; it runs in the background right after loading |
| A column has the wrong role | Check the Overview → Columns table; roles come from values and names (e.g. names ending in *Id*, *Code*, *Key*, *No* are keys) |
| An answer used an unexpected date column | Mention the event in the question ("visited", "ordered", "shipped") or add a year |
