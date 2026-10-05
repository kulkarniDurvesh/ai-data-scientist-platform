# User guide

How to use each part of the dashboard and how to read its output.
Start it with `python app.py` (or `python app.py --file data.xlsx`) and open <http://127.0.0.1:8050>.

- [Loading data](#loading-data)
- [Overview tab](#overview-tab)
- [Auto insights tab](#auto-insights-tab)
- [Chart builder tab](#chart-builder-tab)
- [Ask tab](#ask-tab)
- [Target tab](#target-tab)
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

**Deep links:** add `?tab=overview`, `auto`, `builder`, `ask`, `target` or `board` to the URL to open a tab directly.

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

## My board tab

Every chart pinned from any tab, including charts answered from another sheet. Use **Remove** on a card to unpin it.

## Troubleshooting

| Problem | Fix |
|---|---|
| Changes to the code don't show | Restart `python app.py` (the server loads code at start) |
| First question on a workbook is slow | Linking sheets takes a few seconds; it runs in the background right after loading |
| A column has the wrong role | Check the Overview → Columns table; roles come from values and names (e.g. names ending in *Id*, *Code*, *Key*, *No* are keys) |
| An answer used an unexpected date column | Mention the event in the question ("visited", "ordered", "shipped") or add a year |
