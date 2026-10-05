# CLAUDE.md

Guidance for Claude Code (and any contributor) working in this repository.

## Project

**AI Data Scientist Platform**: upload any dataset; the platform understands it, answers plain-English questions, prepares it for machine learning, and builds models from a goal (predict, rank, recommend, forecast, segment). LLM, RAG and agentic layers will orchestrate and explain on top of deterministic Python pipelines. Roadmap: `docs/ROADMAP.md`. Architecture: `docs/ARCHITECTURE.md`.

## The rule that overrides everything: stay generic

This platform must work on **any** dataset. The pharma SFA workbook is only a test case.

- **Never hardcode** sheet/table names, column names, category values, ID formats, specific dates or domain terms in code. No `if column == "DoctorId"`, no `df["Territory"]`, no `"Order Placed"` in logic.
- **Derive everything from the data**: column roles (`core/schema_inference.py`), values and cardinalities, links between sheets (`core/workbook.py`), name tokens (`name_tokens`, `key_entity_name`, `same_word`), date grains, statistics.
- **Generic word lists are allowed only as hints** (English words such as `id`, `code`, `next`, `success`, `not`), never as domain vocabulary. Every value proposed from such a hint must be shown to the user and be overridable.
- **User-facing text must not contain domain words.** Build nouns from the data (e.g. `key_entity_name("DoctorId")` → "doctor") or use neutral examples ("customer", "account").
- **Domain specifics** belong only in optional configuration files (`domains/*.yaml`, read by `core/kpi/`), never in code, and the platform must work without them.
- Comments and docstrings may cite examples from any dataset; mark them as examples.
- Before finishing any change, search for hardcoded names: `grep -rniE "doctor|territor|mrid|visit|pharma" core ui visualization` should only hit comments and docstrings.

## Engineering principles

- **Compute, don't guess**: every number shown comes from deterministic code. LLMs (future phases) plan, ask and explain; they never produce numbers and their structured output is validated against the real columns.
- **Honest ML**: leakage checks, time-aware splits for dated or panel data, a baseline in every comparison, model selection on validation data only, warnings when nothing beats the baseline.
- **Refuse rather than mislead**: unknown values, impossible requests or missing data produce a clear message, never silent wrong results (e.g. never return every row when a filter isn't understood).
- **Dashboard-only runtime**: no `print` statements; errors are shown in the UI (`_safe`, `ui.message`). Long work (training, planning, forecasting) runs as a background job (`DatasetBundle._run_job`) with progress messages.
- Match the surrounding style: dataclasses for results, small pure functions in `core/`, no analysis logic in `ui/` or `api/` (both go through `service/session.py`), `core/` never imports `ui/`, `api/` or `service/`.
- New capabilities get an API endpoint too (Pydantic models in `api/schemas.py`, JSON view in `api/results.py`) and a test in `tests/test_api.py`.

## Tests

- `python -m pytest -q` must pass.
- Tests use **synthetic data with names unrelated to the pharma workbook** (retail, stores, shipments, sales reps, customers) and **planted signals** (a known leak, group, outlier, trend) that the code must find. Tests never read the pharma workbook.
- New behaviour needs a test; bugs get a regression test.

## Definition of done for a change

1. Tests pass and no `print` statements were added.
2. The hardcoding check above is clean.
3. The feature was exercised on the real dashboard (`python app.py --file <workbook>`), including any download.
4. Docs updated: `README.md` (status, features, example results), `docs/USER_GUIDE.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`; screenshot in `docs/images/` refreshed if the UI changed.

## Layout

```
core/                 domain-agnostic engine
  loader, schema_inference, data_quality, eda_pipeline (+ insight modules)
  date_parts          calendar expressions in questions
  workbook            multi-sheet links and lookups
  target_analysis     target-aware EDA
  NLP/                Ask engine: parser -> planner -> engine -> answer
  modeling/           goal -> features -> split -> candidates -> explain -> score -> registry
  recommend/          roles -> point-in-time history -> success model -> plan -> backtest
  forecast/           series -> models -> rolling backtest -> forecast + intervals
  segment/            units -> correlations -> segments -> anomalies
  why/                period comparison -> drill-down -> unusual groups -> attention
  kpi/                domain-file config, safe KPI evaluator, starter generator
visualization/        chart specs, recommender, engine, renderer, theme
service/              session layer (session.py) shared by the dashboard and the API
ui/                   Dash app: layout and callbacks (app.py), tab content (panels.py)
api/                  FastAPI service: endpoints (main.py), Pydantic schemas, result views
domains/              optional domain files (*.yaml): the ONLY place domain terms may appear
tests/                pytest, synthetic data only
docs/                 user guide, architecture, roadmap, images, examples
```

## Commands

```
pip install -r requirements.txt -r requirements-dev.txt
python app.py --file path/to/data.xlsx      # http://127.0.0.1:8050, ?tab=<name> deep links
python -m api                               # http://127.0.0.1:8000/docs
docker compose up --build                   # API :8000 + dashboard :8050
python -m pytest -q
```
