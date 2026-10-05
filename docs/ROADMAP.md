# Roadmap

The project grows from intelligent EDA into a goal-driven **AI Data Scientist** platform: the user says what they want to achieve and the platform formulates the problem, builds and evaluates models, and explains the result. LLMs orchestrate; tested Python pipelines compute.

Legend: ✅ done · 🔜 next · 📋 planned

## Done

| Phase | Delivered |
|---|---|
| ✅ 0 | Generic schema inference: column roles, kinds, primary time axis, no hardcoded names |
| ✅ 1 | Dash dashboard: overview and data quality, recommended charts, ranked insights, chart builder, Ask box, pin board |
| ✅ 1.5 | Ask box v2 (lists, dates, chart requests, fuzzy IDs/columns, run-together words, friendly refusals) and multi-sheet workbooks (key links, lookups, question routing) |
| ✅ 2 | Target-aware EDA: target detection, segment rates with significance, feature signal, leakage and future-date checks, panel detection, time-split advice, drift, Markdown report |

## Model builder (Phase 3)

| Phase | Deliverable | Done when |
|---|---|---|
| 🔜 3a | **Build model** tab: goal picker (predict yes/no, predict a number, rank), problem formulation from Phase 2, generic feature pipelines, baseline + candidate models (linear, random forest, gradient boosting), time-aware validation, metrics in plain words, explanations, scoring, local model registry | Doctor next-month-order model beats the baseline on the time-split test period; works unchanged on another dataset |
| 📋 3b | **Recommendation:** generic user/item/interaction roles, next-best-visit ranking with point-in-time features, configurable rules, daily plan per MR and territory, backtest | Weekly plan respects capacity and visit gaps; backtest shows lift over historical visits and the rule-based script |
| 📋 3c | **Forecasting:** series detection, trend/seasonality, baselines vs ETS/ARIMA/lag-feature models, walk-forward backtest, intervals | Territory forecast beats seasonal-naive |
| 📋 3d | **Clustering, segmentation, anomaly detection, correlation:** auto-k clustering with profiles, Isolation Forest with reasons, correlation matrix and redundant features | Planted clusters/outliers recovered in tests |

## Analysis and service (Phases 4–5)

| Phase | Deliverable |
|---|---|
| 📋 4a | "Why" tools: period comparison, change decomposition with drill-down, per-entity anomalies, attention ranking, Investigate tab |
| 📋 4b | KPI definitions layer (one config file per domain; the only place domain terms live) |
| 📋 5 | FastAPI service exposing every capability as typed endpoints; Docker / docker-compose |

## AI layers (Phases 6–8)

| Phase | Deliverable |
|---|---|
| 📋 6 | LLM layer with one interface for local (Ollama) and Azure OpenAI models; structured output; LLM fallback for the Ask box; dataset and model narratives; evaluation sets |
| 📋 7 | RAG over synthetic SOP/policy/product documents and project documents; hybrid search; answers with citations; retrieval and groundedness metrics |
| 📋 8 | Agentic AI: hand-written tool-calling loop → **free-text AutoML orchestrator** (goal → plan → clarifying questions → confirmation → pipelines → explanation), analyst agent, MCP server, small multi-agent setup, agent evaluation |

## Cloud, integration and governance (Phases 9–12)

| Phase | Deliverable |
|---|---|
| 📋 9 | Azure via Bicep deployment stacks: one command creates, one deletes (also on app exit); Azure OpenAI, Azure AI Search (Free), Container Apps, Key Vault, managed identity, Application Insights, budget alert |
| 📋 10 | Pharma SFA integration (ASP.NET Core + Angular call the Python service) and a .NET Manager Agent (Semantic Kernel / Agent Framework); morning briefing |
| 📋 11 | Evaluation gates in CI (GitHub Actions), tracing, Entra ID and role-based data scope, Responsible AI notes, drift monitoring |
| 📋 12 | Documentation, demo scripts and results summary |

## Principles that apply to every phase

- Generic: no dataset-specific names in code.
- Every number from deterministic code; AI layers explain and orchestrate.
- Guardrails first: leakage checks, honest validation, baselines.
- Each phase ships tested and demoable.
