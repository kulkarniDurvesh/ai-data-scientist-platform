# HTTP API

The platform's capabilities as typed HTTP endpoints (FastAPI + Pydantic). Interactive documentation is served at `/docs`, the machine-readable schema at `/openapi.json` — generate a typed client from it (e.g. NSwag or Kiota for .NET, `openapi-typescript` for Angular).

- [Run it](#run-it)
- [Concepts](#concepts)
- [Endpoints](#endpoints)
- [Examples](#examples)
- [Security and settings](#security-and-settings)

---

## Run it

```bash
python -m api                     # http://127.0.0.1:8000  (docs: /docs)
python -m api --host 0.0.0.0 --port 8000 --reload
docker compose up --build         # API on :8000 and dashboard on :8050
```

## Concepts

| Concept | Meaning |
|---|---|
| **Dataset** | Upload a file once (`POST /datasets`); every other call uses its `id`. Excel workbooks keep all sheets, linked by shared key columns; endpoints that work on one table take an optional `sheet`. Datasets live in memory (last 8). |
| **Automatic defaults** | Request fields left empty get the same choices the dashboard makes (target, date column, measure, roles, period…). Send a field to override it. |
| **Jobs** | Long work (models, plans, forecasts, segments) starts a background job: the `POST` returns `job_id` and `status_url`; poll `GET /datasets/{id}/jobs/{job_id}` until `status` is `done` (with `result`) or `error`. |
| **Tables** | Tabular results are `{columns, rows, total_rows}`; long tables return the first rows (`rows` query parameter where offered). |
| **Errors** | `400` with a readable `detail` for unusable requests (unknown column, nothing to forecast…), `404` for unknown datasets, jobs, models or domain files, `401` without the API key when one is configured. |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, version, datasets in memory |
| POST | `/datasets` | Upload a file (`multipart/form-data`: `file`, optional `sheet`) → dataset info with column roles |
| GET | `/datasets` | Datasets in memory |
| GET | `/datasets/{id}` | Rows, columns, sheets, primary time, column roles |
| GET | `/datasets/{id}/quality` | Data-quality findings |
| GET | `/datasets/{id}/insights?top=10` | Ranked insights |
| GET | `/datasets/{id}/narrative?kind=dataset\|model&use_model=false` | Plain-language summary from computed facts (with the facts); `use_model=true` asks the language model, whose text is kept only if every number is in the facts |
| POST | `/datasets/{id}/ask` | `{question}` → answer, table, chart spec, the computed query `plan`, `by_model` (also "why did … change?") |
| GET | `/datasets/{id}/target?column=` | Target analysis: rate, findings, split, segments, feature signal, Markdown report |
| POST | `/datasets/{id}/models` | `{goal_type: rank|classify|regress, target?}` → job; result: candidates, metrics vs baseline, importance, lift, scored rows |
| GET | `/datasets/{id}/models/latest?rows=` | The last trained model's result |
| POST | `/datasets/{id}/models/save` | Save the last model to the registry |
| GET | `/models` | Saved models with goal, metrics and data fingerprint |
| POST | `/models/{name}/score` | `{rows: [...]}` → predictions from a saved model |
| GET | `/datasets/{id}/jobs/{job_id}` | Job status, progress messages, result |
| POST | `/datasets/{id}/recommend` | Plan contacts: roles (detected if empty), `period: days|month`, `days`, `start`, `capacity`, `min_gap_days`, `one_owner` → job |
| GET | `/datasets/{id}/recommend/latest?user=` | The last plan, optionally for one user |
| POST | `/datasets/{id}/forecast` | `{sheet?, time?, measure?, aggregation?, group?, freq?, horizon?}` → job |
| POST | `/datasets/{id}/segments` | `{sheet?, unit?, features?, method: kmeans|bands, k?, band_measure?, anomaly_share?}` → job |
| POST | `/datasets/{id}/why` | Explain a change (answered directly): explanation chain, breakdowns, trend, attention, unusual groups |
| GET | `/domains?dataset_id=` | Domain files and how many of their KPIs a dataset supports |
| POST | `/datasets/{id}/kpis` | `{domain}` (file name) or `{domain_yaml}` (inline), `freq`, `group_role` → KPIs, breakdown, trend, unavailable |
| GET | `/datasets/{id}/kpis/starter` | A starter domain file for the dataset |
| GET | `/agents` | Agents and their tools |
| POST | `/datasets/{id}/agents/{agent}` | `{message, conversation_id?, engine: loop\|framework}` → `{job_id, status_url, conversation_id}`; the job result has the `answer`, `status` (done / needs_input / step_limit / error), every `step` (tool, arguments, observation) and a `note` if the answer had unsupported numbers. Needs a language model |
| GET | `/documents?refresh=` | Indexed documents: files, passages, embedder, build time |
| POST | `/documents/search` | `{query, k, mode: hybrid\|keyword\|vector}` → ranked passages with section, source, ranks, similarity |
| POST | `/documents/ask` | `{question, use_model}` → answer with numbered `sources`, `covered`, `method` (extractive / language model), `grounded`, `problems` |
| POST | `/documents` | Upload a .md / .txt document (multipart `file`); the index is rebuilt |
| POST | `/documents/reindex` | Rebuild the index |
| GET | `/llm/status?refresh=` | Which language model is in use (Ollama, Azure OpenAI or none, with the reason) |
| GET | `/datasets/{id}/goals/suggestions` | Goals the dataset supports, each validated and ready to run |
| POST | `/datasets/{id}/goals/interpret` | `{text}` → plan: spec, summary, choices made, warnings, questions, allowed options per field |
| POST | `/datasets/{id}/goals/check` | An edited spec (e.g. answers to the questions) → validated plan |
| POST | `/datasets/{id}/goals/run` | A complete spec → `{tab, job_id, status_url}` (poll the job) or `{tab, answer}` for questions |

## Examples

### curl

```bash
# Upload a workbook
curl -s -F "file=@Pharma_AI_SFA_ML_Training_Dataset.xlsx" http://127.0.0.1:8000/datasets
# -> {"id": "8f1c...", "sheet": "ML_TrainingData", "rows": 10500, ...}

# Ask a question
curl -s -X POST http://127.0.0.1:8000/datasets/8f1c.../ask \
     -H "Content-Type: application/json" \
     -d '{"question": "give list of doctors visited in month of March by MR01"}'

# Next-month day-wise plan, then one MR's schedule
curl -s -X POST http://127.0.0.1:8000/datasets/8f1c.../recommend \
     -H "Content-Type: application/json" \
     -d '{"period": "month", "capacity": 4, "min_gap_days": 10}'
curl -s http://127.0.0.1:8000/datasets/8f1c.../jobs/<job_id>
curl -s "http://127.0.0.1:8000/datasets/8f1c.../recommend/latest?user=MR001"

# KPIs from the pharma domain file, by territory
curl -s -X POST http://127.0.0.1:8000/datasets/8f1c.../kpis \
     -H "Content-Type: application/json" \
     -d '{"domain": "pharma_sfa", "group_role": "territory"}'
```

### C# (.NET, `HttpClient`)

```csharp
using System.Net.Http.Json;

var api = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:8000") };
// api.DefaultRequestHeaders.Add("X-API-Key", "<key>");   // when AIDS_API_KEY is set

using var form = new MultipartFormDataContent();
form.Add(new ByteArrayContent(File.ReadAllBytes("Pharma_AI_SFA_ML_Training_Dataset.xlsx")), "file", "data.xlsx");
var dataset = await (await api.PostAsync("/datasets", form)).Content.ReadFromJsonAsync<JsonElement>();
var id = dataset.GetProperty("id").GetString();

var start = await (await api.PostAsJsonAsync($"/datasets/{id}/recommend",
    new { period = "month", capacity = 4, min_gap_days = 10 })).Content.ReadFromJsonAsync<JsonElement>();
var statusUrl = start.GetProperty("status_url").GetString();

JsonElement status;
do {
    await Task.Delay(1000);
    status = await api.GetFromJsonAsync<JsonElement>(statusUrl);
} while (status.GetProperty("status").GetString() == "running");

var plan = await api.GetFromJsonAsync<JsonElement>($"/datasets/{id}/recommend/latest?user=MR001");
```

For typed models, generate a client from `/openapi.json` instead of using `JsonElement`.

### Goal box: plain words → plan → run

```bash
curl -s -X POST http://127.0.0.1:8000/datasets/8f1c.../goals/interpret \
     -H "Content-Type: application/json" \
     -d '{"text": "plan calls to doctors for each MR next month"}'
# -> {"ready": true, "tab": "recommend", "summary": "Plan which 'DoctorId' each 'MRId' should contact next month ...",
#     "spec": {"task": "recommend", "sheet": "Visits", "user": "MRId", "item": "DoctorId", ...},
#     "assumptions": ["Date column: 'VisitDate' (chosen automatically).", ...], "questions": [], ...}

# Not ready? Answer the questions by editing the spec, check it again, then run it:
curl -s -X POST http://127.0.0.1:8000/datasets/8f1c.../goals/run \
     -H "Content-Type: application/json" -d '<the spec from the plan>'
```

## Security and settings

| Variable | Effect |
|---|---|
| `AIDS_API_KEY` | When set, every request needs the header `X-API-Key: <value>` (otherwise `401`) |
| `AIDS_CORS_ORIGINS` | Comma-separated origins allowed by CORS, e.g. `http://localhost:4200` for an Angular dev server (default `*`) |
| `AIDS_EMBED_PROVIDER`, `AIDS_EMBED_MODEL` | Document vectors: `auto` (default: Ollama embedding model if installed, else local LSA), `ollama`, `lsa`; model default `nomic-embed-text` |
| `AIDS_KNOWLEDGE_PATHS` | Folders / files to index, separated by the OS path separator (default: `knowledge/`, `docs/*.md`, `README.md`) |
| `AIDS_LLM_PROVIDER` | `none` (default, rules only), `ollama`, `azure`, or `auto` (Azure OpenAI if configured, else Ollama if running) |
| `AIDS_LLM_NUM_CTX`, `AIDS_LLM_THINK` | Ollama context window (default 8192 tokens) and reasoning mode (default off: thinking takes minutes on a CPU) |
| `AIDS_LLM_MODEL`, `AIDS_OLLAMA_HOST` | Ollama model tag (default `qwen3.5:4b`) and address (default `http://127.0.0.1:11434`) |
| `AIDS_AZURE_OPENAI_ENDPOINT`, `_DEPLOYMENT`, `_KEY`, `_API_VERSION` | Azure OpenAI settings (API version default `2024-10-21`) |

Planned (Phase 11): Entra ID sign-in and role-based data scope instead of a shared key.
