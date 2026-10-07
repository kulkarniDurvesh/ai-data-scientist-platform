# MCP server

The platform's tools over the [Model Context Protocol](https://modelcontextprotocol.io), so an MCP client's model (Claude Desktop, Claude Code, an IDE assistant) can load a dataset and use the platform: the same deterministic tools as the platform's own agents, with every number computed by its pipelines.

## Run it

```bash
python -m mcp_server              # stdio: started by the MCP client
python -m mcp_server --http 8001  # streamable HTTP at http://127.0.0.1:8001/mcp
```

## Connect a client

**Claude Code** (from the project folder):

```bash
claude mcp add ai-data-scientist -- python -m mcp_server
```

**Claude Desktop** (`claude_desktop_config.json`; use your Python and project paths):

```json
{
  "mcpServers": {
    "ai-data-scientist": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "D:/numpyPython/Intelligent_eda",
      "env": {"AIDS_EMBED_PROVIDER": "auto"}
    }
  }
}
```

Then ask the client, for example: *"Load D:/data/sales.xlsx and tell me which accounts each rep should contact next month."*

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `load_dataset` | `path`, `sheet?` | `dataset_id`, rows, tables, columns with roles |
| `list_datasets` | – | Loaded datasets |
| `describe_dataset` | `dataset_id` | Facts (size, roles, time span, quality, patterns), columns |
| `ask_data` | `dataset_id`, `question` | Computed answer and a small table, or why it can't be answered |
| `ask_documents` | `question` | Answer with numbered sources, or `covered: false` |
| `search_documents` | `query` | Matching passages with section and file |
| `suggest_goals` | `dataset_id` | Goals the data supports, each with a ready spec |
| `interpret_goal` | `dataset_id`, `text` | Checked plan: summary, spec, choices made, questions |
| `run_goal` | `dataset_id`, `spec` | Result summary of the pipeline (model, plan, forecast, segments, explanation) |
| `summarize_model` | `dataset_id` | Summary and facts of the last model |

Every tool except `run_goal` is marked read-only; `run_goal` computes but changes no files. Errors (unknown dataset, missing file, a spec that isn't ready) come back as error results with a readable message.

## Design

- The MCP server wraps the same tool functions as the platform's agents (`service/agents.py`), so MCP clients, the Assistant tab and the HTTP API behave the same.
- Datasets live in the server process (last 8); a client loads them by path, which suits local use. Remote use (HTTP) gets authentication in Phase 11.
- Tools never call a language model: the client's model plans, the platform computes.
