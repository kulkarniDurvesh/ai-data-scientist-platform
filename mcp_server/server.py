"""
MCP server: the platform's tools for any MCP client (Phase 8b).

    python -m mcp_server              # stdio (Claude Desktop, Claude Code, IDEs)
    python -m mcp_server --http 8001  # streamable HTTP at http://127.0.0.1:8001/mcp

The tools are the agents' tools (service/agents.py) plus loading a
dataset, so a client's model gets exactly what the platform's own agents
get: deterministic tools whose numbers are computed by the pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from service.agents import build_tools
from service.knowledge import knowledge as default_knowledge
from service.session import DatasetBundle, DatasetStore

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
COMPUTES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
MAX_FILE_BYTES = 200_000_000
SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}

INSTRUCTIONS = """Tools of the AI Data Scientist Platform. Load a local data file with load_dataset, then use its
dataset_id with the other tools. Numbers come from the platform's pipelines: report them as returned.
For a goal (predict, rank, plan, forecast, segment, explain a change), call interpret_goal first; if the plan has
questions, ask the user, then call run_goal with the spec."""


def build_server(store: DatasetStore | None = None, knowledge=None) -> MCPServer:
    store = store or DatasetStore(capacity=8)
    knowledge = knowledge or default_knowledge
    server = MCPServer("ai-data-scientist", instructions=INSTRUCTIONS, version="0.8.0")

    def bundle(dataset_id: str) -> DatasetBundle:
        found = store.get(dataset_id)
        if found is None:
            raise ToolError(f"Dataset '{dataset_id}' is not loaded; call load_dataset first.")
        return found

    def call(dataset_id: str, name: str, **arguments: Any) -> Any:
        tool = build_tools(bundle(dataset_id))[name]
        try:
            return tool.function(**arguments)
        except (ValueError, KeyError, TypeError) as error:
            raise ToolError(str(error)) from error

    @server.tool(annotations=READ_ONLY)
    def load_dataset(path: str, sheet: str | None = None) -> dict:
        """Load a local data file (CSV, TSV, Excel, JSON, Parquet); returns its dataset_id, tables and columns."""

        file = Path(path).expanduser()
        if not file.is_file():
            raise ToolError(f"No file at '{path}'.")
        if file.suffix.lower() not in SUFFIXES:
            raise ToolError(f"Unsupported file type '{file.suffix}'.")
        if file.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("The file is larger than 200 MB.")
        loaded = store.load(file.read_bytes(), file.name, sheet)
        return {
            "dataset_id": loaded.id, "name": loaded.name, "rows": len(loaded.df),
            "loaded table": loaded.sheet, "tables": loaded.sheets,
            "columns": [f"{c} ({loaded.schema.role_of(c)})" for c in loaded.df.columns][:60],
        }

    @server.tool(annotations=READ_ONLY)
    def list_datasets() -> list[dict]:
        """Datasets loaded in this server."""

        return [{"dataset_id": b.id, "name": b.name, "rows": len(b.df), "table": b.sheet} for b in store._items.values()]

    @server.tool(annotations=READ_ONLY)
    def describe_dataset(dataset_id: str) -> dict:
        """Rows, columns with their roles, tables, time span and data quality."""

        return call(dataset_id, "describe_dataset")

    @server.tool(annotations=READ_ONLY)
    def ask_data(dataset_id: str, question: str) -> dict:
        """Answer a question by computing on the data (counts, totals, averages, lists, top N, filters, by group)."""

        return call(dataset_id, "ask_data", question=question)

    @server.tool(annotations=READ_ONLY)
    def ask_documents(question: str) -> dict:
        """Answer from the indexed documents (company policies, platform docs) with numbered sources."""

        answer = knowledge.ask(question)
        return {"covered": answer.covered, "answer": answer.text,
                "sources": [f"[{s.number}] {s.hit.chunk.section} ({s.hit.chunk.source})" for s in answer.sources]}

    @server.tool(annotations=READ_ONLY)
    def search_documents(query: str) -> list[dict]:
        """Find document passages about a topic."""

        return [{"section": h.chunk.section, "source": h.chunk.source, "text": h.chunk.text[:600]}
                for h in knowledge.search(query, 4)]

    @server.tool(annotations=READ_ONLY)
    def suggest_goals(dataset_id: str) -> list[dict]:
        """Goals this dataset supports, each with a ready-to-run spec."""

        return call(dataset_id, "suggest_goals")

    @server.tool(annotations=READ_ONLY)
    def interpret_goal(dataset_id: str, text: str) -> dict:
        """Turn a goal in plain words into a checked plan: summary, spec, choices made, questions to ask the user."""

        return call(dataset_id, "interpret_goal", text=text)

    @server.tool(annotations=COMPUTES)
    def run_goal(dataset_id: str, spec: dict) -> dict:
        """Run a ready plan's spec (model, plan, forecast, segments or explanation) and return the result summary."""

        return call(dataset_id, "run_goal", spec=spec)

    @server.tool(annotations=READ_ONLY)
    def summarize_model(dataset_id: str) -> dict:
        """Plain-language summary and facts of the last model trained on the dataset."""

        return call(dataset_id, "summarize_model")

    return server


server = build_server()
