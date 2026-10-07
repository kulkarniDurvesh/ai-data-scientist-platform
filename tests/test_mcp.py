"""
Phase 8b: MCP server.

The platform's tools served over MCP, checked in-process and over the
real stdio protocol with the MCP client (as Claude Desktop or Claude Code
would call them). Synthetic retail data only.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from mcp_server.server import build_server
from tests.sample_data import retail_dataframe

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def data_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDS_EMBED_PROVIDER", "lsa")
    path = tmp_path / "retail.csv"
    retail_dataframe(600).to_csv(path, index=False)
    return path


def _json(result):
    return json.loads(result.content[0].text)


def test_tools_in_process(data_file):
    server = build_server()

    async def scenario():
        names = [t.name for t in await server.list_tools()]
        loaded = _json(await server.call_tool("load_dataset", {"path": str(data_file)}))
        ds = loaded["dataset_id"]
        answer = _json(await server.call_tool("ask_data", {"dataset_id": ds, "question": "how many customers are in North region"}))
        plan = _json(await server.call_tool("interpret_goal", {"dataset_id": ds, "text": "forecast sales by region for the next 3 months"}))
        result = _json(await server.call_tool("run_goal", {"dataset_id": ds, "spec": plan["spec"]}))
        return names, loaded, answer, plan, result

    names, loaded, answer, plan, result = asyncio.run(scenario())
    assert {"load_dataset", "ask_data", "ask_documents", "interpret_goal", "run_goal"} <= set(names)
    assert loaded["rows"] == 600 and "Sales (measure)" in loaded["columns"]
    assert answer["answer"].startswith("There are")
    assert plan["ready"] and result["tab"] == "forecast" and "MASE" in result["summary"]


def test_read_only_hints():
    tools = {t.name: t for t in asyncio.run(build_server().list_tools())}
    assert tools["ask_data"].annotations.read_only_hint is True
    assert tools["run_goal"].annotations.read_only_hint is False and tools["run_goal"].annotations.destructive_hint is False


def test_stdio_protocol(data_file):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], cwd=str(ROOT),
                                       env={**os.environ, "AIDS_EMBED_PROVIDER": "lsa"})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                info = await session.initialize()
                tools = await session.list_tools()
                loaded = _json(await session.call_tool("load_dataset", {"path": str(data_file)}))
                total = _json(await session.call_tool("ask_data", {"dataset_id": loaded["dataset_id"], "question": "total sales by region"}))
                missing = await session.call_tool("describe_dataset", {"dataset_id": "nope"})
                return info, tools, total, missing

    info, tools, total, missing = asyncio.run(scenario())
    assert info.server_info.name == "ai-data-scientist"
    assert len(tools.tools) == 10
    assert "Region" in total["table"]
    # Tool errors reach the client as error results with a readable message.
    assert missing.is_error and "call load_dataset first" in missing.content[0].text
