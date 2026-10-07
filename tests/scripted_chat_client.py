"""A scripted Agent Framework chat client for tests: replays tool calls, then a final text."""

from __future__ import annotations

import json
import uuid

from agent_framework import BaseChatClient, ChatResponse, Content, FunctionInvocationLayer, Message


class ScriptedChatClient(FunctionInvocationLayer, BaseChatClient):
    """Each script item is {"tool": name, "arguments": {...}} or {"text": "final answer"}."""

    def __init__(self, script: list[dict]):
        super().__init__()
        self.script = list(script)
        self.requests: list[list[Message]] = []

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.requests.append(list(messages))
        item = self.script.pop(0) if self.script else {"text": "Done."}
        if "tool" in item:
            content = Content.from_function_call(
                call_id=uuid.uuid4().hex[:8], name=item["tool"], arguments=json.dumps(item.get("arguments", {}))
            )
        else:
            content = Content.from_text(item["text"])
        return ChatResponse(messages=Message("assistant", [content]))
