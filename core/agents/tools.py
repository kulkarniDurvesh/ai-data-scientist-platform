"""
Tools an agent can call: a name, a description, typed arguments, a function.

Arguments are a Pydantic model, so the model sees a JSON schema and every
call is validated before it runs; a bad call becomes an observation the
agent can correct, never an exception in the app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

MAX_OBSERVATION_CHARS = 2500


class NoArguments(BaseModel):
    pass


@dataclass
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    function: Callable[..., Any]
    stops: bool = False          # ends the turn (e.g. asking the user a question)

    def schema(self) -> dict:
        schema = self.arguments.model_json_schema()
        return {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}

    def describe(self) -> str:
        properties = self.schema()["properties"]
        args = ", ".join(
            f"{name}: {spec.get('type', 'any')}" + (f" ({spec['description']})" if spec.get("description") else "")
            for name, spec in properties.items()
        )
        return f"- {self.name}({args}): {self.description}"

    def call(self, arguments: dict | None) -> tuple[bool, str]:
        """(ok, observation text) - validation and tool errors are observations too."""

        try:
            parsed = self.arguments.model_validate(arguments or {})
        except ValidationError as error:
            first = error.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())) or "arguments"
            return False, f"Invalid arguments for {self.name}: {where}: {first.get('msg')}"
        try:
            result = self.function(**parsed.model_dump())
        except (ValueError, KeyError, TypeError) as error:
            return False, f"{self.name} failed: {error}"
        return True, render(result)


def render(result: Any) -> str:
    text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
    if len(text) > MAX_OBSERVATION_CHARS:
        text = text[:MAX_OBSERVATION_CHARS] + " ... (truncated)"
    return text
