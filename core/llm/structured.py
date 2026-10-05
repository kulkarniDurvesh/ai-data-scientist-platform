"""
Structured output: ask a model for JSON matching a Pydantic model.

The schema is sent with the request (Ollama `format`, Azure
`response_format`), the reply is parsed and validated, and an invalid
reply is retried once with the validation error, so callers get a typed
object or an LLMError, never half-parsed text.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .provider import LLMError, Provider

T = TypeVar("T", bound=BaseModel)


def parse_json(text: str) -> dict:
    """The JSON object in a reply, tolerating code fences or text around it."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start: end + 1])


def structured(
    provider: Provider,
    system: str,
    user: str,
    model: type[T],
    retries: int = 1,
) -> T:
    schema = model.model_json_schema()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = ""

    for _ in range(retries + 1):
        reply = provider.chat(messages, schema=schema)
        try:
            return model.model_validate(parse_json(reply))
        except (ValueError, ValidationError) as error:
            last_error = str(error).splitlines()[0][:300]
            messages = messages + [
                {"role": "assistant", "content": reply[:2000]},
                {"role": "user", "content": f"That reply was not valid ({last_error}). Reply again with only a JSON object matching the schema."},
            ]

    raise LLMError(f"The model's reply did not match the expected format: {last_error}")
