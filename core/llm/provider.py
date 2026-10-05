"""
One interface for language models: local (Ollama), Azure OpenAI, or none.

    provider = get_provider()          # from environment settings, may be None
    text = provider.chat(messages, schema=json_schema)

The platform works without a model: callers must accept `None` and fall
back to rules. Settings (environment variables):

    AIDS_LLM_PROVIDER   none (default) | ollama | azure | auto
    AIDS_LLM_MODEL      Ollama model tag (default qwen3.5:4b)
    AIDS_OLLAMA_HOST    default http://127.0.0.1:11434
    AIDS_LLM_NUM_CTX    Ollama context window in tokens (default 8192)
    AIDS_LLM_THINK      1 = let reasoning models think first (slow on CPU; default off)
    AIDS_AZURE_OPENAI_ENDPOINT, AIDS_AZURE_OPENAI_DEPLOYMENT,
    AIDS_AZURE_OPENAI_KEY, AIDS_AZURE_OPENAI_API_VERSION (default 2024-10-21)

The default is no model: a 4B model on a laptop CPU takes about a minute
per request, so a model is used only when switched on. "auto" uses Azure
OpenAI when its endpoint is set, else Ollama when it is running and has
the model, else no model.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 8192
DEFAULT_AZURE_API_VERSION = "2024-10-21"
TIMEOUT_SECONDS = 120


class LLMError(RuntimeError):
    """The model could not be reached or gave no usable answer."""


class Provider(ABC):
    name: str = "provider"
    model: str = ""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], schema: dict | None = None, temperature: float = 0.0) -> str:
        """The model's reply text; with `schema`, the reply is JSON for that schema."""

    def describe(self) -> str:
        return f"{self.name} ({self.model})" if self.model else self.name


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = TIMEOUT_SECONDS) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"{error.code} from the model service: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise LLMError(f"Model service not reachable: {error}") from error


class OllamaProvider(Provider):
    name = "Ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("AIDS_LLM_MODEL") or DEFAULT_OLLAMA_MODEL
        self.host = (host or os.environ.get("AIDS_OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.num_ctx = int(os.environ.get("AIDS_LLM_NUM_CTX") or DEFAULT_NUM_CTX)
        self.think = os.environ.get("AIDS_LLM_THINK", "0").strip().lower() in ("1", "true", "yes")

    def chat(self, messages, schema=None, temperature=0.0) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # The default 4096-token window can cut a large table catalogue.
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
            # Reasoning models think at length before answering: minutes on a CPU.
            "think": self.think,
        }
        if schema is not None:
            payload["format"] = schema
        try:
            reply = _post_json(f"{self.host}/api/chat", payload)
        except LLMError as error:
            if "think" not in str(error).lower():
                raise
            payload.pop("think")  # models without a thinking switch
            reply = _post_json(f"{self.host}/api/chat", payload)
        content = (reply.get("message") or {}).get("content", "")
        if not content:
            raise LLMError("The model returned an empty reply.")
        return content

    def installed_models(self, timeout: float = 1.5) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return []
        return [item.get("name", "") for item in data.get("models", [])]


class AzureOpenAIProvider(Provider):
    name = "Azure OpenAI"

    def __init__(self, endpoint: str | None = None, deployment: str | None = None, key: str | None = None, api_version: str | None = None):
        self.endpoint = (endpoint or os.environ.get("AIDS_AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        self.model = deployment or os.environ.get("AIDS_AZURE_OPENAI_DEPLOYMENT", "")
        self.key = key or os.environ.get("AIDS_AZURE_OPENAI_KEY", "")
        self.api_version = api_version or os.environ.get("AIDS_AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)
        if not (self.endpoint and self.model and self.key):
            raise LLMError("Azure OpenAI needs AIDS_AZURE_OPENAI_ENDPOINT, _DEPLOYMENT and _KEY.")

    def chat(self, messages, schema=None, temperature=0.0) -> str:
        payload: dict[str, Any] = {"messages": messages, "temperature": temperature}
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": False},
            }
        url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
        reply = _post_json(url, payload, headers={"api-key": self.key})
        try:
            return reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as error:
            raise LLMError("Unexpected reply from Azure OpenAI.") from error


class ScriptedProvider(Provider):
    """Replies from a fixed list, in order: for tests and offline demos."""

    name = "Scripted"

    def __init__(self, replies: list[str | dict]):
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, schema=None, temperature=0.0) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("No scripted reply left.")
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


_cache: dict[str, Any] = {"at": 0.0, "provider": None, "status": None}
CACHE_SECONDS = 30


def get_provider(refresh: bool = False) -> Provider | None:
    """The configured model, or None when none is configured or reachable."""

    if not refresh and time.time() - _cache["at"] < CACHE_SECONDS:
        return _cache["provider"]

    choice = os.environ.get("AIDS_LLM_PROVIDER", "none").strip().lower()
    provider: Provider | None = None
    status: dict[str, Any] = {"provider": None, "model": None, "available": False, "detail": ""}

    if choice == "none":
        status["detail"] = "Language model switched off (set AIDS_LLM_PROVIDER=ollama or azure to use one)."
    elif choice == "azure" or (choice == "auto" and os.environ.get("AIDS_AZURE_OPENAI_ENDPOINT")):
        try:
            provider = AzureOpenAIProvider()
            status.update(provider="Azure OpenAI", model=provider.model, available=True, detail="Configured.")
        except LLMError as error:
            status["detail"] = str(error)
    else:
        ollama = OllamaProvider()
        status.update(provider="Ollama", model=ollama.model)
        installed = ollama.installed_models()
        if not installed:
            status["detail"] = f"Ollama is not running at {ollama.host} (or has no models); using rules only."
        elif not _has_model(installed, ollama.model):
            status["detail"] = f"Model '{ollama.model}' is not installed: run `ollama pull {ollama.model}`."
        else:
            provider = ollama
            status.update(available=True, detail="Running.")

    _cache.update(at=time.time(), provider=provider, status=status)
    return provider


def _has_model(installed: list[str], model: str) -> bool:
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in installed or model in installed


def provider_status(refresh: bool = False) -> dict[str, Any]:
    """Which model is in use and, if none, why not."""

    get_provider(refresh)
    return dict(_cache["status"])


def use_provider(provider: Provider | None) -> None:
    """Pin a provider (tests, demos); `get_provider` returns it until refreshed."""

    status = {
        "provider": provider.name if provider else None,
        "model": provider.model if provider else None,
        "available": provider is not None,
        "detail": "Set by the application." if provider else "Language model switched off.",
    }
    _cache.update(at=float("inf"), provider=provider, status=status)


def reset_provider() -> None:
    _cache.update(at=0.0, provider=None, status=None)
