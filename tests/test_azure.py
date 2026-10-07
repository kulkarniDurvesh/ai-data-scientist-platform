"""
Phase 9: Azure readiness without an Azure account.

Entra ID sign-in for Azure OpenAI (a fake credential, no network), and
static checks of the infrastructure files: the stack deletes what it
creates, nothing is billed while idle, and no keys are used.
"""

import re
from pathlib import Path

import pytest

from core.llm import AzureOpenAIProvider, LLMError
from core.llm import provider as provider_module

ROOT = Path(__file__).resolve().parent.parent
INFRA = ROOT / "infra"


class FakeToken:
    token = "token-123"


class FakeCredential:
    def __init__(self):
        self.scopes = []

    def get_token(self, scope):
        self.scopes.append(scope)
        return FakeToken()


def test_azure_openai_uses_entra_id_without_a_key(monkeypatch):
    sent = {}

    def fake_post(url, payload, headers=None, timeout=0):
        sent.update(url=url, headers=headers)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(provider_module, "_post_json", fake_post)
    credential = FakeCredential()
    provider = AzureOpenAIProvider(endpoint="https://oai-test.openai.azure.com/", deployment="gpt-4.1-mini", credential=credential)
    assert provider.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert sent["headers"] == {"Authorization": "Bearer token-123"}
    assert credential.scopes == ["https://cognitiveservices.azure.com/.default"]
    assert sent["url"].startswith("https://oai-test.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions")

    keyed = AzureOpenAIProvider(endpoint="https://x.openai.azure.com", deployment="m", key="secret")
    assert keyed._auth_headers() == {"api-key": "secret"}

    with pytest.raises(LLMError, match="ENDPOINT"):
        AzureOpenAIProvider(endpoint="", deployment="m", key="k")


def test_sign_in_failures_are_reported(monkeypatch):
    class Broken:
        def get_token(self, scope):
            raise RuntimeError("no identity")

    provider = AzureOpenAIProvider(endpoint="https://x.openai.azure.com", deployment="m", credential=Broken())
    with pytest.raises(LLMError, match="Could not sign in"):
        provider.chat([{"role": "user", "content": "hi"}])


def _read(path):
    return (INFRA / path).read_text(encoding="utf-8")


def test_infrastructure_is_cheap_keyless_and_deletable():
    main, platform, budget = _read("main.bicep"), _read("modules/platform.bicep"), _read("modules/budget.bicep")
    up, down, run = _read("scripts/azure-up.ps1"), _read("scripts/azure-down.ps1"), _read("scripts/run-cloud.ps1")

    assert "targetScope = 'subscription'" in main
    assert "--action-on-unmanage deleteAll" in up and "--action-on-unmanage deleteAll" in down
    assert "minReplicas: 0" in platform                      # scale to zero
    assert "name: 'free'" in platform                        # AI Search Free tier
    assert "dailyQuotaGb" in platform                        # capped log ingestion
    assert "disableLocalAuth: true" in platform              # Azure OpenAI: managed identity only
    assert "enablePurgeProtection: null" in platform         # purgeable on teardown
    assert "purge" in down and "keyvault purge" in down
    assert "finally" in run                                  # teardown even on Ctrl+C
    assert "Microsoft.Consumption/budgets" in budget and "Forecasted" in budget
    assert not re.search(r"(?i)(api[-_]?key|password)\s*[:=]\s*'[^']+'", main + platform)  # no literal secrets


def test_image_workflow_publishes_to_ghcr():
    workflow = (ROOT / ".github/workflows/image.yml").read_text(encoding="utf-8")
    assert "ghcr.io" in workflow and "packages: write" in workflow
    assert "ghcr.io/kulkarnidurvesh/ai-data-scientist-platform" in _read("main.bicep")
