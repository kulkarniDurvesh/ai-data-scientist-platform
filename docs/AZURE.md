# Azure deployment

The whole platform on Azure as **one deployment stack**: one command creates it, one deletes everything (resource group included) and purges soft-deleted resources, so nothing keeps existing or billing. Designed to cost close to nothing.

## What gets created

| Resource | Purpose | Cost design |
|---|---|---|
| Resource group `rg-aidsp` | Holds everything below | – |
| Container Apps environment + 2 apps (`ca-aidsp-api`, `ca-aidsp-web`) | HTTP API and dashboard | Scale to zero; the monthly free grant (180,000 vCPU-s, 360,000 GiB-s, 2M requests) covers demos |
| User-assigned managed identity `id-aidsp` | The apps sign in to Azure OpenAI, AI Search and Key Vault without keys | Free |
| Azure OpenAI (`gpt-4.1-mini`, Global Standard) | Fast language model for goals, Ask, documents and agents | Per token only; key access disabled |
| Azure AI Search, **Free** tier | Cloud document index (used by the app from Phase 9b) | Free (one per subscription) |
| Key Vault | The API key (`api-key`), read by the API app through the identity | ~Free |
| Log Analytics + Application Insights | Logs and traces | Ingestion capped at 0.1 GB/day |
| Budget (default 1 in your billing currency) | Emails at 10%, 50%, 100% actual and 100% forecast | Free |

The container image comes from **GitHub Container Registry** (`ghcr.io/kulkarnidurvesh/ai-data-scientist-platform`), built by `.github/workflows/image.yml` on every push — free for a public repository, so no Azure Container Registry is needed.

## One-time setup

1. **Azure CLI** (PowerShell): `winget install Microsoft.AzureCLI`, open a new terminal, then `az login` and `az bicep install`.
2. **Container image:** push to GitHub (the *image* workflow runs), then on GitHub open *Packages → ai-data-scientist-platform → Package settings → Change visibility → Public* so Azure can pull it without credentials.
3. **Check the subscription** (creates nothing):

   ```powershell
   ./infra/scripts/precheck.ps1 -Location centralindia
   ```

   It shows the account and spending limit, resource providers, whether `gpt-4.1-mini` and its quota are available in the region, whether a Free search service already exists, and leftovers from earlier runs.

## Use

```powershell
# Preview: shows what would be created, creates nothing
./infra/scripts/azure-up.ps1 -BudgetEmail you@example.com -WhatIf

# Create, open the dashboard, and delete everything when you press Enter / Ctrl+C / after 3 hours
./infra/scripts/run-cloud.ps1 -BudgetEmail you@example.com

# Or step by step
./infra/scripts/azure-up.ps1 -BudgetEmail you@example.com
./infra/scripts/azure-down.ps1
```

Options: `-SkipOpenAi` (no Azure OpenAI, language model off), `-SkipSearch` (no AI Search), `-OpenAiLocation swedencentral` (if the model isn't offered in Central India), `-OpenAiModelVersion` (match what precheck lists), `-BudgetAmount 5`.

The first request after idle takes 30–60 s while an app scales up from zero.

## Rough cost

| Scenario | Estimate |
|---|---|
| 2-hour demo, then `azure-down` | well under 1 USD (mostly Azure OpenAI tokens: ~0.40 / 1.60 USD per million input / output tokens for gpt-4.1-mini) |
| Forgotten for a month, idle | ~0 for compute (scaled to zero); a little for logs; the budget email arrives at 10% |
| Forgotten and used heavily | bounded by the free grant, the log cap and token use — the budget alerts early |

Prices change and differ by region and currency: check the Azure pricing calculator for your subscription.

## Design

- **Deployment stack with `--action-on-unmanage deleteAll`**: Azure tracks every resource the template created; deleting the stack deletes them and the resource group. `azure-down` also removes a resource group left by a failed run and purges soft-deleted Azure OpenAI and Key Vault resources (their names and quota stay reserved otherwise).
- **No keys**: Azure OpenAI has key access disabled; the apps use the managed identity (`AZURE_CLIENT_ID`) through `azure-identity`. Locally, the same code signs in with `az login`.
- **Same image as local Docker**: settings come from environment variables set by the template (`AIDS_LLM_PROVIDER=azure`, endpoint, deployment, API key from Key Vault).
- Not yet: Azure AI Search as the document index and OpenTelemetry traces to Application Insights (Phase 9b / 11), Entra ID sign-in for users (Phase 11).

## Troubleshooting

| Problem | Fix |
|---|---|
| `InsufficientQuota` / model not available | Run precheck; use `-OpenAiLocation` with a region that lists the model, or `-SkipOpenAi` |
| `ServiceQuotaExceeded` for search | A Free search service already exists in the subscription: delete it or use `-SkipSearch` |
| Container app fails to pull the image | Make the GitHub package public (setup step 2) and check the workflow ran |
| Secret / role errors on first run | Role assignments can take a minute to apply: run `azure-up` again |
| Something left after `azure-down` | `az resource list -g rg-aidsp -o table`, then run `azure-down` again |
