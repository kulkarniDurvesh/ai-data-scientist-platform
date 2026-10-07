<#
.SYNOPSIS
  Create the platform on Azure as one deployment stack (az stack sub create).

.EXAMPLE
  ./infra/scripts/azure-up.ps1 -BudgetEmail you@example.com
  ./infra/scripts/azure-up.ps1 -BudgetEmail you@example.com -SkipOpenAi -SkipSearch   # app only
  ./infra/scripts/azure-up.ps1 -BudgetEmail you@example.com -WhatIf                   # preview, creates nothing
#>
param(
    [Parameter(Mandatory = $true)][string]$BudgetEmail,
    [string]$Location = "centralindia",
    [string]$OpenAiLocation = "",
    [string]$Prefix = "aidsp",
    [string]$Image = "ghcr.io/kulkarnidurvesh/ai-data-scientist-platform:latest",
    [string]$OpenAiModel = "gpt-4.1-mini",
    [string]$OpenAiModelVersion = "2025-04-14",
    [int]$BudgetAmount = 1,
    [switch]$SkipOpenAi,
    [switch]$SkipSearch,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$template = Join-Path $root "infra/main.bicep"
$stack = "stack-$Prefix"
if (-not $OpenAiLocation) { $OpenAiLocation = $Location }

az account show -o none 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Not signed in: run 'az login' first." -ForegroundColor Red; exit 1 }

$parameters = @(
    "location=$Location", "openAiLocation=$OpenAiLocation", "prefix=$Prefix", "image=$Image",
    "openAiModel=$OpenAiModel", "openAiModelVersion=$OpenAiModelVersion",
    "budgetAmount=$BudgetAmount", "budgetEmail=$BudgetEmail",
    "deployOpenAi=$(if ($SkipOpenAi) { 'false' } else { 'true' })",
    "deploySearch=$(if ($SkipSearch) { 'false' } else { 'true' })"
)

if ($WhatIf) {
    Write-Host "Preview (what-if): nothing will be created." -ForegroundColor Cyan
    az deployment sub what-if --location $Location --template-file $template --parameters $parameters
    exit $LASTEXITCODE
}

foreach ($p in "Microsoft.App", "Microsoft.CognitiveServices", "Microsoft.Search", "Microsoft.OperationalInsights", "Microsoft.Insights", "Microsoft.KeyVault", "Microsoft.ManagedIdentity", "Microsoft.Consumption") {
    if ((az provider show --namespace $p --query registrationState -o tsv) -ne "Registered") {
        Write-Host "Registering $p ..."
        az provider register --namespace $p --wait -o none
    }
}

Write-Host "Creating deployment stack '$stack' in $Location (a few minutes) ..." -ForegroundColor Cyan
$started = Get-Date
az stack sub create --name $stack --location $Location --template-file $template --parameters $parameters `
    --action-on-unmanage deleteAll --deny-settings-mode none --yes -o none
if ($LASTEXITCODE -ne 0) {
    Write-Host "Deployment failed. Fix the error above, then run azure-up again (or azure-down to remove what was created)." -ForegroundColor Red
    exit 1
}

$outputs = az stack sub show --name $stack --query outputs -o json | ConvertFrom-Json
$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host "`nReady in $minutes min." -ForegroundColor Green
Write-Host "Dashboard : $($outputs.dashboardUrl.value)"
Write-Host "API       : $($outputs.apiUrl.value)/docs   (header X-API-Key: see Key Vault '$($outputs.keyVaultName.value)', secret 'api-key')"
if ($outputs.openAiEndpoint.value) { Write-Host "Azure OpenAI: $($outputs.openAiEndpoint.value) (managed identity, no keys)" }
if ($outputs.searchEndpoint.value) { Write-Host "AI Search : $($outputs.searchEndpoint.value)" }
Write-Host "`nThe first request after idle takes ~30-60 s (scale from zero). Remove everything with: ./infra/scripts/azure-down.ps1" -ForegroundColor Yellow
