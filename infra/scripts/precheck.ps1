<#
.SYNOPSIS
  Check what the signed-in Azure subscription allows before deploying (creates nothing).

.EXAMPLE
  ./infra/scripts/precheck.ps1 -Location centralindia
#>
param(
    [string]$Location = "centralindia",
    [string]$OpenAiModel = "gpt-4.1-mini"
)

$ErrorActionPreference = "Stop"

function Section($title) { Write-Host "`n== $title ==" -ForegroundColor Cyan }

Section "Account"
$account = az account show --query "{subscription:name, id:id, user:user.name, tenant:tenantId}" -o json | ConvertFrom-Json
if (-not $account) { Write-Host "Not signed in: run 'az login' first." -ForegroundColor Red; exit 1 }
$account | Format-List
$spending = az account show --query "subscriptionPolicies.spendingLimit" -o tsv 2>$null
Write-Host "Spending limit: $spending  (On = free trial / credit subscription: spending stops at the credit)"

Section "Resource providers (registered = ready)"
$providers = "Microsoft.App", "Microsoft.CognitiveServices", "Microsoft.Search", "Microsoft.OperationalInsights",
             "Microsoft.Insights", "Microsoft.KeyVault", "Microsoft.ManagedIdentity", "Microsoft.Consumption"
foreach ($p in $providers) {
    $state = az provider show --namespace $p --query registrationState -o tsv 2>$null
    $color = if ($state -eq "Registered") { "Green" } else { "Yellow" }
    Write-Host ("{0,-32} {1}" -f $p, $state) -ForegroundColor $color
}
Write-Host "Register a missing one with: az provider register --namespace <name>   (azure-up does this too)"

Section "Azure OpenAI model '$OpenAiModel' in $Location"
$models = az cognitiveservices model list --location $Location --query "[?model.name=='$OpenAiModel'].{version:model.version, skus:model.skus[].name}" -o json 2>$null | ConvertFrom-Json
if (-not $models) {
    Write-Host "Not offered in $Location (or Azure OpenAI is not available to this subscription)." -ForegroundColor Yellow
    Write-Host "Deploy with -SkipOpenAi, or try another -OpenAiLocation (e.g. swedencentral, eastus2)."
} else {
    $models | ForEach-Object { Write-Host ("version {0}: {1}" -f $_.version, ($_.skus -join ", ")) }
    Write-Host "The template uses GlobalStandard; pass -OpenAiModelVersion to match a version above."
}

Section "Azure OpenAI quota in $Location"
$usage = az cognitiveservices usage list --location $Location --query "[?contains(name.value, '$OpenAiModel')].{name:name.value, used:currentValue, limit:limit}" -o json 2>$null | ConvertFrom-Json
if ($usage) { $usage | Format-Table } else { Write-Host "No quota found for $OpenAiModel (a limit of 0 means the model can't be deployed)." -ForegroundColor Yellow }

Section "Azure AI Search Free tier"
$free = az search service list --query "[?sku.name=='free'].name" -o tsv 2>$null
if ($free) { Write-Host "A Free search service already exists ($free): deploy with -SkipSearch or delete it." -ForegroundColor Yellow }
else { Write-Host "No Free search service yet: one can be created." -ForegroundColor Green }

Section "Leftovers from earlier deployments"
az stack sub list --query "[].{stack:name, state:provisioningState}" -o table 2>$null
az cognitiveservices account list-deleted --query "[].name" -o tsv 2>$null | ForEach-Object { Write-Host "Soft-deleted Azure OpenAI: $_ (azure-down purges these)" -ForegroundColor Yellow }
az keyvault list-deleted --query "[].name" -o tsv 2>$null | ForEach-Object { Write-Host "Soft-deleted Key Vault: $_ (azure-down purges these)" -ForegroundColor Yellow }
Write-Host "`nDone. Nothing was created."
