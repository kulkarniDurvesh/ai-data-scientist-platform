<#
.SYNOPSIS
  Delete everything the stack created (resource group included) and purge soft-deleted
  Azure OpenAI and Key Vault resources, so nothing keeps existing or billing.

.EXAMPLE
  ./infra/scripts/azure-down.ps1
#>
param(
    [string]$Prefix = "aidsp",
    [string]$Location = "centralindia"
)

$ErrorActionPreference = "Continue"
$stack = "stack-$Prefix"
$group = "rg-$Prefix"

az account show -o none 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Not signed in: run 'az login' first." -ForegroundColor Red; exit 1 }

$exists = az stack sub show --name $stack --query name -o tsv 2>$null
if ($exists) {
    Write-Host "Deleting deployment stack '$stack' and everything it manages ..." -ForegroundColor Cyan
    az stack sub delete --name $stack --action-on-unmanage deleteAll --yes -o none
} else {
    Write-Host "No stack '$stack' found."
}

# A resource group left behind by a failed deployment.
if ((az group exists --name $group) -eq "true") {
    Write-Host "Deleting leftover resource group '$group' ..."
    az group delete --name $group --yes -o none
}

# Soft-deleted resources keep their names reserved (and Azure OpenAI can keep quota) until purged.
foreach ($name in (az cognitiveservices account list-deleted --query "[?contains(name, '$Prefix')].name" -o tsv 2>$null)) {
    $loc = az cognitiveservices account list-deleted --query "[?name=='$name'].location | [0]" -o tsv
    Write-Host "Purging soft-deleted Azure OpenAI '$name' ($loc) ..."
    az cognitiveservices account purge --name $name --resource-group $group --location $loc -o none
}
foreach ($name in (az keyvault list-deleted --query "[?contains(name, '$Prefix')].name" -o tsv 2>$null)) {
    Write-Host "Purging soft-deleted Key Vault '$name' ..."
    az keyvault purge --name $name -o none
}

$left = az resource list --query "[?contains(name, '$Prefix')].name" -o tsv 2>$null
if ($left) { Write-Host "Still present (check in the portal): $left" -ForegroundColor Yellow }
else { Write-Host "Everything is deleted." -ForegroundColor Green }
