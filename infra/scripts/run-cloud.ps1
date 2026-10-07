<#
.SYNOPSIS
  Create the platform on Azure, open it, and delete everything when you are done
  (press Enter, close with Ctrl+C, or after -Hours).

.EXAMPLE
  ./infra/scripts/run-cloud.ps1 -BudgetEmail you@example.com
  ./infra/scripts/run-cloud.ps1 -BudgetEmail you@example.com -Hours 2
#>
param(
    [Parameter(Mandatory = $true)][string]$BudgetEmail,
    [string]$Location = "centralindia",
    [string]$Prefix = "aidsp",
    [double]$Hours = 3,
    [switch]$SkipOpenAi,
    [switch]$SkipSearch
)

$up = Join-Path $PSScriptRoot "azure-up.ps1"
$down = Join-Path $PSScriptRoot "azure-down.ps1"

try {
    & $up -BudgetEmail $BudgetEmail -Location $Location -Prefix $Prefix -SkipOpenAi:$SkipOpenAi -SkipSearch:$SkipSearch
    if ($LASTEXITCODE -ne 0) { throw "Deployment failed." }

    $url = az stack sub show --name "stack-$Prefix" --query "outputs.dashboardUrl.value" -o tsv
    if ($url) { Start-Process $url }

    $deadline = (Get-Date).AddHours($Hours)
    Write-Host "`nRunning until $($deadline.ToString('HH:mm')). Press Enter to shut down now." -ForegroundColor Yellow
    while ((Get-Date) -lt $deadline) {
        if ([Console]::KeyAvailable -and [Console]::ReadKey($true).Key -eq "Enter") { break }
        Start-Sleep -Seconds 1
    }
}
finally {
    # Runs on Enter, on the time limit, on Ctrl+C and on errors.
    Write-Host "`nShutting down: deleting all Azure resources ..." -ForegroundColor Cyan
    & $down -Prefix $Prefix -Location $Location
}
