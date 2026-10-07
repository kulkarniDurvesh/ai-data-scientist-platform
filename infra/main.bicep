// AI Data Scientist Platform: everything it needs on Azure, as one deployment stack.
//
//   ./infra/scripts/azure-up.ps1     create (az stack sub create ... --action-on-unmanage deleteAll)
//   ./infra/scripts/azure-down.ps1   delete everything, including the resource group
//
// Cost design: Container Apps scale to zero (free monthly grant), AI Search on the
// Free tier, image from GitHub Container Registry (free for public repos), log
// ingestion capped per day, Azure OpenAI billed per token, a budget alert.

targetScope = 'subscription'

@description('Region for every resource.')
param location string = 'centralindia'

@description('Region for Azure OpenAI (Global Standard deployments are available in many regions).')
param openAiLocation string = location

@description('Short name used in resource names (lowercase letters and digits).')
@minLength(3)
@maxLength(10)
param prefix string = 'aidsp'

@description('Container image of the platform (built by .github/workflows/image.yml).')
param image string = 'ghcr.io/kulkarnidurvesh/ai-data-scientist-platform:latest'

@description('Create Azure OpenAI and use it as the language model.')
param deployOpenAi bool = true

@description('Create Azure AI Search on the Free tier (one free service per subscription).')
param deploySearch bool = true

@description('Azure OpenAI chat model and version.')
param openAiModel string = 'gpt-4.1-mini'
param openAiModelVersion string = '2025-04-14'

@description('Model capacity in thousands of tokens per minute.')
param openAiCapacity int = 10

@description('Monthly budget in the billing currency; alerts at 10%, 50% and 100%.')
param budgetAmount int = 1

@description('Email for budget alerts.')
param budgetEmail string

@description('First day of the budget period (defaults to this month).')
param budgetStart string = '${utcNow('yyyy-MM')}-01'

@description('Key for the HTTP API (X-API-Key); generated if not given.')
@secure()
param apiKey string = newGuid()

var resourceGroupName = 'rg-${prefix}'
var tags = {
  app: 'ai-data-scientist-platform'
  managedBy: 'deployment-stack'
}

resource group 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module platform 'modules/platform.bicep' = {
  name: 'platform'
  scope: group
  params: {
    location: location
    openAiLocation: openAiLocation
    prefix: prefix
    image: image
    deployOpenAi: deployOpenAi
    deploySearch: deploySearch
    openAiModel: openAiModel
    openAiModelVersion: openAiModelVersion
    openAiCapacity: openAiCapacity
    apiKey: apiKey
    tags: tags
  }
}

module budget 'modules/budget.bicep' = {
  name: 'budget'
  params: {
    name: 'budget-${prefix}'
    amount: budgetAmount
    email: budgetEmail
    startDate: budgetStart
    resourceGroupName: resourceGroupName
  }
}

output resourceGroup string = resourceGroupName
output apiUrl string = platform.outputs.apiUrl
output dashboardUrl string = platform.outputs.dashboardUrl
output openAiEndpoint string = platform.outputs.openAiEndpoint
output searchEndpoint string = platform.outputs.searchEndpoint
output keyVaultName string = platform.outputs.keyVaultName
output openAiAccountName string = platform.outputs.openAiAccountName
