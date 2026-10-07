// Resources inside the platform's resource group.

param location string
param openAiLocation string
param prefix string
param image string
param deployOpenAi bool
param deploySearch bool
param openAiModel string
param openAiModelVersion string
param openAiCapacity int
@secure()
param apiKey string
param tags object

var suffix = uniqueString(resourceGroup().id)
var roles = {
  openAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'          // Cognitive Services OpenAI User
  searchIndexContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7' // Search Index Data Contributor
  searchServiceContributor: '7ca78c08-252a-4471-8644-bb5ff32d4ba0' // Search Service Contributor
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'  // Key Vault Secrets User
}

// ---------------------------------------------------------------- identity

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${prefix}'
  location: location
  tags: tags
}

// -------------------------------------------------------------- monitoring

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${prefix}-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: json('0.1') }   // caps ingestion cost
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${prefix}'
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

// --------------------------------------------------------------- key vault

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${prefix}-${take(suffix, 8)}'
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: null   // off, so azure-down can purge it
  }
}

resource apiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'api-key'
  properties: { value: apiKey }
}

resource vaultAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, identity.id, roles.keyVaultSecretsUser)
  scope: vault
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsUser)
  }
}

// ------------------------------------------------------------ azure openai

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployOpenAi) {
  name: 'oai-${prefix}-${suffix}'
  location: openAiLocation
  kind: 'OpenAI'
  sku: { name: 'S0' }
  tags: tags
  properties: {
    customSubDomainName: 'oai-${prefix}-${suffix}'
    disableLocalAuth: true          // managed identity only, no keys
    publicNetworkAccess: 'Enabled'
  }
}

resource chatModel 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployOpenAi) {
  parent: openAi
  name: openAiModel
  sku: { name: 'GlobalStandard', capacity: openAiCapacity }
  properties: {
    model: { format: 'OpenAI', name: openAiModel, version: openAiModelVersion }
  }
}

resource openAiAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployOpenAi) {
  name: guid(resourceGroup().id, 'openai', identity.id, roles.openAiUser)
  scope: openAi
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.openAiUser)
  }
}

// --------------------------------------------------------------- ai search

resource search 'Microsoft.Search/searchServices@2023-11-01' = if (deploySearch) {
  name: 'srch-${prefix}-${suffix}'
  location: location
  sku: { name: 'free' }
  tags: tags
  properties: {
    replicaCount: 1
    partitionCount: 1
    authOptions: { aadOrApiKey: { aadAuthFailureMode: 'http401WithBearerChallenge' } }
  }
}

resource searchData 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deploySearch) {
  name: guid(resourceGroup().id, 'search-data', identity.id)
  scope: search
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchIndexContributor)
  }
}

resource searchService 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deploySearch) {
  name: guid(resourceGroup().id, 'search-service', identity.id)
  scope: search
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchServiceContributor)
  }
}

// ---------------------------------------------------------- container apps

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${prefix}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var openAiEnv = deployOpenAi ? [
  { name: 'AIDS_LLM_PROVIDER', value: 'azure' }
  { name: 'AIDS_AZURE_OPENAI_ENDPOINT', value: openAi.properties.endpoint }
  { name: 'AIDS_AZURE_OPENAI_DEPLOYMENT', value: openAiModel }
] : [
  { name: 'AIDS_LLM_PROVIDER', value: 'none' }
]
var searchEnv = deploySearch ? [
  { name: 'AIDS_AZURE_SEARCH_ENDPOINT', value: 'https://${search.name}.search.windows.net' }
] : []
var commonEnv = concat([
  { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: insights.properties.ConnectionString }
  { name: 'AIDS_EMBED_PROVIDER', value: 'lsa' }
], openAiEnv, searchEnv)

var apps = [
  {
    name: 'ca-${prefix}-api'
    port: 8000
    command: [ 'python', '-m', 'uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000' ]
    extraEnv: [ { name: 'AIDS_API_KEY', secretRef: 'api-key' } ]
  }
  {
    name: 'ca-${prefix}-web'
    port: 8050
    command: [ 'python', 'app.py', '--host', '0.0.0.0', '--port', '8050' ]
    extraEnv: []
  }
]

resource containerApps 'Microsoft.App/containerApps@2024-03-01' = [for app in apps: {
  name: app.name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: { external: true, targetPort: app.port, transport: 'auto' }
      secrets: [
        { name: 'api-key', keyVaultUrl: apiKeySecret.properties.secretUri, identity: identity.id }
      ]
    }
    template: {
      containers: [
        {
          name: 'app'
          image: image
          command: app.command
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat(commonEnv, app.extraEnv)
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 1 }   // scale to zero: no charge when idle
    }
  }
  dependsOn: [ vaultAccess ]
}]

output apiUrl string = 'https://${containerApps[0].properties.configuration.ingress.fqdn}'
output dashboardUrl string = 'https://${containerApps[1].properties.configuration.ingress.fqdn}'
output openAiEndpoint string = deployOpenAi ? openAi.properties.endpoint : ''
output openAiAccountName string = deployOpenAi ? openAi.name : ''
output searchEndpoint string = deploySearch ? 'https://${search.name}.search.windows.net' : ''
output keyVaultName string = vault.name
