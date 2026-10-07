// Monthly budget on the platform's resource group, with email alerts.

targetScope = 'subscription'

param name string
param amount int
param email string
param startDate string
param resourceGroupName string

resource budget 'Microsoft.Consumption/budgets@2023-05-01' = {
  name: name
  properties: {
    category: 'Cost'
    amount: amount
    timeGrain: 'Monthly'
    timePeriod: { startDate: startDate }
    filter: {
      dimensions: { name: 'ResourceGroupName', operator: 'In', values: [ resourceGroupName ] }
    }
    notifications: {
      actual10: { enabled: true, operator: 'GreaterThanOrEqualTo', threshold: 10, contactEmails: [ email ], thresholdType: 'Actual' }
      actual50: { enabled: true, operator: 'GreaterThanOrEqualTo', threshold: 50, contactEmails: [ email ], thresholdType: 'Actual' }
      actual100: { enabled: true, operator: 'GreaterThanOrEqualTo', threshold: 100, contactEmails: [ email ], thresholdType: 'Actual' }
      forecast100: { enabled: true, operator: 'GreaterThanOrEqualTo', threshold: 100, contactEmails: [ email ], thresholdType: 'Forecasted' }
    }
  }
}
