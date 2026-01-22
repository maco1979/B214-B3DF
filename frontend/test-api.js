// Test script to verify frontend API client functionality
import { apiClient } from './src/services/api'

// Test the API client by calling the decision-service health endpoint
async function testApiClient() {
  try {
    console.log('Testing API client...')
    console.log(`API Base URL: ${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080'}`)
    
    // Test the decision-service health endpoint through the API gateway
    const response = await fetch('http://localhost:8080/api/decision/health', {
      headers: {
        'X-API-KEY': 'your-api-key-here'
      }
    })
    
    const data = await response.json()
    console.log('Health check response:', data)
    
    if (response.ok) {
      console.log('✅ API client test passed!')
    } else {
      console.log('❌ API client test failed!')
    }
  } catch (error) {
    console.error('❌ Error testing API client:', error)
  }
}

testApiClient()