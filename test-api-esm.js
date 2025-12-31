// ES module test script for the complete flow
import fetch from 'node-fetch';

const API_KEY = 'your-api-key-here';
const GATEWAY_URL = 'http://localhost:8080';

// Helper function for making API requests
async function makeRequest(endpoint, options = {}) {
    const url = `${GATEWAY_URL}${endpoint}`;
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                'X-API-KEY': API_KEY,
                ...options.headers,
            },
            ...options,
        });
        
        const data = await response.json();
        console.log(`Request to ${endpoint}:`);
        console.log(`Status: ${response.status}`);
        console.log(`Response:`, JSON.stringify(data, null, 2));
        console.log('---');
        
        return { status: response.status, data };
    } catch (error) {
        console.error(`Request to ${endpoint} failed:`, error.message);
        console.log('---');
        return { status: 500, error: error.message };
    }
}

// Test the complete flow
async function testCompleteFlow() {
    console.log('Testing complete flow: frontend → API gateway → decision-service');
    console.log('='.repeat(50));
    console.log();
    
    // Test 1: System health check
    console.log('Test 1: System health check');
    await makeRequest('/system/health');
    
    // Test 2: Decision-service health check through gateway
    console.log('Test 2: Decision-service health check through gateway');
    await makeRequest('/api/decision/health');
    
    // Test 3: Try without API key (should fail)
    console.log('Test 3: Try without API key (should fail)');
    await fetch(`${GATEWAY_URL}/system/health`)
        .then(res => res.json())
        .then(data => {
            console.log(`Request to /system/health without API key:`);
            console.log(`Status: ${res.status}`);
            console.log(`Response:`, JSON.stringify(data, null, 2));
            console.log('---');
        })
        .catch(error => {
            console.error('Request failed:', error.message);
            console.log('---');
        });
    
    console.log('All tests completed!');
}

// Run the tests
testCompleteFlow().catch(console.error);
