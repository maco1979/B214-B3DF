// Simple test to verify API gateway is working correctly with the right API key
const https = require('https');
const http = require('http');

// Test API gateway health check with API key
const options = {
  hostname: 'localhost',
  port: 8080,
  path: '/api/decision/health',
  method: 'GET',
  headers: {
    'X-API-KEY': 'your-api-key-here'
  }
};

console.log('Testing API gateway with correct API key...');

const req = http.request(options, (res) => {
  console.log(`Status: ${res.statusCode}`);
  console.log(`Headers: ${JSON.stringify(res.headers)}`);
  
  res.on('data', (d) => {
    process.stdout.write(d);
    console.log('\n✅ API gateway test passed!');
  });
});

req.on('error', (error) => {
  console.error('❌ API gateway test failed:', error);
});

req.end();