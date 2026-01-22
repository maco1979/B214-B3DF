#!/bin/bash

echo "Testing complete flow: frontend → API gateway → decision-service"
echo "="
echo ""

API_KEY="your-api-key-here"
DECISION_SERVICE_URL="http://localhost:8001"
GATEWAY_URL="http://localhost:8080"

# Test 1: Decision-service health check directly
echo "Test 1: Decision-service health check directly"
curl -X GET "$DECISION_SERVICE_URL/health"
echo ""
echo ""

# Test 2: API gateway health check with API key
echo "Test 2: API gateway health check with API key"
curl -X GET "$GATEWAY_URL/system/health" -H "X-API-KEY: $API_KEY"
echo ""
echo ""

# Test 3: API gateway routing to decision-service health
echo "Test 3: API gateway routing to decision-service health"
curl -X GET "$GATEWAY_URL/api/decision/health" -H "X-API-KEY: $API_KEY"
echo ""
echo ""

# Test 4: API gateway without API key
echo "Test 4: API gateway without API key (should fail)"
curl -X GET "$GATEWAY_URL/system/health"
echo ""
echo ""

# Test 5: API gateway with invalid API key
echo "Test 5: API gateway with invalid API key (should fail)"
curl -X GET "$GATEWAY_URL/system/health" -H "X-API-KEY: invalid-key"
echo ""
echo ""

echo "All tests completed!"
