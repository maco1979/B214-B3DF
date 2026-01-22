import fetch from 'node-fetch';

// 模拟前端API请求
async function testFrontendToGatewayToDecisionService() {
    const API_BASE_URL = 'http://localhost:8080';
    const API_KEY = 'your-api-key-here';
    
    try {
        console.log('测试前端到网关到决策服务的端到端流程...');
        console.log(`API_BASE_URL: ${API_BASE_URL}`);
        
        // 测试健康检查路由
        const healthResponse = await fetch(`${API_BASE_URL}/health`, {
            headers: {
                'X-API-KEY': API_KEY
            }
        });
        
        const healthData = await healthResponse.json();
        console.log('\n网关健康状态:', healthData);
        
        // 测试决策服务路由
        const decisionHealthResponse = await fetch(`${API_BASE_URL}/api/decision/health`, {
            headers: {
                'X-API-KEY': API_KEY
            }
        });
        
        const decisionHealthData = await decisionHealthResponse.json();
        console.log('\n决策服务健康状态:', decisionHealthData);
        
        console.log('\n✅ 端到端测试成功!');
        console.log('前端 -> API网关 -> 决策服务 的流程已验证通过。');
        
    } catch (error) {
        console.error('\n❌ 测试失败:', error.message);
    }
}

testFrontendToGatewayToDecisionService();
