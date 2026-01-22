const fetch = require('node-fetch');

// 测试前端API配置
const API_BASE_URL = 'http://localhost:8001/api';
const LOGIN_ENDPOINT = `${API_BASE_URL}/auth/login`;

// 测试数据
const testCredentials = {
  email: 'user@example.com',
  password: 'password123'
};

async function testLogin() {
  try {
    console.log(`测试登录API: ${LOGIN_ENDPOINT}`);
    console.log('测试凭据:', testCredentials);
    
    const response = await fetch(LOGIN_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': 'your-api-key-here'
      },
      body: JSON.stringify(testCredentials)
    });
    
    console.log('响应状态:', response.status);
    console.log('响应头:', Object.fromEntries(response.headers.entries()));
    
    const data = await response.json();
    console.log('响应数据:', JSON.stringify(data, null, 2));
    
    if (response.status === 200) {
      console.log('\n✅ 登录API测试成功!');
    } else {
      console.log('\n❌ 登录API测试失败!');
    }
  } catch (error) {
    console.error('\n❌ 测试过程中出错:', error.message);
    console.error('详细错误:', error);
  }
}

// 运行测试
testLogin();