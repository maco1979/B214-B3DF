import requests

# 测试健康检查端点
try:
    response = requests.get('http://localhost:8000/system/health')
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
    print("健康检查端点测试成功!")
except Exception as e:
    print(f"健康检查端点测试失败: {e}")
