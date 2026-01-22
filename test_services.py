import requests
import time

def test_service(name, url):
    """测试服务是否可以访问"""
    try:
        print(f"\n🔍 测试 {name} ({url})...")
        start_time = time.time()
        response = requests.get(url, timeout=5)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            print(f"✅ {name} 响应成功")
            print(f"   状态码: {response.status_code}")
            print(f"   响应时间: {response_time:.2f} ms")
            print(f"   响应内容: {response.text[:100]}...")
            return True
        else:
            print(f"❌ {name} 响应失败")
            print(f"   状态码: {response.status_code}")
            return False
    except requests.ConnectionError:
        print(f"❌ {name} 无法连接")
        return False
    except requests.Timeout:
        print(f"⏱️ {name} 请求超时")
        return False
    except Exception as e:
        print(f"⚠️ {name} 测试出错: {e}")
        return False

# 测试各服务
print("🚀 开始测试微服务系统...")

# 测试后端服务
test_service("后端服务", "http://localhost:8000/health")

# 测试决策服务
test_service("决策服务", "http://localhost:8001/health")

# 测试API网关
test_service("API网关", "http://localhost:8080/health")

# 测试通过API网关访问后端
test_service("API网关 -> 后端", "http://localhost:8080/api/backend/health")

# 测试通过API网关访问决策服务
test_service("API网关 -> 决策服务", "http://localhost:8080/api/decision/health")

print("\n📋 测试完成！")