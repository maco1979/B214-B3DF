import requests
import time

def test_health_check():
    # 正确的URL应该包含/system前缀
    url = "http://localhost:8000/system/health"
    print(f"正在测试健康检查端点: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        end_time = time.time()
        
        print(f"请求耗时: {end_time - start_time:.2f}秒")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("健康检查通过!")
            print("响应内容:")
            print(response.json())
            return True
        else:
            print(f"健康检查失败，状态码: {response.status_code}")
            print("响应内容:")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("请求超时")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"连接错误: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

if __name__ == "__main__":
    test_health_check()