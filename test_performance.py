import requests
import time
import statistics

# 测试配置
BASE_URL = "http://localhost:8001"
ENDPOINTS = [
    ("/api/system/health", "健康检查"),
    ("/api/system/info", "系统信息"),
    ("/api/system/stats", "系统统计"),
    ("/api/system/metrics", "系统指标")
]
TEST_COUNT = 10  # 每个端点测试次数

print("=== 系统API性能测试 ===\n")

for endpoint, name in ENDPOINTS:
    print(f"测试端点: {name} ({endpoint})")
    response_times = []
    error_count = 0
    
    for i in range(TEST_COUNT):
        start_time = time.time()
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = (end_time - start_time) * 1000  # 转换为毫秒
                response_times.append(response_time)
                print(f"  请求 {i+1:2d}: {response_time:.2f} ms (200 OK)")
            else:
                error_count += 1
                print(f"  请求 {i+1:2d}: 错误 - 状态码 {response.status_code}")
        except Exception as e:
            error_count += 1
            print(f"  请求 {i+1:2d}: 异常 - {str(e)}")
    
    if response_times:
        avg_time = statistics.mean(response_times)
        med_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n  测试结果 ({TEST_COUNT} 次请求):")
        print(f"  平均响应时间: {avg_time:.2f} ms")
        print(f"  中位数响应时间: {med_time:.2f} ms")
        print(f"  最小响应时间: {min_time:.2f} ms")
        print(f"  最大响应时间: {max_time:.2f} ms")
        print(f"  错误率: {error_count/TEST_COUNT*100:.1f}%")
    else:
        print("  所有请求都失败了\n")
    
    print("-" * 50)

print("\n=== 测试完成 ===")
