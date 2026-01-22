import subprocess
import time
import requests
import sys

# 启动极简FastAPI应用
server_process = subprocess.Popen([sys.executable, "test_simple_app.py"])

try:
    # 等待服务器启动
    time.sleep(3)
    
    print("=== 极简FastAPI应用性能测试 ===")
    print("测试端点: /")
    
    # 测试根路径
    total_time = 0
    errors = 0
    num_requests = 5
    
    for i in range(1, num_requests + 1):
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8002/")
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            total_time += response_time
            
            print(f"  请求 {i}: {response_time:.2f} ms ({response.status_code} {response.reason})")
        except Exception as e:
            errors += 1
            print(f"  请求 {i}: 异常 - {str(e)}")
    
    if num_requests - errors > 0:
        avg_time = total_time / (num_requests - errors)
        print(f"\n  测试结果 ({num_requests - errors} 次成功请求, {errors} 次失败):")
        print(f"  平均响应时间: {avg_time:.2f} ms")
    else:
        print(f"\n  所有请求都失败了")
    
    # 测试health端点
    print("\n测试端点: /health")
    
    total_time = 0
    errors = 0
    
    for i in range(1, num_requests + 1):
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8002/health")
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            total_time += response_time
            
            print(f"  请求 {i}: {response_time:.2f} ms ({response.status_code} {response.reason})")
        except Exception as e:
            errors += 1
            print(f"  请求 {i}: 异常 - {str(e)}")
    
    if num_requests - errors > 0:
        avg_time = total_time / (num_requests - errors)
        print(f"\n  测试结果 ({num_requests - errors} 次成功请求, {errors} 次失败):")
        print(f"  平均响应时间: {avg_time:.2f} ms")
    else:
        print(f"\n  所有请求都失败了")
        
finally:
    # 无论测试结果如何，都终止服务器进程
    server_process.terminate()
    server_process.wait(timeout=5)
    print("\n服务器已关闭")
