#!/usr/bin/env python3
"""
简化系统测试脚本
验证AI平台的核心功能
"""

import subprocess
import time
import sys
import os

def check_dependencies():
    """检查系统依赖"""
    print("🔍 检查系统依赖...")
    
    dependencies = [
        ("Python", ["python", "--version"]),
        ("Node.js", ["node", "--version"]),
        ("Docker", ["docker", "--version"])
    ]
    
    for name, cmd in dependencies:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: 未安装")
        except Exception:
            print(f"❌ {name}: 检查失败")

def check_backend_service():
    """检查后端服务状态"""
    print("\n🔍 检查后端服务...")
    
    # 检查端口是否被占用
    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True)
        if ":8000" in result.stdout:
            print("✅ 后端服务端口8000正在监听")
            return True
        else:
            print("❌ 后端服务未在端口8000运行")
            return False
    except Exception as e:
        print(f"❌ 端口检查失败: {e}")
        return False

def start_backend_service():
    """启动后端服务"""
    print("\n🚀 启动后端服务...")
    
    try:
        # 在后台启动服务
        backend_dir = os.path.join(os.path.dirname(__file__), "backend")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待服务启动
        time.sleep(8)
        
        # 检查是否启动成功
        if check_backend_service():
            print("✅ 后端服务启动成功")
            return proc
        else:
            print("❌ 后端服务启动失败")
            proc.terminate()
            return None
            
    except Exception as e:
        print(f"❌ 启动后端服务失败: {e}")
        return None

def test_api_health():
    """测试API健康检查"""
    print("\n🔍 测试API健康检查...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ API健康检查通过")
            print(f"   响应内容: {response.json()}")
            return True
        else:
            print(f"❌ API健康检查失败: 状态码 {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API测试异常: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = ["backend", "frontend", "infrastructure"]
    required_files = ["README.md", "docker-compose.yml", "run_tests.py"]
    
    all_passed = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"✅ 目录存在: {dir_name}")
        else:
            print(f"❌ 目录缺失: {dir_name}")
            all_passed = False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ 文件存在: {file_name}")
        else:
            print(f"❌ 文件缺失: {file_name}")
            all_passed = False
    
    return all_passed

def run_performance_check():
    """运行性能检查"""
    print("\n⚡ 运行性能检查...")
    
    try:
        import requests
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                start_time = time.time()
                response = requests.get("http://localhost:8000/health", timeout=5)
                end_time = time.time()
                
                results.put({
                    'success': response.status_code == 200,
                    'response_time': end_time - start_time
                })
            except Exception:
                results.put({'success': False, 'response_time': 10.0})
        
        # 创建5个并发请求
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 分析结果
        successful_requests = 0
        total_response_time = 0
        
        while not results.empty():
            result = results.get()
            if result['success']:
                successful_requests += 1
                total_response_time += result['response_time']
        
        success_rate = successful_requests / 5
        avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 10.0
        
        print(f"📊 性能检查结果:")
        print(f"   成功率: {success_rate:.1%}")
        print(f"   平均响应时间: {avg_response_time:.3f}秒")
        
        # 性能标准
        if success_rate >= 0.8 and avg_response_time < 2.0:
            print("✅ 性能检查通过")
            return True
        else:
            print("❌ 性能检查未达标")
            return False
            
    except Exception as e:
        print(f"❌ 性能检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🤖 AI平台系统测试和性能优化")
    print("=" * 60)
    
    # 检查依赖
    check_dependencies()
    
    # 检查项目结构
    structure_ok = check_project_structure()
    
    # 检查后端服务
    backend_running = check_backend_service()
    
    # 如果后端服务未运行，则启动它
    backend_proc = None
    if not backend_running:
        backend_proc = start_backend_service()
        backend_running = backend_proc is not None
    
    # 测试API
    api_ok = False
    if backend_running:
        api_ok = test_api_health()
    
    # 性能检查
    performance_ok = False
    if api_ok:
        performance_ok = run_performance_check()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总:")
    print("=" * 60)
    
    results = {
        "项目结构": "✅ 通过" if structure_ok else "❌ 失败",
        "后端服务": "✅ 运行中" if backend_running else "❌ 未运行",
        "API测试": "✅ 通过" if api_ok else "❌ 失败",
        "性能检查": "✅ 通过" if performance_ok else "❌ 失败"
    }
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # 生成优化建议
    print("\n💡 优化建议:")
    print("-" * 40)
    
    if not structure_ok:
        print("• 检查并完善项目目录结构")
    
    if not backend_running:
        print("• 修复后端服务启动问题")
    
    if not api_ok:
        print("• 优化API接口实现")
    
    if not performance_ok:
        print("• 优化系统性能，考虑使用缓存和异步处理")
    
    # 清理资源
    if backend_proc:
        backend_proc.terminate()
    
    # 总体评估
    all_passed = structure_ok and backend_running and api_ok and performance_ok
    
    if all_passed:
        print("\n🎉 系统测试全部通过，AI平台运行正常！")
        return 0
    else:
        print("\n⚠️  部分测试未通过，需要进一步优化")
        return 1

if __name__ == "__main__":
    sys.exit(main())