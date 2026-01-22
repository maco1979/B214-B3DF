#!/usr/bin/env python3
"""
系统测试和性能优化脚本
运行完整的系统测试套件
"""

import subprocess
import sys
import time
import requests
import json
from typing import Dict, Any


def run_command(command: list) -> bool:
    """运行命令并检查结果"""
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command[0]} 执行成功")
            return True
        else:
            print(f"❌ {command[0]} 执行失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 执行命令失败: {e}")
        return False


def test_backend_health() -> bool:
    """测试后端健康状态"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ 后端健康检查通过")
            return True
        else:
            print(f"❌ 后端健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 后端连接失败: {e}")
        return False


def test_api_endpoints() -> bool:
    """测试API端点"""
    endpoints = [
        ("/", "根端点"),
        ("/models", "模型管理"),
        ("/blockchain/status", "区块链状态"),
        ("/federated/status", "联邦学习状态"),
        ("/system/health", "系统健康")
    ]
    
    all_passed = True
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {description} API测试通过")
            else:
                print(f"❌ {description} API测试失败: {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"❌ {description} API测试异常: {e}")
            all_passed = False
    
    return all_passed


def run_performance_test() -> bool:
    """运行性能测试"""
    print("🚀 开始性能测试...")
    
    # 模拟并发请求
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
    
    # 创建10个并发请求
    threads = []
    for i in range(10):
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
    
    success_rate = successful_requests / 10
    avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 10.0
    
    print(f"📊 性能测试结果:")
    print(f"   成功率: {success_rate:.1%}")
    print(f"   平均响应时间: {avg_response_time:.3f}秒")
    
    # 性能标准
    if success_rate >= 0.9 and avg_response_time < 1.0:
        print("✅ 性能测试通过")
        return True
    else:
        print("❌ 性能测试未达标")
        return False


def generate_test_report() -> Dict[str, Any]:
    """生成测试报告"""
    report = {
        'timestamp': time.time(),
        'tests': {
            'backend_health': test_backend_health(),
            'api_endpoints': test_api_endpoints(),
            'performance': run_performance_test()
        },
        'recommendations': []
    }
    
    # 生成优化建议
    if not report['tests']['performance']:
        report['recommendations'].append("优化API响应时间，考虑使用缓存和异步处理")
    
    if not report['tests']['api_endpoints']:
        report['recommendations'].append("检查API端点实现，确保错误处理完善")
    
    return report


def main():
    """主测试函数"""
    print("=" * 60)
    print("🤖 AI平台系统测试和性能优化")
    print("=" * 60)
    
    # 检查依赖
    print("\n1. 检查系统依赖...")
    dependencies = [
        ["python", "--version"],
        ["docker", "--version"],
        ["node", "--version"]
    ]
    
    for dep in dependencies:
        run_command(dep)
    
    # 运行测试
    print("\n2. 运行系统测试...")
    report = generate_test_report()
    
    # 显示测试结果
    print("\n3. 测试结果汇总:")
    print("-" * 40)
    
    for test_name, result in report['tests'].items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    # 显示优化建议
    if report['recommendations']:
        print("\n4. 优化建议:")
        print("-" * 40)
        for recommendation in report['recommendations']:
            print(f"• {recommendation}")
    else:
        print("\n🎉 所有测试通过，系统运行正常！")
    
    # 保存测试报告
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 测试报告已保存到: test_report.json")
    
    # 返回测试结果
    all_passed = all(report['tests'].values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())