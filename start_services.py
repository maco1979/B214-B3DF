"""
简化版服务启动脚本
用于启动后端服务进行测试
"""

import subprocess
import sys
import os
import time
import threading
import requests
from pathlib import Path


def start_backend_service():
    """启动后端服务"""
    print("🚀 启动后端服务...")
    
    project_root = Path("d:\\1.5\\backend")
    
    # 启动FastAPI后端服务
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.api.simple_app:app", 
        "--host", "127.0.0.1", 
        "--port", "8000",
        "--reload"
    ]
    
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待服务启动
    time.sleep(10)
    
    # 检查服务是否启动成功
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ 后端服务启动成功")
            return process
        else:
            print(f"❌ 后端服务启动失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 后端服务启动失败: {e}")
        return None


def test_basic_functionality():
    """测试基本功能"""
    print("\n🔍 测试基本功能...")
    
    tests = [
        {
            "name": "健康检查",
            "url": "http://127.0.0.1:8000/health",
            "method": "GET"
        },
        {
            "name": "获取模型列表",
            "url": "http://127.0.0.1:8000/api/models",
            "method": "GET"
        },
        {
            "name": "用户统计信息",
            "url": "http://127.0.0.1:8000/api/user/stats?user_id=test_user",
            "method": "GET"
        },
        {
            "name": "企业信息",
            "url": "http://127.0.0.1:8000/api/enterprise/test_business",
            "method": "GET"
        }
    ]
    
    results = []
    for test in tests:
        try:
            response = requests.request(
                method=test["method"],
                url=test["url"],
                timeout=10
            )
            status = "✅" if response.status_code == 200 else "❌"
            results.append({
                "name": test["name"],
                "status": status,
                "code": response.status_code
            })
            print(f"  {status} {test['name']}: {response.status_code}")
        except Exception as e:
            results.append({
                "name": test["name"],
                "status": "❌",
                "error": str(e)
            })
            print(f"  ❌ {test['name']}: {str(e)}")
    
    return results


def run_continuous_test():
    """运行连续测试"""
    print("\n🔄 开始连续功能测试...")
    
    test_count = 0
    failed_tests = 0
    
    try:
        while True:
            test_count += 1
            print(f"\n--- 测试周期 #{test_count} ---")
            
            # 运行基本功能测试
            results = test_basic_functionality()
            
            # 检查失败的测试
            failed_in_cycle = [r for r in results if r["status"] == "❌"]
            if failed_in_cycle:
                failed_tests += len(failed_in_cycle)
                print(f"⚠️  周期 #{test_count} 发现 {len(failed_in_cycle)} 个失败测试")
            else:
                print(f"✅ 周期 #{test_count} 所有测试通过")
            
            # 每10个周期输出摘要
            if test_count % 10 == 0:
                success_rate = ((test_count * 5 - failed_tests) / (test_count * 5)) * 100
                print(f"\n📊 摘要 (周期 1-{test_count}):")
                print(f"   总测试: {test_count * 5}")
                print(f"   失败: {failed_tests}")
                print(f"   成功率: {success_rate:.2f}%")
            
            # 等待一段时间再进行下一轮测试
            print("⏳ 等待下一轮测试 (30秒)...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n⏸️ 测试被用户中断")
        print(f"📈 最终统计:")
        print(f"   完成周期: {test_count}")
        print(f"   总测试: {test_count * 5}")
        print(f"   失败: {failed_tests}")
        if test_count > 0:
            success_rate = ((test_count * 5 - failed_tests) / (test_count * 5)) * 100
            print(f"   成功率: {success_rate:.2f}%")


def main():
    """主函数"""
    print("🚀 开始本地服务部署测试")
    print("="*50)
    
    # 启动后端服务
    backend_process = start_backend_service()
    
    if backend_process is None:
        print("❌ 无法启动后端服务，测试终止")
        return
    
    try:
        # 运行基本功能测试
        print("\n" + "="*50)
        initial_results = test_basic_functionality()
        
        # 检查初始测试结果
        failed_initial = [r for r in initial_results if r["status"] == "❌"]
        if failed_initial:
            print(f"\n⚠️  初始测试发现 {len(failed_initial)} 个问题")
            for failed in failed_initial:
                print(f"   - {failed['name']}")
        else:
            print(f"\n✅ 初始测试全部通过")
        
        print("\n" + "="*50)
        print("开始长时间运行测试...")
        print("按 Ctrl+C 停止测试")
        print("="*50)
        
        # 开始连续测试
        run_continuous_test()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
    finally:
        # 停止后端服务
        print("\n🛑 停止后端服务...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ 服务已停止")


if __name__ == "__main__":
    main()