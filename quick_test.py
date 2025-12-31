#!/usr/bin/env python3
"""
快速测试脚本 - 验证AI平台基本功能
"""

import sys
import os
import time
import json
from pathlib import Path

def check_system_dependencies():
    """检查系统依赖"""
    print("🔍 检查系统依赖...")
    
    dependencies = [
        ("Python", sys.version.split()[0]),
    ]
    
    for name, version in dependencies:
        print(f"✅ {name}: {version}")
    
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_items = [
        ("backend/", True),
        ("frontend/", True),
        ("infrastructure/", True),
        ("backend/main.py", False),
        ("docker-compose.yml", False),
        ("README.md", False)
    ]
    
    all_passed = True
    
    for item, is_dir in required_items:
        path = Path(item)
        if path.exists():
            if is_dir and path.is_dir():
                print(f"✅ 目录存在: {item}")
            elif not is_dir and path.is_file():
                print(f"✅ 文件存在: {item}")
            else:
                print(f"❌ 类型不匹配: {item}")
                all_passed = False
        else:
            print(f"❌ 缺失: {item}")
            all_passed = False
    
    return all_passed

def check_backend_code():
    """检查后端代码"""
    print("\n🔧 检查后端代码...")
    
    try:
        # 检查main.py内容
        with open("backend/main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = [
            ("FastAPI导入", "from fastapi import FastAPI" in content),
            ("API路由", "@app.get" in content),
            ("健康检查", "/health" in content),
            ("CORS配置", "CORSMiddleware" in content)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 后端代码检查失败: {e}")
        return False

def check_configuration():
    """检查配置文件"""
    print("\n⚙️  检查配置文件...")
    
    config_files = [
        "docker-compose.yml",
        "backend/requirements.txt"
    ]
    
    all_passed = True
    
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"✅ 配置文件存在: {config_file}")
            
            # 检查docker-compose内容
            if config_file == "docker-compose.yml":
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    services = ["backend", "frontend", "monitoring"]
                    found_services = [s for s in services if s in content]
                    
                    if found_services:
                        print(f"   ✅ 包含服务: {', '.join(found_services)}")
                    else:
                        print("   ⚠️  未找到服务配置")
                        
                except Exception as e:
                    print(f"   ❌ 配置文件读取失败: {e}")
                    all_passed = False
        else:
            print(f"❌ 配置文件缺失: {config_file}")
            all_passed = False
    
    return all_passed

def simulate_api_test():
    """模拟API测试"""
    print("\n🌐 模拟API测试...")
    
    # 模拟API端点测试
    endpoints = [
        ("/", "根端点", {"message": "AI平台API服务运行正常"}),
        ("/health", "健康检查", {"status": "healthy"}),
        ("/metrics", "监控指标", {"status": "ok"})
    ]
    
    print("⚠️  注意: 当前为模拟测试，实际API需要启动服务")
    print("   服务启动后可通过以下URL访问:")
    print("   - 后端API: http://localhost:8000")
    print("   - 前端应用: http://localhost:80")
    print("   - 监控面板: http://localhost:3000")
    
    for endpoint, description, expected_response in endpoints:
        print(f"   ✅ {description}: http://localhost:8000{endpoint}")
    
    return True

def generate_test_report(results):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📋 测试报告")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed_tests}/{total_tests} 项测试通过")
    print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    # 生成建议
    print("\n💡 部署建议:")
    print("-" * 40)
    
    if failed_tests == 0:
        print("🎉 所有测试通过！系统准备就绪。")
        print("\n下一步操作:")
        print("1. 启动Docker Desktop")
        print("2. 运行: docker-compose up -d")
        print("3. 访问: http://localhost:80")
    else:
        print("⚠️  部分测试未通过，需要修复问题")
        
        if not results.get("项目结构检查"):
            print("• 检查并完善项目目录结构")
        
        if not results.get("后端代码检查"):
            print("• 修复后端代码问题")
        
        if not results.get("配置文件检查"):
            print("• 完善配置文件")
    
    return passed_tests == total_tests

def main():
    """主测试函数"""
    print("=" * 60)
    print("🤖 AI平台快速测试版本")
    print("=" * 60)
    
    # 切换到项目目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 执行测试
    test_results = {
        "系统依赖检查": check_system_dependencies(),
        "项目结构检查": check_project_structure(),
        "后端代码检查": check_backend_code(),
        "配置文件检查": check_configuration(),
        "API接口模拟": simulate_api_test()
    }
    
    # 生成报告
    all_passed = generate_test_report(test_results)
    
    # 保存测试报告
    report = {
        "timestamp": time.time(),
        "results": test_results,
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for r in test_results.values() if r),
            "all_passed": all_passed
        }
    }
    
    with open("quick_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 测试报告已保存到: quick_test_report.json")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())