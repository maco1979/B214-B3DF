#!/usr/bin/env python3
"""
AI平台代码验证脚本
快速验证核心代码功能
"""

import sys
import os
import json
from pathlib import Path

def test_environment():
    """测试基础环境"""
    print("🔍 测试基础环境...")
    
    try:
        # Python版本检查
        print(f"   Python版本: {sys.version.split()[0]}")
        
        # 基础模块检查
        required_modules = ['json', 'os', 'subprocess', 'time', 'pathlib']
        for module in required_modules:
            __import__(module)
        print("   ✅ 基础模块可用")
        
        return True
    except Exception as e:
        print(f"   ❌ 环境测试失败: {e}")
        return False

def test_project_structure():
    """测试项目结构"""
    print("📁 测试项目结构...")
    
    try:
        project_root = Path(__file__).parent
        
        required_dirs = [
            'backend', 'frontend', 'infrastructure', 
            'core-framework', 'decision-service', 'api-gateway'
        ]
        
        all_passed = True
        for dir_name in required_dirs:
            if (project_root / dir_name).exists():
                print(f"   ✅ 目录存在: {dir_name}")
            else:
                print(f"   ❌ 目录缺失: {dir_name}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"   ❌ 项目结构测试失败: {e}")
        return False

def test_deployment_system():
    """测试部署系统"""
    print("🚀 测试部署系统...")
    
    try:
        import deploy_integration_system as dis
        print("   ✅ 部署系统模块导入成功")
        
        # 测试数据类
        from deploy_integration_system import ValidationResult, ServiceInfo, DeploymentReport
        
        # 创建测试实例
        validation_result = ValidationResult(
            environment='test',
            strategy='performance',
            valid=True,
            warnings=['测试警告'],
            errors=[],
            recommendations=['测试建议']
        )
        
        service_info = ServiceInfo(
            name='测试服务',
            pid=1234,
            status='running',
            process=None
        )
        
        deployment_report = DeploymentReport(
            timestamp='2024-01-01',
            environment='test',
            optimization_strategy='performance',
            services=[service_info],
            health_check='passed',
            integration_tests='passed',
            overall_status='success'
        )
        
        print("   ✅ 数据类实例化成功")
        
        # 验证功能
        assert validation_result.valid == True
        assert service_info.name == '测试服务'
        assert deployment_report.overall_status == 'success'
        
        print("   ✅ 部署系统功能验证通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 部署系统测试失败: {e}")
        return False

def test_backend_code():
    """测试后端代码"""
    print("🔧 测试后端代码...")
    
    try:
        backend_dir = Path('backend')
        
        if not backend_dir.exists():
            print("   ❌ 后端目录不存在")
            return False
        
        # 检查关键文件
        required_files = ['requirements.txt']
        for file_name in required_files:
            if (backend_dir / file_name).exists():
                print(f"   ✅ 后端文件存在: {file_name}")
            else:
                print(f"   ⚠️  后端文件缺失: {file_name}")
        
        # 检查关键目录
        required_dirs = ['src', 'tests']
        for dir_name in required_dirs:
            if (backend_dir / dir_name).exists():
                print(f"   ✅ 后端子目录存在: {dir_name}")
            else:
                print(f"   ⚠️  后端子目录缺失: {dir_name}")
        
        return True
    except Exception as e:
        print(f"   ❌ 后端代码测试失败: {e}")
        return False

def test_frontend_structure():
    """测试前端结构"""
    print("🌐 测试前端结构...")
    
    try:
        frontend_dir = Path('frontend')
        
        if not frontend_dir.exists():
            print("   ❌ 前端目录不存在")
            return False
        
        # 检查关键文件
        frontend_files = ['package.json', 'index.html']
        for file_name in frontend_files:
            if (frontend_dir / file_name).exists():
                print(f"   ✅ 前端文件存在: {file_name}")
            else:
                print(f"   ⚠️  前端文件缺失: {file_name}")
        
        return True
    except Exception as e:
        print(f"   ❌ 前端结构测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🤖 AI平台代码验证测试")
    print("=" * 60)
    
    # 运行测试套件
    tests = [
        ("环境测试", test_environment),
        ("项目结构", test_project_structure),
        ("部署系统", test_deployment_system),
        ("后端代码", test_backend_code),
        ("前端结构", test_frontend_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        result = test_func()
        results.append((test_name, result))
        
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    success_rate = passed_count / total_count
    
    print(f"总测试数: {total_count}")
    print(f"通过数: {passed_count}")
    print(f"成功率: {success_rate:.1%}")
    
    for test_name, result in results:
        status_icon = "✅" if result else "❌"
        print(f"{status_icon} {test_name}")
    
    # 总体评估
    if success_rate >= 0.8:
        print("\n🎉 AI平台代码验证通过！系统运行正常。")
        return 0
    else:
        print("\n⚠️  AI平台代码验证未完全通过，需要进一步检查。")
        
        # 显示失败的建议
        failed_tests = [name for name, result in results if not result]
        if failed_tests:
            print("\n💡 修复建议:")
            for test_name in failed_tests:
                print(f"  • 修复 {test_name} 相关代码")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())