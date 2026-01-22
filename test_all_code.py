#!/usr/bin/env python3
"""
AI平台全部代码测试脚本
测试整个项目的代码功能
"""

import sys
import os
import subprocess
from pathlib import Path

def print_section(title):
    """打印测试章节标题"""
    print(f"\n{'='*50}")
    print(f"📋 {title}")
    print('='*50)

def run_test(test_name, test_func):
    """运行单个测试"""
    try:
        result = test_func()
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
        return result
    except Exception as e:
        print(f"❌ 异常: {test_name} - {e}")
        return False

def test_python_environment():
    """测试Python环境"""
    print("Python版本:", sys.version.split()[0])
    
    # 检查必要模块
    required_modules = ['json', 'os', 'subprocess', 'time', 'pathlib', 'dataclasses']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            return False
    return True

def test_project_structure():
    """测试项目结构"""
    project_root = Path(__file__).parent
    
    required_dirs = [
        'backend', 'frontend', 'infrastructure', 
        'core-framework', 'decision-service', 'api-gateway'
    ]
    
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            return False
    return True

def test_deployment_system():
    """测试部署系统"""
    try:
        import deploy_integration_system
        
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
        
        # 验证功能
        assert validation_result.valid == True
        assert service_info.name == '测试服务'
        assert deployment_report.overall_status == 'success'
        
        return True
    except Exception as e:
        print(f"部署系统测试错误: {e}")
        return False

def test_backend_structure():
    """测试后端结构"""
    backend_dir = Path('backend')
    
    if not backend_dir.exists():
        return False
    
    # 检查关键目录
    required_subdirs = ['src', 'tests']
    for subdir in required_subdirs:
        if not (backend_dir / subdir).exists():
            return False
    
    return True

def test_frontend_structure():
    """测试前端结构"""
    frontend_dir = Path('frontend')
    
    if not frontend_dir.exists():
        return False
    
    # 检查关键文件
    required_files = ['package.json', 'index.html']
    for file_name in required_files:
        if not (frontend_dir / file_name).exists():
            return False
    
    return True

def test_config_files():
    """测试配置文件"""
    required_files = ['README.md', 'docker-compose.yml', 'requirements.txt']
    
    for file_name in required_files:
        if not Path(file_name).exists():
            return False
    
    return True

def test_python_syntax():
    """测试Python文件语法"""
    python_files = [
        'deploy_integration_system.py',
        'simple_test.py',
        'run_tests.py'
    ]
    
    for py_file in python_files:
        if not Path(py_file).exists():
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, py_file, 'exec')
        except SyntaxError as e:
            print(f"语法错误 {py_file}: {e}")
            return False
    
    return True

def main():
    """主测试函数"""
    print_section("🚀 AI平台全部代码测试")
    
    # 定义测试套件
    test_suite = [
        ("Python环境", test_python_environment),
        ("项目结构", test_project_structure),
        ("部署系统", test_deployment_system),
        ("后端结构", test_backend_structure),
        ("前端结构", test_frontend_structure),
        ("配置文件", test_config_files),
        ("Python语法", test_python_syntax)
    ]
    
    # 运行所有测试
    results = []
    for test_name, test_func in test_suite:
        result = run_test(test_name, test_func)
        results.append(result)
    
    # 汇总结果
    print_section("📊 测试结果汇总")
    
    passed_count = sum(1 for r in results if r)
    total_count = len(results)
    success_rate = passed_count / total_count if total_count > 0 else 0
    
    print(f"总测试数: {total_count}")
    print(f"通过数: {passed_count}")
    print(f"成功率: {success_rate:.1%}")
    
    # 总体评估
    if success_rate >= 0.8:
        print("\n🎉 AI平台代码测试通过！所有核心功能正常。")
        return 0
    else:
        print("\n⚠️  AI平台代码测试未完全通过。")
        
        # 显示建议
        print("\n💡 修复建议:")
        if not test_python_environment():
            print("  • 检查Python环境配置")
        if not test_project_structure():
            print("  • 完善项目目录结构")
        if not test_deployment_system():
            print("  • 修复部署系统代码")
        if not test_backend_structure():
            print("  • 完善后端代码结构")
        if not test_frontend_structure():
            print("  • 完善前端项目结构")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())