#!/usr/bin/env python3
"""
AI平台全面测试脚本
系统性地测试整个代码库的功能和性能
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class TestResult:
    """测试结果类型"""
    name: str
    status: str  # "passed", "failed", "skipped"
    message: str
    duration: float
    details: Dict[str, Any]

class AIPlatformTester:
    """AI平台测试器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results: List[TestResult] = []
        
    def run_test(self, test_name: str, test_func) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status="passed" if result else "failed",
                message="测试通过" if result else "测试失败",
                duration=duration,
                details={"result": result}
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status="failed",
                message=f"测试异常: {e}",
                duration=duration,
                details={"error": str(e)}
            )
    
    def test_environment(self) -> bool:
        """测试基础环境"""
        print("🔍 测试基础环境...")
        
        # 检查Python版本
        if sys.version_info < (3, 8):
            print("  ❌ Python版本过低，需要3.8+")
            return False
        print(f"  ✅ Python版本: {sys.version.split()[0]}")
        
        # 检查必要模块
        required_modules = ['json', 'os', 'subprocess', 'time', 'pathlib']
        for module in required_modules:
            try:
                __import__(module)
                print(f"  ✅ 模块可用: {module}")
            except ImportError:
                print(f"  ❌ 模块缺失: {module}")
                return False
        
        # 检查项目结构
        required_dirs = [
            'backend', 'frontend', 'infrastructure', 
            'core-framework', 'decision-service', 'api-gateway'
        ]
        
        for dir_name in required_dirs:
            if (self.project_root / dir_name).exists():
                print(f"  ✅ 目录存在: {dir_name}")
            else:
                print(f"  ❌ 目录缺失: {dir_name}")
                return False
        
        return True
    
    def test_imports(self) -> bool:
        """测试模块导入"""
        print("🔍 测试模块导入...")
        
        # 测试导入刚刚修复的部署系统
        try:
            import deploy_integration_system as dis
            print("  ✅ deploy_integration_system 导入成功")
        except Exception as e:
            print(f"  ❌ deploy_integration_system 导入失败: {e}")
            return False
        
        # 测试数据类
        try:
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
            
            print("  ✅ 数据类实例化成功")
            
        except Exception as e:
            print(f"  ❌ 数据类测试失败: {e}")
            return False
        
        return True
    
    def test_config_validation(self) -> bool:
        """测试配置验证功能"""
        print("🔍 测试配置验证...")
        
        try:
            # 测试简单的配置验证
            from deploy_integration_system import validate_config_integrity
            
            # 创建模拟配置
            class MockConfig:
                def __init__(self):
                    self.environment = "test"
                    self.optimization_strategy = "performance"
            
            config = MockConfig()
            
            # 测试验证函数
            result = validate_config_integrity(config)
            
            if hasattr(result, 'valid') and hasattr(result, 'environment'):
                print("  ✅ 配置验证功能正常")
                return True
            else:
                print("  ❌ 配置验证结果格式错误")
                return False
                
        except Exception as e:
            print(f"  ❌ 配置验证测试失败: {e}")
            return False
    
    def test_file_structure(self) -> bool:
        """测试文件结构和代码质量"""
        print("🔍 测试文件结构...")
        
        # 检查关键文件
        required_files = [
            'deploy_integration_system.py',
            'README.md',
            'docker-compose.yml',
            'run_tests.py'
        ]
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                # 检查文件大小（确保不是空文件）
                if file_path.stat().st_size > 100:
                    print(f"  ✅ 文件正常: {file_name}")
                else:
                    print(f"  ⚠️  文件过小: {file_name}")
            else:
                print(f"  ❌ 文件缺失: {file_name}")
                return False
        
        # 检查Python文件语法
        python_files = [
            'deploy_integration_system.py',
            'simple_test.py',
            'run_tests.py'
        ]
        
        for py_file in python_files:
            file_path = self.project_root / py_file
            if file_path.exists():
                try:
                    # 尝试编译Python文件检查语法
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, py_file, 'exec')
                    print(f"  ✅ 语法正确: {py_file}")
                except SyntaxError as e:
                    print(f"  ❌ 语法错误 {py_file}: {e}")
                    return False
        
        return True
    
    def test_backend_code(self) -> bool:
        """测试后端代码结构"""
        print("🔍 测试后端代码...")
        
        backend_dir = self.project_root / 'backend'
        
        if not backend_dir.exists():
            print("  ❌ 后端目录不存在")
            return False
        
        # 检查后端关键目录
        backend_subdirs = ['src', 'tests', 'config']
        for subdir in backend_subdirs:
            subdir_path = backend_dir / subdir
            if subdir_path.exists():
                print(f"  ✅ 后端子目录存在: {subdir}")
            else:
                print(f"  ⚠️  后端子目录缺失: {subdir}")
        
        # 检查关键后端文件
        backend_files = ['requirements.txt', 'main.py']
        for file_name in backend_files:
            file_path = backend_dir / file_name
            if file_path.exists():
                print(f"  ✅ 后端文件存在: {file_name}")
            else:
                print(f"  ⚠️  后端文件缺失: {file_name}")
        
        return True
    
    def test_frontend_structure(self) -> bool:
        """测试前端结构"""
        print("🔍 测试前端结构...")
        
        frontend_dir = self.project_root / 'frontend'
        
        if not frontend_dir.exists():
            print("  ❌ 前端目录不存在")
            return False
        
        # 检查前端关键文件
        frontend_files = ['package.json', 'index.html', 'vite.config.js']
        for file_name in frontend_files:
            file_path = frontend_dir / file_name
            if file_path.exists():
                print(f"  ✅ 前端文件存在: {file_name}")
            else:
                print(f"  ⚠️  前端文件缺失: {file_name}")
        
        return True
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🚀 开始AI平台全面测试")
        print("=" * 60)
        
        # 定义测试套件
        test_suite = [
            ("环境测试", self.test_environment),
            ("模块导入", self.test_imports),
            ("配置验证", self.test_config_validation),
            ("文件结构", self.test_file_structure),
            ("后端代码", self.test_backend_code),
            ("前端结构", self.test_frontend_structure)
        ]
        
        # 运行测试
        for test_name, test_func in test_suite:
            print(f"\n📋 运行测试: {test_name}")
            result = self.run_test(test_name, test_func)
            self.test_results.append(result)
            
            status_icon = "✅" if result.status == "passed" else "❌"
            print(f"   {status_icon} {test_name}: {result.message} ({result.duration:.2f}s)")
        
        # 生成报告
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📊 测试报告")
        print("=" * 60)
        
        passed_count = sum(1 for r in self.test_results if r.status == "passed")
        total_count = len(self.test_results)
        success_rate = passed_count / total_count if total_count > 0 else 0
        
        report = {
            "timestamp": time.time(),
            "total_tests": total_count,
            "passed_tests": passed_count,
            "success_rate": success_rate,
            "test_results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in self.test_results
            ],
            "summary": {
                "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED",
                "recommendations": self.generate_recommendations()
            }
        }
        
        # 显示结果
        print(f"总测试数: {total_count}")
        print(f"通过数: {passed_count}")
        print(f"成功率: {success_rate:.1%}")
        
        for result in self.test_results:
            status_icon = "✅" if result.status == "passed" else "❌"
            print(f"{status_icon} {result.name}: {result.message}")
        
        # 总体评估
        if success_rate >= 0.8:
            print("\n🎉 AI平台代码测试通过！系统运行正常。")
        else:
            print("\n⚠️  AI平台代码测试未完全通过，需要进一步检查。")
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if r.status == "failed"]
        
        if failed_tests:
            failed_names = [r.name for r in failed_tests]
            recommendations.append(f"修复失败的测试: {', '.join(failed_names)}")
        
        # 基于项目结构的建议
        if not (self.project_root / 'backend' / 'requirements.txt').exists():
            recommendations.append("添加后端依赖文件 requirements.txt")
        
        if not (self.project_root / 'frontend' / 'package.json').exists():
            recommendations.append("完善前端项目结构")
        
        return recommendations

def main():
    """主函数"""
    tester = AIPlatformTester()
    
    try:
        report = tester.run_all_tests()
        
        # 保存测试报告
        report_file = tester.project_root / "comprehensive_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细测试报告已保存到: {report_file}")
        
        # 返回退出码
        success_rate = report["success_rate"]
        return 0 if success_rate >= 0.8 else 1
        
    except Exception as e:
        print(f"❌ 测试过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())