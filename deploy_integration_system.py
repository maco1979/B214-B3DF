#!/usr/bin/env python3
"""
迁移学习和边缘计算集成系统部署脚本
自动化部署完整的集成系统，包括性能监控和优化功能
"""

import os
import sys
import json
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 为类型检查器提供明确的类型定义
if TYPE_CHECKING:
    from backend.config.migration_edge_integration_config import (
        IntegrationConfigManager, 
        DeploymentEnvironment, 
        OptimizationStrategy,
        IntegrationConfig
    )
else:
    # 运行时占位符类
    class IntegrationConfigManager:
        def get_optimized_config(self, env, strategy):
            return None
        
        def validate_deployment(self, config):
            return {"valid": True, "warnings": [], "recommendations": [], "errors": []}
    
    class DeploymentEnvironment:
        def __init__(self, value):
            self.value = value
    
    class OptimizationStrategy:
        def __init__(self, value):
            self.value = value
    
    class IntegrationConfig:
        def __init__(self, environment=None, optimization_strategy=None):
            self.environment = environment or DeploymentEnvironment("production")
            self.optimization_strategy = optimization_strategy or OptimizationStrategy("performance")
        
        def to_dict(self):
            return {}

# 精确的类型定义
@dataclass
class ValidationResult:
    """配置验证结果类型"""
    environment: str
    strategy: str
    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

@dataclass
class ServiceInfo:
    """服务信息类型"""
    name: str
    pid: int
    status: str
    process: Optional[subprocess.Popen[bytes]] = None

@dataclass
class DeploymentReport:
    """部署报告类型"""
    timestamp: str
    environment: str
    optimization_strategy: str
    services: list[ServiceInfo]
    health_check: str
    integration_tests: str
    overall_status: str

# 异常类型定义
class ConfigurationValidationError(Exception):
    """配置验证异常"""
    def __init__(self, message: str, validation_result: ValidationResult):
        super().__init__(message)
        self.validation_result = validation_result

class ServiceStartupError(Exception):
    """服务启动异常"""
    def __init__(self, message: str, service_name: str):
        super().__init__(f"{service_name}: {message}")
        self.service_name = service_name

class DeploymentError(Exception):
    """部署过程异常"""
    def __init__(self, message: str, step: str):
        super().__init__(f"{step}: {message}")
        self.step = step

# 实际导入配置模块
try:
    from backend.config.migration_edge_integration_config import (
        IntegrationConfigManager, 
        DeploymentEnvironment, 
        OptimizationStrategy
    )
except ImportError:
    # 使用占位符类
    print("⚠️  配置模块未找到，使用占位符实现")


def setup_environment():
    """设置部署环境"""
    print("🚀 设置部署环境...")
    
    # 创建必要的目录
    directories = [
        "logs",
        "data",
        "models",
        "config",
        "tmp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  创建目录: {directory}")
    
    # 设置环境变量
    env_vars = {
        "PYTHONPATH": str(project_root),
        "PROJECT_ROOT": str(project_root),
        "LOG_LEVEL": "INFO"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  设置环境变量: {key}={value}")
    
    print("✅ 环境设置完成")


def install_dependencies():
    """安装项目依赖"""
    print("📦 安装项目依赖...")
    
    # 检查并安装后端依赖
    if os.path.exists("backend/requirements.txt"):
        print("  安装后端依赖...")
        try:
            _ = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", "backend/requirements.txt"
            ], check=True)
            print("  ✅ 后端依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 后端依赖安装失败: {e}")
            return False
    
    # 检查并安装前端依赖
    if os.path.exists("frontend/package.json"):
        print("  安装前端依赖...")
        try:
            # 检查是否安装了Node.js
            _ = subprocess.run(["node", "--version"], check=True, capture_output=True)
            
            # 安装npm包
            os.chdir("frontend")
            _ = subprocess.run(["npm", "install"], check=True)
            os.chdir("..")
            print("  ✅ 前端依赖安装完成")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ⚠️  前端依赖安装跳过: {e}")
    
    print("✅ 依赖安装完成")
    return True


def load_configuration(environment: str, strategy: str) -> tuple[Optional[IntegrationConfig], Optional[str]]:
    """加载配置"""
    print("⚙️  加载配置...")
    
    config_manager = IntegrationConfigManager()
    
    try:
        # 获取优化配置
        config = config_manager.get_optimized_config(
            DeploymentEnvironment(environment),
            OptimizationStrategy(strategy)
        )
        
        # 保存配置到文件
        config_file = "config/integration_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 配置已保存到: {config_file}")
        print(f"  环境: {environment}")
        print(f"  优化策略: {strategy}")
        
        return config, config_file
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return None, None


def validate_config_integrity(config: IntegrationConfig) -> ValidationResult:
    """验证配置完整性，返回类型安全的验证结果"""
    config_manager = IntegrationConfigManager()
    raw_validation_result = config_manager.validate_deployment(config)
    
    # 类型安全的验证结果处理
    validation_result = ValidationResult(
        environment=str(raw_validation_result.get("environment", "")),
        strategy=str(raw_validation_result.get("strategy", "")),
        valid=bool(raw_validation_result.get("valid", False)),
        warnings=list(raw_validation_result.get("warnings", [])),
        errors=list(raw_validation_result.get("errors", [])),
        recommendations=list(raw_validation_result.get("recommendations", []))
    )
    
    return validation_result


def validate_configuration(config: IntegrationConfig) -> bool:
    """验证配置"""
    print("🔍 验证配置...")
    
    try:
        validation_result = validate_config_integrity(config)
        
        if validation_result.valid:
            print("  ✅ 配置验证通过")
            
            # 显示警告和建议
            if validation_result.warnings:
                print("  ⚠️  警告:")
                for warning in validation_result.warnings:
                    print(f"    - {warning}")
            
            if validation_result.recommendations:
                print("  💡 建议:")
                for recommendation in validation_result.recommendations:
                    print(f"    - {recommendation}")
            
            return True
        else:
            error_message = f"配置验证失败: {', '.join(validation_result.errors)}"
            raise ConfigurationValidationError(error_message, validation_result)
            
    except Exception as e:
        print(f"  ❌ 配置验证过程异常: {e}")
        return False


def start_backend_services(config: IntegrationConfig) -> list[ServiceInfo]:
    """启动后端服务"""
    print("🔧 启动后端服务...")
    
    services = [
        {
            "name": "API服务",
            "module": "src.api",
            "env": {
                "CONFIG_FILE": "config/integration_config.json"
            }
        },
        {
            "name": "性能监控服务",
            "module": "src.performance.performance_monitor",
            "env": {
                "PERFORMANCE_MONITORING_ENABLED": "true"
            }
        },
        {
            "name": "边缘计算同步服务",
            "module": "src.edge_computing.cloud_edge_sync",
            "env": {
                "EDGE_COMPUTING_ENABLED": str(getattr(getattr(config, 'edge_computing', None), 'enabled', False)).lower()
            }
        }
    ]
    
    processes: list[ServiceInfo] = []
    
    for service in services:
        print(f"  启动 {service['name']}...")
        
        # 设置环境变量
        env = os.environ.copy()
        service_env = service.get("env", {})
        # 确保env是字典类型，避免类型检查错误
        if isinstance(service_env, dict):
            env.update(service_env)
        
        try:
            # 确保模块路径是字符串类型
            module_path = str(service["module"]) if service["module"] else ""
            process = subprocess.Popen([
                sys.executable, "-m", module_path
            ], env=env, cwd="backend")
            
            # 类型安全的服务信息创建
            service_info = ServiceInfo(
                name=str(service["name"]),
                pid=int(process.pid) if process.pid else 0,
                status="running" if process.poll() is None else "stopped",
                process=process
            )
                
            processes.append(service_info)
            
            print(f"  ✅ {service['name']} 已启动 (PID: {process.pid})")
            
            # 给服务一些启动时间
            time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ 启动 {service['name']} 失败: {e}")
    
    return processes


def start_frontend_service():
    """启动前端服务"""
    print("🌐 启动前端服务...")
    
    if not os.path.exists("frontend/package.json"):
        print("  ⚠️  前端项目不存在，跳过前端启动")
        return None
    
    try:
        # 切换到前端目录
        os.chdir("frontend")
        
        # 启动开发服务器
        process = subprocess.Popen([
            "npm", "run", "dev"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 返回项目根目录
        os.chdir("..")
        
        print(f"  ✅ 前端服务已启动 (PID: {process.pid})")
        
        # 等待前端服务启动
        time.sleep(5)
        
        return process
        
    except Exception as e:
        print(f"  ❌ 启动前端服务失败: {e}")
        return None


def run_health_check():
    """运行健康检查"""
    print("🏥 运行健康检查...")
    
    health_endpoints = [
        ("API服务", "http://localhost:8000/system/health"),
        ("性能监控", "http://localhost:8000/performance/summary"),
    ]
    
    import requests
    
    all_healthy = True
    
    for service_name, endpoint in health_endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ {service_name} 健康检查通过")
            else:
                print(f"  ❌ {service_name} 健康检查失败: HTTP {response.status_code}")
                all_healthy = False
                
        except Exception as e:
            print(f"  ❌ {service_name} 健康检查失败: {e}")
            all_healthy = False
    
    return all_healthy


def run_integration_tests():
    """运行集成测试"""
    print("🧪 运行集成测试...")
    
    test_modules = [
        "tests.integration.test_migration_integration",
        "tests.integration.test_edge_integration",
        "tests.integration.test_decision_integration"
    ]
    
    all_passed = True
    
    for test_module in test_modules:
        print(f"  运行 {test_module}...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"backend/{test_module.replace('.', '/')}.py",
                "-v"
            ], cwd=".", capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ {test_module} 测试通过")
            else:
                print(f"  ❌ {test_module} 测试失败")
                print(f"    错误输出: {result.stderr}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ {test_module} 测试执行失败: {e}")
            all_passed = False
    
    return all_passed


def generate_deployment_report(config: IntegrationConfig, processes: list[ServiceInfo], health_check_passed: bool, tests_passed: bool) -> DeploymentReport:
    """生成部署报告"""
    print("📊 生成部署报告...")
    
    # 类型安全的部署报告创建
    report = DeploymentReport(
        timestamp=datetime.now().isoformat(),
        environment=str(config.environment.value),
        optimization_strategy=str(config.optimization_strategy.value),
        services=processes,
        health_check="passed" if health_check_passed else "failed",
        integration_tests="passed" if tests_passed else "failed",
        overall_status="success" if (health_check_passed and tests_passed) else "partial"
    )
    
    # 保存报告到JSON文件
    report_dict = {
        "timestamp": report.timestamp,
        "environment": report.environment,
        "optimization_strategy": report.optimization_strategy,
        "services": [
            {
                "name": service.name,
                "pid": service.pid,
                "status": service.status
            }
            for service in report.services
        ],
        "health_check": report.health_check,
        "integration_tests": report.integration_tests,
        "overall_status": report.overall_status
    }
    
    report_file = "logs/deployment_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 部署报告已保存到: {report_file}")
    
    # 显示摘要
    print("\n📋 部署摘要:")
    print(f"  环境: {config.environment.value}")
    print(f"  优化策略: {config.optimization_strategy.value}")
    
    # 类型安全的运行服务计数
    running_services = [s for s in report.services if s.status == "running"]
    print(f"  运行服务: {len(running_services)}")
    
    print(f"  健康检查: {report.health_check}")
    print(f"  集成测试: {report.integration_tests}")
    print(f"  总体状态: {report.overall_status}")
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='迁移学习和边缘计算集成系统部署脚本')
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'testing', 'staging', 'production', 'edge'],
                       default='development',
                       help='部署环境')
    parser.add_argument('--strategy', '-s',
                       choices=['performance', 'accuracy', 'resource_efficiency', 'latency', 'cost'],
                       default='performance',
                       help='优化策略')
    parser.add_argument('--skip-tests', action='store_true',
                       help='跳过集成测试')
    parser.add_argument('--skip-frontend', action='store_true',
                       help='跳过前端部署')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 迁移学习和边缘计算集成系统部署")
    print("=" * 60)
    
    try:
        # 1. 设置环境
        setup_environment()
        
        # 2. 安装依赖
        if not install_dependencies():
            print("❌ 依赖安装失败，退出部署")
            return 1
        
        # 3. 加载和验证配置
        config, _ = load_configuration(args.environment, args.strategy)
        if not config:
            print("❌ 配置加载失败，退出部署")
            return 1
        
        if not validate_configuration(config):
            print("❌ 配置验证失败，退出部署")
            return 1
        
        # 4. 启动后端服务
        backend_processes = start_backend_services(config)
        if not backend_processes:
            print("❌ 后端服务启动失败，退出部署")
            return 1
        
        # 5. 启动前端服务（可选）
        frontend_process = None
        if not args.skip_frontend:
            frontend_process = start_frontend_service()
        
        # 6. 等待服务启动
        print("⏳ 等待服务启动...")
        time.sleep(10)
        
        # 7. 运行健康检查
        health_check_passed = run_health_check()
        
        # 8. 运行集成测试（可选）
        tests_passed = True
        if not args.skip_tests:
            tests_passed = run_integration_tests()
        
        # 9. 生成部署报告
        report = generate_deployment_report(config, backend_processes, 
                                          health_check_passed, tests_passed)
        
        # 10. 显示部署结果
        print("\n" + "=" * 60)
        if report.overall_status == "success":
            print("🎉 部署成功完成!")
        else:
            print("⚠️  部署完成，但存在一些问题")
        
        print("\n📡 服务访问信息:")
        print("  后端API: http://localhost:8000")
        if frontend_process:
            print("  前端界面: http://localhost:5173")
        print("  性能监控: http://localhost:8000/performance")
        
        print("\n🔧 管理命令:")
        print("  停止服务: Ctrl+C")
        print("  查看日志: tail -f logs/*.log")
        
        print("\n💡 下一步:")
        print("  1. 访问前端界面验证功能")
        print("  2. 查看性能监控仪表板")
        print("  3. 运行基准测试验证性能")
        
        print("\n" + "=" * 60)
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号，停止服务...")
        
        # 清理进程
        print("🧹 清理进程...")
        for process_info in backend_processes:
            if process_info.process:
                try:
                    process_info.process.terminate()
                    process_info.process.wait(timeout=5)
                    print(f"  ✅ 停止 {process_info.name}")
                except:
                    process_info.process.kill()
                    print(f"  ⚠️  强制停止 {process_info.name}")
        
        if frontend_process:
            try:
                frontend_process.terminate()
                frontend_process.wait(timeout=5)
                print("  ✅ 停止前端服务")
            except:
                frontend_process.kill()
                print("  ⚠️  强制停止前端服务")
        
        print("✅ 清理完成")
        
        return 0 if report.overall_status == "success" else 1
        
    except Exception as e:
        print(f"❌ 部署过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())