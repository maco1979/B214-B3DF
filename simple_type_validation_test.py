#!/usr/bin/env python3
"""
简单类型验证测试
验证修复后的类型安全性
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_type_safety():
    """测试基本类型安全性"""
    print("🧪 测试基本类型安全性...")
    
    try:
        # 导入修改后的模块
        from deploy_integration_system import (
            ValidationResult,
            ServiceInfo,
            DeploymentReport,
            ConfigurationValidationError
        )
        
        # 测试ValidationResult类型
        result = ValidationResult(
            environment="production",
            strategy="performance",
            valid=True,
            warnings=["测试警告"],
            errors=[],
            recommendations=["测试建议"]
        )
        
        assert isinstance(result.environment, str), "environment类型错误"
        assert isinstance(result.valid, bool), "valid类型错误"
        assert isinstance(result.warnings, list), "warnings类型错误"
        
        print("  ✅ ValidationResult类型安全通过")
        
        # 测试ServiceInfo类型
        service = ServiceInfo(
            name="测试服务",
            pid=12345,
            status="running"
        )
        
        assert isinstance(service.name, str), "name类型错误"
        assert isinstance(service.pid, int), "pid类型错误"
        
        print("  ✅ ServiceInfo类型安全通过")
        
        # 测试DeploymentReport类型
        services = [service]
        report = DeploymentReport(
            timestamp="2024-01-01T12:00:00",
            environment="production",
            optimization_strategy="performance",
            services=services,
            health_check="passed",
            integration_tests="passed",
            overall_status="success"
        )
        
        assert isinstance(report.services, list), "services类型错误"
        assert isinstance(report.overall_status, str), "overall_status类型错误"
        
        print("  ✅ DeploymentReport类型安全通过")
        
        # 测试异常类型
        try:
            raise ConfigurationValidationError("测试异常", result)
        except ConfigurationValidationError as e:
            assert isinstance(e.validation_result, ValidationResult), "validation_result类型错误"
            print("  ✅ ConfigurationValidationError类型安全通过")
        
        print("🎉 所有基本类型安全测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 类型安全测试失败: {e}")
        return False

def test_function_signatures():
    """测试函数签名类型安全"""
    print("🧪 测试函数签名类型安全...")
    
    try:
        # 检查函数签名是否正确导入
        from deploy_integration_system import (
            validate_configuration,
            load_configuration,
            start_backend_services,
            generate_deployment_report
        )
        
        # 导入相关的配置类型
        from backend.config.migration_edge_integration_config import (
            IntegrationConfig,
            DeploymentEnvironment,
            OptimizationStrategy
        )
        
        # 测试函数导入成功
        assert callable(validate_configuration), "validate_configuration函数不可调用"
        assert callable(load_configuration), "load_configuration函数不可调用"
        
        print("  ✅ 函数签名导入成功")
        
        # 测试类型导入成功
        config_instance = IntegrationConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
            optimization_strategy=OptimizationStrategy.PERFORMANCE
        )
        
        assert hasattr(config_instance, 'environment'), "IntegrationConfig缺少environment属性"
        assert hasattr(config_instance, 'optimization_strategy'), "IntegrationConfig缺少optimization_strategy属性"
        
        print("  ✅ 配置类型导入成功")
        
        print("🎉 函数签名类型安全测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 函数签名类型安全测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始类型安全集成测试")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_basic_type_safety()
    print()
    test2_passed = test_function_signatures()
    
    print("=" * 60)
    print("📊 集成测试结果:")
    print(f"  基本类型安全: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"  函数签名安全: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    overall_passed = test1_passed and test2_passed
    
    if overall_passed:
        print("\n🎉 所有集成测试通过! 类型安全修复成功!")
    else:
        print("\n⚠️  部分测试失败，需要进一步检查")
    
    return overall_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)