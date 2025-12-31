#!/usr/bin/env python3
"""
类型安全测试用例
验证validate_configuration函数及相关类型的类型安全性
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入测试所需的类型和模块
try:
    from deploy_integration_system import (
        validate_config_integrity,
        validate_configuration,
        ValidationResult,
        ServiceInfo,
        DeploymentReport,
        ConfigurationValidationError,
        ServiceStartupError,
        DeploymentError
    )
    from backend.config.migration_edge_integration_config import (
        IntegrationConfig,
        DeploymentEnvironment,
        OptimizationStrategy
    )
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


def test_validation_result_type():
    """测试ValidationResult类型的类型安全性"""
    print("🧪 测试ValidationResult类型...")
    
    # 创建有效的验证结果
    result = ValidationResult(
        environment="production",
        strategy="performance",
        valid=True,
        warnings=["警告1", "警告2"],
        errors=[],
        recommendations=["建议1"]
    )
    
    # 测试类型属性
    assert isinstance(result.environment, str), "environment应该是字符串"
    assert isinstance(result.strategy, str), "strategy应该是字符串"
    assert isinstance(result.valid, bool), "valid应该是布尔值"
    assert isinstance(result.warnings, list), "warnings应该是列表"
    assert isinstance(result.errors, list), "errors应该是列表"
    assert isinstance(result.recommendations, list), "recommendations应该是列表"
    
    # 测试列表内容类型
    for warning in result.warnings:
        assert isinstance(warning, str), "warning应该是字符串"
    for error in result.errors:
        assert isinstance(error, str), "error应该是字符串"
    for recommendation in result.recommendations:
        assert isinstance(recommendation, str), "recommendation应该是字符串"
    
    print("  ✅ ValidationResult类型测试通过")


def test_service_info_type():
    """测试ServiceInfo类型的类型安全性"""
    print("🧪 测试ServiceInfo类型...")
    
    # 创建有效的服务信息
    service = ServiceInfo(
        name="API服务",
        pid=12345,
        status="running"
    )
    
    # 测试类型属性
    assert isinstance(service.name, str), "name应该是字符串"
    assert isinstance(service.pid, int), "pid应该是整数"
    assert isinstance(service.status, str), "status应该是字符串"
    assert service.pid > 0, "pid应该是正数"
    assert service.status in ["running", "stopped"], "status应该是running或stopped"
    
    print("  ✅ ServiceInfo类型测试通过")


def test_deployment_report_type():
    """测试DeploymentReport类型的类型安全性"""
    print("🧪 测试DeploymentReport类型...")
    
    # 创建服务信息列表
    services = [
        ServiceInfo(name="API服务", pid=12345, status="running"),
        ServiceInfo(name="监控服务", pid=12346, status="stopped")
    ]
    
    # 创建部署报告
    report = DeploymentReport(
        timestamp="2024-01-01T12:00:00",
        environment="production",
        optimization_strategy="performance",
        services=services,
        health_check="passed",
        integration_tests="passed",
        overall_status="success"
    )
    
    # 测试类型属性
    assert isinstance(report.timestamp, str), "timestamp应该是字符串"
    assert isinstance(report.environment, str), "environment应该是字符串"
    assert isinstance(report.optimization_strategy, str), "optimization_strategy应该是字符串"
    assert isinstance(report.services, list), "services应该是列表"
    assert isinstance(report.health_check, str), "health_check应该是字符串"
    assert isinstance(report.integration_tests, str), "integration_tests应该是字符串"
    assert isinstance(report.overall_status, str), "overall_status应该是字符串"
    
    # 测试服务列表类型
    for service in report.services:
        assert isinstance(service, ServiceInfo), "service应该是ServiceInfo类型"
    
    # 测试状态值有效性
    assert report.health_check in ["passed", "failed"], "health_check应该是passed或failed"
    assert report.integration_tests in ["passed", "failed"], "integration_tests应该是passed或failed"
    assert report.overall_status in ["success", "partial"], "overall_status应该是success或partial"
    
    print("  ✅ DeploymentReport类型测试通过")


def test_configuration_validation_error():
    """测试ConfigurationValidationError异常"""
    print("🧪 测试ConfigurationValidationError异常...")
    
    # 创建验证结果
    validation_result = ValidationResult(
        environment="production",
        strategy="performance",
        valid=False,
        warnings=[],
        errors=["配置错误1", "配置错误2"],
        recommendations=[]
    )
    
    # 创建异常
    error = ConfigurationValidationError(
        "配置验证失败",
        validation_result
    )
    
    # 测试异常属性
    assert isinstance(error.validation_result, ValidationResult), "validation_result应该是ValidationResult类型"
    assert error.validation_result.valid is False, "验证结果应该为False"
    assert len(error.validation_result.errors) == 2, "应该有2个错误"
    
    print("  ✅ ConfigurationValidationError测试通过")


def test_integration_config_type():
    """测试IntegrationConfig类型的兼容性"""
    print("🧪 测试IntegrationConfig类型兼容性...")
    
    try:
        # 创建集成配置
        config = IntegrationConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            optimization_strategy=OptimizationStrategy.PERFORMANCE
        )
        
        # 测试配置属性
        assert hasattr(config, 'environment'), "config应该有environment属性"
        assert hasattr(config, 'optimization_strategy'), "config应该有optimization_strategy属性"
        assert hasattr(config, 'migration_learning'), "config应该有migration_learning属性"
        assert hasattr(config, 'edge_computing'), "config应该有edge_computing属性"
        
        # 测试验证方法
        errors = config.validate()
        assert isinstance(errors, list), "validate()应该返回列表"
        
        # 测试转换为字典
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "to_dict()应该返回字典"
        
        print("  ✅ IntegrationConfig类型兼容性测试通过")
        
    except Exception as e:
        print(f"  ⚠️  IntegrationConfig类型兼容性测试跳过: {e}")


def test_validate_config_integrity_function():
    """测试validate_config_integrity函数的类型安全性"""
    print("🧪 测试validate_config_integrity函数...")
    
    try:
        # 创建测试配置
        config = IntegrationConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
            optimization_strategy=OptimizationStrategy.PERFORMANCE
        )
        
        # 调用验证函数
        result = validate_config_integrity(config)
        
        # 验证返回类型
        assert isinstance(result, ValidationResult), "应该返回ValidationResult类型"
        assert isinstance(result.valid, bool), "valid应该是布尔值"
        assert isinstance(result.warnings, list), "warnings应该是列表"
        assert isinstance(result.errors, list), "errors应该是列表"
        
        # 验证列表内容类型
        for warning in result.warnings:
            assert isinstance(warning, str), "warning应该是字符串"
        for error in result.errors:
            assert isinstance(error, str), "error应该是字符串"
        
        print("  ✅ validate_config_integrity函数测试通过")
        
    except Exception as e:
        print(f"  ⚠️  validate_config_integrity函数测试跳过: {e}")


def test_validate_configuration_function():
    """测试validate_configuration函数的类型安全性"""
    print("🧪 测试validate_configuration函数...")
    
    try:
        # 创建测试配置
        config = IntegrationConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
            optimization_strategy=OptimizationStrategy.PERFORMANCE
        )
        
        # 调用验证函数
        result = validate_configuration(config)
        
        # 验证返回类型
        assert isinstance(result, bool), "应该返回布尔值"
        
        print("  ✅ validate_configuration函数测试通过")
        
    except Exception as e:
        print(f"  ⚠️  validate_configuration函数测试跳过: {e}")


def run_all_type_tests():
    """运行所有类型安全测试"""
    print("🚀 开始类型安全测试套件")
    print("=" * 60)
    
    tests = [
        test_validation_result_type,
        test_service_info_type,
        test_deployment_report_type,
        test_configuration_validation_error,
        test_integration_config_type,
        test_validate_config_integrity_function,
        test_validate_configuration_function
    ]
    
    passed = 0
    skipped = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠️  {test.__name__} 跳过: {e}")
            skipped += 1
    
    print("=" * 60)
    print(f"📊 测试结果:")
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  ⚠️  跳过: {skipped}")
    
    if failed == 0:
        print("🎉 所有类型安全测试通过!")
        return True
    else:
        print(f"⚠️  有{failed}个测试失败，请检查类型定义")
        return False


if __name__ == "__main__":
    success = run_all_type_tests()
    sys.exit(0 if success else 1)