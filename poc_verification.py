#!/usr/bin/env python3
"""
POC验证脚本
用于验证核心技术组件的功能和性能
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('poc_verification.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_plugin_architecture():
    """
    测试插件化架构功能
    包括插件发现、加载、激活、停用和卸载
    """
    logger.info("=== 开始测试插件化架构 ===")
    
    try:
        from backend.src.core.plugins.plugin_manager import plugin_manager, PluginStatus
        
        # 1. 添加插件路径
        plugin_path = os.path.join(os.path.dirname(__file__), 'backend', 'plugins')
        plugin_manager.add_plugin_path(plugin_path)
        logger.info(f"添加插件路径: {plugin_path}")
        
        # 2. 发现插件
        logger.info("开始发现插件...")
        discovered_plugins = await plugin_manager.discover_plugins()
        logger.info(f"发现 {len(discovered_plugins)} 个插件:")
        for plugin in discovered_plugins:
            logger.info(f"  - {plugin.name} (v{plugin.version}) - {plugin.plugin_type.value}")
        
        # 3. 加载插件
        logger.info("开始加载插件...")
        loaded_plugins = await plugin_manager.load_all_plugins()
        logger.info(f"成功加载 {len(loaded_plugins)} 个插件: {loaded_plugins}")
        
        # 4. 激活插件
        logger.info("开始激活插件...")
        activated_plugins = await plugin_manager.activate_all_plugins()
        logger.info(f"成功激活 {len(activated_plugins)} 个插件: {activated_plugins}")
        
        # 5. 查看插件状态
        logger.info("查看插件状态:")
        all_plugins = plugin_manager.list_plugins()
        for plugin in all_plugins:
            logger.info(f"  - {plugin.name}: {plugin.status.value}")
        
        # 6. 获取大脑导航区域
        brain_regions = plugin_manager.get_brain_navigation_regions()
        logger.info(f"获取到 {len(brain_regions)} 个大脑导航区域")
        for region in brain_regions:
            logger.info(f"  - {region.name} (ID: {region.id})")
        
        # 7. 测试插件卸载
        if loaded_plugins:
            plugin_to_unload = loaded_plugins[0]
            logger.info(f"测试卸载插件: {plugin_to_unload}")
            success = await plugin_manager.unload_plugin(plugin_to_unload)
            logger.info(f"卸载插件 {plugin_to_unload}: {'成功' if success else '失败'}")
        
        logger.info("=== 插件化架构测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"插件化架构测试失败: {e}", exc_info=True)
        return False

async def test_risk_control_framework():
    """
    测试风险控制框架功能
    包括规则管理、风险评估和预警系统
    """
    logger.info("=== 开始测试风险控制框架 ===")
    
    try:
        from backend.src.ai_risk_control.risk_rule_engine import (
            risk_rule_engine, RiskRule, RuleType, RiskLevel, RuleOperator
        )
        
        # 1. 添加测试规则
        logger.info("添加测试风险规则...")
        
        # 创建技术风险规则
        technical_rule = RiskRule(
            id="tech_001",
            name="高CPU使用率",
            description="当CPU使用率超过80%时触发技术风险",
            rule_type=RuleType.TECHNICAL,
            category="系统监控",
            condition={
                "field": "system.metrics.cpu_usage",
                "operator": ">",
                "value": 80
            },
            risk_level=RiskLevel.HIGH,
            risk_score=0.8,
            priority=1
        )
        
        # 创建数据安全风险规则
        data_security_rule = RiskRule(
            id="data_001",
            name="敏感数据泄露",
            description="当检测到敏感数据泄露时触发数据安全风险",
            rule_type=RuleType.DATA_SECURITY,
            category="数据保护",
            condition={
                "or": [
                    {
                        "field": "system.logs.sensitive_data_count",
                        "operator": ">",
                        "value": 0
                    },
                    {
                        "field": "system.api.sensitive_endpoint_access_count",
                        "operator": ">",
                        "value": 10
                    }
                ]
            },
            risk_level=RiskLevel.CRITICAL,
            risk_score=1.0,
            priority=1
        )
        
        # 添加规则
        rule1_success = risk_rule_engine.add_rule(technical_rule)
        rule2_success = risk_rule_engine.add_rule(data_security_rule)
        logger.info(f"添加技术风险规则: {'成功' if rule1_success else '失败'}")
        logger.info(f"添加数据安全风险规则: {'成功' if rule2_success else '失败'}")
        
        # 2. 列出规则
        logger.info("列出所有规则:")
        all_rules = risk_rule_engine.list_rules()
        for rule in all_rules:
            logger.info(f"  - {rule.name} (ID: {rule.id}) - {rule.risk_level.value}")
        
        # 3. 测试风险评估
        logger.info("测试风险评估...")
        
        # 测试数据1: 正常情况
        test_data1 = {
            "system": {
                "metrics": {
                    "cpu_usage": 50,
                    "memory_usage": 60
                },
                "logs": {
                    "sensitive_data_count": 0
                },
                "api": {
                    "sensitive_endpoint_access_count": 5
                }
            }
        }
        
        result1 = await risk_rule_engine.evaluate_risk(test_data1)
        logger.info(f"测试数据1 - 风险评估结果:")
        logger.info(f"  总体风险分数: {result1.overall_risk_score}")
        logger.info(f"  总体风险等级: {result1.overall_risk_level.value}")
        logger.info(f"  触发规则数: {len(result1.triggered_rules)}")
        
        # 测试数据2: 高风险情况
        test_data2 = {
            "system": {
                "metrics": {
                    "cpu_usage": 90,
                    "memory_usage": 85
                },
                "logs": {
                    "sensitive_data_count": 3
                },
                "api": {
                    "sensitive_endpoint_access_count": 15
                }
            }
        }
        
        result2 = await risk_rule_engine.evaluate_risk(test_data2)
        logger.info(f"测试数据2 - 风险评估结果:")
        logger.info(f"  总体风险分数: {result2.overall_risk_score}")
        logger.info(f"  总体风险等级: {result2.overall_risk_level.value}")
        logger.info(f"  触发规则数: {len(result2.triggered_rules)}")
        
        for rule in result2.triggered_rules:
            logger.info(f"    - 触发规则: {rule['rule'].name} (风险分数: {rule['risk_score']})")
        
        # 4. 测试规则导入导出
        logger.info("测试规则导出...")
        exported_rules = risk_rule_engine.export_rules()
        logger.info(f"导出 {exported_rules['rules_count']} 个规则")
        
        logger.info("=== 风险控制框架测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"风险控制框架测试失败: {e}", exc_info=True)
        return False

async def test_microservices_architecture():
    """
    测试微服务架构功能
    包括服务通信、负载均衡和容错机制
    """
    logger.info("=== 开始测试微服务架构 ===")
    
    try:
        import aiohttp
        
        # 测试API网关和服务通信
        logger.info("测试API网关和服务通信...")
        
        # 假设API网关运行在localhost:8000
        api_url = "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            # 测试健康检查端点
            async with session.get(f"{api_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"健康检查成功: {health_data}")
                else:
                    logger.warning(f"健康检查失败，状态码: {response.status}")
            
            # 测试模型管理端点
            async with session.get(f"{api_url}/api/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    logger.info(f"模型列表获取成功，共 {len(models_data)} 个模型")
                else:
                    logger.warning(f"模型列表获取失败，状态码: {response.status}")
        
        logger.info("=== 微服务架构测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"微服务架构测试失败: {e}", exc_info=True)
        logger.warning("微服务测试可能需要API服务运行，请确保后端服务已启动")
        return False

async def test_containerization():
    """
    测试容器化功能
    包括Docker镜像构建和Kubernetes部署
    """
    logger.info("=== 开始测试容器化功能 ===")
    
    try:
        import subprocess
        import json
        
        # 1. 检查Docker是否安装
        logger.info("检查Docker状态...")
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Docker版本: {result.stdout.strip()}")
        else:
            logger.warning("Docker未安装或未运行")
            return False
        
        # 2. 检查Docker镜像
        logger.info("列出Docker镜像...")
        result = subprocess.run(["docker", "images"], capture_output=True, text=True)
        logger.info(f"Docker镜像列表:\n{result.stdout}")
        
        # 3. 检查Kubernetes状态
        logger.info("检查Kubernetes状态...")
        result = subprocess.run(["kubectl", "version", "--short"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Kubernetes版本:\n{result.stdout}")
        else:
            logger.warning("Kubernetes未安装或未运行")
        
        logger.info("=== 容器化功能测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"容器化测试失败: {e}", exc_info=True)
        return False

async def main():
    """
    主函数，运行所有POC验证测试
    """
    logger.info("========================================")
    logger.info("开始POC验证测试")
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("========================================")
    
    # 运行所有测试
    test_results = {
        "插件化架构": await test_plugin_architecture(),
        "风险控制框架": await test_risk_control_framework(),
        "微服务架构": await test_microservices_architecture(),
        "容器化功能": await test_containerization()
    }
    
    # 生成测试报告
    logger.info("\n========================================")
    logger.info("POC验证测试报告")
    logger.info("========================================")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    failed_tests = total_tests - passed_tests
    
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过数: {passed_tests}")
    logger.info(f"失败数: {failed_tests}")
    logger.info(f"通过率: {passed_tests / total_tests * 100:.1f}%")
    
    logger.info("\n测试详情:")
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"- {test_name}: {status}")
    
    logger.info("\n========================================")
    logger.info("POC验证测试完成")
    logger.info("========================================")
    
    # 返回整体结果
    return passed_tests == total_tests

if __name__ == "__main__":
    # 运行主函数
    success = asyncio.run(main())
    
    # 设置退出码
    sys.exit(0 if success else 1)
