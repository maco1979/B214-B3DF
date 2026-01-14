#!/usr/bin/env python3
"""
系统集成测试

测试整个AGI农业系统的核心功能集成
"""

import pytest
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_cross_domain_transfer(mock_dependencies):
    """测试跨领域知识迁移功能"""
    logger.info("测试跨领域知识迁移功能...")
    
    from src.core.cross_domain_transfer import cross_domain_transfer_service
    
    # 测试获取迁移统计信息
    stats = cross_domain_transfer_service.get_transfer_statistics()
    assert isinstance(stats, dict), "迁移统计信息应为字典类型"
    assert "total_transfers" in stats, "迁移统计信息应包含total_transfers字段"
    
    logger.info("✅ 跨领域知识迁移功能测试通过")

@pytest.mark.asyncio
async def test_meta_cognitive_control(mock_dependencies):
    """测试元认知控制功能"""
    logger.info("测试元认知控制功能...")
    
    from src.core.meta_cognitive_controller import meta_cognitive_system
    
    # 初始化系统
    await meta_cognitive_system.initialize()
    
    # 执行系统更新
    meta_cognitive_system.update()
    
    # 获取系统状态
    status = meta_cognitive_system.get_status()
    assert isinstance(status, dict), "系统状态应为字典类型"
    assert "self_awareness_level" in status, "系统状态应包含self_awareness_level字段"
    
    logger.info("✅ 元认知控制功能测试通过")

@pytest.mark.asyncio
async def test_performance_monitor(mock_dependencies):
    """测试性能监控功能"""
    logger.info("测试性能监控功能...")
    
    from src.performance.performance_monitor import PerformanceMonitor
    
    # 创建性能监控实例
    monitor = PerformanceMonitor(max_history_size=1000)
    
    # 记录测试指标
    await monitor.record_metric(
        metric_type="cpu_usage",
        value=40.0,
        unit="%",
        source="test"
    )
    
    await monitor.record_metric(
        metric_type="memory_usage",
        value=60.0,
        unit="%",
        source="test"
    )
    
    # 获取系统性能报告
    report = monitor.get_system_performance_report()
    assert isinstance(report, dict), "性能报告应为字典类型"
    assert "overall_health" in report, "性能报告应包含overall_health字段"
    
    logger.info("✅ 性能监控功能测试通过")

@pytest.mark.asyncio
async def test_emotional_processing(mock_dependencies):
    """测试情感处理功能"""
    logger.info("测试情感处理功能...")
    
    from src.core.services.hume_evi_service import hume_evi_service
    
    # 测试文本情感分析
    test_text = "我很高兴今天天气这么好，心情非常愉快！"
    emotions = await hume_evi_service.analyze_emotions(test_text)
    assert isinstance(emotions, dict), "情感分析结果应为字典类型"
    assert "joy" in emotions, "情感分析结果应包含joy字段"
    assert emotions["joy"] > 0.5, "积极文本应检测到较高的joy值"
    
    # 测试获取主导情绪
    dominant_emotions = hume_evi_service.get_dominant_emotions(emotions, top_n=3)
    assert isinstance(dominant_emotions, list), "主导情绪应为列表类型"
    assert len(dominant_emotions) > 0, "应检测到至少一种主导情绪"
    
    logger.info("✅ 情感处理功能测试通过")

@pytest.mark.asyncio
async def test_creativity_service(mock_dependencies):
    """测试创造性内容生成功能"""
    logger.info("测试创造性内容生成功能...")
    
    from src.core.services.creativity_service import creativity_service
    
    # 测试生成创造性想法
    prompt = "农业病虫害防治"
    ideas = creativity_service.generate_creative_ideas(prompt, num_ideas=2)
    assert isinstance(ideas, list), "创造性想法应为列表类型"
    assert len(ideas) > 0, "应生成至少一个创造性想法"
    
    # 测试生成创造性故事
    story_prompt = "未来农业机器人"
    story = creativity_service.generate_creative_story(story_prompt, genre="科幻", length=100)
    assert isinstance(story, dict), "故事生成结果应为字典类型"
    assert "story" in story, "故事生成结果应包含story字段"
    
    logger.info("✅ 创造性内容生成功能测试通过")

@pytest.mark.asyncio
async def test_ethical_rule_engine(mock_dependencies):
    """测试伦理规则引擎功能"""
    logger.info("测试伦理规则引擎功能...")
    
    from src.core.ethical_rule_engine import get_ethical_rule_engine
    
    # 获取伦理规则引擎实例
    ethical_engine = get_ethical_rule_engine()
    
    # 测试伦理决策评估
    test_action = "帮助用户解决农业病虫害问题"
    test_context = {"user": "农民", "domain": "农业", "task": "病虫害防治"}
    
    result = ethical_engine.evaluate_ethical_decision(test_action, test_context)
    assert isinstance(result, dict), "伦理评估结果应为字典类型"
    assert "ethical_evaluation" in result, "伦理评估结果应包含ethical_evaluation字段"
    assert "rule_execution_results" in result, "伦理评估结果应包含rule_execution_results字段"
    
    logger.info("✅ 伦理规则引擎功能测试通过")

@pytest.mark.asyncio
async def test_cognitive_architecture_integration(mock_dependencies):
    """测试认知架构集成功能"""
    logger.info("测试认知架构集成功能...")
    
    from src.core.cognitive_architecture import cognitive_architecture
    
    # 初始化认知架构
    await cognitive_architecture.initialize()
    
    # 测试输入处理
    test_input = "今天的收成太好了，我非常开心！"
    result = cognitive_architecture.process_input(test_input)
    
    assert isinstance(result, dict), "输入处理结果应为字典类型"
    assert "text" in result, "输入处理结果应包含text字段"
    
    logger.info("✅ 认知架构集成功能测试通过")

@pytest.mark.asyncio
async def test_ai_model_service(mock_dependencies):
    """测试AI模型服务功能"""
    logger.info("测试AI模型服务功能...")
    
    from src.core.services.ai_model_service import aimodel_service
    
    # 测试意图识别
    test_input = "生成一些农业创新的创意"
    intent = aimodel_service.recognize_intent(test_input)
    
    assert isinstance(intent, str), "意图识别结果应为字符串类型"
    
    # 测试生成响应
    test_context = {"text": test_input}
    response = aimodel_service.generate_response(test_context)
    
    assert isinstance(response, dict), "生成响应结果应为字典类型"
    assert "text" in response, "生成响应结果应包含text字段"
    
    logger.info("✅ AI模型服务功能测试通过")

@pytest.mark.asyncio
async def test_comvas_service_integration(mock_dependencies):
    """测试ComVas服务集成功能"""
    logger.info("测试ComVas服务集成功能...")
    
    from src.core.services.comvas_service import comvas_service
    
    # 测试获取当前价值系统
    current_system = comvas_service.current_value_system
    assert isinstance(current_system, str), "当前价值系统应为字符串类型"
    
    # 测试获取价值系统
    value_system = comvas_service.get_value_system(current_system)
    assert isinstance(value_system, dict), "价值系统应为字典类型"
    assert "name" in value_system, "价值系统应包含name字段"
    
    logger.info("✅ ComVas服务集成功能测试通过")

@pytest.mark.asyncio
async def test_logical_consistency_check(mock_dependencies):
    """测试逻辑一致性检查功能"""
    logger.info("测试逻辑一致性检查功能...")
    
    from src.core.logical_consistency.consistency_checker import LogicalConsistencyChecker
    
    # 创建逻辑一致性检查器实例
    checker = LogicalConsistencyChecker()
    
    # 测试一致性检查
    test_decision = "使用农药A防治病虫害"
    test_history = ["农药A对作物安全", "农药A有效防治病虫害"]
    
    result = checker.check_consistency(test_decision, test_history)
    assert isinstance(result, dict), "一致性检查结果应为字典类型"
    assert "is_consistent" in result, "一致性检查结果应包含is_consistent字段"
    assert "consistency_score" in result, "一致性检查结果应包含consistency_score字段"
    
    logger.info("✅ 逻辑一致性检查功能测试通过")

@pytest.mark.asyncio
async def test_system_health_check(mock_dependencies):
    """测试系统健康检查功能"""
    logger.info("测试系统健康检查功能...")
    
    from src.performance.performance_monitor import PerformanceMonitor
    
    # 创建性能监控实例
    monitor = PerformanceMonitor(max_history_size=1000)
    
    # 获取健康状态
    health_status = monitor.get_health_status()
    
    assert isinstance(health_status, dict), "健康状态应为字典类型"
    assert "status" in health_status, "健康状态应包含status字段"
    assert "health_score" in health_status, "健康状态应包含health_score字段"
    
    logger.info("✅ 系统健康检查功能测试通过")
