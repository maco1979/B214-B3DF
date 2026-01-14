#!/usr/bin/env python3
"""
测试伦理规则引擎

验证伦理规则引擎的功能集成和效果
"""

import logging
from src.core.ethical_rule_engine import get_ethical_rule_engine, EthicalRuleType
from src.core.services.comvas_service import comvas_service
from src.core.rule_engine import Rule, Condition, Action, ActionType, ConditionOperator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ethical_rule_engine_init():
    """测试伦理规则引擎初始化"""
    logger.info("测试伦理规则引擎初始化...")
    
    # 获取伦理规则引擎实例
    ethical_engine = get_ethical_rule_engine()
    
    # 检查默认伦理规则是否添加成功
    ethical_rules = ethical_engine.get_ethical_rules()
    logger.info(f"默认伦理规则数量: {len(ethical_rules)}")
    
    for rule in ethical_rules:
        logger.info(f"伦理规则: {rule.name} (ID: {rule.id}, 优先级: {rule.priority})")
    
    # 检查动作函数是否注册成功
    action_functions = ethical_engine.action_functions.keys()
    logger.info(f"注册的动作函数: {action_functions}")
    
    assert len(ethical_rules) > 0, "没有添加默认伦理规则"
    assert "evaluate_ethical_decision" in action_functions, "伦理评估动作函数未注册"
    assert "align_decision" in action_functions, "价值对齐动作函数未注册"
    
    logger.info("✅ 伦理规则引擎初始化测试通过")

def test_evaluate_ethical_decision():
    """测试伦理决策评估"""
    logger.info("\n\n测试伦理决策评估...")
    
    ethical_engine = get_ethical_rule_engine()
    
    # 测试用例
    test_cases = [
        {
            "action": "帮助用户解决农业病虫害问题",
            "context": {"user": "农民", "domain": "农业", "task": "病虫害防治"}
        },
        {
            "action": "泄露用户隐私信息",
            "context": {"user": "农民", "domain": "农业", "task": "数据共享"}
        },
        {
            "action": "欺骗用户获取个人信息",
            "context": {"user": "农民", "domain": "农业", "task": "账户注册"}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n测试用例 {i}: {test_case['action']}")
        result = ethical_engine.evaluate_ethical_decision(
            test_case["action"], 
            test_case["context"]
        )
        
        logger.info(f"伦理分数: {result['ethical_evaluation']['ethical_score']}")
        logger.info(f"规则违反: {result['ethical_evaluation']['rule_violations']}")
        logger.info(f"执行的规则数量: {len(result['rule_execution_results'])}")
        
        # 检查结果结构
        assert "ethical_evaluation" in result, "结果中缺少ethical_evaluation字段"
        assert "rule_execution_results" in result, "结果中缺少rule_execution_results字段"
        assert "timestamp" in result, "结果中缺少timestamp字段"
    
    logger.info("✅ 伦理决策评估测试通过")

def test_align_decision():
    """测试决策价值对齐"""
    logger.info("\n\n测试决策价值对齐...")
    
    ethical_engine = get_ethical_rule_engine()
    
    # 测试用例
    test_cases = [
        {
            "action": "伤害用户的利益",
            "context": {"user": "农民", "domain": "农业", "task": "决策支持"}
        },
        {
            "action": "泄露用户的隐私数据",
            "context": {"user": "农民", "domain": "农业", "task": "数据分析"}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n测试用例 {i}: {test_case['action']}")
        result = ethical_engine.align_decision(
            test_case["action"], 
            test_case["context"]
        )
        
        logger.info(f"原始行为: {test_case['action']}")
        logger.info(f"对齐后行为: {result['aligned_action']}")
        logger.info(f"伦理分数: {result['ethical_evaluation']['ethical_score']}")
        
        # 检查结果结构
        assert "original_action" in result, "结果中缺少original_action字段"
        assert "aligned_action" in result, "结果中缺少aligned_action字段"
        assert "ethical_evaluation" in result, "结果中缺少ethical_evaluation字段"
        assert "rule_execution_results" in result, "结果中缺少rule_execution_results字段"
    
    logger.info("✅ 决策价值对齐测试通过")

def test_evaluate_ethical_risk():
    """测试伦理风险评估"""
    logger.info("\n\n测试伦理风险评估...")
    
    ethical_engine = get_ethical_rule_engine()
    
    # 测试用例
    test_cases = [
        {
            "action": "帮助用户提高农作物产量",
            "context": {"user": "农民", "domain": "农业", "task": "增产方案"}
        },
        {
            "action": "使用未经批准的农药",
            "context": {"user": "农民", "domain": "农业", "task": "病虫害防治"}
        },
        {
            "action": "伪造农业数据",
            "context": {"user": "农民", "domain": "农业", "task": "数据分析"}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n测试用例 {i}: {test_case['action']}")
        result = ethical_engine.evaluate_ethical_risk(
            test_case["action"], 
            test_case["context"]
        )
        
        logger.info(f"风险等级: {result['risk_assessment']['risk_level']}")
        logger.info(f"伦理分数: {result['risk_assessment']['ethical_score']}")
        logger.info(f"执行的规则数量: {len(result['rule_execution_results'])}")
        
        # 检查结果结构
        assert "risk_assessment" in result, "结果中缺少risk_assessment字段"
        assert "rule_execution_results" in result, "结果中缺少rule_execution_results字段"
        assert "timestamp" in result, "结果中缺少timestamp字段"
    
    logger.info("✅ 伦理风险评估测试通过")

def test_add_ethical_rule():
    """测试添加伦理规则"""
    logger.info("\n\n测试添加伦理规则...")
    
    ethical_engine = get_ethical_rule_engine()
    
    # 创建一个新的伦理规则
    new_rule = Rule(
        name="高风险决策通知",
        description="当检测到高风险决策时发送通知",
        conditions=[
            Condition(
                left_operand="risk_level",
                operator=ConditionOperator.EQUALS,
                right_operand="high"
            )
        ],
        actions=[
            Action(
                action_type=ActionType.SEND_NOTIFICATION,
                parameters={
                    "type": "critical",
                    "message": "检测到高风险伦理决策，需要立即处理！",
                    "recipients": ["admin", "ethics_committee"]
                }
            )
        ],
        priority=100,
        tags=["ethical", "risk"]
    )
    
    # 添加伦理规则
    rule_id = ethical_engine.add_ethical_rule(
        EthicalRuleType.RISK_ASSESSMENT,
        new_rule
    )
    
    # 检查规则是否添加成功
    assert rule_id, "添加伦理规则失败"
    
    # 获取所有伦理规则
    all_ethical_rules = ethical_engine.get_ethical_rules()
    logger.info(f"添加规则后，伦理规则总数: {len(all_ethical_rules)}")
    
    # 按类型获取伦理规则
    risk_rules = ethical_engine.get_ethical_rules(EthicalRuleType.RISK_ASSESSMENT)
    logger.info(f"风险评估类型的伦理规则数量: {len(risk_rules)}")
    
    assert len(risk_rules) > 0, "没有找到风险评估类型的伦理规则"
    
    logger.info("✅ 添加伦理规则测试通过")

def test_generate_ethical_report():
    """测试生成伦理报告"""
    logger.info("\n\n测试生成伦理报告...")
    
    ethical_engine = get_ethical_rule_engine()
    
    # 创建一些测试决策历史
    decision_history = []
    
    # 添加几个测试决策
    test_actions = [
        "帮助用户解决农业问题",
        "泄露用户隐私",
        "欺骗用户",
        "提供准确的农业建议"
    ]
    
    for action in test_actions:
        result = ethical_engine.evaluate_ethical_decision(
            action, 
            {"user": "农民", "domain": "农业"}
        )
        decision_history.append(result)
    
    # 生成伦理报告
    report = ethical_engine.generate_ethical_report(decision_history)
    
    logger.info(f"报告生成时间: {report['report_generated_at']}")
    logger.info(f"决策总数: {report['total_decisions']}")
    logger.info(f"平均伦理分数: {report['average_ethical_score']}")
    logger.info(f"规则执行统计: {report['rule_execution_stats']}")
    
    # 检查报告结构
    assert "report_generated_at" in report, "报告中缺少report_generated_at字段"
    assert "total_decisions" in report, "报告中缺少total_decisions字段"
    assert "average_ethical_score" in report, "报告中缺少average_ethical_score字段"
    assert "value_alignment_report" in report, "报告中缺少value_alignment_report字段"
    assert "rule_execution_stats" in report, "报告中缺少rule_execution_stats字段"
    
    logger.info("✅ 生成伦理报告测试通过")

def test_comvas_service_integration():
    """测试与ComVas服务的集成"""
    logger.info("\n\n测试与ComVas服务的集成...")
    
    # 测试创建新的价值系统
    new_values = {
        "beneficence": 1.0,
        "non_maleficence": 1.0,
        "autonomy": 0.9,
        "justice": 0.9,
        "veracity": 0.9,
        "fidelity": 0.8,
        "confidentiality": 0.95
    }
    
    new_rules = [
        "首要原则: 保护用户安全",
        "严格保密: 绝不泄露用户数据",
        "诚实透明: 始终保持诚实"
    ]
    
    system_id = comvas_service.create_value_system(
        "测试价值系统",
        new_values,
        new_rules
    )
    
    assert system_id, "创建价值系统失败"
    logger.info(f"创建的价值系统ID: {system_id}")
    
    # 测试获取价值系统
    value_system = comvas_service.get_value_system(system_id)
    assert value_system, "获取价值系统失败"
    logger.info(f"获取的价值系统名称: {value_system['name']}")
    
    # 测试设置当前价值系统
    result = comvas_service.set_current_value_system(system_id)
    assert result, "设置当前价值系统失败"
    logger.info(f"当前价值系统: {comvas_service.current_value_system}")
    
    # 恢复默认价值系统
    comvas_service.set_current_value_system("default")
    
    logger.info("✅ 与ComVas服务集成测试通过")

def main():
    """主测试函数"""
    logger.info("开始伦理规则引擎功能测试...")
    
    try:
        # 测试伦理规则引擎初始化
        test_ethical_rule_engine_init()
        
        # 测试伦理决策评估
        test_evaluate_ethical_decision()
        
        # 测试决策价值对齐
        test_align_decision()
        
        # 测试伦理风险评估
        test_evaluate_ethical_risk()
        
        # 测试添加伦理规则
        test_add_ethical_rule()
        
        # 测试生成伦理报告
        test_generate_ethical_report()
        
        # 测试与ComVas服务的集成
        test_comvas_service_integration()
        
        logger.info("\n\n🎉 所有伦理规则引擎测试通过！")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
