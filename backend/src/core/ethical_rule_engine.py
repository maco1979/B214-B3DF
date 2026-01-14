"""
伦理规则引擎

将伦理决策与规则引擎结合，实现伦理驱动的决策系统
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from datetime import datetime

from .rule_engine import RuleEngine, Rule, Condition, Action, ActionType, ConditionOperator, RuleExecutionResult
from .services.comvas_service import comvas_service

logger = logging.getLogger(__name__)


class EthicalRuleType(Enum):
    """伦理规则类型枚举"""
    ETHICAL_EVALUATION = "ethical_evaluation"  # 伦理评估规则
    VALUE_ALIGNMENT = "value_alignment"  # 价值对齐规则
    RULE_COMPLIANCE = "rule_compliance"  # 规则合规性检查
    RISK_ASSESSMENT = "risk_assessment"  # 伦理风险评估
    SUGGESTION_GENERATION = "suggestion_generation"  # 伦理建议生成


class EthicalRuleEngine(RuleEngine):
    """伦理规则引擎，扩展自通用规则引擎"""
    
    def __init__(self):
        super().__init__()
        self.ethical_rules: Dict[str, Dict[str, Any]] = {}
        self.ethical_rule_prefix = "ethical_"
        
        # 注册伦理相关的动作函数
        self._register_ethical_action_functions()
        
        # 添加默认伦理规则
        self._add_default_ethical_rules()
        
        logger.info("✅ 伦理规则引擎初始化成功")
    
    def _register_ethical_action_functions(self):
        """注册伦理相关的动作函数"""
        # 注册伦理评估动作
        self.register_action_function(
            "evaluate_ethical_decision", 
            self._action_evaluate_ethical_decision
        )
        
        # 注册价值对齐动作
        self.register_action_function(
            "align_decision", 
            self._action_align_decision
        )
        
        # 注册伦理建议生成动作
        self.register_action_function(
            "generate_ethical_suggestions", 
            self._action_generate_ethical_suggestions
        )
        
        # 注册伦理风险评估动作
        self.register_action_function(
            "assess_ethical_risk", 
            self._action_assess_ethical_risk
        )
    
    def _action_evaluate_ethical_decision(self, action: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行伦理评估动作"""
        return comvas_service.evaluate_ethical_decision(action, context)
    
    def _action_align_decision(self, action: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行价值对齐动作"""
        return comvas_service.align_decision(action, context)
    
    def _action_generate_ethical_suggestions(self, **kwargs) -> List[str]:
        """执行伦理建议生成动作"""
        # 从kwargs中提取参数
        ethical_score = kwargs.get("ethical_score", 0.0)
        rule_violations = kwargs.get("rule_violations", [])
        
        # 转换规则违反为列表
        violations = []
        if isinstance(rule_violations, int):
            # 如果是计数，转换为相应数量的占位符
            violations = ["违反规则"] * rule_violations
        elif isinstance(rule_violations, list):
            # 如果已经是列表，直接使用
            violations = rule_violations
        
        # 获取当前价值系统
        value_system = comvas_service.value_systems[comvas_service.current_value_system]
        return comvas_service._generate_ethical_suggestions(ethical_score, violations, value_system)
    
    def _action_assess_ethical_risk(self, action: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行伦理风险评估动作"""
        evaluation = comvas_service.evaluate_ethical_decision(action, context)
        
        # 基于伦理分数评估风险等级
        risk_level = "low"
        if evaluation["ethical_score"] < 0.3:
            risk_level = "high"
        elif evaluation["ethical_score"] < 0.6:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "ethical_score": evaluation["ethical_score"],
            "rule_violations": evaluation["rule_violations"],
            "risk_factors": evaluation["value_evaluations"]
        }
    
    def _add_default_ethical_rules(self):
        """添加默认伦理规则"""
        # 伦理分数检查规则
        ethical_score_rule = Rule(
            name="伦理分数阈值检查",
            description="检查决策的伦理分数是否低于阈值",
            conditions=[
                Condition(
                    left_operand="ethical_score",
                    operator=ConditionOperator.LESS_THAN,
                    right_operand=0.6
                )
            ],
            actions=[
                Action(
                    action_type=ActionType.EXECUTE_FUNCTION,
                    parameters={
                        "function_name": "generate_ethical_suggestions",
                        "parameters": {}
                    }
                ),
                Action(
                    action_type=ActionType.SEND_NOTIFICATION,
                    parameters={
                        "type": "warning",
                        "message": "检测到伦理分数较低的决策，请检查！",
                        "recipients": ["admin"]
                    }
                )
            ],
            priority=90,
            tags=["ethical", "compliance"]
        )
        self.add_rule(ethical_score_rule)
        
        # 规则违反检查规则
        rule_violation_rule = Rule(
            name="伦理规则违反检查",
            description="检查决策是否违反了伦理规则",
            conditions=[
                Condition(
                    left_operand="rule_violations",
                    operator=ConditionOperator.GREATER_THAN,
                    right_operand=0
                )
            ],
            actions=[
                Action(
                    action_type=ActionType.EXECUTE_FUNCTION,
                    parameters={
                        "function_name": "generate_ethical_suggestions",
                        "parameters": {}
                    }
                ),
                Action(
                    action_type=ActionType.LOG_EVENT,
                    parameters={
                        "event_type": "ethical_violation",
                        "data": {}
                    }
                )
            ],
            priority=95,
            tags=["ethical", "compliance"]
        )
        self.add_rule(rule_violation_rule)
    
    def evaluate_ethical_decision(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估伦理决策并执行相关规则"""
        # 1. 执行伦理评估
        ethical_evaluation = comvas_service.evaluate_ethical_decision(action, context)
        
        # 2. 构建规则上下文
        rule_context = {
            **context,
            "action": action,
            "ethical_score": ethical_evaluation["ethical_score"],
            "rule_violations": len(ethical_evaluation["rule_violations"]),
            "value_evaluations": ethical_evaluation["value_evaluations"]
        }
        
        # 3. 评估并执行相关规则
        rule_results = self.evaluate_rules(rule_context)
        
        # 4. 生成最终结果
        result = {
            "ethical_evaluation": ethical_evaluation,
            "rule_execution_results": [result.to_dict() for result in rule_results],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"伦理决策评估完成，伦理分数: {ethical_evaluation['ethical_score']}, 规则执行数量: {len(rule_results)}")
        return result
    
    def align_decision(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """对齐决策到伦理标准"""
        # 1. 使用ComVas服务进行价值对齐
        aligned_decision = comvas_service.align_decision(action, context)
        
        # 2. 构建规则上下文
        rule_context = {
            **context,
            "original_action": action,
            "aligned_action": aligned_decision["aligned_action"],
            "ethical_score": aligned_decision["ethical_evaluation"]["ethical_score"],
            "rule_violations": len(aligned_decision["ethical_evaluation"]["rule_violations"])
        }
        
        # 3. 评估并执行相关规则
        rule_results = self.evaluate_rules(rule_context)
        
        # 4. 生成最终结果
        result = {
            **aligned_decision,
            "rule_execution_results": [result.to_dict() for result in rule_results],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"决策价值对齐完成，原始行为: {action}, 对齐后行为: {aligned_decision['aligned_action']}")
        return result
    
    def add_ethical_rule(self, rule_type: EthicalRuleType, rule: Rule) -> str:
        """添加伦理规则"""
        rule_id = self.add_rule(rule)
        
        # 确保规则包含ethical标签
        if "ethical" not in rule.tags:
            rule.tags.append("ethical")
        
        # 记录伦理规则
        self.ethical_rules[rule_id] = {
            "rule_type": rule_type.value,
            "rule_id": rule_id,
            "added_at": datetime.now().isoformat()
        }
        
        logger.info(f"添加伦理规则: {rule.name} (类型: {rule_type.value}, ID: {rule_id})")
        return rule_id
    
    def get_ethical_rules(self, rule_type: Optional[EthicalRuleType] = None) -> List[Rule]:
        """获取伦理规则"""
        if rule_type:
            # 根据规则类型过滤
            rule_ids = [rid for rid, info in self.ethical_rules.items() if info["rule_type"] == rule_type.value]
            return [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            # 返回所有带有ethical标签的规则
            return [rule for rule in self.rules.values() if "ethical" in rule.tags]
    
    def evaluate_ethical_risk(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估伦理风险"""
        # 1. 执行伦理风险评估
        risk_assessment = self._action_assess_ethical_risk(action, context)
        
        # 2. 构建规则上下文
        rule_context = {
            **context,
            "action": action,
            "risk_level": risk_assessment["risk_level"],
            "ethical_score": risk_assessment["ethical_score"],
            "rule_violations": risk_assessment["rule_violations"]
        }
        
        # 3. 评估并执行相关规则
        rule_results = self.evaluate_rules(rule_context)
        
        # 4. 生成最终结果
        result = {
            "risk_assessment": risk_assessment,
            "rule_execution_results": [result.to_dict() for result in rule_results],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"伦理风险评估完成，风险等级: {risk_assessment['risk_level']}")
        return result
    
    def generate_ethical_report(self, decision_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成伦理报告"""
        # 1. 使用ComVas服务生成价值对齐报告
        value_alignment_report = comvas_service.get_value_alignment_report(decision_history)
        
        # 2. 计算额外的伦理指标
        total_decisions = len(decision_history)
        ethical_score_sum = sum(d["ethical_evaluation"]["ethical_score"] for d in decision_history)
        average_ethical_score = ethical_score_sum / total_decisions if total_decisions > 0 else 0.0
        
        # 3. 统计规则执行情况
        rule_execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "rules_matched": 0
        }
        
        for decision in decision_history:
            if "rule_execution_results" in decision:
                for result in decision["rule_execution_results"]:
                    rule_execution_stats["total_executions"] += 1
                    if result["executed"]:
                        rule_execution_stats["successful_executions"] += 1
                    if result["error"]:
                        rule_execution_stats["failed_executions"] += 1
                    if result["conditions_met"]:
                        rule_execution_stats["rules_matched"] += 1
        
        # 4. 生成综合伦理报告
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "total_decisions": total_decisions,
            "average_ethical_score": round(average_ethical_score, 2),
            "value_alignment_report": value_alignment_report,
            "rule_execution_stats": rule_execution_stats,
            "decision_history": decision_history
        }
        
        logger.info(f"生成伦理报告，包含 {total_decisions} 个决策")
        return report


# 创建伦理规则引擎实例（单例模式）
ethical_rule_engine_instance = None


def get_ethical_rule_engine() -> EthicalRuleEngine:
    """获取伦理规则引擎实例（单例模式）"""
    global ethical_rule_engine_instance
    if ethical_rule_engine_instance is None:
        ethical_rule_engine_instance = EthicalRuleEngine()
    return ethical_rule_engine_instance
