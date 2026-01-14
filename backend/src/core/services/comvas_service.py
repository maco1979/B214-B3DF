"""ComVas Dynamic Value Alignment Service

实现ComVas动态价值对齐系统，用于伦理决策
"""

import logging
from typing import Dict, List, Optional, Any
import uuid

logger = logging.getLogger(__name__)

class ComVasService:
    """ComVas动态价值对齐服务类"""
    
    def __init__(self):
        self.value_systems = {
            "default": {
                "name": "默认价值系统",
                "values": {
                    "beneficence": 0.9,
                    "non_maleficence": 0.95,
                    "autonomy": 0.85,
                    "justice": 0.8,
                    "veracity": 0.8,
                    "fidelity": 0.75,
                    "confidentiality": 0.85
                },
                "rules": [
                    "不伤害原则: 避免对用户造成任何形式的伤害",
                    "尊重隐私: 保护用户的隐私和数据安全",
                    "诚实透明: 保持诚实，不误导用户",
                    "公平公正: 公平对待所有用户",
                    "尊重自主权: 尊重用户的自主选择"
                ]
            }
        }
        
        self.current_value_system = "default"
        
        logger.info("✅ ComVas动态价值对齐服务初始化成功")
    
    def create_value_system(self, name: str, values: Dict[str, float], rules: List[str]) -> str:
        """创建新的价值系统
        
        Args:
            name: 价值系统名称
            values: 价值权重字典
            rules: 伦理规则列表
            
        Returns:
            价值系统ID
        """
        try:
            # 生成唯一ID
            system_id = str(uuid.uuid4())
            
            # 创建价值系统
            self.value_systems[system_id] = {
                "name": name,
                "values": values,
                "rules": rules
            }
            
            logger.info(f"创建新价值系统: {name} (ID: {system_id})")
            return system_id
        except Exception as e:
            logger.error(f"创建价值系统失败: {e}")
            return ""
    
    def get_value_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """获取价值系统
        
        Args:
            system_id: 价值系统ID
            
        Returns:
            价值系统字典
        """
        return self.value_systems.get(system_id)
    
    def set_current_value_system(self, system_id: str) -> bool:
        """设置当前价值系统
        
        Args:
            system_id: 价值系统ID
            
        Returns:
            是否成功设置
        """
        if system_id in self.value_systems:
            self.current_value_system = system_id
            logger.info(f"设置当前价值系统: {self.value_systems[system_id]['name']}")
            return True
        else:
            logger.error(f"价值系统ID不存在: {system_id}")
            return False
    
    def evaluate_ethical_decision(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估伦理决策
        
        Args:
            action: 要评估的行为
            context: 决策上下文
            
        Returns:
            伦理评估结果
        """
        try:
            # 获取当前价值系统
            value_system = self.value_systems[self.current_value_system]
            
            # 评估各项价值
            value_evaluations = {}
            for value, weight in value_system["values"].items():
                # 简单的价值评估逻辑（实际应使用更复杂的算法）
                evaluation_score = self._evaluate_single_value(value, action, context, weight)
                value_evaluations[value] = {
                    "score": evaluation_score,
                    "weight": weight
                }
            
            # 计算综合伦理分数
            total_score = sum(eval["score"] * eval["weight"] for eval in value_evaluations.values())
            total_weight = sum(eval["weight"] for eval in value_evaluations.values())
            ethical_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # 检查规则违反情况
            rule_violations = self._check_rule_violations(action, context, value_system["rules"])
            
            # 生成伦理建议
            ethical_suggestions = self._generate_ethical_suggestions(ethical_score, rule_violations, value_system)
            
            return {
                "ethical_score": round(ethical_score, 2),
                "value_evaluations": value_evaluations,
                "rule_violations": rule_violations,
                "suggestions": ethical_suggestions,
                "value_system": self.current_value_system
            }
        except Exception as e:
            logger.error(f"评估伦理决策失败: {e}")
            return {
                "ethical_score": 0.0,
                "error": str(e)
            }
    
    def _evaluate_single_value(self, value: str, action: str, context: Dict[str, Any], weight: float) -> float:
        """评估单个价值
        
        Args:
            value: 价值名称
            action: 行为描述
            context: 上下文信息
            weight: 价值权重
            
        Returns:
            价值评估分数
        """
        # 简单的价值评估逻辑（实际应使用更复杂的算法）
        base_score = 0.7  # 默认基础分数
        
        # 根据价值类型调整分数
        if value == "beneficence":
            # 善意原则：行为是否有益
            if "帮助" in action or "支持" in action or "改善" in action:
                base_score += 0.2
        elif value == "non_maleficence":
            # 不伤害原则：行为是否有害
            if "伤害" in action or "损害" in action or "威胁" in action:
                base_score -= 0.3
        elif value == "autonomy":
            # 自主原则：是否尊重用户选择
            if "用户选择" in action or "尊重" in action or "自主" in action:
                base_score += 0.2
        elif value == "justice":
            # 公正原则：是否公平
            if "公平" in action or "平等" in action or "公正" in action:
                base_score += 0.2
        elif value == "veracity":
            # 诚实原则：是否诚实
            if "诚实" in action or "透明" in action or "真实" in action:
                base_score += 0.2
        elif value == "confidentiality":
            # 保密原则：是否保护隐私
            if "隐私" in action or "保密" in action or "安全" in action:
                base_score += 0.2
        
        # 限制分数在0-1之间
        return min(1.0, max(0.0, base_score))
    
    def _check_rule_violations(self, action: str, context: Dict[str, Any], rules: List[str]) -> List[str]:
        """检查规则违反情况
        
        Args:
            action: 行为描述
            context: 上下文信息
            rules: 伦理规则列表
            
        Returns:
            违反的规则列表
        """
        violations = []
        
        # 简单的规则违反检测（实际应使用更复杂的算法）
        for rule in rules:
            # 检测是否违反规则
            if any(violation_keyword in action for violation_keyword in ["伤害", "欺骗", "不公正", "泄露隐私", "不尊重"]):
                violations.append(rule)
        
        return violations
    
    def _generate_ethical_suggestions(self, ethical_score: float, violations: List[str], value_system: Dict[str, Any]) -> List[str]:
        """生成伦理建议
        
        Args:
            ethical_score: 伦理分数
            violations: 规则违反列表
            value_system: 价值系统
            
        Returns:
            伦理建议列表
        """
        suggestions = []
        
        if ethical_score < 0.5:
            suggestions.append("该行为的伦理分数较低，建议重新考虑")
        elif ethical_score < 0.7:
            suggestions.append("该行为的伦理分数一般，建议优化")
        else:
            suggestions.append("该行为符合伦理标准")
        
        if violations:
            suggestions.append(f"违反了以下伦理规则: {', '.join(violations)}")
            suggestions.append("建议调整行为以符合伦理规则")
        
        return suggestions
    
    def align_decision(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """对齐决策到当前价值系统
        
        Args:
            action: 要对齐的行为
            context: 决策上下文
            
        Returns:
            对齐后的决策结果
        """
        try:
            # 评估伦理决策
            evaluation = self.evaluate_ethical_decision(action, context)
            
            # 根据评估结果生成对齐决策
            aligned_decision = {
                "original_action": action,
                "ethical_evaluation": evaluation,
                "aligned_action": self._generate_aligned_action(action, evaluation),
                "timestamp": str(uuid.uuid4())
            }
            
            return aligned_decision
        except Exception as e:
            logger.error(f"对齐决策失败: {e}")
            return {
                "original_action": action,
                "error": str(e)
            }
    
    def _generate_aligned_action(self, action: str, evaluation: Dict[str, Any]) -> str:
        """生成对齐后的行为
        
        Args:
            action: 原始行为
            evaluation: 伦理评估结果
            
        Returns:
            对齐后的行为
        """
        if evaluation["ethical_score"] > 0.7:
            # 伦理分数较高，保持原始行为
            return action
        else:
            # 伦理分数较低，调整行为
            aligned_action = action
            
            # 简单的行为调整逻辑
            if "伤害" in action:
                aligned_action = aligned_action.replace("伤害", "保护")
            if "欺骗" in action:
                aligned_action = aligned_action.replace("欺骗", "诚实告知")
            if "泄露隐私" in action:
                aligned_action = aligned_action.replace("泄露隐私", "保护隐私")
            
            return aligned_action
    
    def get_value_alignment_report(self, decision_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成价值对齐报告
        
        Args:
            decision_history: 决策历史记录
            
        Returns:
            价值对齐报告
        """
        try:
            # 计算统计数据
            if not decision_history:
                return {
                    "total_decisions": 0,
                    "average_ethical_score": 0.0,
                    "high_ethical_decisions": 0,
                    "medium_ethical_decisions": 0,
                    "low_ethical_decisions": 0,
                    "rule_violations": 0
                }
            
            total_decisions = len(decision_history)
            total_score = sum(decision["ethical_evaluation"]["ethical_score"] for decision in decision_history)
            average_score = total_score / total_decisions
            
            high_ethical = sum(1 for decision in decision_history if decision["ethical_evaluation"]["ethical_score"] >= 0.8)
            medium_ethical = sum(1 for decision in decision_history if 0.5 <= decision["ethical_evaluation"]["ethical_score"] < 0.8)
            low_ethical = sum(1 for decision in decision_history if decision["ethical_evaluation"]["ethical_score"] < 0.5)
            
            total_violations = sum(len(decision["ethical_evaluation"].get("rule_violations", [])) for decision in decision_history)
            
            return {
                "total_decisions": total_decisions,
                "average_ethical_score": round(average_score, 2),
                "high_ethical_decisions": high_ethical,
                "medium_ethical_decisions": medium_ethical,
                "low_ethical_decisions": low_ethical,
                "rule_violations": total_violations
            }
        except Exception as e:
            logger.error(f"生成价值对齐报告失败: {e}")
            return {
                "error": str(e)
            }

# 创建单例实例
comvas_service = ComVasService()
