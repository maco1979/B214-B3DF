"""
风险规则引擎
实现可配置的风险规则管理和评估，支持用户自定义风险规则
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
from datetime import datetime


logger = logging.getLogger(__name__)


class RuleType(Enum):
    """风险规则类型"""
    TECHNICAL = "technical"  # 技术风险规则
    DATA_SECURITY = "data_security"  # 数据安全风险规则
    ALGORITHM_BIAS = "algorithm_bias"  # 算法偏见风险规则
    BUSINESS = "business"  # 业务风险规则
    COMPLIANCE = "compliance"  # 合规风险规则
    CUSTOM = "custom"  # 自定义风险规则


class RuleOperator(Enum):
    """规则运算符"""
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"  # 正则匹配
    IN = "in"
    NOT_IN = "not_in"


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskRule:
    """风险规则"""
    id: str  # 规则ID
    name: str  # 规则名称
    description: str  # 规则描述
    rule_type: RuleType  # 规则类型
    category: str  # 规则分类
    condition: Dict[str, Any]  # 规则条件
    risk_level: RiskLevel  # 风险等级
    risk_score: float  # 风险分数 (0-1)
    enabled: bool = True  # 是否启用
    created_at: datetime = datetime.utcnow()  # 创建时间
    updated_at: datetime = datetime.utcnow()  # 更新时间
    priority: int = 5  # 优先级 (1-10, 1最高)
    tags: List[str] = None  # 规则标签
    action: Optional[Dict[str, Any]] = None  # 触发后执行的操作
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RiskRuleSet:
    """风险规则集"""
    id: str  # 规则集ID
    name: str  # 规则集名称
    description: str  # 规则集描述
    rules: List[RiskRule]  # 规则列表
    enabled: bool = True  # 是否启用
    created_at: datetime = datetime.utcnow()  # 创建时间
    updated_at: datetime = datetime.utcnow()  # 更新时间
    tags: List[str] = None  # 规则集标签
    industry: Optional[str] = None  # 适用行业
    scenario: Optional[str] = None  # 适用场景
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    overall_risk_score: float  # 总体风险分数
    overall_risk_level: RiskLevel  # 总体风险等级
    triggered_rules: List[Dict[str, Any]]  # 触发的规则
    risk_breakdown: Dict[str, float]  # 风险分类明细
    timestamp: datetime = datetime.utcnow()  # 评估时间
    context: Optional[Dict[str, Any]] = None  # 评估上下文


class RiskRuleEngine:
    """风险规则引擎
    负责风险规则的管理和评估
    """
    
    def __init__(self):
        self.rules: Dict[str, RiskRule] = {}  # 所有规则
        self.rule_sets: Dict[str, RiskRuleSet] = {}  # 规则集
        self.rule_factories: Dict[str, Callable] = {}  # 规则评估工厂
        
        # 初始化规则评估工厂
        self._init_rule_factories()
    
    def _init_rule_factories(self):
        """初始化规则评估工厂"""
        # 注册基本规则评估器
        self.rule_factories[RuleOperator.EQUAL.value] = self._evaluate_equal
        self.rule_factories[RuleOperator.NOT_EQUAL.value] = self._evaluate_not_equal
        self.rule_factories[RuleOperator.GREATER_THAN.value] = self._evaluate_greater_than
        self.rule_factories[RuleOperator.GREATER_EQUAL.value] = self._evaluate_greater_equal
        self.rule_factories[RuleOperator.LESS_THAN.value] = self._evaluate_less_than
        self.rule_factories[RuleOperator.LESS_EQUAL.value] = self._evaluate_less_equal
        self.rule_factories[RuleOperator.CONTAINS.value] = self._evaluate_contains
        self.rule_factories[RuleOperator.NOT_CONTAINS.value] = self._evaluate_not_contains
        self.rule_factories[RuleOperator.MATCHES.value] = self._evaluate_matches
        self.rule_factories[RuleOperator.IN.value] = self._evaluate_in
        self.rule_factories[RuleOperator.NOT_IN.value] = self._evaluate_not_in
    
    # 规则评估方法
    async def _evaluate_equal(self, value: Any, expected: Any) -> bool:
        """评估相等条件"""
        return value == expected
    
    async def _evaluate_not_equal(self, value: Any, expected: Any) -> bool:
        """评估不等条件"""
        return value != expected
    
    async def _evaluate_greater_than(self, value: Any, expected: Any) -> bool:
        """评估大于条件"""
        try:
            return float(value) > float(expected)
        except (ValueError, TypeError):
            return False
    
    async def _evaluate_greater_equal(self, value: Any, expected: Any) -> bool:
        """评估大于等于条件"""
        try:
            return float(value) >= float(expected)
        except (ValueError, TypeError):
            return False
    
    async def _evaluate_less_than(self, value: Any, expected: Any) -> bool:
        """评估小于条件"""
        try:
            return float(value) < float(expected)
        except (ValueError, TypeError):
            return False
    
    async def _evaluate_less_equal(self, value: Any, expected: Any) -> bool:
        """评估小于等于条件"""
        try:
            return float(value) <= float(expected)
        except (ValueError, TypeError):
            return False
    
    async def _evaluate_contains(self, value: Any, expected: Any) -> bool:
        """评估包含条件"""
        if isinstance(value, str) and isinstance(expected, str):
            return expected in value
        elif isinstance(value, (list, tuple, set)) and expected in value:
            return True
        return False
    
    async def _evaluate_not_contains(self, value: Any, expected: Any) -> bool:
        """评估不包含条件"""
        return not await self._evaluate_contains(value, expected)
    
    async def _evaluate_matches(self, value: Any, expected: Any) -> bool:
        """评估正则匹配条件"""
        if isinstance(value, str) and isinstance(expected, str):
            try:
                return bool(re.match(expected, value))
            except re.error:
                logger.error(f"无效的正则表达式: {expected}")
                return False
        return False
    
    async def _evaluate_in(self, value: Any, expected: Any) -> bool:
        """评估包含于条件"""
        if isinstance(expected, (list, tuple, set)):
            return value in expected
        return False
    
    async def _evaluate_not_in(self, value: Any, expected: Any) -> bool:
        """评估不包含于条件"""
        return not await self._evaluate_in(value, expected)
    
    # 规则管理方法
    def add_rule(self, rule: RiskRule) -> bool:
        """添加风险规则
        Args:
            rule: 风险规则
        Returns:
            bool: 添加是否成功
        """
        try:
            if rule.id in self.rules:
                logger.warning(f"规则已存在: {rule.id}")
                return False
            
            self.rules[rule.id] = rule
            logger.info(f"规则添加成功: {rule.name} (ID: {rule.id})")
            return True
        except Exception as e:
            logger.error(f"添加规则失败: {e}")
            return False
    
    def update_rule(self, rule: RiskRule) -> bool:
        """更新风险规则
        Args:
            rule: 风险规则
        Returns:
            bool: 更新是否成功
        """
        try:
            if rule.id not in self.rules:
                logger.warning(f"规则不存在: {rule.id}")
                return False
            
            self.rules[rule.id] = rule
            logger.info(f"规则更新成功: {rule.name} (ID: {rule.id})")
            return True
        except Exception as e:
            logger.error(f"更新规则失败: {e}")
            return False
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除风险规则
        Args:
            rule_id: 规则ID
        Returns:
            bool: 删除是否成功
        """
        try:
            if rule_id not in self.rules:
                logger.warning(f"规则不存在: {rule_id}")
                return False
            
            del self.rules[rule_id]
            logger.info(f"规则删除成功: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"删除规则失败: {e}")
            return False
    
    def get_rule(self, rule_id: str) -> Optional[RiskRule]:
        """获取风险规则
        Args:
            rule_id: 规则ID
        Returns:
            Optional[RiskRule]: 风险规则，不存在返回None
        """
        return self.rules.get(rule_id)
    
    def list_rules(self, 
                  rule_type: Optional[RuleType] = None, 
                  enabled: Optional[bool] = None, 
                  category: Optional[str] = None) -> List[RiskRule]:
        """列出风险规则
        Args:
            rule_type: 规则类型，None表示所有类型
            enabled: 是否启用，None表示所有状态
            category: 规则分类，None表示所有分类
        Returns:
            List[RiskRule]: 风险规则列表
        """
        rules = list(self.rules.values())
        
        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        
        if enabled is not None:
            rules = [r for r in rules if r.enabled == enabled]
        
        if category:
            rules = [r for r in rules if r.category == category]
        
        # 按优先级排序
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    # 规则集管理方法
    def add_rule_set(self, rule_set: RiskRuleSet) -> bool:
        """添加规则集
        Args:
            rule_set: 规则集
        Returns:
            bool: 添加是否成功
        """
        try:
            if rule_set.id in self.rule_sets:
                logger.warning(f"规则集已存在: {rule_set.id}")
                return False
            
            self.rule_sets[rule_set.id] = rule_set
            logger.info(f"规则集添加成功: {rule_set.name} (ID: {rule_set.id})")
            return True
        except Exception as e:
            logger.error(f"添加规则集失败: {e}")
            return False
    
    def update_rule_set(self, rule_set: RiskRuleSet) -> bool:
        """更新规则集
        Args:
            rule_set: 规则集
        Returns:
            bool: 更新是否成功
        """
        try:
            if rule_set.id not in self.rule_sets:
                logger.warning(f"规则集不存在: {rule_set.id}")
                return False
            
            self.rule_sets[rule_set.id] = rule_set
            logger.info(f"规则集更新成功: {rule_set.name} (ID: {rule_set.id})")
            return True
        except Exception as e:
            logger.error(f"更新规则集失败: {e}")
            return False
    
    def delete_rule_set(self, rule_set_id: str) -> bool:
        """删除规则集
        Args:
            rule_set_id: 规则集ID
        Returns:
            bool: 删除是否成功
        """
        try:
            if rule_set_id not in self.rule_sets:
                logger.warning(f"规则集不存在: {rule_set_id}")
                return False
            
            del self.rule_sets[rule_set_id]
            logger.info(f"规则集删除成功: {rule_set_id}")
            return True
        except Exception as e:
            logger.error(f"删除规则集失败: {e}")
            return False
    
    def get_rule_set(self, rule_set_id: str) -> Optional[RiskRuleSet]:
        """获取规则集
        Args:
            rule_set_id: 规则集ID
        Returns:
            Optional[RiskRuleSet]: 规则集，不存在返回None
        """
        return self.rule_sets.get(rule_set_id)
    
    def list_rule_sets(self, 
                      enabled: Optional[bool] = None, 
                      industry: Optional[str] = None, 
                      scenario: Optional[str] = None) -> List[RiskRuleSet]:
        """列出规则集
        Args:
            enabled: 是否启用，None表示所有状态
            industry: 适用行业，None表示所有行业
            scenario: 适用场景，None表示所有场景
        Returns:
            List[RiskRuleSet]: 规则集列表
        """
        rule_sets = list(self.rule_sets.values())
        
        if enabled is not None:
            rule_sets = [rs for rs in rule_sets if rs.enabled == enabled]
        
        if industry:
            rule_sets = [rs for rs in rule_sets if rs.industry == industry]
        
        if scenario:
            rule_sets = [rs for rs in rule_sets if rs.scenario == scenario]
        
        return rule_sets
    
    # 规则评估方法
    async def evaluate_risk(self, data: Dict[str, Any], 
                          rule_set_ids: Optional[List[str]] = None, 
                          rule_types: Optional[List[RuleType]] = None) -> RiskAssessmentResult:
        """评估风险
        Args:
            data: 待评估的数据
            rule_set_ids: 规则集ID列表，None表示使用所有规则
            rule_types: 规则类型列表，None表示使用所有类型
        Returns:
            RiskAssessmentResult: 风险评估结果
        """
        try:
            # 获取适用的规则
            applicable_rules = self._get_applicable_rules(rule_set_ids, rule_types)
            if not applicable_rules:
                logger.info("没有适用的风险规则")
                return RiskAssessmentResult(
                    overall_risk_score=0.0,
                    overall_risk_level=RiskLevel.LOW,
                    triggered_rules=[],
                    risk_breakdown={}
                )
            
            # 并行评估所有规则
            evaluated_rules = await asyncio.gather(*[
                self._evaluate_single_rule(rule, data)
                for rule in applicable_rules
            ])
            
            # 收集触发的规则
            triggered_rules = []
            risk_scores_by_category = {}
            total_risk_score = 0.0
            
            for rule_result in evaluated_rules:
                if rule_result["triggered"]:
                    triggered_rules.append(rule_result)
                    # 按分类累加风险分数
                    category = rule_result["rule"].category
                    risk_scores_by_category[category] = risk_scores_by_category.get(category, 0) + rule_result["rule"].risk_score
                    total_risk_score += rule_result["rule"].risk_score
            
            # 计算总体风险等级
            overall_risk_level = self._calculate_overall_risk_level(total_risk_score, len(applicable_rules))
            
            # 生成风险评估结果
            result = RiskAssessmentResult(
                overall_risk_score=round(total_risk_score, 2),
                overall_risk_level=overall_risk_level,
                triggered_rules=triggered_rules,
                risk_breakdown=risk_scores_by_category,
                context={"evaluated_rules_count": len(applicable_rules), "triggered_rules_count": len(triggered_rules)}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return RiskAssessmentResult(
                overall_risk_score=1.0,  # 评估失败视为最高风险
                overall_risk_level=RiskLevel.CRITICAL,
                triggered_rules=[],
                risk_breakdown={}
            )
    
    def _get_applicable_rules(self, 
                            rule_set_ids: Optional[List[str]] = None, 
                            rule_types: Optional[List[RuleType]] = None) -> List[RiskRule]:
        """获取适用的风险规则
        Args:
            rule_set_ids: 规则集ID列表，None表示使用所有规则
            rule_types: 规则类型列表，None表示使用所有类型
        Returns:
            List[RiskRule]: 适用的风险规则列表
        """
        applicable_rules = []
        
        if rule_set_ids:
            # 从指定规则集中获取规则
            for rule_set_id in rule_set_ids:
                rule_set = self.rule_sets.get(rule_set_id)
                if rule_set and rule_set.enabled:
                    applicable_rules.extend([r for r in rule_set.rules if r.enabled])
        else:
            # 获取所有启用的规则
            applicable_rules = [r for r in self.rules.values() if r.enabled]
        
        # 按规则类型过滤
        if rule_types:
            applicable_rules = [r for r in applicable_rules if r.rule_type in rule_types]
        
        # 按优先级排序
        applicable_rules.sort(key=lambda r: r.priority)
        
        return applicable_rules
    
    async def _evaluate_single_rule(self, rule: RiskRule, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个风险规则
        Args:
            rule: 风险规则
            data: 待评估的数据
        Returns:
            Dict[str, Any]: 规则评估结果
        """
        try:
            # 获取规则条件
            condition = rule.condition
            if not condition:
                return {"rule": rule, "triggered": False, "reason": "规则条件为空"}
            
            # 评估条件
            triggered = await self._evaluate_condition(condition, data)
            
            result = {
                "rule": rule,
                "triggered": triggered,
                "risk_score": rule.risk_score if triggered else 0.0,
                "risk_level": rule.risk_level if triggered else RiskLevel.LOW,
                "evaluated_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"评估规则失败 {rule.name}: {e}")
            return {"rule": rule, "triggered": False, "reason": f"评估错误: {str(e)}"}
    
    async def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """评估规则条件
        Args:
            condition: 规则条件
            data: 待评估的数据
        Returns:
            bool: 条件是否满足
        """
        # 支持复杂条件：and/or嵌套
        if "and" in condition:
            # AND条件，所有子条件必须满足
            sub_conditions = condition["and"]
            for sub_condition in sub_conditions:
                if not await self._evaluate_condition(sub_condition, data):
                    return False
            return True
        elif "or" in condition:
            # OR条件，至少一个子条件满足
            sub_conditions = condition["or"]
            for sub_condition in sub_conditions:
                if await self._evaluate_condition(sub_condition, data):
                    return True
            return False
        else:
            # 单个条件
            return await self._evaluate_simple_condition(condition, data)
    
    async def _evaluate_simple_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """评估简单条件
        Args:
            condition: 简单条件
            data: 待评估的数据
        Returns:
            bool: 条件是否满足
        """
        try:
            # 获取条件字段和值
            field = condition.get("field")
            operator = condition.get("operator")
            expected = condition.get("value")
            
            if not field or not operator:
                logger.warning(f"条件缺少字段或运算符: {condition}")
                return False
            
            # 获取数据中的实际值
            actual_value = self._get_nested_value(data, field)
            if actual_value is None:
                logger.debug(f"数据中不存在字段: {field}")
                return False
            
            # 获取评估器
            evaluator = self.rule_factories.get(operator)
            if not evaluator:
                logger.warning(f"不支持的运算符: {operator}")
                return False
            
            # 执行评估
            return await evaluator(actual_value, expected)
            
        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """获取嵌套字段的值
        Args:
            data: 数据字典
            field: 嵌套字段路径，如 "system.metrics.cpu_usage"
        Returns:
            Any: 字段值，不存在返回None
        """
        try:
            keys = field.split(".")
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except Exception as e:
            logger.error(f"获取嵌套字段值失败: {e}")
            return None
    
    def _calculate_overall_risk_level(self, total_score: float, rule_count: int) -> RiskLevel:
        """计算总体风险等级
        Args:
            total_score: 总风险分数
            rule_count: 评估的规则数量
        Returns:
            RiskLevel: 总体风险等级
        """
        if rule_count == 0:
            return RiskLevel.LOW
        
        # 计算平均风险分数
        avg_score = total_score / rule_count
        
        # 根据平均分数确定风险等级
        if avg_score >= 0.8:
            return RiskLevel.CRITICAL
        elif avg_score >= 0.6:
            return RiskLevel.HIGH
        elif avg_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    # 规则导入导出方法
    def export_rules(self, rule_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """导出风险规则
        Args:
            rule_ids: 规则ID列表，None表示导出所有规则
        Returns:
            Dict[str, Any]: 导出的规则数据
        """
        rules_to_export = self.rules.values()
        if rule_ids:
            rules_to_export = [r for r in rules_to_export if r.id in rule_ids]
        
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "rules_count": len(rules_to_export),
            "rules": [{
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "rule_type": rule.rule_type.value,
                "category": rule.category,
                "condition": rule.condition,
                "risk_level": rule.risk_level.value,
                "risk_score": rule.risk_score,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "tags": rule.tags,
                "action": rule.action
            } for rule in rules_to_export]
        }
        
        return export_data
    
    def import_rules(self, import_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """导入风险规则
        Args:
            import_data: 导入的规则数据
        Returns:
            Tuple[bool, Dict[str, Any]]: 导入结果和统计信息
        """
        try:
            if "rules" not in import_data:
                return False, {"message": "导入数据格式错误，缺少rules字段"}
            
            imported_count = 0
            updated_count = 0
            skipped_count = 0
            errors = []
            
            for rule_data in import_data["rules"]:
                try:
                    # 创建风险规则对象
                    rule = RiskRule(
                        id=rule_data["id"],
                        name=rule_data["name"],
                        description=rule_data["description"],
                        rule_type=RuleType(rule_data["rule_type"]),
                        category=rule_data["category"],
                        condition=rule_data["condition"],
                        risk_level=RiskLevel(rule_data["risk_level"]),
                        risk_score=rule_data["risk_score"],
                        enabled=rule_data["enabled"],
                        priority=rule_data["priority"],
                        tags=rule_data.get("tags", []),
                        action=rule_data.get("action")
                    )
                    
                    # 添加或更新规则
                    if rule.id in self.rules:
                        if self.update_rule(rule):
                            updated_count += 1
                        else:
                            skipped_count += 1
                    else:
                        if self.add_rule(rule):
                            imported_count += 1
                        else:
                            skipped_count += 1
                            
                except Exception as e:
                    errors.append(f"规则 {rule_data.get('name', '未知')}: {str(e)}")
                    skipped_count += 1
            
            result = {
                "imported": imported_count,
                "updated": updated_count,
                "skipped": skipped_count,
                "errors": errors
            }
            
            return True, result
            
        except Exception as e:
            return False, {"message": f"导入失败: {str(e)}"}


# 全局风险规则引擎实例
risk_rule_engine = RiskRuleEngine()
