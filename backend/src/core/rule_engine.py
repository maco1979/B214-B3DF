"""
通用规则引擎 - 支持复杂条件和动态规则管理
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class ConditionOperator(Enum):
    """条件操作符枚举"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(Enum):
    """动作类型枚举"""
    EXECUTE_FUNCTION = "execute_function"
    SET_VALUE = "set_value"
    SEND_NOTIFICATION = "send_notification"
    CALL_API = "call_api"
    LOG_EVENT = "log_event"


@dataclass
class Condition:
    """条件表达式"""
    left_operand: str  # 左操作数（变量名）
    operator: ConditionOperator  # 操作符
    right_operand: Any  # 右操作数（常量或变量名）
    is_right_operand_variable: bool = False  # 右操作数是否为变量
    conditions: List['Condition'] = field(default_factory=list)  # 子条件（用于AND/OR/NOT）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "left_operand": self.left_operand,
            "operator": self.operator.value,
            "right_operand": self.right_operand,
            "is_right_operand_variable": self.is_right_operand_variable,
            "conditions": [c.to_dict() for c in self.conditions]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Condition':
        """从字典创建条件对象"""
        return cls(
            left_operand=data["left_operand"],
            operator=ConditionOperator(data["operator"]),
            right_operand=data["right_operand"],
            is_right_operand_variable=data.get("is_right_operand_variable", False),
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])]
        )


@dataclass
class Action:
    """规则动作"""
    action_type: ActionType  # 动作类型
    parameters: Dict[str, Any]  # 动作参数
    function: Optional[Callable] = None  # 函数引用（仅用于EXECUTE_FUNCTION类型）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "action_type": self.action_type.value,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """从字典创建动作对象"""
        return cls(
            action_type=ActionType(data["action_type"]),
            parameters=data["parameters"]
        )


@dataclass
class Rule:
    """规则定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    conditions: List[Condition] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    priority: int = 50  # 优先级（0-100，值越高优先级越高）
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    execution_count: int = 0
    last_executed: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "actions": [a.to_dict() for a in self.actions],
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_count": self.execution_count,
            "last_executed": self.last_executed,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """从字典创建规则对象"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])],
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            priority=data.get("priority", 50),
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            execution_count=data.get("execution_count", 0),
            last_executed=data.get("last_executed"),
            tags=data.get("tags", [])
        )
    
    def update(self) -> None:
        """更新规则的更新时间"""
        self.updated_at = time.time()
    
    def execute(self) -> None:
        """执行规则"""
        self.execution_count += 1
        self.last_executed = time.time()
        self.updated_at = time.time()


@dataclass
class RuleExecutionResult:
    """规则执行结果"""
    rule_id: str
    rule_name: str
    executed: bool
    conditions_met: bool
    actions_executed: int
    execution_time: float
    error: Optional[str] = None
    results: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "executed": self.executed,
            "conditions_met": self.conditions_met,
            "actions_executed": self.actions_executed,
            "execution_time": self.execution_time,
            "error": self.error,
            "results": self.results
        }


class RuleEngine:
    """通用规则引擎核心类"""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.action_functions: Dict[str, Callable] = {}
        self.execution_logs: List[RuleExecutionResult] = []
        self.max_logs: int = 1000  # 最大日志数量
        
        # 监控统计数据
        self.monitoring_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_rules_evaluated": 0,
            "rules_matched": 0,
            "rules_executed": 0,
            "actions_executed": 0,
            "execution_time_history": [],  # 最近100次执行时间
            "rules_by_tag": {},  # 按标签统计规则数量
            "action_type_stats": {},  # 按动作类型统计执行次数
            "error_types": {},  # 错误类型统计
            "last_reset_time": time.time()
        }
    
    def add_rule(self, rule: Rule) -> str:
        """添加规则"""
        self.rules[rule.id] = rule
        
        # 更新标签统计
        for tag in rule.tags:
            if tag not in self.monitoring_stats["rules_by_tag"]:
                self.monitoring_stats["rules_by_tag"][tag] = 0
            self.monitoring_stats["rules_by_tag"][tag] += 1
        
        logger.info(f"添加规则: {rule.name} (ID: {rule.id})")
        return rule.id
    
    def add_rule_from_dict(self, rule_dict: Dict[str, Any]) -> str:
        """从字典添加规则"""
        rule = Rule.from_dict(rule_dict)
        return self.add_rule(rule)
    
    def update_rule(self, rule_id: str, rule: Rule) -> bool:
        """更新规则"""
        if rule_id in self.rules:
            self.rules[rule_id] = rule
            rule.update()
            logger.info(f"更新规则: {rule.name} (ID: {rule_id})")
            return True
        return False
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            rule_name = rule.name
            
            # 更新标签统计
            for tag in rule.tags:
                if tag in self.monitoring_stats["rules_by_tag"]:
                    self.monitoring_stats["rules_by_tag"][tag] -= 1
                    if self.monitoring_stats["rules_by_tag"][tag] <= 0:
                        del self.monitoring_stats["rules_by_tag"][tag]
            
            del self.rules[rule_id]
            logger.info(f"删除规则: {rule_name} (ID: {rule_id})")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[Rule]:
        """获取所有规则"""
        return list(self.rules.values())
    
    def get_rules_by_tag(self, tag: str) -> List[Rule]:
        """根据标签获取规则"""
        return [rule for rule in self.rules.values() if tag in rule.tags]
    
    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.rules[rule_id].update()
            logger.info(f"启用规则: {self.rules[rule_id].name} (ID: {rule_id})")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.rules[rule_id].update()
            logger.info(f"禁用规则: {self.rules[rule_id].name} (ID: {rule_id})")
            return True
        return False
    
    def register_action_function(self, name: str, function: Callable) -> None:
        """注册动作函数"""
        self.action_functions[name] = function
        logger.info(f"注册动作函数: {name}")
    
    def _evaluate_condition(self, condition: Condition, context: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        try:
            if condition.operator in [ConditionOperator.AND, ConditionOperator.OR]:
                # 复合条件处理
                if not condition.conditions:
                    return False
                
                results = [self._evaluate_condition(c, context) for c in condition.conditions]
                
                if condition.operator == ConditionOperator.AND:
                    return all(results)
                else:  # OR
                    return any(results)
            
            elif condition.operator == ConditionOperator.NOT:
                # NOT条件处理
                if not condition.conditions:
                    return True
                return not all([self._evaluate_condition(c, context) for c in condition.conditions])
            
            else:
                # 简单条件处理
                left_value = context.get(condition.left_operand)
                
                if condition.is_right_operand_variable:
                    right_value = context.get(condition.right_operand)
                else:
                    right_value = condition.right_operand
                
                # 处理None值
                if left_value is None or right_value is None:
                    return False
                
                # 执行比较操作
                if condition.operator == ConditionOperator.EQUALS:
                    return left_value == right_value
                elif condition.operator == ConditionOperator.NOT_EQUALS:
                    return left_value != right_value
                elif condition.operator == ConditionOperator.GREATER_THAN:
                    return left_value > right_value
                elif condition.operator == ConditionOperator.GREATER_THAN_EQUALS:
                    return left_value >= right_value
                elif condition.operator == ConditionOperator.LESS_THAN:
                    return left_value < right_value
                elif condition.operator == ConditionOperator.LESS_THAN_EQUALS:
                    return left_value <= right_value
                elif condition.operator == ConditionOperator.CONTAINS:
                    return str(right_value) in str(left_value)
                elif condition.operator == ConditionOperator.NOT_CONTAINS:
                    return str(right_value) not in str(left_value)
                elif condition.operator == ConditionOperator.IN:
                    return left_value in right_value
                elif condition.operator == ConditionOperator.NOT_IN:
                    return left_value not in right_value
                
                return False
        
        except Exception as e:
            logger.error(f"条件评估错误: {e}", exc_info=True)
            return False
    
    def _execute_action(self, action: Action, context: Dict[str, Any]) -> Any:
        """执行动作"""
        try:
            if action.action_type == ActionType.EXECUTE_FUNCTION:
                # 执行函数
                func_name = action.parameters.get("function_name")
                func_params = action.parameters.get("parameters", {})
                
                if func_name in self.action_functions:
                    result = self.action_functions[func_name](**func_params, **context)
                    logger.debug(f"执行函数 {func_name} 成功，结果: {result}")
                    return result
                else:
                    logger.error(f"未找到动作函数: {func_name}")
                    return None
            
            elif action.action_type == ActionType.SET_VALUE:
                # 设置值
                variable_name = action.parameters.get("variable_name")
                value = action.parameters.get("value")
                context[variable_name] = value
                logger.debug(f"设置变量 {variable_name} = {value}")
                return value
            
            elif action.action_type == ActionType.SEND_NOTIFICATION:
                # 发送通知
                notification_type = action.parameters.get("type", "info")
                message = action.parameters.get("message", "")
                recipients = action.parameters.get("recipients", [])
                
                logger.info(f"发送通知 [{notification_type}] 给 {recipients}: {message}")
                return {"type": notification_type, "message": message, "recipients": recipients}
            
            elif action.action_type == ActionType.CALL_API:
                # 调用API
                api_url = action.parameters.get("url")
                method = action.parameters.get("method", "GET")
                headers = action.parameters.get("headers", {})
                data = action.parameters.get("data", {})
                
                logger.info(f"调用API {method} {api_url}")
                # 这里可以添加实际的API调用逻辑
                return {"url": api_url, "method": method, "status": "success"}
            
            elif action.action_type == ActionType.LOG_EVENT:
                # 记录事件
                event_type = action.parameters.get("event_type", "info")
                event_data = action.parameters.get("data", {})
                
                logger.info(f"记录事件 [{event_type}]: {event_data}")
                return {"event_type": event_type, "data": event_data}
            
            else:
                logger.error(f"未知动作类型: {action.action_type}")
                return None
        
        except Exception as e:
            logger.error(f"动作执行错误: {e}", exc_info=True)
            return None
    
    def evaluate_rules(self, context: Dict[str, Any]) -> List[RuleExecutionResult]:
        """评估并执行所有规则"""
        results = []
        
        # 按优先级排序规则
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        # 本次执行统计
        execution_start_time = time.time()
        rules_evaluated = 0
        rules_matched_count = 0
        actions_executed_count = 0
        successful_count = 0
        failed_count = 0
        execution_times = []
        action_type_counts = {}
        error_type_counts = {}
        
        for rule in sorted_rules:
            rules_evaluated += 1
            
            if not rule.enabled:
                continue
            
            rule_start_time = time.time()
            
            try:
                # 评估条件
                conditions_met = all([self._evaluate_condition(cond, context) for cond in rule.conditions])
                
                if conditions_met:
                    rules_matched_count += 1
                    # 执行规则
                    rule.execute()
                    
                    # 执行动作
                    action_results = []
                    for action in rule.actions:
                        result = self._execute_action(action, context)
                        action_results.append(result)
                        actions_executed_count += 1
                        
                        # 统计动作类型
                        action_type = action.action_type.value
                        if action_type not in action_type_counts:
                            action_type_counts[action_type] = 0
                        action_type_counts[action_type] += 1
                    
                    # 记录结果
                    rule_execution_time = time.time() - rule_start_time
                    execution_times.append(rule_execution_time)
                    
                    result = RuleExecutionResult(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        executed=True,
                        conditions_met=True,
                        actions_executed=len(action_results),
                        execution_time=rule_execution_time,
                        results=action_results
                    )
                    successful_count += 1
                else:
                    # 条件不满足
                    rule_execution_time = time.time() - rule_start_time
                    execution_times.append(rule_execution_time)
                    
                    result = RuleExecutionResult(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        executed=False,
                        conditions_met=False,
                        actions_executed=0,
                        execution_time=rule_execution_time
                    )
                
                results.append(result)
                self._add_execution_log(result)
                
            except Exception as e:
                # 规则执行错误
                rule_execution_time = time.time() - rule_start_time
                execution_times.append(rule_execution_time)
                
                error_result = RuleExecutionResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    executed=False,
                    conditions_met=False,
                    actions_executed=0,
                    execution_time=rule_execution_time,
                    error=str(e)
                )
                results.append(error_result)
                self._add_execution_log(error_result)
                
                # 统计错误类型
                error_type = type(e).__name__
                if error_type not in error_type_counts:
                    error_type_counts[error_type] = 0
                error_type_counts[error_type] += 1
                
                failed_count += 1
                logger.error(f"规则执行错误: {rule.name} (ID: {rule.id}): {e}", exc_info=True)
        
        # 更新监控统计
        total_execution_time = time.time() - execution_start_time
        
        # 更新总执行次数
        self.monitoring_stats["total_executions"] += 1
        self.monitoring_stats["successful_executions"] += successful_count
        self.monitoring_stats["failed_executions"] += failed_count
        
        # 更新规则评估统计
        self.monitoring_stats["total_rules_evaluated"] += rules_evaluated
        self.monitoring_stats["rules_matched"] += rules_matched_count
        self.monitoring_stats["rules_executed"] += successful_count
        self.monitoring_stats["actions_executed"] += actions_executed_count
        
        # 更新执行时间统计
        if execution_times:
            avg_exec_time = sum(execution_times) / len(execution_times)
            # 更新执行时间历史
            self.monitoring_stats["execution_time_history"].append(avg_exec_time)
            # 只保留最近100次执行时间
            if len(self.monitoring_stats["execution_time_history"]) > 100:
                self.monitoring_stats["execution_time_history"].pop(0)
            
            # 更新平均执行时间
            total_history_time = sum(self.monitoring_stats["execution_time_history"])
            self.monitoring_stats["average_execution_time"] = total_history_time / len(self.monitoring_stats["execution_time_history"])
        
        # 更新动作类型统计
        for action_type, count in action_type_counts.items():
            if action_type not in self.monitoring_stats["action_type_stats"]:
                self.monitoring_stats["action_type_stats"][action_type] = 0
            self.monitoring_stats["action_type_stats"][action_type] += count
        
        # 更新错误类型统计
        for error_type, count in error_type_counts.items():
            if error_type not in self.monitoring_stats["error_types"]:
                self.monitoring_stats["error_types"][error_type] = 0
            self.monitoring_stats["error_types"][error_type] += count
        
        return results
    
    def evaluate_single_rule(self, rule_id: str, context: Dict[str, Any]) -> Optional[RuleExecutionResult]:
        """评估并执行单个规则"""
        rule = self.get_rule(rule_id)
        if not rule or not rule.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # 评估条件
            conditions_met = all([self._evaluate_condition(cond, context) for cond in rule.conditions])
            
            if conditions_met:
                # 执行规则
                rule.execute()
                
                # 执行动作
                action_results = []
                for action in rule.actions:
                    result = self._execute_action(action, context)
                    action_results.append(result)
                
                # 记录结果
                result = RuleExecutionResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    executed=True,
                    conditions_met=True,
                    actions_executed=len(action_results),
                    execution_time=time.time() - start_time,
                    results=action_results
                )
            else:
                # 条件不满足
                result = RuleExecutionResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    executed=False,
                    conditions_met=False,
                    actions_executed=0,
                    execution_time=time.time() - start_time
                )
            
            self._add_execution_log(result)
            return result
            
        except Exception as e:
            # 规则执行错误
            error_result = RuleExecutionResult(
                rule_id=rule.id,
                rule_name=rule.name,
                executed=False,
                conditions_met=False,
                actions_executed=0,
                execution_time=time.time() - start_time,
                error=str(e)
            )
            self._add_execution_log(error_result)
            logger.error(f"规则执行错误: {rule.name} (ID: {rule.id}): {e}", exc_info=True)
            return error_result
    
    def get_execution_logs(self, limit: Optional[int] = None) -> List[RuleExecutionResult]:
        """获取执行日志"""
        if limit:
            return self.execution_logs[-limit:]
        return self.execution_logs
    
    def clear_execution_logs(self) -> None:
        """清除执行日志"""
        self.execution_logs.clear()
        logger.info("清除所有执行日志")
    
    def _add_execution_log(self, log: RuleExecutionResult) -> None:
        """添加执行日志"""
        self.execution_logs.append(log)
        
        # 限制日志数量
        if len(self.execution_logs) > self.max_logs:
            self.execution_logs.pop(0)
    
    def register_action_handler(self, action_type: ActionType, handler: Callable) -> None:
        """注册动作处理器"""
        # 这里可以添加动作处理器的注册逻辑
        logger.info(f"注册动作处理器: {action_type.value}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计数据"""
        return self.monitoring_stats.copy()
    
    def reset_monitoring_stats(self) -> None:
        """重置监控统计数据"""
        self.monitoring_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_rules_evaluated": 0,
            "rules_matched": 0,
            "rules_executed": 0,
            "actions_executed": 0,
            "execution_time_history": [],
            "rules_by_tag": {},
            "action_type_stats": {},
            "error_types": {},
            "last_reset_time": time.time()
        }
        logger.info("监控统计数据已重置")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if self.monitoring_stats["total_executions"] == 0:
            return {
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "rules_per_second": 0.0,
                "actions_per_second": 0.0
            }
        
        success_rate = self.monitoring_stats["successful_executions"] / self.monitoring_stats["total_executions"]
        rules_per_second = self.monitoring_stats["total_rules_evaluated"] / (time.time() - self.monitoring_stats["last_reset_time"])
        actions_per_second = self.monitoring_stats["actions_executed"] / (time.time() - self.monitoring_stats["last_reset_time"])
        
        return {
            "success_rate": round(success_rate, 4),
            "average_execution_time": round(self.monitoring_stats["average_execution_time"], 4),
            "rules_per_second": round(rules_per_second, 4),
            "actions_per_second": round(actions_per_second, 4)
        }
    
    def get_rule_execution_stats(self) -> Dict[str, Any]:
        """获取规则执行统计"""
        rule_stats = {}
        for rule_id, rule in self.rules.items():
            rule_stats[rule_id] = {
                "name": rule.name,
                "execution_count": rule.execution_count,
                "last_executed": rule.last_executed,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "tags": rule.tags
            }
        
        return rule_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "rules": [rule.to_dict() for rule in self.rules.values()],
            "execution_logs": [log.to_dict() for log in self.execution_logs],
            "stats": {
                "total_rules": len(self.rules),
                "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
                "total_logs": len(self.execution_logs)
            },
            "monitoring": {
                "stats": self.get_monitoring_stats(),
                "performance": self.get_performance_metrics(),
                "rule_execution": self.get_rule_execution_stats()
            }
        }


# 规则引擎实例（单例模式）
rule_engine_instance = None


def get_rule_engine() -> RuleEngine:
    """获取规则引擎实例（单例模式）"""
    global rule_engine_instance
    if rule_engine_instance is None:
        rule_engine_instance = RuleEngine()
    return rule_engine_instance
