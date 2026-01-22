import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import json
import threading
from .scene_manager import scene_manager

logger = logging.getLogger(__name__)

# 事件类型枚举
class EventType(str, Enum):
    DEVICE_STATE_CHANGED = "device_state_changed"
    TIMER_TRIGGERED = "timer_triggered"
    CONDITION_MET = "condition_met"
    ACTION_EXECUTED = "action_executed"
    RULE_CREATED = "rule_created"
    RULE_UPDATED = "rule_updated"
    RULE_DELETED = "rule_deleted"

# 条件操作符枚举
class ConditionOperator(str, Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    ANY = "any"
    ALL = "all"

# 动作类型枚举
class ActionType(str, Enum):
    DEVICE_COMMAND = "device_command"
    DELAY = "delay"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    NOTIFICATION = "notification"
    WEBHOOK = "webhook"
    SCENE_TRIGGER = "scene_trigger"

# 条件模型
class Condition(BaseModel):
    """自动化规则条件"""
    type: str = Field(..., description="条件类型，如device_state、timer、weather等")
    device_id: Optional[str] = Field(None, description="设备ID")
    property: Optional[str] = Field(None, description="设备属性")
    operator: Optional[ConditionOperator] = Field(None, description="比较操作符")
    value: Optional[Union[str, int, float, bool, List[Any]]] = Field(None, description="比较值")
    conditions: Optional[List["Condition"]] = Field(None, description="子条件列表，用于组合条件")
    logic: Optional[str] = Field(None, description="子条件逻辑关系，如and、or")

# 动作模型
class Action(BaseModel):
    """自动化规则动作"""
    type: ActionType = Field(..., description="动作类型")
    device_id: Optional[str] = Field(None, description="设备ID")
    command: Optional[str] = Field(None, description="设备命令")
    parameters: Optional[Dict[str, Any]] = Field(None, description="命令参数")
    actions: Optional[List["Action"]] = Field(None, description="子动作列表")
    delay: Optional[int] = Field(None, description="延迟时间（毫秒）")
    url: Optional[str] = Field(None, description="Webhook URL")
    method: Optional[str] = Field(None, description="Webhook HTTP方法")
    headers: Optional[Dict[str, str]] = Field(None, description="Webhook HTTP头")
    body: Optional[Dict[str, Any]] = Field(None, description="Webhook请求体")
    scene_id: Optional[str] = Field(None, description="场景ID")
    message: Optional[str] = Field(None, description="通知消息")

# 自动化规则模型
class AutomationRule(BaseModel):
    """自动化规则"""
    id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    enabled: bool = Field(default=True, description="规则是否启用")
    conditions: List[Condition] = Field(..., description="规则条件列表")
    actions: List[Action] = Field(..., description="规则动作列表")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    last_executed: Optional[float] = Field(None, description="最后执行时间")
    execution_count: int = Field(default=0, description="执行次数")

# 设备状态模型
class DeviceState(BaseModel):
    """设备状态"""
    device_id: str = Field(..., description="设备ID")
    state: Dict[str, Any] = Field(..., description="设备状态")
    timestamp: float = Field(default_factory=time.time, description="状态更新时间")
    online: bool = Field(default=True, description="设备是否在线")

# 事件模型
class AutomationEvent(BaseModel):
    """自动化事件"""
    event_type: EventType = Field(..., description="事件类型")
    timestamp: float = Field(default_factory=time.time, description="事件时间")
    device_id: Optional[str] = Field(None, description="设备ID")
    rule_id: Optional[str] = Field(None, description="规则ID")
    data: Optional[Dict[str, Any]] = Field(None, description="事件数据")

# 规则执行日志模型
class RuleExecutionLog(BaseModel):
    """规则执行日志"""
    id: str = Field(..., description="日志ID")
    rule_id: str = Field(..., description="规则ID")
    rule_name: str = Field(..., description="规则名称")
    executed_at: float = Field(default_factory=time.time, description="执行时间")
    status: str = Field(..., description="执行状态")
    conditions_result: Dict[str, bool] = Field(..., description="条件评估结果")
    actions_result: List[Dict[str, Any]] = Field(..., description="动作执行结果")
    error: Optional[str] = Field(None, description="执行错误信息")

class DeviceAutomationEngine:
    """设备自动化引擎"""
    
    def __init__(self):
        self.rules: Dict[str, AutomationRule] = {}
        self.device_states: Dict[str, DeviceState] = {}
        self.event_queue = asyncio.Queue()
        self.is_running = False
        self.rule_execution_logs: List[RuleExecutionLog] = []
        self.max_logs = 1000
        self.lock = threading.Lock()
        self.execution_thread: Optional[threading.Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def start(self):
        """启动自动化引擎"""
        if not self.is_running:
            logger.info("启动设备自动化引擎")
            self.is_running = True
            self.event_loop = asyncio.new_event_loop()
            self.execution_thread = threading.Thread(target=self._run_event_loop)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            asyncio.run_coroutine_threadsafe(self._process_events(), self.event_loop)
            asyncio.run_coroutine_threadsafe(self._check_timer_rules(), self.event_loop)
    
    def stop(self):
        """停止自动化引擎"""
        if self.is_running:
            logger.info("停止设备自动化引擎")
            self.is_running = False
            if self.event_loop:
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            if self.execution_thread:
                self.execution_thread.join(timeout=5)
    
    def _run_event_loop(self):
        """运行事件循环"""
        logger.info("设备自动化引擎事件循环已启动")
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()
        logger.info("设备自动化引擎事件循环已停止")
    
    async def _process_events(self):
        """处理事件队列"""
        while self.is_running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._handle_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"处理事件时出错: {str(e)}", exc_info=True)
    
    async def _handle_event(self, event: AutomationEvent):
        """处理单个事件"""
        logger.info(f"处理事件: {event.event_type}, 设备ID: {event.device_id}, 规则ID: {event.rule_id}")
        
        if event.event_type == EventType.DEVICE_STATE_CHANGED:
            await self._on_device_state_changed(event.device_id, event.data)
        elif event.event_type == EventType.TIMER_TRIGGERED:
            await self._on_timer_triggered(event.rule_id)
        elif event.event_type == EventType.CONDITION_MET:
            await self._on_condition_met(event.rule_id, event.data)
    
    async def _on_device_state_changed(self, device_id: str, data: Optional[Dict[str, Any]]):
        """处理设备状态变化事件"""
        # 遍历所有启用的规则，检查条件是否满足
        for rule_id, rule in self.rules.items():
            if rule.enabled:
                await self._evaluate_rule_conditions(rule)
    
    async def _on_timer_triggered(self, rule_id: str):
        """处理定时器触发事件"""
        rule = self.rules.get(rule_id)
        if rule and rule.enabled:
            await self._evaluate_rule_conditions(rule)
    
    async def _on_condition_met(self, rule_id: str, data: Optional[Dict[str, Any]]):
        """处理条件满足事件"""
        rule = self.rules.get(rule_id)
        if rule and rule.enabled:
            await self._execute_rule_actions(rule)
    
    async def _check_timer_rules(self):
        """检查定时器规则"""
        while self.is_running:
            try:
                current_time = time.time()
                # 这里可以添加定时器规则的检查逻辑
                # 例如，检查规则中的定时器条件是否满足
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"检查定时器规则时出错: {str(e)}", exc_info=True)
    
    async def _evaluate_rule_conditions(self, rule: AutomationRule):
        """评估规则条件"""
        try:
            all_conditions_met = True
            conditions_result = {}
            
            for i, condition in enumerate(rule.conditions):
                condition_met = await self._evaluate_condition(condition)
                conditions_result[f"condition_{i}"] = condition_met
                if not condition_met:
                    all_conditions_met = False
            
            if all_conditions_met:
                logger.info(f"规则条件满足: {rule.name} ({rule.id})")
                await self.event_queue.put(AutomationEvent(
                    event_type=EventType.CONDITION_MET,
                    rule_id=rule.id,
                    data={"conditions_result": conditions_result}
                ))
            else:
                logger.debug(f"规则条件不满足: {rule.name} ({rule.id})")
        except Exception as e:
            logger.error(f"评估规则条件时出错: {str(e)}", exc_info=True)
    
    async def _evaluate_condition(self, condition: Condition) -> bool:
        """评估单个条件"""
        try:
            if condition.type == "device_state":
                return await self._evaluate_device_state_condition(condition)
            elif condition.type == "timer":
                return await self._evaluate_timer_condition(condition)
            elif condition.type == "combination":
                return await self._evaluate_combination_condition(condition)
            # 可以添加更多条件类型的评估
            return False
        except Exception as e:
            logger.error(f"评估条件时出错: {str(e)}", exc_info=True)
            return False
    
    async def _evaluate_device_state_condition(self, condition: Condition) -> bool:
        """评估设备状态条件"""
        if not condition.device_id or not condition.property or condition.operator is None:
            return False
        
        device_state = self.device_states.get(condition.device_id)
        if not device_state or not device_state.online:
            return False
        
        # 获取设备属性值
        property_value = device_state.state.get(condition.property)
        if property_value is None:
            return False
        
        # 比较值
        expected_value = condition.value
        operator = condition.operator
        
        logger.debug(f"评估设备状态条件: {condition.device_id}.{condition.property} {operator} {expected_value}, 当前值: {property_value}")
        
        # 根据操作符进行比较
        if operator == ConditionOperator.EQUAL:
            return property_value == expected_value
        elif operator == ConditionOperator.NOT_EQUAL:
            return property_value != expected_value
        elif operator == ConditionOperator.GREATER_THAN:
            return float(property_value) > float(expected_value)
        elif operator == ConditionOperator.LESS_THAN:
            return float(property_value) < float(expected_value)
        elif operator == ConditionOperator.GREATER_EQUAL:
            return float(property_value) >= float(expected_value)
        elif operator == ConditionOperator.LESS_EQUAL:
            return float(property_value) <= float(expected_value)
        elif operator == ConditionOperator.CONTAINS:
            return str(expected_value) in str(property_value)
        elif operator == ConditionOperator.NOT_CONTAINS:
            return str(expected_value) not in str(property_value)
        
        return False
    
    async def _evaluate_timer_condition(self, condition: Condition) -> bool:
        """评估定时器条件"""
        # 简单的定时器条件评估，实际实现可以更复杂
        return True
    
    async def _evaluate_combination_condition(self, condition: Condition) -> bool:
        """评估组合条件"""
        if not condition.conditions:
            return False
        
        if condition.logic == "and":
            # 所有子条件都满足
            for sub_condition in condition.conditions:
                if not await self._evaluate_condition(sub_condition):
                    return False
            return True
        elif condition.logic == "or":
            # 至少一个子条件满足
            for sub_condition in condition.conditions:
                if await self._evaluate_condition(sub_condition):
                    return True
            return False
        
        return False
    
    async def _execute_rule_actions(self, rule: AutomationRule):
        """执行规则动作"""
        logger.info(f"执行规则动作: {rule.name} ({rule.id})")
        
        # 更新规则执行统计
        rule.last_executed = time.time()
        rule.execution_count += 1
        rule.updated_at = time.time()
        
        # 创建执行日志
        log = RuleExecutionLog(
            id=f"log_{time.time()}_{rule.id}",
            rule_id=rule.id,
            rule_name=rule.name,
            status="executing",
            conditions_result={},  # 实际应从条件评估结果获取
            actions_result=[]
        )
        
        actions_result = []
        try:
            for action in rule.actions:
                action_result = await self._execute_action(action)
                actions_result.append(action_result)
            
            log.status = "success"
            log.actions_result = actions_result
        except Exception as e:
            log.status = "failed"
            log.error = str(e)
            log.actions_result = actions_result
            logger.error(f"执行规则动作时出错: {str(e)}", exc_info=True)
        
        # 添加到日志
        self._add_execution_log(log)
        
        # 发送动作执行事件
        await self.event_queue.put(AutomationEvent(
            event_type=EventType.ACTION_EXECUTED,
            rule_id=rule.id,
            data={"actions_result": actions_result}
        ))
    
    async def _execute_action(self, action: Action) -> Dict[str, Any]:
        """执行单个动作"""
        logger.info(f"执行动作: {action.type}, 设备ID: {action.device_id}")
        
        start_time = time.time()
        result = {
            "action_type": action.type,
            "device_id": action.device_id,
            "status": "success",
            "execution_time": 0
        }
        
        try:
            if action.type == ActionType.DEVICE_COMMAND:
                result.update(await self._execute_device_command(action))
            elif action.type == ActionType.DELAY:
                result.update(await self._execute_delay(action))
            elif action.type == ActionType.CONDITIONAL:
                result.update(await self._execute_conditional(action))
            elif action.type == ActionType.PARALLEL:
                result.update(await self._execute_parallel(action))
            elif action.type == ActionType.SEQUENTIAL:
                result.update(await self._execute_sequential(action))
            elif action.type == ActionType.NOTIFICATION:
                result.update(await self._execute_notification(action))
            elif action.type == ActionType.WEBHOOK:
                result.update(await self._execute_webhook(action))
            elif action.type == ActionType.SCENE_TRIGGER:
                result.update(await self._execute_scene_trigger(action))
            else:
                result["status"] = "error"
                result["error"] = f"不支持的动作类型: {action.type}"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"执行动作时出错: {str(e)}", exc_info=True)
        
        result["execution_time"] = time.time() - start_time
        return result
    
    async def _execute_device_command(self, action: Action) -> Dict[str, Any]:
        """执行设备命令动作"""
        if not action.device_id or not action.command:
            return {
                "status": "error",
                "error": "设备ID和命令不能为空"
            }
        
        # 这里应该调用设备控制器发送命令
        # 暂时模拟命令执行
        logger.info(f"发送设备命令: 设备ID: {action.device_id}, 命令: {action.command}, 参数: {action.parameters}")
        
        # 模拟命令执行延迟
        await asyncio.sleep(0.1)
        
        return {
            "command": action.command,
            "parameters": action.parameters,
            "message": "设备命令已发送"
        }
    
    async def _execute_delay(self, action: Action) -> Dict[str, Any]:
        """执行延迟动作"""
        delay = action.delay or 0
        logger.info(f"执行延迟动作: {delay}毫秒")
        
        await asyncio.sleep(delay / 1000)
        
        return {
            "delay": delay,
            "message": f"延迟 {delay} 毫秒已完成"
        }
    
    async def _execute_conditional(self, action: Action) -> Dict[str, Any]:
        """执行条件动作"""
        logger.info(f"执行条件动作")
        
        # 条件动作需要扩展，这里暂时返回成功
        return {
            "message": "条件动作执行成功"
        }
    
    async def _execute_parallel(self, action: Action) -> Dict[str, Any]:
        """执行并行动作"""
        if not action.actions:
            return {
                "status": "error",
                "error": "并行动作列表不能为空"
            }
        
        logger.info(f"执行并行动作: {len(action.actions)}个动作")
        
        # 并行执行所有动作
        tasks = [self._execute_action(sub_action) for sub_action in action.actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "parallel_actions_count": len(action.actions),
            "results": results,
            "message": f"并行执行 {len(action.actions)} 个动作已完成"
        }
    
    async def _execute_sequential(self, action: Action) -> Dict[str, Any]:
        """执行顺序动作"""
        if not action.actions:
            return {
                "status": "error",
                "error": "顺序动作列表不能为空"
            }
        
        logger.info(f"执行顺序动作: {len(action.actions)}个动作")
        
        # 顺序执行所有动作
        results = []
        for sub_action in action.actions:
            result = await self._execute_action(sub_action)
            results.append(result)
            if result["status"] == "error":
                break
        
        return {
            "sequential_actions_count": len(action.actions),
            "results": results,
            "message": f"顺序执行 {len(results)} 个动作已完成"
        }
    
    async def _execute_notification(self, action: Action) -> Dict[str, Any]:
        """执行通知动作"""
        logger.info(f"执行通知动作: {action.message}")
        
        # 这里应该调用通知服务发送通知
        # 暂时模拟通知发送
        return {
            "message": action.message,
            "notification_status": "sent"
        }
    
    async def _execute_webhook(self, action: Action) -> Dict[str, Any]:
        """执行Webhook动作"""
        if not action.url:
            return {
                "status": "error",
                "error": "Webhook URL不能为空"
            }
        
        logger.info(f"执行Webhook动作: {action.method} {action.url}")
        
        # 这里应该调用HTTP客户端发送Webhook请求
        # 暂时模拟Webhook发送
        return {
            "url": action.url,
            "method": action.method,
            "webhook_status": "sent"
        }
    
    async def _execute_scene_trigger(self, action: Action) -> Dict[str, Any]:
        """执行场景触发动作"""
        if not action.scene_id:
            return {
                "status": "error",
                "error": "场景ID不能为空"
            }
        
        logger.info(f"执行场景触发动作: 场景ID {action.scene_id}")
        
        # 调用场景管理器触发场景
        success = await scene_manager.trigger_scene(action.scene_id)
        
        return {
            "scene_id": action.scene_id,
            "scene_status": "triggered" if success else "failed",
            "success": success
        }
    
    def _add_execution_log(self, log: RuleExecutionLog):
        """添加执行日志"""
        with self.lock:
            self.rule_execution_logs.append(log)
            # 限制日志数量
            if len(self.rule_execution_logs) > self.max_logs:
                self.rule_execution_logs.pop(0)
    
    # 规则管理方法
    def create_rule(self, rule: AutomationRule) -> AutomationRule:
        """创建新的自动化规则"""
        with self.lock:
            self.rules[rule.id] = rule
        
        # 发送规则创建事件
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put(AutomationEvent(
                event_type=EventType.RULE_CREATED,
                rule_id=rule.id
            )),
            self.event_loop
        )
        
        logger.info(f"创建规则: {rule.name} ({rule.id})")
        return rule
    
    def get_rule(self, rule_id: str) -> Optional[AutomationRule]:
        """获取指定规则"""
        with self.lock:
            return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[AutomationRule]:
        """获取所有规则"""
        with self.lock:
            return list(self.rules.values())
    
    def update_rule(self, rule_id: str, rule: AutomationRule) -> Optional[AutomationRule]:
        """更新规则"""
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id] = rule
                
                # 发送规则更新事件
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put(AutomationEvent(
                        event_type=EventType.RULE_UPDATED,
                        rule_id=rule_id
                    )),
                    self.event_loop
                )
                
                logger.info(f"更新规则: {rule.name} ({rule_id})")
                return rule
        return None
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        with self.lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                
                # 发送规则删除事件
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put(AutomationEvent(
                        event_type=EventType.RULE_DELETED,
                        rule_id=rule_id
                    )),
                    self.event_loop
                )
                
                logger.info(f"删除规则: {rule_id}")
                return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = True
                self.rules[rule_id].updated_at = time.time()
                logger.info(f"启用规则: {rule_id}")
                return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = False
                self.rules[rule_id].updated_at = time.time()
                logger.info(f"禁用规则: {rule_id}")
                return True
        return False
    
    # 设备状态管理方法
    def update_device_state(self, device_id: str, state: Dict[str, Any]):
        """更新设备状态"""
        with self.lock:
            if device_id in self.device_states:
                self.device_states[device_id].state = state
                self.device_states[device_id].timestamp = time.time()
            else:
                self.device_states[device_id] = DeviceState(
                    device_id=device_id,
                    state=state
                )
        
        # 发送设备状态变化事件
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put(AutomationEvent(
                event_type=EventType.DEVICE_STATE_CHANGED,
                device_id=device_id,
                data=state
            )),
            self.event_loop
        )
        
        logger.debug(f"更新设备状态: {device_id}, 状态: {state}")
    
    def get_device_state(self, device_id: str) -> Optional[DeviceState]:
        """获取设备状态"""
        with self.lock:
            return self.device_states.get(device_id)
    
    def get_all_device_states(self) -> Dict[str, DeviceState]:
        """获取所有设备状态"""
        with self.lock:
            return self.device_states.copy()
    
    def update_device_online_status(self, device_id: str, online: bool):
        """更新设备在线状态"""
        with self.lock:
            if device_id in self.device_states:
                self.device_states[device_id].online = online
                self.device_states[device_id].timestamp = time.time()
            else:
                self.device_states[device_id] = DeviceState(
                    device_id=device_id,
                    state={},
                    online=online
                )
        
        logger.debug(f"更新设备在线状态: {device_id}, 在线: {online}")
    
    # 日志管理方法
    def get_rule_execution_logs(self, rule_id: Optional[str] = None, limit: int = 100) -> List[RuleExecutionLog]:
        """获取规则执行日志"""
        with self.lock:
            if rule_id:
                logs = [log for log in self.rule_execution_logs if log.rule_id == rule_id]
            else:
                logs = self.rule_execution_logs.copy()
            
            # 按执行时间倒序排序
            logs.sort(key=lambda x: x.executed_at, reverse=True)
            
            # 限制返回数量
            return logs[:limit]
    
    def clear_rule_execution_logs(self, rule_id: Optional[str] = None):
        """清除规则执行日志"""
        with self.lock:
            if rule_id:
                self.rule_execution_logs = [log for log in self.rule_execution_logs if log.rule_id != rule_id]
            else:
                self.rule_execution_logs.clear()
        
        logger.info(f"清除规则执行日志, 规则ID: {rule_id}")

# 全局设备自动化引擎实例
device_automation_engine = DeviceAutomationEngine()

# 引擎启动函数
def start_automation_engine():
    """启动设备自动化引擎"""
    device_automation_engine.start()

# 引擎停止函数
def stop_automation_engine():
    """停止设备自动化引擎"""
    device_automation_engine.stop()

# 初始化函数
def init_automation_engine():
    """初始化设备自动化引擎"""
    logger.info("初始化设备自动化引擎")
    # 可以在这里添加初始化逻辑，如加载规则、恢复设备状态等
    start_automation_engine()

# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 启动自动化引擎
    start_automation_engine()
    
    # 创建示例规则
    sample_rule = AutomationRule(
        id="rule_001",
        name="温度过高时关闭空调",
        description="当温度超过28度时，自动关闭空调",
        enabled=True,
        conditions=[
            Condition(
                type="device_state",
                device_id="device_thermostat_001",
                property="temperature",
                operator=ConditionOperator.GREATER_THAN,
                value=28
            )
        ],
        actions=[
            Action(
                type=ActionType.DEVICE_COMMAND,
                device_id="device_ac_001",
                command="turn_off",
                parameters={}
            ),
            Action(
                type=ActionType.NOTIFICATION,
                message="温度过高，已自动关闭空调"
            )
        ]
    )
    
    # 添加规则
    device_automation_engine.create_rule(sample_rule)
    
    # 更新设备状态
    device_automation_engine.update_device_state("device_thermostat_001", {"temperature": 30})
    
    # 等待规则执行
    time.sleep(2)
    
    # 获取规则执行日志
    logs = device_automation_engine.get_rule_execution_logs(rule_id="rule_001")
    for log in logs:
        print(f"规则执行日志: {log.id}, 状态: {log.status}, 执行时间: {log.executed_at}")
    
    # 停止自动化引擎
    stop_automation_engine()
    print("设备自动化引擎示例运行完成")
