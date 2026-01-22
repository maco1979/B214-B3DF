"""
规则引擎API路由 - 处理规则的增删改查和执行
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import time
import uuid

from src.core.rule_engine import (
    RuleEngine, Condition, Action, Rule, RuleExecutionResult,
    ConditionOperator, ActionType,
    get_rule_engine
)

router = APIRouter(prefix="/rule-engine", tags=["rule_engine"])


class ConditionCreate(BaseModel):
    """条件创建请求"""
    left_operand: str
    operator: str  # 操作符字符串
    right_operand: Any
    is_right_operand_variable: bool = False
    conditions: Optional[List['ConditionCreate']] = None


class ActionCreate(BaseModel):
    """动作创建请求"""
    action_type: str  # 动作类型字符串
    parameters: Dict[str, Any]


class RuleCreate(BaseModel):
    """规则创建请求"""
    name: str
    description: str = ""
    conditions: List[ConditionCreate]
    actions: List[ActionCreate]
    priority: int = 50
    enabled: bool = True
    tags: List[str] = []


class RuleUpdate(BaseModel):
    """规则更新请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    conditions: Optional[List[ConditionCreate]] = None
    actions: Optional[List[ActionCreate]] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    tags: Optional[List[str]] = None


class RuleExecuteRequest(BaseModel):
    """规则执行请求"""
    context: Dict[str, Any]
    rule_ids: Optional[List[str]] = None  # 如果提供，只执行指定规则


class RuleStatusUpdate(BaseModel):
    """规则状态更新请求"""
    enabled: bool


class RuleExecutionLogQuery(BaseModel):
    """规则执行日志查询"""
    limit: int = 100
    offset: int = 0


@router.get("/")
async def get_rule_engine_info():
    """获取规则引擎信息"""
    rule_engine = get_rule_engine()
    info = rule_engine.to_dict()
    return {
        "success": True,
        "data": info
    }


@router.get("/rules")
async def get_all_rules():
    """获取所有规则"""
    rule_engine = get_rule_engine()
    rules = rule_engine.get_all_rules()
    return {
        "success": True,
        "data": {
            "rules": [rule.to_dict() for rule in rules],
            "total": len(rules)
        }
    }


@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    """获取单个规则"""
    rule_engine = get_rule_engine()
    rule = rule_engine.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    return {
        "success": True,
        "data": rule.to_dict()
    }


@router.get("/rules/tag/{tag}")
async def get_rules_by_tag(tag: str):
    """根据标签获取规则"""
    rule_engine = get_rule_engine()
    rules = rule_engine.get_rules_by_tag(tag)
    return {
        "success": True,
        "data": {
            "rules": [rule.to_dict() for rule in rules],
            "total": len(rules)
        }
    }


@router.post("/rules")
async def create_rule(rule_create: RuleCreate):
    """创建规则"""
    rule_engine = get_rule_engine()
    
    # 转换条件
    def convert_conditions(condition_creates: List[ConditionCreate]) -> List[Condition]:
        conditions = []
        for cond_create in condition_creates:
            sub_conditions = []
            if cond_create.conditions:
                sub_conditions = convert_conditions(cond_create.conditions)
            
            condition = Condition(
                left_operand=cond_create.left_operand,
                operator=ConditionOperator(cond_create.operator),
                right_operand=cond_create.right_operand,
                is_right_operand_variable=cond_create.is_right_operand_variable,
                conditions=sub_conditions
            )
            conditions.append(condition)
        return conditions
    
    # 转换动作
    def convert_actions(action_creates: List[ActionCreate]) -> List[Action]:
        actions = []
        for action_create in action_creates:
            action = Action(
                action_type=ActionType(action_create.action_type),
                parameters=action_create.parameters
            )
            actions.append(action)
        return actions
    
    # 创建规则
    rule = Rule(
        name=rule_create.name,
        description=rule_create.description,
        conditions=convert_conditions(rule_create.conditions),
        actions=convert_actions(rule_create.actions),
        priority=rule_create.priority,
        enabled=rule_create.enabled,
        tags=rule_create.tags
    )
    
    # 添加规则
    rule_id = rule_engine.add_rule(rule)
    
    return {
        "success": True,
        "data": {
            "rule_id": rule_id,
            "rule": rule.to_dict()
        }
    }


@router.put("/rules/{rule_id}")
async def update_rule(rule_id: str, rule_update: RuleUpdate):
    """更新规则"""
    rule_engine = get_rule_engine()
    existing_rule = rule_engine.get_rule(rule_id)
    
    if not existing_rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    # 更新规则字段
    if rule_update.name is not None:
        existing_rule.name = rule_update.name
    if rule_update.description is not None:
        existing_rule.description = rule_update.description
    if rule_update.priority is not None:
        existing_rule.priority = rule_update.priority
    if rule_update.enabled is not None:
        existing_rule.enabled = rule_update.enabled
    if rule_update.tags is not None:
        existing_rule.tags = rule_update.tags
    
    # 更新条件
    if rule_update.conditions is not None:
        def convert_conditions(condition_creates: List[ConditionCreate]) -> List[Condition]:
            conditions = []
            for cond_create in condition_creates:
                sub_conditions = []
                if cond_create.conditions:
                    sub_conditions = convert_conditions(cond_create.conditions)
                
                condition = Condition(
                    left_operand=cond_create.left_operand,
                    operator=ConditionOperator(cond_create.operator),
                    right_operand=cond_create.right_operand,
                    is_right_operand_variable=cond_create.is_right_operand_variable,
                    conditions=sub_conditions
                )
                conditions.append(condition)
            return conditions
        
        existing_rule.conditions = convert_conditions(rule_update.conditions)
    
    # 更新动作
    if rule_update.actions is not None:
        def convert_actions(action_creates: List[ActionCreate]) -> List[Action]:
            actions = []
            for action_create in action_creates:
                action = Action(
                    action_type=ActionType(action_create.action_type),
                    parameters=action_create.parameters
                )
                actions.append(action)
            return actions
        
        existing_rule.actions = convert_actions(rule_update.actions)
    
    # 保存更新
    success = rule_engine.update_rule(rule_id, existing_rule)
    
    if not success:
        raise HTTPException(status_code=500, detail="更新规则失败")
    
    return {
        "success": True,
        "data": {
            "rule_id": rule_id,
            "rule": existing_rule.to_dict()
        }
    }


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """删除规则"""
    rule_engine = get_rule_engine()
    success = rule_engine.delete_rule(rule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    return {
        "success": True,
        "data": {
            "message": "规则已删除"
        }
    }


@router.post("/rules/{rule_id}/enable")
async def enable_rule(rule_id: str):
    """启用规则"""
    rule_engine = get_rule_engine()
    success = rule_engine.enable_rule(rule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    return {
        "success": True,
        "data": {
            "message": "规则已启用"
        }
    }


@router.post("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str):
    """禁用规则"""
    rule_engine = get_rule_engine()
    success = rule_engine.disable_rule(rule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    return {
        "success": True,
        "data": {
            "message": "规则已禁用"
        }
    }


@router.post("/execute")
async def execute_rules(request: RuleExecuteRequest):
    """执行规则"""
    rule_engine = get_rule_engine()
    
    if request.rule_ids:
        # 执行指定规则
        results = []
        for rule_id in request.rule_ids:
            result = rule_engine.evaluate_single_rule(rule_id, request.context)
            if result:
                results.append(result.to_dict())
    else:
        # 执行所有规则
        execution_results = rule_engine.evaluate_rules(request.context)
        results = [result.to_dict() for result in execution_results]
    
    return {
        "success": True,
        "data": {
            "execution_results": results,
            "total_executed": len(results),
            "context": request.context
        }
    }


@router.post("/rules/{rule_id}/execute")
async def execute_single_rule(rule_id: str, context: Dict[str, Any]):
    """执行单个规则"""
    rule_engine = get_rule_engine()
    result = rule_engine.evaluate_single_rule(rule_id, context)
    
    if not result:
        raise HTTPException(status_code=404, detail="规则不存在或已禁用")
    
    return {
        "success": True,
        "data": {
            "execution_result": result.to_dict(),
            "context": context
        }
    }


@router.get("/logs")
async def get_execution_logs(limit: int = 100, offset: int = 0):
    """获取执行日志"""
    rule_engine = get_rule_engine()
    logs = rule_engine.get_execution_logs(limit)
    
    return {
        "success": True,
        "data": {
            "logs": [log.to_dict() for log in logs],
            "total": len(logs),
            "limit": limit,
            "offset": offset
        }
    }


@router.delete("/logs")
async def clear_execution_logs():
    """清除执行日志"""
    rule_engine = get_rule_engine()
    rule_engine.clear_execution_logs()
    
    return {
        "success": True,
        "data": {
            "message": "执行日志已清除"
        }
    }


@router.get("/operators")
async def get_condition_operators():
    """获取所有条件操作符"""
    operators = [op.value for op in ConditionOperator]
    return {
        "success": True,
        "data": {
            "operators": operators
        }
    }


@router.get("/action-types")
async def get_action_types():
    """获取所有动作类型"""
    action_types = [at.value for at in ActionType]
    return {
        "success": True,
        "data": {
            "action_types": action_types
        }
    }
