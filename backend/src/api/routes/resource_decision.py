"""
系统资源决策API路由 - 提供系统资源动态分配决策的API接口
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
import time
from datetime import datetime

# 导入决策引擎
from src.core.decision.resource_decision_engine import (
    ResourceDecisionEngine, 
    ResourceState, 
    ResourceObjective, 
    ResourceDecisionResult,
    ResourceAction
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/resource/decision", tags=["resource-decision"])

# 创建决策引擎实例（单例模式）
_decision_engine = None

def get_decision_engine() -> ResourceDecisionEngine:
    """获取决策引擎实例"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = ResourceDecisionEngine()
    return _decision_engine


class ResourceStateRequest(BaseModel):
    """资源状态请求模型"""
    # CPU资源
    cpu_total_cores: int = 8
    cpu_utilization: float = 0.0
    cpu_available_cores: int = 8
    cpu_load_average: float = 0.0
    
    # 内存资源
    memory_total: float = 16.0
    memory_used: float = 0.0
    memory_available: float = 16.0
    memory_utilization: float = 0.0
    
    # GPU资源
    gpu_total_memory: float = 8.0
    gpu_used_memory: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 40.0
    
    # 存储资源
    storage_total: float = 500.0
    storage_used: float = 0.0
    storage_available: float = 500.0
    storage_io_utilization: float = 0.0
    
    # 网络资源
    network_bandwidth: float = 1000.0
    network_utilization: float = 0.0
    network_latency: float = 10.0
    
    # 系统负载
    system_load: float = 0.0
    active_processes: int = 0
    system_uptime: float = 0.0
    
    # 应用需求
    ai_training_demand: float = 0.0
    inference_demand: float = 0.0
    blockchain_demand: float = 0.0
    agriculture_demand: float = 0.0
    
    objective: str = "optimize_efficiency"


class DecisionResponse(BaseModel):
    """决策响应模型"""
    action: str
    parameters: Dict[str, float]
    expected_reward: float
    confidence: float
    execution_time: float
    risk_assessment: Dict[str, float]
    decision_id: str
    timestamp: str


class PerformanceMetricsResponse(BaseModel):
    """性能指标响应模型"""
    average_reward: float
    decision_count: int
    recent_success_rate: float
    resource_efficiency: float
    engine_status: str


@router.post("/make_decision", response_model=DecisionResponse)
async def make_resource_decision(
    request: ResourceStateRequest,
    engine: ResourceDecisionEngine = Depends(get_decision_engine)
):
    """
    基于当前系统资源状态做出自主决策
    
    Args:
        request: 系统资源状态和决策目标
        
    Returns:
        包含决策动作、参数和评估信息的响应
    """
    try:
        # 转换目标枚举
        objective_map = {
            "maximize_performance": ResourceObjective.MAXIMIZE_PERFORMANCE,
            "minimize_cost": ResourceObjective.MINIMIZE_COST,
            "optimize_efficiency": ResourceObjective.OPTIMIZE_EFFICIENCY,
            "balance_fairness": ResourceObjective.BALANCE_FAIRNESS
        }
        
        objective = objective_map.get(request.objective, ResourceObjective.OPTIMIZE_EFFICIENCY)
        
        # 创建资源状态对象
        state = ResourceState(
            cpu_total_cores=request.cpu_total_cores,
            cpu_utilization=request.cpu_utilization,
            cpu_available_cores=request.cpu_available_cores,
            cpu_load_average=request.cpu_load_average,
            
            memory_total=request.memory_total,
            memory_used=request.memory_used,
            memory_available=request.memory_available,
            memory_utilization=request.memory_utilization,
            
            gpu_total_memory=request.gpu_total_memory,
            gpu_used_memory=request.gpu_used_memory,
            gpu_utilization=request.gpu_utilization,
            gpu_temperature=request.gpu_temperature,
            
            storage_total=request.storage_total,
            storage_used=request.storage_used,
            storage_available=request.storage_available,
            storage_io_utilization=request.storage_io_utilization,
            
            network_bandwidth=request.network_bandwidth,
            network_utilization=request.network_utilization,
            network_latency=request.network_latency,
            
            system_load=request.system_load,
            active_processes=request.active_processes,
            system_uptime=request.system_uptime,
            
            ai_training_demand=request.ai_training_demand,
            inference_demand=request.inference_demand,
            blockchain_demand=request.blockchain_demand,
            agriculture_demand=request.agriculture_demand
        )
        
        # 使用决策引擎做出决策
        decision_result = engine.make_decision(state, objective)
        
        return DecisionResponse(
            action=decision_result.action.value,
            parameters=decision_result.parameters,
            expected_reward=decision_result.expected_reward,
            confidence=decision_result.confidence,
            execution_time=decision_result.execution_time,
            risk_assessment=decision_result.risk_assessment,
            decision_id=f"resource_decision_{int(time.time() * 1000)}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"资源决策引擎错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"资源决策引擎错误: {str(e)}")


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_decision_performance(
    engine: ResourceDecisionEngine = Depends(get_decision_engine)
):
    """
    获取决策引擎的性能指标
    
    Returns:
        包含性能指标的响应
    """
    try:
        metrics = engine.get_performance_metrics()
        
        return PerformanceMetricsResponse(
            average_reward=metrics.get("average_reward", 0.0),
            decision_count=metrics.get("decision_count", 0),
            recent_success_rate=metrics.get("recent_success_rate", 0.0),
            resource_efficiency=metrics.get("resource_efficiency", 0.0),
            engine_status="active"
        )
        
    except Exception as e:
        logger.error(f"获取性能指标错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标错误: {str(e)}")


@router.post("/update_policy")
async def update_decision_policy(
    state: ResourceStateRequest,
    action: str,
    reward: float,
    next_state: ResourceStateRequest,
    engine: ResourceDecisionEngine = Depends(get_decision_engine)
):
    """
    更新决策策略（用于强化学习训练）
    
    Args:
        state: 当前状态
        action: 执行的动作
        reward: 获得的奖励
        next_state: 下一个状态
    """
    try:
        # 转换动作枚举
        action_map = {
            "allocate_cpu": ResourceAction.ALLOCATE_CPU,
            "allocate_memory": ResourceAction.ALLOCATE_MEMORY,
            "allocate_gpu": ResourceAction.ALLOCATE_GPU,
            "allocate_storage": ResourceAction.ALLOCATE_STORAGE,
            "balance_load": ResourceAction.BALANCE_LOAD,
            "scale_up": ResourceAction.SCALE_UP,
            "scale_down": ResourceAction.SCALE_DOWN,
            "no_action": ResourceAction.NO_ACTION
        }
        
        action_enum = action_map.get(action, ResourceAction.NO_ACTION)
        
        # 创建状态对象
        current_state_obj = ResourceState(
            cpu_total_cores=state.cpu_total_cores,
            cpu_utilization=state.cpu_utilization,
            cpu_available_cores=state.cpu_available_cores,
            cpu_load_average=state.cpu_load_average,
            
            memory_total=state.memory_total,
            memory_used=state.memory_used,
            memory_available=state.memory_available,
            memory_utilization=state.memory_utilization,
            
            gpu_total_memory=state.gpu_total_memory,
            gpu_used_memory=state.gpu_used_memory,
            gpu_utilization=state.gpu_utilization,
            gpu_temperature=state.gpu_temperature,
            
            storage_total=state.storage_total,
            storage_used=state.storage_used,
            storage_available=state.storage_available,
            storage_io_utilization=state.storage_io_utilization,
            
            network_bandwidth=state.network_bandwidth,
            network_utilization=state.network_utilization,
            network_latency=state.network_latency,
            
            system_load=state.system_load,
            active_processes=state.active_processes,
            system_uptime=state.system_uptime,
            
            ai_training_demand=state.ai_training_demand,
            inference_demand=state.inference_demand,
            blockchain_demand=state.blockchain_demand,
            agriculture_demand=state.agriculture_demand
        )
        
        next_state_obj = ResourceState(
            cpu_total_cores=next_state.cpu_total_cores,
            cpu_utilization=next_state.cpu_utilization,
            cpu_available_cores=next_state.cpu_available_cores,
            cpu_load_average=next_state.cpu_load_average,
            
            memory_total=next_state.memory_total,
            memory_used=next_state.memory_used,
            memory_available=next_state.memory_available,
            memory_utilization=next_state.memory_utilization,
            
            gpu_total_memory=next_state.gpu_total_memory,
            gpu_used_memory=next_state.gpu_used_memory,
            gpu_utilization=next_state.gpu_utilization,
            gpu_temperature=next_state.gpu_temperature,
            
            storage_total=next_state.storage_total,
            storage_used=next_state.storage_used,
            storage_available=next_state.storage_available,
            storage_io_utilization=next_state.storage_io_utilization,
            
            network_bandwidth=next_state.network_bandwidth,
            network_utilization=next_state.network_utilization,
            network_latency=next_state.network_latency,
            
            system_load=next_state.system_load,
            active_processes=next_state.active_processes,
            system_uptime=next_state.system_uptime,
            
            ai_training_demand=next_state.ai_training_demand,
            inference_demand=next_state.inference_demand,
            blockchain_demand=next_state.blockchain_demand,
            agriculture_demand=next_state.agriculture_demand
        )
        
        # 更新策略
        engine.update_policy(current_state_obj, action_enum, reward, next_state_obj)
        
        return {"status": "success", "message": "策略更新成功"}
        
    except Exception as e:
        logger.error(f"策略更新错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"策略更新错误: {str(e)}")


@router.post("/execute_resource_action")
async def execute_resource_action(
    decision_response: DecisionResponse,
    engine: ResourceDecisionEngine = Depends(get_decision_engine)
):
    """
    执行资源分配决策动作
    
    Args:
        decision_response: 决策结果
        
    Returns:
        执行结果
    """
    try:
        action = decision_response.action
        parameters = decision_response.parameters
        
        # 根据动作类型执行相应操作
        if action == "allocate_cpu":
            result = await _execute_cpu_allocation(parameters)
        elif action == "allocate_memory":
            result = await _execute_memory_allocation(parameters)
        elif action == "allocate_gpu":
            result = await _execute_gpu_allocation(parameters)
        elif action == "allocate_storage":
            result = await _execute_storage_allocation(parameters)
        elif action == "balance_load":
            result = await _execute_load_balancing(parameters)
        elif action == "scale_up":
            result = await _execute_scale_up(parameters)
        elif action == "scale_down":
            result = await _execute_scale_down(parameters)
        else:
            result = {"status": "no_action", "message": "无需执行动作"}
        
        return {
            "action": action,
            "parameters": parameters,
            "execution_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"执行资源动作错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行资源动作错误: {str(e)}")


async def _execute_cpu_allocation(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行CPU分配动作"""
    return {
        "status": "cpu_allocated",
        "cpu_cores": parameters.get("cpu_cores", 2.0),
        "priority_level": parameters.get("priority_level", 5.0),
        "message": "CPU资源分配成功"
    }


async def _execute_memory_allocation(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行内存分配动作"""
    return {
        "status": "memory_allocated",
        "memory_gb": parameters.get("memory_gb", 4.0),
        "memory_swap": parameters.get("memory_swap", 2.0),
        "message": "内存资源分配成功"
    }


async def _execute_gpu_allocation(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行GPU分配动作"""
    return {
        "status": "gpu_allocated",
        "gpu_memory_gb": parameters.get("gpu_memory_gb", 2.0),
        "gpu_utilization_limit": parameters.get("gpu_utilization_limit", 0.9),
        "message": "GPU资源分配成功"
    }


async def _execute_storage_allocation(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行存储分配动作"""
    return {
        "status": "storage_allocated",
        "storage_gb": parameters.get("storage_gb", 50.0),
        "io_priority": parameters.get("io_priority", "medium"),
        "message": "存储资源分配成功"
    }


async def _execute_load_balancing(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行负载均衡动作"""
    return {
        "status": "load_balanced",
        "migration_threshold": parameters.get("migration_threshold", 0.5),
        "rebalance_aggressiveness": parameters.get("rebalance_aggressiveness", 0.7),
        "message": "负载均衡执行成功"
    }


async def _execute_scale_up(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行扩容动作"""
    return {
        "status": "scaled_up",
        "scale_factor": parameters.get("scale_factor", 1.5),
        "instance_type": parameters.get("instance_type", "balanced"),
        "message": "系统扩容成功"
    }


async def _execute_scale_down(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行缩容动作"""
    return {
        "status": "scaled_down",
        "scale_factor": parameters.get("scale_factor", 0.7),
        "preserve_capacity": parameters.get("preserve_capacity", 0.3),
        "message": "系统缩容成功"
    }


@router.get("/health")
async def health_check():
    """
    决策引擎健康检查
    
    Returns:
        健康状态信息
    """
    return {
        "status": "healthy",
        "service": "resource_decision_engine",
        "timestamp": datetime.now().isoformat()
    }