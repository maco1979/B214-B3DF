"""
AI模型训练决策API路由 - 提供AI模型自动训练决策的API接口
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
import time
from datetime import datetime

# 使用绝对导入
from src.core.decision.model_training_decision_engine import (
    ModelTrainingDecisionEngine, 
    TrainingState, 
    TrainingObjective, 
    TrainingDecisionResult,
    TrainingAction
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training/decision", tags=["training-decision"])

# 创建决策引擎实例（单例模式）
_decision_engine = None

def get_decision_engine() -> ModelTrainingDecisionEngine:
    """获取决策引擎实例"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = ModelTrainingDecisionEngine()
    return _decision_engine


class TrainingStateRequest(BaseModel):
    """训练状态请求模型"""
    current_accuracy: float = 0.0
    best_accuracy: float = 0.0
    current_loss: float = 1.0
    best_loss: float = 1.0
    
    training_epochs: int = 0
    total_training_time: float = 0.0
    convergence_rate: float = 0.0
    
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    cpu_utilization: float = 0.0
    
    dataset_size: int = 0
    data_quality: float = 0.5
    class_imbalance: float = 0.0
    
    is_training: bool = False
    model_complexity: float = 1.0
    recent_improvement: float = 0.0
    
    objective: str = "maximize_accuracy"


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
    training_efficiency: float
    engine_status: str


@router.post("/make_decision", response_model=DecisionResponse)
async def make_training_decision(
    request: TrainingStateRequest,
    engine: ModelTrainingDecisionEngine = Depends(get_decision_engine)
):
    """
    基于当前训练状态做出自主决策
    
    Args:
        request: 训练状态和决策目标
        
    Returns:
        包含决策动作、参数和评估信息的响应
    """
    try:
        # 转换目标枚举
        objective_map = {
            "maximize_accuracy": TrainingObjective.MAXIMIZE_ACCURACY,
            "minimize_training_time": TrainingObjective.MINIMIZE_TRAINING_TIME,
            "optimize_resource_usage": TrainingObjective.OPTIMIZE_RESOURCE_USAGE,
            "balance_performance_resource": TrainingObjective.BALANCE_PERFORMANCE_RESOURCE
        }
        
        objective = objective_map.get(request.objective, TrainingObjective.MAXIMIZE_ACCURACY)
        
        # 创建训练状态对象
        state = TrainingState(
            current_accuracy=request.current_accuracy,
            best_accuracy=request.best_accuracy,
            current_loss=request.current_loss,
            best_loss=request.best_loss,
            
            training_epochs=request.training_epochs,
            total_training_time=request.total_training_time,
            convergence_rate=request.convergence_rate,
            
            gpu_utilization=request.gpu_utilization,
            memory_usage=request.memory_usage,
            cpu_utilization=request.cpu_utilization,
            
            dataset_size=request.dataset_size,
            data_quality=request.data_quality,
            class_imbalance=request.class_imbalance,
            
            is_training=request.is_training,
            model_complexity=request.model_complexity,
            recent_improvement=request.recent_improvement
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
            decision_id=f"training_decision_{int(time.time() * 1000)}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"训练决策引擎错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"训练决策引擎错误: {str(e)}")


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_decision_performance(
    engine: ModelTrainingDecisionEngine = Depends(get_decision_engine)
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
            training_efficiency=metrics.get("training_efficiency", 0.0),
            engine_status="active"
        )
        
    except Exception as e:
        logger.error(f"获取性能指标错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标错误: {str(e)}")


@router.post("/update_policy")
async def update_decision_policy(
    state: TrainingStateRequest,
    action: str,
    reward: float,
    next_state: TrainingStateRequest,
    engine: ModelTrainingDecisionEngine = Depends(get_decision_engine)
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
            "start_training": TrainingAction.START_TRAINING,
            "stop_training": TrainingAction.STOP_TRAINING,
            "adjust_hyperparams": TrainingAction.ADJUST_HYPERPARAMS,
            "switch_model_type": TrainingAction.SWITCH_MODEL_TYPE,
            "data_augmentation": TrainingAction.DATA_AUGMENTATION,
            "no_action": TrainingAction.NO_ACTION
        }
        
        action_enum = action_map.get(action, TrainingAction.NO_ACTION)
        
        # 创建状态对象
        current_state_obj = TrainingState(
            current_accuracy=state.current_accuracy,
            best_accuracy=state.best_accuracy,
            current_loss=state.current_loss,
            best_loss=state.best_loss,
            
            training_epochs=state.training_epochs,
            total_training_time=state.total_training_time,
            convergence_rate=state.convergence_rate,
            
            gpu_utilization=state.gpu_utilization,
            memory_usage=state.memory_usage,
            cpu_utilization=state.cpu_utilization,
            
            dataset_size=state.dataset_size,
            data_quality=state.data_quality,
            class_imbalance=state.class_imbalance,
            
            is_training=state.is_training,
            model_complexity=state.model_complexity,
            recent_improvement=state.recent_improvement
        )
        
        next_state_obj = TrainingState(
            current_accuracy=next_state.current_accuracy,
            best_accuracy=next_state.best_accuracy,
            current_loss=next_state.current_loss,
            best_loss=next_state.best_loss,
            
            training_epochs=next_state.training_epochs,
            total_training_time=next_state.total_training_time,
            convergence_rate=next_state.convergence_rate,
            
            gpu_utilization=next_state.gpu_utilization,
            memory_usage=next_state.memory_usage,
            cpu_utilization=next_state.cpu_utilization,
            
            dataset_size=next_state.dataset_size,
            data_quality=next_state.data_quality,
            class_imbalance=next_state.class_imbalance,
            
            is_training=next_state.is_training,
            model_complexity=next_state.model_complexity,
            recent_improvement=next_state.recent_improvement
        )
        
        # 更新策略
        engine.update_policy(current_state_obj, action_enum, reward, next_state_obj)
        
        return {"status": "success", "message": "策略更新成功"}
        
    except Exception as e:
        logger.error(f"策略更新错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"策略更新错误: {str(e)}")


@router.post("/execute_training_action")
async def execute_training_action(
    decision_response: DecisionResponse,
    engine: ModelTrainingDecisionEngine = Depends(get_decision_engine)
):
    """
    执行训练决策动作
    
    Args:
        decision_response: 决策结果
        
    Returns:
        执行结果
    """
    try:
        action = decision_response.action
        parameters = decision_response.parameters
        
        # 根据动作类型执行相应操作
        if action == "start_training":
            result = await _execute_start_training(parameters)
        elif action == "stop_training":
            result = await _execute_stop_training(parameters)
        elif action == "adjust_hyperparams":
            result = await _execute_adjust_hyperparams(parameters)
        elif action == "switch_model_type":
            result = await _execute_switch_model(parameters)
        elif action == "data_augmentation":
            result = await _execute_data_augmentation(parameters)
        else:
            result = {"status": "no_action", "message": "无需执行动作"}
        
        return {
            "action": action,
            "parameters": parameters,
            "execution_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"执行训练动作错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行训练动作错误: {str(e)}")


async def _execute_start_training(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行开始训练动作"""
    # 这里应该调用实际的训练服务
    # 简化实现，返回模拟结果
    return {
        "status": "training_started",
        "learning_rate": parameters.get("learning_rate", 1e-4),
        "batch_size": int(parameters.get("batch_size", 32)),
        "num_epochs": int(parameters.get("num_epochs", 100)),
        "message": "训练任务已启动"
    }


async def _execute_stop_training(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行停止训练动作"""
    immediate_stop = parameters.get("immediate_stop", 0.0) > 0.5
    save_checkpoint = parameters.get("save_checkpoint", 1.0) > 0.5
    
    return {
        "status": "training_stopped",
        "immediate_stop": immediate_stop,
        "save_checkpoint": save_checkpoint,
        "message": "训练任务已停止"
    }


async def _execute_adjust_hyperparams(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行超参数调整动作"""
    return {
        "status": "hyperparams_adjusted",
        "learning_rate_multiplier": parameters.get("learning_rate_multiplier", 1.0),
        "batch_size_multiplier": parameters.get("batch_size_multiplier", 1.0),
        "message": "超参数已调整"
    }


async def _execute_switch_model(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行模型切换动作"""
    return {
        "status": "model_switched",
        "target_model_type": parameters.get("target_model_type", "balanced"),
        "complexity_adjustment": parameters.get("complexity_adjustment", 1.0),
        "message": "模型类型已切换"
    }


async def _execute_data_augmentation(parameters: Dict[str, float]) -> Dict[str, Any]:
    """执行数据增强动作"""
    return {
        "status": "data_augmented",
        "augmentation_intensity": parameters.get("augmentation_intensity", 0.7),
        "rotation_range": parameters.get("rotation_range", 15.0),
        "message": "数据增强已应用"
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
        "service": "training_decision_engine",
        "timestamp": datetime.now().isoformat()
    }