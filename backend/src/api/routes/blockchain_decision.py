"""
区块链决策API路由 - 提供区块链积分分配和交易决策的API接口
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

# 导入决策引擎
from ...core.decision.blockchain_decision_engine import (
    BlockchainDecisionEngine, 
    BlockchainState, 
    BlockchainObjective, 
    BlockchainDecisionResult
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/blockchain/decision", tags=["blockchain-decision"])

# 创建决策引擎实例（单例模式）
_decision_engine = None

def get_decision_engine() -> BlockchainDecisionEngine:
    """获取决策引擎实例"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = BlockchainDecisionEngine()
    return _decision_engine


class BlockchainStateRequest(BaseModel):
    """区块链状态请求模型"""
    user_contribution: float = 0.0
    user_activity: float = 0.0
    user_reputation: float = 0.0
    user_balance: float = 0.0
    
    market_demand: float = 0.0
    market_supply: float = 0.0
    transaction_volume: float = 0.0
    average_transaction_value: float = 0.0
    
    total_points_issued: float = 0.0
    points_in_circulation: float = 0.0
    system_utilization: float = 0.0
    risk_level: float = 0.0
    
    time_since_last_decision: float = 0.0
    
    objective: str = "maximize_ecosystem_growth"


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
    risk_control_effectiveness: float
    engine_status: str


@router.post("/make_decision", response_model=DecisionResponse)
async def make_blockchain_decision(
    request: BlockchainStateRequest,
    engine: BlockchainDecisionEngine = Depends(get_decision_engine)
):
    """
    基于当前区块链状态做出自主决策
    
    Args:
        request: 区块链状态和决策目标
        
    Returns:
        包含决策动作、参数和评估信息的响应
    """
    try:
        # 转换目标枚举
        objective_map = {
            "maximize_ecosystem_growth": BlockchainObjective.MAXIMIZE_ECOSYSTEM_GROWTH,
            "optimize_fairness": BlockchainObjective.OPTIMIZE_FAIRNESS,
            "minimize_risk": BlockchainObjective.MINIMIZE_RISK,
            "maximize_efficiency": BlockchainObjective.MAXIMIZE_EFFICIENCY
        }
        
        objective = objective_map.get(request.objective, BlockchainObjective.MAXIMIZE_ECOSYSTEM_GROWTH)
        
        # 创建区块链状态对象
        state = BlockchainState(
            user_contribution=request.user_contribution,
            user_activity=request.user_activity,
            user_reputation=request.user_reputation,
            user_balance=request.user_balance,
            
            market_demand=request.market_demand,
            market_supply=request.market_supply,
            transaction_volume=request.transaction_volume,
            average_transaction_value=request.average_transaction_value,
            
            total_points_issued=request.total_points_issued,
            points_in_circulation=request.points_in_circulation,
            system_utilization=request.system_utilization,
            risk_level=request.risk_level,
            
            time_since_last_decision=request.time_since_last_decision
        )
        
        # 使用决策引擎做出决策
        decision_result = engine.make_decision(state, objective)
        
        import time
        from datetime import datetime
        
        return DecisionResponse(
            action=decision_result.action.value,
            parameters=decision_result.parameters,
            expected_reward=decision_result.expected_reward,
            confidence=decision_result.confidence,
            execution_time=decision_result.execution_time,
            risk_assessment=decision_result.risk_assessment,
            decision_id=f"decision_{int(time.time() * 1000)}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"决策引擎错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"决策引擎错误: {str(e)}")


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_decision_performance(
    engine: BlockchainDecisionEngine = Depends(get_decision_engine)
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
            risk_control_effectiveness=metrics.get("risk_control_effectiveness", 0.0),
            engine_status="active"
        )
        
    except Exception as e:
        logger.error(f"获取性能指标错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标错误: {str(e)}")


@router.post("/update_policy")
async def update_decision_policy(
    state: BlockchainStateRequest,
    action: str,
    reward: float,
    next_state: BlockchainStateRequest,
    engine: BlockchainDecisionEngine = Depends(get_decision_engine)
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
            "allocate_points": engine.BlockchainAction.ALLOCATE_POINTS,
            "approve_transaction": engine.BlockchainAction.APPROVE_TRANSACTION,
            "set_transaction_fee": engine.BlockchainAction.SET_TRANSACTION_FEE,
            "adjust_incentives": engine.BlockchainAction.ADJUST_INCENTIVES,
            "risk_control": engine.BlockchainAction.RISK_CONTROL,
            "no_action": engine.BlockchainAction.NO_ACTION
        }
        
        action_enum = action_map.get(action, engine.BlockchainAction.NO_ACTION)
        
        # 创建状态对象
        current_state_obj = BlockchainState(
            user_contribution=state.user_contribution,
            user_activity=state.user_activity,
            user_reputation=state.user_reputation,
            user_balance=state.user_balance,
            
            market_demand=state.market_demand,
            market_supply=state.market_supply,
            transaction_volume=state.transaction_volume,
            average_transaction_value=state.average_transaction_value,
            
            total_points_issued=state.total_points_issued,
            points_in_circulation=state.points_in_circulation,
            system_utilization=state.system_utilization,
            risk_level=state.risk_level,
            
            time_since_last_decision=state.time_since_last_decision
        )
        
        next_state_obj = BlockchainState(
            user_contribution=next_state.user_contribution,
            user_activity=next_state.user_activity,
            user_reputation=next_state.user_reputation,
            user_balance=next_state.user_balance,
            
            market_demand=next_state.market_demand,
            market_supply=next_state.market_supply,
            transaction_volume=next_state.transaction_volume,
            average_transaction_value=next_state.average_transaction_value,
            
            total_points_issued=next_state.total_points_issued,
            points_in_circulation=next_state.points_in_circulation,
            system_utilization=next_state.system_utilization,
            risk_level=next_state.risk_level,
            
            time_since_last_decision=next_state.time_since_last_decision
        )
        
        # 更新策略
        engine.update_policy(current_state_obj, action_enum, reward, next_state_obj)
        
        return {"status": "success", "message": "策略更新成功"}
        
    except Exception as e:
        logger.error(f"策略更新错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"策略更新错误: {str(e)}")


@router.get("/health")
async def health_check():
    """
    决策引擎健康检查
    
    Returns:
        健康状态信息
    """
    return {
        "status": "healthy",
        "service": "blockchain_decision_engine",
        "timestamp": "2025-12-20T10:00:00Z"
    }