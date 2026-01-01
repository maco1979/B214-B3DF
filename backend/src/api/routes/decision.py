"""
AI自主决策API路由 - 处理农业参数优化、区块链积分分配等决策功能
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import time

from src.core.decision.agriculture_decision_engine import (
    AgricultureDecisionEngine, AgricultureState, DecisionAction, DecisionObjective, DecisionResult
)
from src.core.models.agriculture_model import AgricultureAIService, SpectrumConfig
from src.integration.decision_integration import DecisionIntegrationManager
from src.core.ai_organic_core import organic_ai_core, get_organic_ai_core

router = APIRouter(prefix="/decision", tags=["decision"])

# 全局决策引擎实例（延迟初始化）
agriculture_service = AgricultureAIService()
decision_engine = None
# 决策集成管理器
decision_integration_manager = None

# 有机体AI核心
organic_ai_core_instance = None

async def initialize_decision_services():
    """初始化决策服务"""
    global decision_engine, decision_integration_manager, organic_ai_core_instance
    if decision_engine is None:
        decision_engine = AgricultureDecisionEngine(agriculture_service)
        decision_integration_manager = DecisionIntegrationManager(decision_engine)
    if organic_ai_core_instance is None:
        organic_ai_core_instance = await get_organic_ai_core()


class DecisionRequest(BaseModel):
    """决策请求"""
    # 环境参数
    temperature: float
    humidity: float
    co2_level: float = 400.0
    light_intensity: float
    
    # 光谱参数
    spectrum_config: Dict[str, float]
    
    # 作物信息
    crop_type: str
    growth_day: int
    growth_rate: float
    health_score: float
    yield_potential: float
    
    # 系统状态
    energy_consumption: float
    resource_utilization: float
    
    # 决策目标
    objective: str  # "maximize_yield", "improve_quality", "enhance_resistance", "optimize_efficiency"
    
    # 任务信息
    task_type: str  # "routine_monitoring", "high_priority", "critical_decision"
    risk_level: str = "medium"  # "low", "medium", "high"


class BatchDecisionRequest(BaseModel):
    """批量决策请求"""
    requests: List[DecisionRequest]
    batch_id: str
    priority: str = "normal"  # "low", "normal", "high", "critical"


class DecisionFeedback(BaseModel):
    """决策反馈"""
    decision_id: str
    actual_reward: float
    next_state: Dict[str, Any]
    success_indicator: bool
    feedback_notes: Optional[str] = None


class DecisionHistoryQuery(BaseModel):
    """决策历史查询"""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    crop_type: Optional[str] = None
    objective: Optional[str] = None
    limit: int = 100
    offset: int = 0


@router.post("/agriculture")
async def make_agriculture_decision(request: DecisionRequest):
    """农业参数优化决策"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        # 将请求转换为字典格式
        decision_request_dict = {
            "temperature": request.temperature,
            "humidity": request.humidity,
            "co2_level": request.co2_level,
            "light_intensity": request.light_intensity,
            "spectrum_config": request.spectrum_config,
            "crop_type": request.crop_type,
            "growth_day": request.growth_day,
            "growth_rate": request.growth_rate,
            "health_score": request.health_score,
            "yield_potential": request.yield_potential,
            "energy_consumption": request.energy_consumption,
            "resource_utilization": request.resource_utilization,
            "objective": request.objective,
            "task_type": request.task_type,
            "risk_level": request.risk_level
        }
        
        # 首先尝试使用有机体AI核心进行决策
        start_time = time.time()
        try:
            organic_decision = await organic_ai_core_instance.make_decision(decision_request_dict)
            decision_result = {
                "action": organic_decision.action,
                "parameters": organic_decision.parameters,
                "expected_reward": organic_decision.expected_reward,
                "confidence": organic_decision.confidence,
                "risk_assessment": organic_decision.risk_assessment
            }
            execution_time = time.time() - start_time
        except Exception as e:
            print(f"有机体AI核心决策失败，回退到传统决策: {e}")
            # 如果有机体AI核心失败，回退到传统决策
            decision_result = await decision_integration_manager.integrated_decision_making(decision_request_dict)
            execution_time = time.time() - start_time
        
        # 生成决策ID
        decision_id = f"abri_decision_{int(time.time()*1000)}"
        
        # 获取建议
        action = decision_result.get("action", "no_action")
        parameters = decision_result.get("parameters", {})
        recommendations = _get_action_recommendations(action, parameters)
        
        return {
            "success": True,
            "data": {
                "decision_id": decision_id,
                "action": action,
                "parameters": parameters,
                "expected_reward": decision_result.get("expected_reward", 0.0),
                "confidence": decision_result.get("confidence", 0.0),
                "execution_time": execution_time,
                "recommendations": recommendations,
                "risk_assessment": decision_result.get("risk_assessment", {})
            }
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Agriculture Decision API Error: {e}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/risk")
async def make_risk_analysis_decision(request: DecisionRequest):
    """农业风险分析决策"""
    return await make_agriculture_decision(request)


@router.post("/agriculture/batch")
async def make_batch_agriculture_decisions(request: BatchDecisionRequest):
    """批量农业决策"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        results = []
        total_execution_time = 0
        
        for i, req in enumerate(request.requests):
            # 转换请求为字典格式
            decision_request_dict = {
                "temperature": req.temperature,
                "humidity": req.humidity,
                "co2_level": req.co2_level,
                "light_intensity": req.light_intensity,
                "spectrum_config": req.spectrum_config,
                "crop_type": req.crop_type,
                "growth_day": req.growth_day,
                "growth_rate": req.growth_rate,
                "health_score": req.health_score,
                "yield_potential": req.yield_potential,
                "energy_consumption": req.energy_consumption,
                "resource_utilization": req.resource_utilization,
                "objective": req.objective,
                "task_type": req.task_type,
                "risk_level": req.risk_level
            }
            
            # 使用有机体AI核心进行批量决策
            start_time = time.time()
            try:
                organic_decision = await organic_ai_core_instance.make_decision(decision_request_dict)
                exec_time = time.time() - start_time
                total_execution_time += exec_time
                
                results.append({
                    "decision_id": f"batch_{request.batch_id}_decision_{i}",
                    "crop_type": req.crop_type,
                    "action": organic_decision.action,
                    "parameters": organic_decision.parameters,
                    "expected_reward": organic_decision.expected_reward,
                    "confidence": organic_decision.confidence,
                    "execution_time": exec_time
                })
            except Exception as e:
                print(f"有机体AI核心批量决策失败，回退到传统决策: {e}")
                # 回退到传统决策
                state = AgricultureState(
                    temperature=req.temperature,
                    humidity=req.humidity,
                    co2_level=req.co2_level,
                    light_intensity=req.light_intensity,
                    spectrum_config=SpectrumConfig(**req.spectrum_config),
                    crop_type=req.crop_type,
                    growth_day=req.growth_day,
                    growth_rate=req.growth_rate,
                    health_score=req.health_score,
                    yield_potential=req.yield_potential,
                    energy_consumption=req.energy_consumption,
                    resource_utilization=req.resource_utilization
                )
                
                objective_map = {
                    "maximize_yield": DecisionObjective.MAXIMIZE_YIELD,
                    "improve_quality": DecisionObjective.IMPROVE_QUALITY,
                    "enhance_resistance": DecisionObjective.ENHANCE_RESISTANCE,
                    "optimize_efficiency": DecisionObjective.OPTIMIZE_EFFICIENCY
                }
                objective = objective_map.get(req.objective, DecisionObjective.MAXIMIZE_YIELD)
                
                decision_result = decision_engine.make_decision(state, objective)
                exec_time = time.time() - start_time
                total_execution_time += exec_time
                
                results.append({
                    "decision_id": f"batch_{request.batch_id}_decision_{i}",
                    "crop_type": req.crop_type,
                    "action": decision_result.action.value,
                    "parameters": decision_result.parameters,
                    "expected_reward": decision_result.expected_reward,
                    "confidence": decision_result.confidence,
                    "execution_time": exec_time
                })
        
        return {
            "success": True,
            "data": {
                "batch_id": request.batch_id,
                "total_decisions": len(results),
                "total_execution_time": total_execution_time,
                "average_execution_time": total_execution_time / len(results) if results else 0,
                "decisions": results
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/feedback")
async def provide_decision_feedback(feedback: DecisionFeedback):
    """提供决策反馈"""
    try:
        # 这里应该实现反馈处理逻辑
        # 目前只是记录反馈
        
        feedback_record = {
            "decision_id": feedback.decision_id,
            "actual_reward": feedback.actual_reward,
            "next_state": feedback.next_state,
            "success_indicator": feedback.success_indicator,
            "feedback_notes": feedback.feedback_notes,
            "timestamp": time.time()
        }
        
        # 在实际系统中，这里应该更新决策引擎的策略
        # decision_engine.update_policy(...)
        
        return {
            "success": True,
            "data": {
                "feedback_id": f"feedback_{int(time.time()*1000)}",
                "message": "反馈已记录，将用于改进决策策略"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/performance")
async def get_decision_performance():
    """获取决策性能指标"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        
        # 获取有机体AI核心状态
        organic_status = organic_ai_core_instance.get_status()
        
        # 获取传统决策引擎性能指标
        metrics = decision_engine.get_performance_metrics()
        
        # 合并性能指标
        performance_data = {
            "organic_ai_core": organic_status,
            "traditional_engine": metrics,
            "system_status": {
                "decision_engine_ready": True,
                "agriculture_service_ready": True,
                "organic_ai_core_ready": True,
                "last_decision_time": time.time()
            }
        }
        
        return {
            "success": True,
            "data": performance_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/history")
async def get_decision_history(query: DecisionHistoryQuery):
    """查询决策历史"""
    try:
        # 这里应该从数据库查询历史记录
        # 目前返回模拟数据
        
        mock_history = [
            {
                "decision_id": f"decision_{i}",
                "timestamp": time.time() - i * 3600,  # 模拟时间戳
                "crop_type": "番茄",
                "action": "adjust_spectrum",
                "objective": "maximize_yield",
                "expected_reward": 0.85 + i * 0.01,
                "confidence": 0.92 - i * 0.02
            }
            for i in range(min(query.limit, 10))
        ]
        
        return {
            "success": True,
            "data": {
                "total_count": len(mock_history),
                "decisions": mock_history,
                "query_info": {
                    "limit": query.limit,
                    "offset": query.offset
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/optimize")
async def optimize_decision_parameters():
    """优化决策参数"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        
        # 这里应该实现决策参数的优化算法
        # 目前返回模拟优化结果
        
        optimization_result = {
            "optimization_id": f"opt_{int(time.time()*1000)}",
            "status": "completed",
            "improvements": {
                "policy_convergence": 0.15,
                "reward_improvement": 0.08,
                "execution_efficiency": 0.12
            },
            "new_parameters": {
                "learning_rate": 0.001,
                "exploration_rate": 0.1,
                "discount_factor": 0.95
            }
        }
        
        return {
            "success": True,
            "data": optimization_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.post("/organic-core/activate-iteration")
async def activate_organic_ai_iteration():
    """激活有机体AI核心的主动迭代功能"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        
        # 启动有机体AI核心的主动迭代
        await organic_ai_core_instance.start_active_iteration()
        
        return {
            "success": True,
            "message": "有机体AI核心主动迭代已激活",
            "data": organic_ai_core_instance.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/organic-core/deactivate-iteration")
async def deactivate_organic_ai_iteration():
    """停用有机体AI核心的主动迭代功能"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        
        # 停止有机体AI核心的主动迭代
        await organic_ai_core_instance.stop_active_iteration()
        
        return {
            "success": True,
            "message": "有机体AI核心主动迭代已停用",
            "data": organic_ai_core_instance.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/organic-core/status")
async def get_organic_ai_status():
    """获取有机体AI核心状态"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        
        status = organic_ai_core_instance.get_status()
        
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/organic-core/evolve-structure")
async def evolve_organic_ai_structure():
    """演化有机体AI核心的网络结构"""
    try:
        # 初始化决策服务
        await initialize_decision_services()
        
        # 执行网络结构演化
        await organic_ai_core_instance.evolve_network_structure()
        
        return {
            "success": True,
            "message": "有机体AI核心网络结构演化完成",
            "data": organic_ai_core_instance.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_action_recommendations(action: DecisionAction, parameters: Dict[str, float]) -> List[str]:
    """获取动作执行建议"""
    recommendations = []
    
    if action == DecisionAction.ADJUST_SPECTRUM:
        recommendations.append("调整LED光谱配比")
        if parameters.get("red_660nm_adjustment", 0) > 0:
            recommendations.append("增加红光强度促进光合作用")
        if parameters.get("uv_380nm_adjustment", 0) > 0:
            recommendations.append("增加紫外线促进次生代谢")
    
    elif action == DecisionAction.ADJUST_TEMPERATURE:
        adjustment = parameters.get("temperature_adjustment", 0)
        if adjustment > 0:
            recommendations.append(f"提高温度 {adjustment*5:.1f}°C")
        else:
            recommendations.append(f"降低温度 {abs(adjustment)*5:.1f}°C")
    
    elif action == DecisionAction.ADJUST_HUMIDITY:
        adjustment = parameters.get("humidity_adjustment", 0)
        if adjustment > 0:
            recommendations.append(f"增加湿度 {adjustment*20:.1f}%")
        else:
            recommendations.append(f"降低湿度 {abs(adjustment)*20:.1f}%")
    
    elif action == DecisionAction.ADJUST_NUTRIENTS:
        recommendations.append("调整营养液配比")
        if parameters.get("phosphorus_boost", 0) > 0:
            recommendations.append("增加磷肥促进开花")
        if parameters.get("potassium_boost", 0) > 0:
            recommendations.append("增加钾肥促进果实发育")
    
    else:
        recommendations.append("维持当前参数设置")
    
    return recommendations