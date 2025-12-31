"""
AI决策监控API路由 - 提供决策效果监控和反馈的API接口
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import time
from datetime import datetime

# 导入监控系统
from src.core.decision.decision_monitoring_system import (
    DecisionMonitoringSystem, 
    DecisionModule, 
    DecisionRecord,
    DecisionPerformance
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/decision/monitoring", tags=["decision-monitoring"])

# 创建监控系统实例（单例模式）
_monitoring_system = None

def get_monitoring_system() -> DecisionMonitoringSystem:
    """获取监控系统实例"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = DecisionMonitoringSystem()
    return _monitoring_system


class DecisionRecordRequest(BaseModel):
    """决策记录请求模型"""
    decision_id: str
    module: str
    action: str
    parameters: Dict[str, float]
    expected_reward: float
    confidence: float
    execution_time: float
    risk_assessment: Optional[Dict[str, float]] = None


class DecisionResultRequest(BaseModel):
    """决策结果请求模型"""
    decision_id: str
    actual_reward: float


class PerformanceMetricsRequest(BaseModel):
    """性能指标请求模型"""
    module: Optional[str] = None
    time_range_hours: int = 24


class AnalyticsRequest(BaseModel):
    """分析数据请求模型"""
    module: Optional[str] = None
    time_range_hours: int = 24


@router.post("/record_decision")
async def record_decision(
    request: DecisionRecordRequest,
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    记录决策
    
    Args:
        request: 决策记录信息
        
    Returns:
        记录结果
    """
    try:
        # 转换模块枚举
        module_map = {
            "agriculture": DecisionModule.AGRICULTURE,
            "blockchain": DecisionModule.BLOCKCHAIN,
            "model_training": DecisionModule.MODEL_TRAINING,
            "resource_allocation": DecisionModule.RESOURCE_ALLOCATION
        }
        
        module = module_map.get(request.module)
        if not module:
            raise HTTPException(status_code=400, detail=f"不支持的决策模块: {request.module}")
        
        # 记录决策
        record_id = monitoring_system.record_decision(
            decision_id=request.decision_id,
            module=module,
            action=request.action,
            parameters=request.parameters,
            expected_reward=request.expected_reward,
            confidence=request.confidence,
            execution_time=request.execution_time,
            risk_assessment=request.risk_assessment
        )
        
        return {
            "status": "success",
            "record_id": record_id,
            "message": "决策记录成功",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"记录决策错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"记录决策错误: {str(e)}")


@router.post("/update_result")
async def update_decision_result(
    request: DecisionResultRequest,
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    更新决策结果
    
    Args:
        request: 决策结果信息
        
    Returns:
        更新结果
    """
    try:
        success = monitoring_system.update_decision_result(
            decision_id=request.decision_id,
            actual_reward=request.actual_reward
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"未找到决策记录: {request.decision_id}")
        
        return {
            "status": "success",
            "message": "决策结果更新成功",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"更新决策结果错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新决策结果错误: {str(e)}")


@router.get("/performance/{module}")
async def get_performance_metrics(
    module: str,
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    获取指定模块的性能指标
    
    Args:
        module: 决策模块名称
        
    Returns:
        性能指标
    """
    try:
        # 转换模块枚举
        module_map = {
            "agriculture": DecisionModule.AGRICULTURE,
            "blockchain": DecisionModule.BLOCKCHAIN,
            "model_training": DecisionModule.MODEL_TRAINING,
            "resource_allocation": DecisionModule.RESOURCE_ALLOCATION
        }
        
        decision_module = module_map.get(module)
        if not decision_module:
            raise HTTPException(status_code=400, detail=f"不支持的决策模块: {module}")
        
        # 计算性能指标
        metrics = monitoring_system.calculate_performance_metrics(decision_module)
        
        return {
            "module": module,
            "metrics": metrics.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取性能指标错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标错误: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    获取所有预警信息
    
    Returns:
        预警列表
    """
    try:
        alerts = monitoring_system.check_alerts()
        
        return {
            "total_alerts": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取预警信息错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取预警信息错误: {str(e)}")


@router.get("/feedback/{module}")
async def get_feedback(
    module: str,
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    获取指定模块的反馈建议
    
    Args:
        module: 决策模块名称
        
    Returns:
        反馈建议
    """
    try:
        # 转换模块枚举
        module_map = {
            "agriculture": DecisionModule.AGRICULTURE,
            "blockchain": DecisionModule.BLOCKCHAIN,
            "model_training": DecisionModule.MODEL_TRAINING,
            "resource_allocation": DecisionModule.RESOURCE_ALLOCATION
        }
        
        decision_module = module_map.get(module)
        if not decision_module:
            raise HTTPException(status_code=400, detail=f"不支持的决策模块: {module}")
        
        # 生成反馈建议
        feedback = monitoring_system.generate_feedback(decision_module)
        
        return feedback
        
    except Exception as e:
        logger.error(f"获取反馈建议错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取反馈建议错误: {str(e)}")


@router.post("/analytics")
async def get_decision_analytics(
    request: AnalyticsRequest,
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    获取决策分析数据
    
    Args:
        request: 分析请求参数
        
    Returns:
        分析数据
    """
    try:
        # 转换模块枚举
        module = None
        if request.module:
            module_map = {
                "agriculture": DecisionModule.AGRICULTURE,
                "blockchain": DecisionModule.BLOCKCHAIN,
                "model_training": DecisionModule.MODEL_TRAINING,
                "resource_allocation": DecisionModule.RESOURCE_ALLOCATION
            }
            module = module_map.get(request.module)
            if not module:
                raise HTTPException(status_code=400, detail=f"不支持的决策模块: {request.module}")
        
        # 获取分析数据
        analytics = monitoring_system.get_decision_analytics(
            module=module,
            time_range_hours=request.time_range_hours
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"获取分析数据错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取分析数据错误: {str(e)}")


@router.get("/report")
async def export_performance_report(
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    导出性能报告
    
    Returns:
        性能报告
    """
    try:
        report = monitoring_system.export_performance_report()
        
        return report
        
    except Exception as e:
        logger.error(f"导出性能报告错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出性能报告错误: {str(e)}")


@router.post("/auto_optimize")
async def auto_optimize_system(
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    自动优化决策系统
    
    Returns:
        优化结果
    """
    try:
        optimization_results = monitoring_system.auto_optimize()
        
        return {
            "status": "success",
            "optimization_results": optimization_results,
            "message": "系统自动优化完成",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"自动优化错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"自动优化错误: {str(e)}")


@router.get("/modules")
async def list_decision_modules():
    """
    列出所有决策模块
    
    Returns:
        模块列表
    """
    modules = [
        {
            "name": "agriculture",
            "description": "农业参数优化决策模块",
            "endpoints": [
                "/api/v1/agriculture/decision/make_decision",
                "/api/v1/agriculture/decision/performance"
            ]
        },
        {
            "name": "blockchain",
            "description": "区块链积分分配决策模块",
            "endpoints": [
                "/api/v1/blockchain/decision/make_decision",
                "/api/v1/blockchain/decision/performance"
            ]
        },
        {
            "name": "model_training",
            "description": "AI模型自动训练决策模块",
            "endpoints": [
                "/api/v1/training/decision/make_decision",
                "/api/v1/training/decision/performance"
            ]
        },
        {
            "name": "resource_allocation",
            "description": "系统资源动态分配决策模块",
            "endpoints": [
                "/api/v1/resource/decision/make_decision",
                "/api/v1/resource/decision/performance"
            ]
        }
    ]
    
    return {
        "total_modules": len(modules),
        "modules": modules,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def health_check(
    monitoring_system: DecisionMonitoringSystem = Depends(get_monitoring_system)
):
    """
    监控系统健康检查
    
    Returns:
        健康状态信息
    """
    try:
        # 检查系统状态
        total_records = len(monitoring_system.decision_records)
        
        return {
            "status": "healthy",
            "service": "decision_monitoring_system",
            "total_decision_records": total_records,
            "monitoring_since": monitoring_system.last_update_time.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"健康检查错误: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "decision_monitoring_system",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }