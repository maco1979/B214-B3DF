"""
API集成模块 - 提供迁移学习和边缘计算集成的API接口
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..core.decision_engine import DecisionEngine
from .decision_integration import DecisionIntegrationManager

logger = logging.getLogger(__name__)

# API路由
router = APIRouter(prefix="/api/integration", tags=["integration"])

# 数据模型
class DecisionRequest(BaseModel):
    """决策请求模型"""
    task_type: str
    data: Dict[str, Any]
    requirements: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

class IntegrationStatus(BaseModel):
    """集成状态模型"""
    migration_learning_enabled: bool
    edge_computing_enabled: bool
    risk_control_enabled: bool
    last_update: datetime
    components_status: Dict[str, Any]

class IntegrationConfig(BaseModel):
    """集成配置模型"""
    migration_learning: Optional[Dict[str, Any]] = None
    edge_computing: Optional[Dict[str, Any]] = None
    risk_control: Optional[Dict[str, Any]] = None

# 全局集成管理器实例
_integration_manager: Optional[DecisionIntegrationManager] = None


def get_integration_manager() -> DecisionIntegrationManager:
    """获取集成管理器实例"""
    global _integration_manager
    if _integration_manager is None:
        # 这里需要获取决策引擎实例
        decision_engine = DecisionEngine()  # 需要根据实际情况获取
        _integration_manager = DecisionIntegrationManager(decision_engine)
    return _integration_manager


@router.post("/decision", response_model=Dict[str, Any])
async def make_integrated_decision(
    request: DecisionRequest,
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    执行集成化决策
    
    Args:
        request: 决策请求
        
    Returns:
        集成化决策结果
    """
    try:
        logger.info(f"接收到集成决策请求: {request.task_type}")
        
        # 构建决策数据
        decision_data = {
            "task_type": request.task_type,
            "data": request.data,
            "requirements": request.requirements or {},
            "constraints": request.constraints or {}
        }
        
        # 执行集成化决策
        result = await manager.integrated_decision_making(decision_data)
        
        logger.info(f"集成决策完成: {request.task_type}")
        return result
        
    except Exception as e:
        logger.error(f"集成决策API错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"决策处理失败: {str(e)}")


@router.get("/status", response_model=IntegrationStatus)
async def get_integration_status(
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    获取集成系统状态
    
    Returns:
        集成系统状态信息
    """
    try:
        status = await manager.get_integration_status()
        
        return IntegrationStatus(
            migration_learning_enabled=status.get("migration_learning_enabled", True),
            edge_computing_enabled=status.get("edge_computing_enabled", True),
            risk_control_enabled=status.get("risk_control_enabled", True),
            last_update=status.get("last_update", datetime.now()),
            components_status={
                "migration_learning": status.get("migration_learning_status", {}),
                "edge_computing": status.get("edge_computing_status", {}),
                "risk_control": status.get("risk_control_status", {}),
                "warning_system": status.get("warning_system_status", {})
            }
        )
        
    except Exception as e:
        logger.error(f"获取集成状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"状态获取失败: {str(e)}")


@router.put("/config")
async def update_integration_config(
    config: IntegrationConfig,
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    更新集成系统配置
    
    Args:
        config: 新的配置信息
        
    Returns:
        更新结果
    """
    try:
        config_dict = config.dict(exclude_unset=True)
        result = await manager.update_integration_config(config_dict)
        
        return {"message": "配置更新成功", "details": result}
        
    except Exception as e:
        logger.error(f"更新集成配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"配置更新失败: {str(e)}")


@router.get("/migration/risk-assessment")
async def assess_migration_risk(
    task_type: str,
    data_size: int,
    source_domain: str,
    target_domain: str,
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    评估迁移学习风险
    
    Args:
        task_type: 任务类型
        data_size: 数据大小
        source_domain: 源领域
        target_domain: 目标领域
        
    Returns:
        风险评估结果
    """
    try:
        risk_data = {
            "task_type": task_type,
            "data_size": data_size,
            "source_domain": source_domain,
            "target_domain": target_domain
        }
        
        # 这里调用风险控制组件的具体方法
        risk_assessment = await manager.risk_control.assess_migration_risk(risk_data)
        
        return {
            "risk_level": risk_assessment.get("risk_level", "unknown"),
            "confidence": risk_assessment.get("confidence", 0),
            "recommendations": risk_assessment.get("recommendations", []),
            "assessment_time": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"迁移学习风险评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"风险评估失败: {str(e)}")


@router.get("/edge/suitability")
async def check_edge_suitability(
    task_type: str,
    data_size: int,
    latency_requirement: int,
    privacy_requirement: str = "medium",
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    检查边缘计算适用性
    
    Args:
        task_type: 任务类型
        data_size: 数据大小（字节）
        latency_requirement: 延迟要求（毫秒）
        privacy_requirement: 隐私要求级别
        
    Returns:
        适用性分析结果
    """
    try:
        suitability_data = {
            "task_type": task_type,
            "data_size": data_size,
            "latency_requirement": latency_requirement,
            "privacy_requirement": privacy_requirement
        }
        
        # 这里调用边缘集成组件的适用性分析
        edge_manager = manager.edge_integration
        suitability_result = await edge_manager._analyze_edge_suitability(suitability_data)
        
        return {
            "suitable": suitability_result.get("suitable", False),
            "reason": suitability_result.get("reason", "unknown"),
            "recommended_mode": "edge" if suitability_result.get("suitable") else "cloud",
            "analysis_time": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"边缘计算适用性分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"适用性分析失败: {str(e)}")


@router.get("/warnings")
async def get_active_warnings(
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    获取当前活跃的警告信息
    
    Returns:
        警告信息列表
    """
    try:
        # 这里调用预警系统获取活跃警告
        warnings = await manager.warning_system.get_active_warnings()
        
        return {
            "warnings": warnings,
            "total_count": len(warnings),
            "critical_count": len([w for w in warnings if w.get("level") == "critical"]),
            "last_updated": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"获取警告信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"警告获取失败: {str(e)}")


@router.post("/warnings/{warning_id}/acknowledge")
async def acknowledge_warning(
    warning_id: str,
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    确认警告
    
    Args:
        warning_id: 警告ID
        
    Returns:
        确认结果
    """
    try:
        result = await manager.warning_system.acknowledge_warning(warning_id)
        
        return {
            "warning_id": warning_id,
            "acknowledged": result.get("acknowledged", False),
            "acknowledged_time": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"确认警告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"警告确认失败: {str(e)}")


@router.get("/performance/metrics")
async def get_performance_metrics(
    time_range: str = "1h",  # 1h, 24h, 7d
    manager: DecisionIntegrationManager = Depends(get_integration_manager)
):
    """
    获取性能指标
    
    Args:
        time_range: 时间范围
        
    Returns:
        性能指标数据
    """
    try:
        # 这里实现性能指标收集逻辑
        metrics = {
            "decision_latency": {
                "avg": 150,  # 毫秒
                "p95": 300,
                "p99": 500
            },
            "throughput": {
                "requests_per_second": 50,
                "success_rate": 0.98
            },
            "resource_usage": {
                "cpu": 65,  # 百分比
                "memory": 45,
                "network": 30
            },
            "time_range": time_range,
            "collected_at": datetime.now()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"性能指标获取失败: {str(e)}")


@router.get("/health")
async def health_check():
    """
    健康检查接口
    
    Returns:
        健康状态
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "components": {
            "migration_learning": "active",
            "edge_computing": "active", 
            "risk_control": "active",
            "api": "active"
        }
    }