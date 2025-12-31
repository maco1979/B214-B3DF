"""
AI自主决策风险控制系统API接口

提供RESTful API接口，方便外部系统集成和调用风险控制功能，
支持实时风险评估、警报查询、应急响应等功能。
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio

from .risk_monitoring_system import (
    AIRiskMonitoringSystem, SystemRiskReport, RiskAlert, 
    SystemStatus, AlertPriority, RiskCategory
)
from .technical_risk_controller import TechnicalRiskAssessment
from .data_security_controller import DataSecurityAssessment
from .algorithm_bias_controller import BiasRiskAssessment
from .governance_conflict_controller import GovernanceConflictAssessment


# API数据模型
class RiskAssessmentRequest(BaseModel):
    """风险评估请求"""
    ai_decisions: Optional[List[Dict[str, Any]]] = None
    blockchain_data: Optional[Dict[str, Any]] = None
    system_metrics: Optional[Dict[str, Any]] = None
    user_activity: Optional[Dict[str, Any]] = None
    security_logs: Optional[List[Dict[str, Any]]] = None


class RiskAssessmentResponse(BaseModel):
    """风险评估响应"""
    report_id: str
    timestamp: datetime
    system_status: str
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_breakdown: Dict[str, float]
    active_alerts: List[Dict[str, Any]]
    recommendations: List[str]


class AlertQueryResponse(BaseModel):
    """警报查询响应"""
    total_count: int
    critical_count: int
    alerts: List[Dict[str, Any]]


class EmergencyResponseRequest(BaseModel):
    """应急响应请求"""
    alert_ids: List[str]
    response_type: str  # "manual", "auto"
    action_plan: Optional[List[str]] = None


class EmergencyResponseResult(BaseModel):
    """应急响应结果"""
    response_id: str
    status: str
    actions_taken: List[str]
    effectiveness_score: float
    completion_time: Optional[datetime]


class SystemStatusResponse(BaseModel):
    """系统状态响应"""
    status: str
    overall_risk_score: float
    active_alerts_count: int
    last_assessment_time: datetime
    next_assessment_time: datetime


# 创建FastAPI应用
app = FastAPI(
    title="AI自主决策风险控制系统API",
    description="区块链经济模型中AI自主决策风险控制系统的API接口",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENVIRONMENT") == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局风险监控系统实例
risk_monitor = None


def get_risk_monitor() -> AIRiskMonitoringSystem:
    """获取风险监控系统实例"""
    global risk_monitor
    if risk_monitor is None:
        risk_monitor = AIRiskMonitoringSystem()
        # 启动监控
        asyncio.create_task(risk_monitor.start_monitoring())
    return risk_monitor


# API路由
@app.get("/", response_model=Dict[str, str])
async def root():
    """API根路径"""
    return {
        "message": "AI自主决策风险控制系统API",
        "version": "1.0.0",
        "status": "运行中"
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    monitor: AIRiskMonitoringSystem = Depends(get_risk_monitor)
):
    """获取系统状态"""
    # 执行快速评估
    risk_report = await monitor.perform_comprehensive_assessment()
    
    return SystemStatusResponse(
        status=risk_report.system_status.value,
        overall_risk_score=risk_report.overall_risk_score,
        active_alerts_count=len(risk_report.active_alerts),
        last_assessment_time=risk_report.timestamp,
        next_assessment_time=risk_report.next_assessment_time
    )


@app.post("/assess", response_model=RiskAssessmentResponse)
async def assess_risk(
    request: RiskAssessmentRequest,
    background_tasks: BackgroundTasks,
    monitor: AIRiskMonitoringSystem = Depends(get_risk_monitor)
):
    """执行综合风险评估"""
    try:
        # 构建系统数据
        system_data = {
            "ai_decisions": request.ai_decisions or [],
            "blockchain_data": request.blockchain_data or {},
            "system_metrics": request.system_metrics or {},
            "user_activity": request.user_activity or {},
            "security_logs": request.security_logs or []
        }
        
        # 执行评估
        risk_report = await monitor.perform_comprehensive_assessment()
        
        # 转换响应格式
        risk_breakdown = {k.value: v for k, v in risk_report.risk_breakdown.items()}
        
        alerts_data = []
        for alert in risk_report.active_alerts:
            alerts_data.append({
                "alert_id": alert.alert_id,
                "category": alert.risk_category.value,
                "priority": alert.priority.value,
                "description": alert.description,
                "risk_score": alert.risk_score,
                "affected_components": alert.affected_components,
                "recommended_actions": alert.recommended_actions,
                "timestamp": alert.timestamp.isoformat(),
                "expires_at": alert.expires_at.isoformat()
            })
        
        return RiskAssessmentResponse(
            report_id=risk_report.report_id,
            timestamp=risk_report.timestamp,
            system_status=risk_report.system_status.value,
            overall_risk_score=risk_report.overall_risk_score,
            risk_breakdown=risk_breakdown,
            active_alerts=alerts_data,
            recommendations=risk_report.recommendations
        )
        
    except Exception as e:
        logging.error(f"风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"风险评估失败: {str(e)}")


@app.get("/alerts", response_model=AlertQueryResponse)
async def get_alerts(
    category: Optional[str] = None,
    priority: Optional[str] = None,
    active_only: bool = True,
    limit: int = 50,
    monitor: AIRiskMonitoringSystem = Depends(get_risk_monitor)
):
    """查询警报"""
    try:
        # 获取当前活跃警报
        current_alerts = monitor.active_alerts if active_only else []
        
        # 过滤条件
        filtered_alerts = current_alerts
        
        if category:
            filtered_alerts = [a for a in filtered_alerts if a.risk_category.value == category]
        
        if priority:
            filtered_alerts = [a for a in filtered_alerts if a.priority.value == priority]
        
        # 限制数量
        filtered_alerts = filtered_alerts[:limit]
        
        # 转换响应格式
        alerts_data = []
        for alert in filtered_alerts:
            alerts_data.append({
                "alert_id": alert.alert_id,
                "category": alert.risk_category.value,
                "priority": alert.priority.value,
                "description": alert.description,
                "risk_score": alert.risk_score,
                "affected_components": alert.affected_components,
                "recommended_actions": alert.recommended_actions,
                "timestamp": alert.timestamp.isoformat(),
                "expires_at": alert.expires_at.isoformat()
            })
        
        critical_count = len([a for a in filtered_alerts if a.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]])
        
        return AlertQueryResponse(
            total_count=len(filtered_alerts),
            critical_count=critical_count,
            alerts=alerts_data
        )
        
    except Exception as e:
        logging.error(f"警报查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"警报查询失败: {str(e)}")


@app.post("/emergency", response_model=EmergencyResponseResult)
async def trigger_emergency_response(
    request: EmergencyResponseRequest,
    background_tasks: BackgroundTasks,
    monitor: AIRiskMonitoringSystem = Depends(get_risk_monitor)
):
    """触发应急响应"""
    try:
        # 查找相关警报
        target_alerts = []
        for alert_id in request.alert_ids:
            for alert in monitor.active_alerts:
                if alert.alert_id == alert_id:
                    target_alerts.append(alert)
                    break
        
        if not target_alerts:
            raise HTTPException(status_code=404, detail="未找到指定的警报")
        
        # 执行应急响应
        # 这里简化实现，实际应用中应有更复杂的应急响应逻辑
        response_id = f"MANUAL_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        actions_taken = ["手动触发应急响应"]
        if request.action_plan:
            actions_taken.extend(request.action_plan)
        
        # 记录响应
        result = EmergencyResponseResult(
            response_id=response_id,
            status="completed",
            actions_taken=actions_taken,
            effectiveness_score=0.9,  # 假设效果良好
            completion_time=datetime.utcnow()
        )
        
        logging.info(f"应急响应执行完成: {response_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"应急响应失败: {e}")
        raise HTTPException(status_code=500, detail=f"应急响应失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI风险控制系统"
    }


# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logging.info("AI风险控制系统API启动中...")
    # 初始化风险监控系统
    get_risk_monitor()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logging.info("AI风险控制系统API关闭中...")
    global risk_monitor
    if risk_monitor:
        await risk_monitor.stop_monitoring()
        risk_monitor = None


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)