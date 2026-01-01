"""
监控路由 - 对接Prometheus/Grafana的审计日志API端点
"""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
from datetime import datetime
from typing import Optional

router = APIRouter(prefix="/monitoring", tags=["监控"])

# 尝试导入监控服务
try:
    from src.core.services.audit_monitoring_service import get_audit_monitoring_service
    _monitoring_available = True
except ImportError:
    _monitoring_available = False


@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    获取Prometheus格式指标
    
    用于Prometheus抓取的/metrics端点
    """
    if not _monitoring_available:
        return Response(
            content="# Monitoring service not available\n",
            media_type="text/plain"
        )
    
    service = get_audit_monitoring_service()
    metrics = await service.get_prometheus_metrics()
    return Response(content=metrics, media_type="text/plain")


@router.get("/dashboard")
async def get_dashboard_data():
    """
    获取监控仪表板数据
    
    返回所有监控指标和告警状态
    """
    if not _monitoring_available:
        return {
            "error": "Monitoring service not available",
            "metrics": {},
            "active_alerts": [],
            "timestamp": datetime.now().isoformat()
        }
    
    service = get_audit_monitoring_service()
    return await service.get_monitoring_dashboard_data()


@router.get("/alerts")
async def get_alerts(hours: int = 24):
    """
    获取告警历史
    
    Args:
        hours: 查询最近多少小时的告警，默认24小时
    """
    if not _monitoring_available:
        return {"alerts": [], "count": 0}
    
    service = get_audit_monitoring_service()
    alerts = await service.get_alert_history(hours)
    return {"alerts": alerts, "count": len(alerts)}


@router.get("/health")
async def monitoring_health():
    """监控服务健康检查"""
    return {
        "status": "healthy" if _monitoring_available else "unavailable",
        "service": "audit_monitoring",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/agent/action")
async def record_agent_action(
    agent_id: str,
    agent_type: str,
    action: str,
    status: str = "success",
    duration_ms: float = 0
):
    """
    记录智能体动作
    
    手动记录智能体动作用于审计
    """
    if not _monitoring_available:
        return {"success": False, "error": "Monitoring service not available"}
    
    service = get_audit_monitoring_service()
    await service.record_agent_action(agent_id, agent_type, action, status, duration_ms)
    return {"success": True, "message": "Action recorded"}


@router.post("/permission/check")
async def record_permission_check(
    agent_id: str,
    permission: str,
    granted: bool
):
    """
    记录权限检查
    
    手动记录权限检查结果用于审计
    """
    if not _monitoring_available:
        return {"success": False, "error": "Monitoring service not available"}
    
    service = get_audit_monitoring_service()
    await service.record_permission_check(agent_id, permission, granted)
    return {"success": True, "message": "Permission check recorded"}


@router.post("/cache/access")
async def record_cache_access(cache_type: str, hit: bool):
    """
    记录缓存访问
    
    手动记录缓存访问用于审计
    """
    if not _monitoring_available:
        return {"success": False, "error": "Monitoring service not available"}
    
    service = get_audit_monitoring_service()
    await service.record_cache_access(cache_type, hit)
    return {"success": True, "message": "Cache access recorded"}


@router.post("/decision")
async def record_decision(
    module: str,
    objective: str,
    status: str,
    latency_ms: float = 0
):
    """
    记录决策
    
    手动记录决策结果用于审计
    """
    if not _monitoring_available:
        return {"success": False, "error": "Monitoring service not available"}
    
    service = get_audit_monitoring_service()
    await service.record_decision(module, objective, status, latency_ms)
    return {"success": True, "message": "Decision recorded"}
