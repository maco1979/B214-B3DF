"""
健康检查路由
"""

from fastapi import APIRouter
from datetime import datetime
import psutil
import os

router = APIRouter(prefix="/api/health", tags=["健康检查"])


@router.get("/")
async def health_check():
    """基础健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ai-platform-backend",
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check():
    """详细健康检查"""
    # 获取系统信息
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ai-platform-backend",
        "version": "1.0.0",
        "system": {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        },
        "process": {
            "pid": os.getpid(),
            "memory_usage_mb": round(psutil.Process().memory_info().rss / (1024**2), 2)
        }
    }


@router.get("/ready")
async def readiness_check():
    """就绪检查"""
    # 检查关键服务是否就绪
    checks = {
        "database_connection": True,  # 实际应用中应检查数据库连接
        "redis_connection": True,     # 实际应用中应检查Redis连接
        "model_loading": True,        # 实际应用中应检查模型加载状态
    }
    
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.now().isoformat(),
        "checks": checks
    }


@router.get("/live")
async def liveness_check():
    """存活检查"""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }