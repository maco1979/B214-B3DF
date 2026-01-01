"""
审计日志监控服务 - 对接Prometheus/Grafana
提供智能体行为审计指标导出和API端点
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PrometheusMetric:
    """Prometheus指标"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""
    timestamp: Optional[float] = None


class AuditMetricsCollector:
    """审计指标收集器 - 导出Prometheus格式指标"""
    
    def __init__(self):
        self._metrics: Dict[str, PrometheusMetric] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        
        # 初始化标准指标
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """初始化标准审计指标"""
        # 智能体相关指标
        self._register_counter(
            "agent_actions_total",
            "Total number of agent actions",
            ["agent_id", "agent_type", "action", "status"]
        )
        
        self._register_counter(
            "agent_permission_checks_total",
            "Total number of permission checks",
            ["agent_id", "permission", "result"]
        )
        
        self._register_gauge(
            "active_agents_count",
            "Number of currently active agents",
            ["agent_type"]
        )
        
        self._register_histogram(
            "agent_action_duration_seconds",
            "Agent action duration in seconds",
            ["agent_id", "action"]
        )
        
        # 缓存相关指标
        self._register_counter(
            "cache_hits_total",
            "Total number of cache hits",
            ["cache_type"]
        )
        
        self._register_counter(
            "cache_misses_total",
            "Total number of cache misses",
            ["cache_type"]
        )
        
        self._register_gauge(
            "cache_size_bytes",
            "Current cache size in bytes",
            ["cache_type"]
        )
        
        # 决策相关指标
        self._register_counter(
            "decisions_total",
            "Total number of decisions made",
            ["module", "objective", "status"]
        )
        
        self._register_histogram(
            "decision_latency_seconds",
            "Decision latency in seconds",
            ["module"]
        )
        
        # 系统健康指标
        self._register_gauge(
            "system_health_status",
            "System health status (1=healthy, 0=unhealthy)",
            ["component"]
        )
    
    def _register_counter(self, name: str, help_text: str, labels: List[str]):
        """注册计数器指标"""
        key = f"counter:{name}"
        self._metrics[key] = PrometheusMetric(
            name=name,
            value=0,
            metric_type=MetricType.COUNTER,
            help_text=help_text
        )
    
    def _register_gauge(self, name: str, help_text: str, labels: List[str]):
        """注册仪表指标"""
        key = f"gauge:{name}"
        self._metrics[key] = PrometheusMetric(
            name=name,
            value=0,
            metric_type=MetricType.GAUGE,
            help_text=help_text
        )
    
    def _register_histogram(self, name: str, help_text: str, labels: List[str]):
        """注册直方图指标"""
        key = f"histogram:{name}"
        self._metrics[key] = PrometheusMetric(
            name=name,
            value=0,
            metric_type=MetricType.HISTOGRAM,
            help_text=help_text
        )
    
    async def increment_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1):
        """增加计数器"""
        async with self._lock:
            label_str = self._labels_to_string(labels or {})
            key = f"{name}{label_str}"
            self._counters[key] += value
    
    async def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表值"""
        async with self._lock:
            label_str = self._labels_to_string(labels or {})
            key = f"{name}{label_str}"
            self._gauges[key] = value
    
    async def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """观察直方图值"""
        async with self._lock:
            label_str = self._labels_to_string(labels or {})
            key = f"{name}{label_str}"
            self._histograms[key].append(value)
            # 保持最近1000个观察值
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]
    
    def _labels_to_string(self, labels: Dict[str, str]) -> str:
        """将标签转换为字符串"""
        if not labels:
            return ""
        sorted_labels = sorted(labels.items())
        return "{" + ",".join(f'{k}="{v}"' for k, v in sorted_labels) + "}"
    
    async def export_prometheus_format(self) -> str:
        """导出Prometheus格式的指标"""
        async with self._lock:
            lines = []
            
            # 导出计数器
            for key, value in self._counters.items():
                name = key.split("{")[0] if "{" in key else key
                labels = key[len(name):] if "{" in key else ""
                lines.append(f"# HELP {name} Counter metric")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels} {value}")
            
            # 导出仪表
            for key, value in self._gauges.items():
                name = key.split("{")[0] if "{" in key else key
                labels = key[len(name):] if "{" in key else ""
                lines.append(f"# HELP {name} Gauge metric")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels} {value}")
            
            # 导出直方图摘要
            for key, values in self._histograms.items():
                if not values:
                    continue
                name = key.split("{")[0] if "{" in key else key
                labels = key[len(name):] if "{" in key else ""
                
                # 计算统计值
                count = len(values)
                total = sum(values)
                avg = total / count if count > 0 else 0
                
                lines.append(f"# HELP {name} Histogram metric")
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{labels} {count}")
                lines.append(f"{name}_sum{labels} {total}")
            
            return "\n".join(lines)
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        async with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: {
                        "count": len(v),
                        "sum": sum(v) if v else 0,
                        "avg": sum(v) / len(v) if v else 0,
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0
                    }
                    for k, v in self._histograms.items()
                },
                "timestamp": datetime.now().isoformat()
            }


class AuditLogMonitoringService:
    """审计日志监控服务"""
    
    def __init__(self):
        self.metrics_collector = AuditMetricsCollector()
        self._alert_rules: List[Dict[str, Any]] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        
        # 初始化默认告警规则
        self._init_default_alert_rules()
    
    def _init_default_alert_rules(self):
        """初始化默认告警规则"""
        self._alert_rules = [
            {
                "name": "HighPermissionDenialRate",
                "description": "High rate of permission denials",
                "condition": "permission_denial_rate > 0.3",
                "severity": "warning",
                "for": "5m"
            },
            {
                "name": "AgentDeactivated",
                "description": "An agent has been deactivated",
                "condition": "agent_deactivated",
                "severity": "info",
                "for": "0m"
            },
            {
                "name": "UnusualAgentActivity",
                "description": "Unusual agent activity detected",
                "condition": "agent_action_rate > 100",
                "severity": "warning",
                "for": "1m"
            },
            {
                "name": "CacheHitRateLow",
                "description": "Cache hit rate is low",
                "condition": "cache_hit_rate < 0.5",
                "severity": "warning",
                "for": "10m"
            }
        ]
    
    async def record_agent_action(self, 
                                  agent_id: str,
                                  agent_type: str,
                                  action: str,
                                  status: str,
                                  duration_ms: float = 0):
        """记录智能体动作"""
        # 更新计数器
        await self.metrics_collector.increment_counter(
            "agent_actions_total",
            {"agent_id": agent_id, "agent_type": agent_type, "action": action, "status": status}
        )
        
        # 记录持续时间
        if duration_ms > 0:
            await self.metrics_collector.observe_histogram(
                "agent_action_duration_seconds",
                duration_ms / 1000,
                {"agent_id": agent_id, "action": action}
            )
    
    async def record_permission_check(self,
                                      agent_id: str,
                                      permission: str,
                                      granted: bool):
        """记录权限检查"""
        result = "granted" if granted else "denied"
        await self.metrics_collector.increment_counter(
            "agent_permission_checks_total",
            {"agent_id": agent_id, "permission": permission, "result": result}
        )
    
    async def record_cache_access(self, cache_type: str, hit: bool):
        """记录缓存访问"""
        metric_name = "cache_hits_total" if hit else "cache_misses_total"
        await self.metrics_collector.increment_counter(
            metric_name,
            {"cache_type": cache_type}
        )
    
    async def record_decision(self,
                             module: str,
                             objective: str,
                             status: str,
                             latency_ms: float = 0):
        """记录决策"""
        await self.metrics_collector.increment_counter(
            "decisions_total",
            {"module": module, "objective": objective, "status": status}
        )
        
        if latency_ms > 0:
            await self.metrics_collector.observe_histogram(
                "decision_latency_seconds",
                latency_ms / 1000,
                {"module": module}
            )
    
    async def update_active_agents(self, agent_type: str, count: int):
        """更新活跃智能体数量"""
        await self.metrics_collector.set_gauge(
            "active_agents_count",
            count,
            {"agent_type": agent_type}
        )
    
    async def update_system_health(self, component: str, healthy: bool):
        """更新系统健康状态"""
        await self.metrics_collector.set_gauge(
            "system_health_status",
            1.0 if healthy else 0.0,
            {"component": component}
        )
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """检查告警条件"""
        async with self._lock:
            triggered_alerts = []
            metrics_summary = await self.metrics_collector.get_metrics_summary()
            
            # 简化的告警检查逻辑
            counters = metrics_summary.get("counters", {})
            
            # 检查权限拒绝率
            total_checks = sum(v for k, v in counters.items() if "permission_checks" in k)
            denied_checks = sum(v for k, v in counters.items() if "permission_checks" in k and 'denied' in k)
            
            if total_checks > 0:
                denial_rate = denied_checks / total_checks
                if denial_rate > 0.3:
                    alert = {
                        "name": "HighPermissionDenialRate",
                        "severity": "warning",
                        "value": denial_rate,
                        "timestamp": datetime.now().isoformat()
                    }
                    triggered_alerts.append(alert)
                    self._alert_history.append(alert)
            
            return triggered_alerts
    
    async def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警历史"""
        async with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            return [
                a for a in self._alert_history
                if datetime.fromisoformat(a["timestamp"]) > cutoff
            ]
    
    async def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式指标"""
        return await self.metrics_collector.export_prometheus_format()
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        metrics_summary = await self.metrics_collector.get_metrics_summary()
        alerts = await self.check_alerts()
        
        return {
            "metrics": metrics_summary,
            "active_alerts": alerts,
            "alert_rules": self._alert_rules,
            "timestamp": datetime.now().isoformat()
        }


# 全局实例
_audit_monitoring_service: Optional[AuditLogMonitoringService] = None


def get_audit_monitoring_service() -> AuditLogMonitoringService:
    """获取审计监控服务单例"""
    global _audit_monitoring_service
    if _audit_monitoring_service is None:
        _audit_monitoring_service = AuditLogMonitoringService()
    return _audit_monitoring_service


# FastAPI路由
def create_monitoring_router():
    """创建监控API路由"""
    from fastapi import APIRouter, Response
    from fastapi.responses import PlainTextResponse
    
    router = APIRouter(prefix="/monitoring", tags=["监控"])
    
    @router.get("/metrics", response_class=PlainTextResponse)
    async def get_prometheus_metrics():
        """获取Prometheus格式指标"""
        service = get_audit_monitoring_service()
        metrics = await service.get_prometheus_metrics()
        return Response(content=metrics, media_type="text/plain")
    
    @router.get("/dashboard")
    async def get_dashboard_data():
        """获取监控仪表板数据"""
        service = get_audit_monitoring_service()
        return await service.get_monitoring_dashboard_data()
    
    @router.get("/alerts")
    async def get_alerts(hours: int = 24):
        """获取告警历史"""
        service = get_audit_monitoring_service()
        alerts = await service.get_alert_history(hours)
        return {"alerts": alerts, "count": len(alerts)}
    
    @router.get("/health")
    async def monitoring_health():
        """监控服务健康检查"""
        return {
            "status": "healthy",
            "service": "audit_monitoring",
            "timestamp": datetime.now().isoformat()
        }
    
    return router
