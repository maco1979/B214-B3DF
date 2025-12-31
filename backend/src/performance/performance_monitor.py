"""
性能监控模块 - 监控迁移学习和边缘计算集成的性能指标
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: datetime
    metric_type: str
    value: float
    tags: Dict[str, Any]
    

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """初始化性能阈值"""
        return {
            "decision_latency": {
                "warning": 500.0,    # 500ms
                "critical": 1000.0   # 1s
            },
            "edge_deployment_latency": {
                "warning": 2000.0,   # 2s
                "critical": 5000.0   # 5s
            },
            "migration_learning_latency": {
                "warning": 1000.0,   # 1s
                "critical": 3000.0   # 3s
            },
            "cpu_usage": {
                "warning": 80.0,     # 80%
                "critical": 95.0     # 95%
            },
            "memory_usage": {
                "warning": 85.0,     # 85%
                "critical": 95.0     # 95%
            },
            "throughput": {
                "warning": 10.0,     # 10 req/s
                "critical": 5.0      # 5 req/s
            }
        }
    
    async def record_metric(self, metric_type: str, value: float, tags: Dict[str, Any] = None):
        """记录性能指标"""
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            tags=tags or {}
        )
        
        self.metrics_history[metric_type].append(metric)
        
        # 检查阈值并触发警报
        await self._check_thresholds(metric)
        
        logger.debug(f"记录性能指标: {metric_type} = {value}")
    
    async def _check_thresholds(self, metric: PerformanceMetric):
        """检查性能阈值"""
        
        if metric.metric_type not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric.metric_type]
        
        # 检查临界阈值
        if metric.value >= thresholds.get("critical", float('inf')):
            await self._trigger_alert(metric, "critical")
        # 检查警告阈值
        elif metric.value >= thresholds.get("warning", float('inf')):
            await self._trigger_alert(metric, "warning")
    
    async def _trigger_alert(self, metric: PerformanceMetric, level: str):
        """触发性能警报"""
        
        alert = {
            "id": f"alert_{len(self.alerts)}_{int(time.time())}",
            "timestamp": datetime.now(),
            "level": level,
            "metric_type": metric.metric_type,
            "value": metric.value,
            "threshold": self.thresholds[metric.metric_type][level],
            "tags": metric.tags,
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        logger.warning(f"性能警报 [{level}]: {metric.metric_type} = {metric.value}")
        
        # 这里可以添加通知逻辑（邮件、短信、Webhook等）
    
    def get_metrics_summary(self, metric_type: str, time_range: str = "1h") -> Dict[str, Any]:
        """获取指标摘要"""
        
        if metric_type not in self.metrics_history:
            return {"error": f"未知指标类型: {metric_type}"}
        
        # 计算时间范围
        end_time = datetime.now()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(days=1)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)  # 默认1小时
        
        # 过滤时间范围内的指标
        metrics_in_range = [
            m for m in self.metrics_history[metric_type]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not metrics_in_range:
            return {"count": 0, "message": "指定时间范围内无数据"}
        
        values = [m.value for m in metrics_in_range]
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "count": len(metrics_in_range),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p95": statistics.quantiles(values, n=20)[18],  # 95%分位数
            "p99": statistics.quantiles(values, n=100)[98],  # 99%分位数
            "latest_value": values[-1] if values else 0,
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势（上升/下降/稳定）"""
        
        if len(values) < 2:
            return "stable"
        
        # 使用线性回归计算趋势
        x = list(range(len(values)))
        y = values
        
        # 简单趋势计算
        first_half_avg = statistics.mean(values[:len(values)//2])
        second_half_avg = statistics.mean(values[len(values)//2:])
        
        if second_half_avg > first_half_avg * 1.1:  # 上升10%
            return "rising"
        elif second_half_avg < first_half_avg * 0.9:  # 下降10%
            return "falling"
        else:
            return "stable"
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """获取系统性能报告"""
        
        report = {
            "timestamp": datetime.now(),
            "overall_health": "healthy",
            "components": {},
            "alerts": {
                "critical": len([a for a in self.alerts if a["level"] == "critical" and not a["acknowledged"]]),
                "warning": len([a for a in self.alerts if a["level"] == "warning" and not a["acknowledged"]])
            },
            "recommendations": []
        }
        
        # 分析各个组件性能
        for metric_type in self.thresholds.keys():
            summary = self.get_metrics_summary(metric_type, "1h")
            
            if "error" not in summary:
                report["components"][metric_type] = summary
                
                # 评估健康状态
                if summary["p95"] >= self.thresholds[metric_type].get("critical", float('inf')):
                    report["overall_health"] = "critical"
                    report["recommendations"].append(f"{metric_type} 性能达到临界值")
                elif summary["p95"] >= self.thresholds[metric_type].get("warning", float('inf')):
                    if report["overall_health"] == "healthy":
                        report["overall_health"] = "warning"
                    report["recommendations"].append(f"{metric_type} 性能达到警告值")
        
        return report
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """确认警报"""
        
        for alert in self.alerts:
            if alert["id"] == alert_id and not alert["acknowledged"]:
                alert["acknowledged"] = True
                alert["acknowledged_time"] = datetime.now()
                logger.info(f"警报已确认: {alert_id}")
                return True
        
        return False
    
    def clear_old_metrics(self, older_than_days: int = 7):
        """清理旧指标数据"""
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        cleared_count = 0
        
        for metric_type, metrics in self.metrics_history.items():
            original_count = len(metrics)
            # 保留最近的数据
            self.metrics_history[metric_type] = deque(
                [m for m in metrics if m.timestamp >= cutoff_time],
                maxlen=self.max_history_size
            )
            cleared_count += (original_count - len(self.metrics_history[metric_type]))
        
        logger.info(f"清理了 {cleared_count} 条旧指标数据")
        return cleared_count


class IntegrationPerformanceMonitor(PerformanceMonitor):
    """集成性能监控器 - 专门监控迁移学习和边缘计算集成性能"""
    
    def __init__(self):
        super().__init__()
        self.integration_specific_thresholds = self._initialize_integration_thresholds()
        
    def _initialize_integration_thresholds(self) -> Dict[str, Dict[str, float]]:
        """初始化集成特定阈值"""
        
        integration_thresholds = {
            "migration_learning_accuracy_loss": {
                "warning": 0.05,   # 5%精度损失
                "critical": 0.10   # 10%精度损失
            },
            "edge_computing_latency_reduction": {
                "warning": 0.3,    # 30%延迟降低（期望值）
                "critical": 0.1    # 10%延迟降低（最低要求）
            },
            "risk_control_false_positive_rate": {
                "warning": 0.05,   # 5%误报率
                "critical": 0.10   # 10%误报率
            },
            "integration_success_rate": {
                "warning": 0.95,   # 95%成功率
                "critical": 0.90   # 90%成功率
            },
            "data_validation_throughput": {
                "warning": 100.0,  # 100 req/s
                "critical": 50.0   # 50 req/s
            }
        }
        
        # 合并基础阈值和集成特定阈值
        self.thresholds.update(integration_thresholds)
        return integration_thresholds
    
    async def record_integration_metric(self, integration_type: str, 
                                      operation: str, 
                                      duration: float, 
                                      success: bool,
                                      additional_tags: Dict[str, Any] = None):
        """记录集成操作指标"""
        
        tags = {
            "integration_type": integration_type,
            "operation": operation,
            "success": success
        }
        
        if additional_tags:
            tags.update(additional_tags)
        
        # 记录延迟指标
        await self.record_metric(f"{integration_type}_{operation}_latency", duration, tags)
        
        # 记录成功率指标
        success_rate_metric = 1.0 if success else 0.0
        await self.record_metric(f"{integration_type}_{operation}_success_rate", 
                               success_rate_metric, tags)
    
    async def record_migration_learning_performance(self, 
                                                  source_domain: str,
                                                  target_domain: str,
                                                  accuracy: float,
                                                  baseline_accuracy: float,
                                                  processing_time: float):
        """记录迁移学习性能指标"""
        
        accuracy_loss = baseline_accuracy - accuracy
        accuracy_loss_percentage = (accuracy_loss / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
        
        tags = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "baseline_accuracy": baseline_accuracy,
            "achieved_accuracy": accuracy
        }
        
        # 记录精度损失
        await self.record_metric("migration_learning_accuracy_loss", 
                               accuracy_loss_percentage, tags)
        
        # 记录处理时间
        await self.record_metric("migration_learning_processing_time", 
                               processing_time, tags)
        
        # 记录迁移效果（负损失表示改进）
        migration_effectiveness = -accuracy_loss_percentage
        await self.record_metric("migration_learning_effectiveness", 
                               migration_effectiveness, tags)
    
    async def record_edge_computing_performance(self,
                                              node_id: str,
                                              task_type: str,
                                              edge_latency: float,
                                              cloud_latency: float,
                                              resource_utilization: Dict[str, float]):
        """记录边缘计算性能指标"""
        
        # 计算延迟降低百分比
        latency_reduction = ((cloud_latency - edge_latency) / cloud_latency) * 100 if cloud_latency > 0 else 0
        
        tags = {
            "node_id": node_id,
            "task_type": task_type,
            "cloud_latency": cloud_latency,
            "edge_latency": edge_latency
        }
        
        # 记录延迟降低
        await self.record_metric("edge_computing_latency_reduction", 
                               latency_reduction, tags)
        
        # 记录资源利用率
        for resource, utilization in resource_utilization.items():
            await self.record_metric(f"edge_{resource}_utilization", 
                                   utilization, tags)
        
        # 记录效率比（延迟降低/资源使用）
        avg_utilization = statistics.mean(resource_utilization.values()) if resource_utilization else 0
        efficiency_ratio = latency_reduction / avg_utilization if avg_utilization > 0 else 0
        await self.record_metric("edge_computing_efficiency_ratio", 
                               efficiency_ratio, tags)
    
    def get_integration_performance_report(self) -> Dict[str, Any]:
        """获取集成性能专项报告"""
        
        base_report = self.get_system_performance_report()
        
        # 添加集成特定分析
        integration_report = {
            "migration_learning_effectiveness": self._analyze_migration_effectiveness(),
            "edge_computing_efficiency": self._analyze_edge_efficiency(),
            "risk_control_performance": self._analyze_risk_control_performance(),
            "integration_reliability": self._analyze_integration_reliability()
        }
        
        base_report["integration_analysis"] = integration_report
        return base_report
    
    def _analyze_migration_effectiveness(self) -> Dict[str, Any]:
        """分析迁移学习效果"""
        
        effectiveness_metrics = self.get_metrics_summary("migration_learning_effectiveness", "24h")
        accuracy_loss_metrics = self.get_metrics_summary("migration_learning_accuracy_loss", "24h")
        
        return {
            "effectiveness_score": effectiveness_metrics.get("avg", 0),
            "accuracy_loss": accuracy_loss_metrics.get("avg", 0),
            "recommendation": self._get_migration_recommendation(
                effectiveness_metrics.get("avg", 0),
                accuracy_loss_metrics.get("avg", 0)
            )
        }
    
    def _analyze_edge_efficiency(self) -> Dict[str, Any]:
        """分析边缘计算效率"""
        
        latency_reduction = self.get_metrics_summary("edge_computing_latency_reduction", "24h")
        efficiency_ratio = self.get_metrics_summary("edge_computing_efficiency_ratio", "24h")
        
        return {
            "latency_reduction": latency_reduction.get("avg", 0),
            "efficiency_ratio": efficiency_ratio.get("avg", 0),
            "recommendation": self._get_edge_recommendation(
                latency_reduction.get("avg", 0),
                efficiency_ratio.get("avg", 0)
            )
        }
    
    def _analyze_risk_control_performance(self) -> Dict[str, Any]:
        """分析风险控制性能"""
        
        # 这里可以添加更复杂的风险控制性能分析
        return {
            "false_positive_rate": 0.02,  # 示例数据
            "detection_accuracy": 0.98,
            "response_time": 150.0
        }
    
    def _analyze_integration_reliability(self) -> Dict[str, Any]:
        """分析集成可靠性"""
        
        success_rate = self.get_metrics_summary("integration_success_rate", "24h")
        
        return {
            "success_rate": success_rate.get("avg", 0) * 100,  # 转换为百分比
            "reliability_level": "high" if success_rate.get("avg", 0) > 0.95 else "medium",
            "improvement_suggestions": []
        }
    
    def _get_migration_recommendation(self, effectiveness: float, accuracy_loss: float) -> str:
        """获取迁移学习推荐"""
        
        if effectiveness > 5:  # 效果显著
            return "迁移学习效果良好，可扩大应用范围"
        elif accuracy_loss > 10:  # 精度损失过大
            return "精度损失较大，建议优化迁移策略或减少迁移范围"
        else:
            return "迁移学习效果适中，保持当前配置"
    
    def _get_edge_recommendation(self, latency_reduction: float, efficiency_ratio: float) -> str:
        """获取边缘计算推荐"""
        
        if latency_reduction > 30 and efficiency_ratio > 2:
            return "边缘计算效果显著，适合更多实时任务"
        elif latency_reduction < 10:
            return "延迟降低有限，建议重新评估边缘计算适用性"
        else:
            return "边缘计算效果适中，可优化资源分配"