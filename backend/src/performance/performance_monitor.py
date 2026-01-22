"""
性能监控模块 - 监控系统各组件的性能指标，实现实时监控和警报
"""

import logging
import time
import asyncio
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: datetime
    metric_type: str
    value: float
    tags: Dict[str, Any]
    source: str = "system"
    unit: str = ""


@dataclass
class AlertConfig:
    """警报配置"""
    threshold: float
    operator: Callable[[float, float], bool]
    message: str
    severity: str


class PerformanceMonitor:
    """性能监控器 - 实时监控系统各组件的性能指标"""
    
    def __init__(self, max_history_size: int = 5000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = self._initialize_thresholds()
        self.alert_configs = self._initialize_alert_configs()
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.monitoring_interval = 5.0  # 默认监控间隔（秒）
        self.system_metrics_lock = threading.Lock()
        self.last_gc_time = time.time()
        
        # 系统资源监控
        self.process = psutil.Process()
        
        # 组件性能跟踪
        self.component_performance: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "latency": deque(maxlen=100),
                "throughput": deque(maxlen=100),
                "error_count": 0,
                "success_count": 0,
                "last_activity": datetime.now()
            }
        )
        
        logger.info(f"性能监控器初始化完成，最大历史记录: {max_history_size}")
        
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """初始化性能阈值"""
        return {
            # 系统资源阈值
            "cpu_usage": {
                "warning": 80.0,     # 80%
                "critical": 95.0     # 95%
            },
            "memory_usage": {
                "warning": 85.0,     # 85%
                "critical": 95.0     # 95%
            },
            "disk_usage": {
                "warning": 80.0,     # 80%
                "critical": 95.0     # 95%
            },
            "network_io": {
                "warning": 100.0,    # 100 MB/s
                "critical": 200.0    # 200 MB/s
            },
            
            # 组件性能阈值
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
            "api_response_time": {
                "warning": 200.0,    # 200ms
                "critical": 500.0    # 500ms
            },
            "database_query_time": {
                "warning": 100.0,    # 100ms
                "critical": 300.0    # 300ms
            },
            
            # 业务指标阈值
            "throughput": {
                "warning": 10.0,     # 10 req/s
                "critical": 5.0      # 5 req/s
            },
            "error_rate": {
                "warning": 5.0,      # 5%
                "critical": 10.0     # 10%
            },
            "success_rate": {
                "warning": 95.0,     # 95%
                "critical": 90.0     # 90%
            },
            
            # 迁移学习特定阈值
            "migration_learning_accuracy_loss": {
                "warning": 5.0,      # 5%精度损失
                "critical": 10.0     # 10%精度损失
            },
            "migration_learning_effectiveness": {
                "warning": 0.0,      # 0%效果
                "critical": -5.0     # -5%效果
            },
            
            # 边缘计算特定阈值
            "edge_computing_latency_reduction": {
                "warning": 30.0,     # 30%延迟降低
                "critical": 10.0     # 10%延迟降低
            },
            "edge_computing_efficiency_ratio": {
                "warning": 1.5,      # 1.5效率比
                "critical": 0.5      # 0.5效率比
            }
        }
    
    def _initialize_alert_configs(self) -> Dict[str, List[AlertConfig]]:
        """初始化警报配置"""
        configs = {
            "cpu_usage": [
                AlertConfig(
                    threshold=self.thresholds["cpu_usage"]["warning"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"CPU使用率超过警告阈值 ({self.thresholds['cpu_usage']['warning']}%)",
                    severity="warning"
                ),
                AlertConfig(
                    threshold=self.thresholds["cpu_usage"]["critical"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"CPU使用率超过临界阈值 ({self.thresholds['cpu_usage']['critical']}%)",
                    severity="critical"
                )
            ],
            "memory_usage": [
                AlertConfig(
                    threshold=self.thresholds["memory_usage"]["warning"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"内存使用率超过警告阈值 ({self.thresholds['memory_usage']['warning']}%)",
                    severity="warning"
                ),
                AlertConfig(
                    threshold=self.thresholds["memory_usage"]["critical"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"内存使用率超过临界阈值 ({self.thresholds['memory_usage']['critical']}%)",
                    severity="critical"
                )
            ],
            "error_rate": [
                AlertConfig(
                    threshold=self.thresholds["error_rate"]["warning"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"错误率超过警告阈值 ({self.thresholds['error_rate']['warning']}%)",
                    severity="warning"
                ),
                AlertConfig(
                    threshold=self.thresholds["error_rate"]["critical"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"错误率超过临界阈值 ({self.thresholds['error_rate']['critical']}%)",
                    severity="critical"
                )
            ],
            "decision_latency": [
                AlertConfig(
                    threshold=self.thresholds["decision_latency"]["warning"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"决策延迟超过警告阈值 ({self.thresholds['decision_latency']['warning']}ms)",
                    severity="warning"
                ),
                AlertConfig(
                    threshold=self.thresholds["decision_latency"]["critical"],
                    operator=lambda value, threshold: value >= threshold,
                    message=f"决策延迟超过临界阈值 ({self.thresholds['decision_latency']['critical']}ms)",
                    severity="critical"
                )
            ]
        }
        
        # 为所有阈值类型添加默认警报配置
        for metric_type, thresholds in self.thresholds.items():
            if metric_type not in configs:
                configs[metric_type] = [
                    AlertConfig(
                        threshold=thresholds["warning"],
                        operator=lambda value, threshold: value >= threshold,
                        message=f"{metric_type}超过警告阈值 ({thresholds['warning']})",
                        severity="warning"
                    ),
                    AlertConfig(
                        threshold=thresholds["critical"],
                        operator=lambda value, threshold: value >= threshold,
                        message=f"{metric_type}超过临界阈值 ({thresholds['critical']})",
                        severity="critical"
                    )
                ]
        
        return configs
    
    async def record_metric(self, 
                          metric_type: str, 
                          value: float, 
                          tags: Dict[str, Any] = None,
                          source: str = "system",
                          unit: str = "",
                          component: str = None):
        """记录性能指标"""
        
        tags = tags or {}
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            tags=tags,
            source=source,
            unit=unit
        )
        
        # 记录到历史
        with self.system_metrics_lock:
            self.metrics_history[metric_type].append(metric)
        
        # 更新组件性能统计
        if component:
            await self._update_component_performance(component, metric_type, value, tags.get('success', True))
        
        # 检查阈值并触发警报
        await self._check_thresholds(metric)
        
        logger.debug(f"记录性能指标: {metric_type} = {value} ({unit}) from {source}")
    
    async def _update_component_performance(self, component: str, metric_type: str, value: float, success: bool):
        """更新组件性能统计"""
        perf_data = self.component_performance[component]
        
        if metric_type.endswith('latency'):
            perf_data['latency'].append(value)
        elif metric_type == 'throughput':
            perf_data['throughput'].append(value)
        
        if success:
            perf_data['success_count'] += 1
        else:
            perf_data['error_count'] += 1
        
        perf_data['last_activity'] = datetime.now()
    
    async def start_monitoring(self, interval: float = 5.0):
        """启动性能监控"""
        if self.is_running:
            logger.warning("性能监控已在运行中")
            return
        
        self.is_running = True
        self.monitoring_interval = interval
        
        # 启动系统资源监控任务
        self.monitoring_tasks['system_resources'] = asyncio.create_task(self._monitor_system_resources())
        
        # 启动组件性能监控任务
        self.monitoring_tasks['component_performance'] = asyncio.create_task(self._monitor_component_performance())
        
        # 启动警报检查任务
        self.monitoring_tasks['alert_check'] = asyncio.create_task(self._periodic_alert_check())
        
        logger.info(f"性能监控已启动，监控间隔: {interval}秒")
    
    async def stop_monitoring(self):
        """停止性能监控"""
        if not self.is_running:
            logger.warning("性能监控未在运行中")
            return
        
        self.is_running = False
        
        # 取消所有监控任务
        for task_name, task in self.monitoring_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.monitoring_tasks.clear()
        logger.info("性能监控已停止")
    
    async def _monitor_system_resources(self):
        """监控系统资源"""
        while self.is_running:
            try:
                # 收集系统资源指标
                await self._collect_system_metrics()
                
                # 定期执行垃圾回收
                current_time = time.time()
                if current_time - self.last_gc_time > 300:  # 每5分钟执行一次
                    gc.collect()
                    self.last_gc_time = current_time
                    logger.info("执行定期垃圾回收")
                
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"系统资源监控错误: {str(e)}")
                await asyncio.sleep(10)  # 错误后等待10秒
    
    async def _collect_system_metrics(self):
        """收集系统资源指标"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=0.1)
            await self.record_metric(
                metric_type="cpu_usage",
                value=cpu_usage,
                unit="%",
                source="psutil"
            )
            
            # 内存使用率
            memory = psutil.virtual_memory()
            await self.record_metric(
                metric_type="memory_usage",
                value=memory.percent,
                unit="%",
                source="psutil"
            )
            
            # 进程内存使用
            process_memory = self.process.memory_info()
            await self.record_metric(
                metric_type="process_memory",
                value=process_memory.rss / (1024 * 1024),  # MB
                unit="MB",
                source="psutil"
            )
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            await self.record_metric(
                metric_type="disk_usage",
                value=disk.percent,
                unit="%",
                source="psutil"
            )
            
            # 网络IO
            net_io = psutil.net_io_counters()
            await self.record_metric(
                metric_type="network_io_sent",
                value=net_io.bytes_sent / (1024 * 1024),  # MB/s
                unit="MB",
                source="psutil"
            )
            await self.record_metric(
                metric_type="network_io_recv",
                value=net_io.bytes_recv / (1024 * 1024),  # MB/s
                unit="MB",
                source="psutil"
            )
            
            # 进程CPU使用率
            process_cpu = self.process.cpu_percent(interval=0.1)
            await self.record_metric(
                metric_type="process_cpu",
                value=process_cpu,
                unit="%",
                source="psutil"
            )
            
            # 线程数
            thread_count = self.process.num_threads()
            await self.record_metric(
                metric_type="thread_count",
                value=thread_count,
                unit="",
                source="psutil"
            )
            
        except Exception as e:
            logger.error(f"收集系统指标错误: {str(e)}")
    
    async def _monitor_component_performance(self):
        """监控组件性能"""
        while self.is_running:
            try:
                # 检查组件活跃度
                await self._check_component_activity()
                
                await asyncio.sleep(self.monitoring_interval * 2)  # 每2个监控间隔检查一次
            except Exception as e:
                logger.error(f"组件性能监控错误: {str(e)}")
                await asyncio.sleep(10)
    
    async def _check_component_activity(self):
        """检查组件活跃度"""
        now = datetime.now()
        inactive_components = []
        
        for component, data in self.component_performance.items():
            # 检查超过5分钟无活动的组件
            if (now - data['last_activity']).total_seconds() > 300:
                inactive_components.append(component)
        
        if inactive_components:
            logger.info(f"发现无活动组件: {inactive_components}")
    
    async def _periodic_alert_check(self):
        """定期检查警报"""
        while self.is_running:
            try:
                await self._check_all_thresholds()
                await asyncio.sleep(self.monitoring_interval * 5)  # 每5个监控间隔检查一次
            except Exception as e:
                logger.error(f"定期警报检查错误: {str(e)}")
                await asyncio.sleep(10)
    
    async def _check_all_thresholds(self):
        """检查所有指标的阈值"""
        with self.system_metrics_lock:
            for metric_type, metrics in self.metrics_history.items():
                if metrics:
                    # 检查最新的指标
                    latest_metric = metrics[-1]
                    await self._check_thresholds(latest_metric)
    
    async def record_component_metric(self, 
                                    component: str, 
                                    operation: str, 
                                    duration: float, 
                                    success: bool,
                                    additional_tags: Dict[str, Any] = None):
        """记录组件操作指标"""
        tags = {
            "component": component,
            "operation": operation,
            "success": success
        }
        
        if additional_tags:
            tags.update(additional_tags)
        
        # 记录延迟指标
        await self.record_metric(
            metric_type=f"{component}_{operation}_latency",
            value=duration,
            unit="ms",
            source="component",
            tags=tags,
            component=component
        )
        
        # 计算并记录成功率
        if success:
            await self.record_metric(
                metric_type=f"{component}_success_count",
                value=1.0,
                unit="count",
                source="component",
                tags=tags,
                component=component
            )
        else:
            await self.record_metric(
                metric_type=f"{component}_error_count",
                value=1.0,
                unit="count",
                source="component",
                tags=tags,
                component=component
            )
        
        # 更新组件吞吐量
        await self.record_metric(
            metric_type=f"{component}_throughput",
            value=1.0,
            unit="ops",
            source="component",
            tags=tags,
            component=component
        )
    
    async def _check_thresholds(self, metric: PerformanceMetric):
        """检查性能阈值"""
        
        if metric.metric_type not in self.alert_configs:
            return
        
        alert_configs = self.alert_configs[metric.metric_type]
        
        for config in alert_configs:
            if config.operator(metric.value, config.threshold):
                await self._trigger_alert(metric, config)
                break  # 只触发最严重的警报
    
    async def _trigger_alert(self, metric: PerformanceMetric, alert_config: AlertConfig):
        """触发性能警报"""
        
        # 检查是否已经存在类似的未确认警报
        existing_alert = None
        for alert in self.alerts:
            if (not alert["acknowledged"] and 
                alert["metric_type"] == metric.metric_type and 
                alert["level"] == alert_config.severity):
                existing_alert = alert
                break
        
        # 如果已有未确认的相同警报，更新时间和值，不创建新警报
        if existing_alert:
            existing_alert["timestamp"] = datetime.now()
            existing_alert["value"] = metric.value
            existing_alert["tags"] = metric.tags
            logger.debug(f"更新现有警报: {metric.metric_type} = {metric.value}")
            return
        
        # 创建新警报
        alert = {
            "id": f"alert_{len(self.alerts)}_{int(time.time())}",
            "timestamp": datetime.now(),
            "level": alert_config.severity,
            "metric_type": metric.metric_type,
            "value": metric.value,
            "threshold": alert_config.threshold,
            "unit": metric.unit,
            "source": metric.source,
            "tags": metric.tags,
            "message": alert_config.message,
            "acknowledged": False,
            "acknowledged_time": None,
            "acknowledged_by": None
        }
        
        self.alerts.append(alert)
        
        logger.warning(f"性能警报 [{alert_config.severity}]: {alert_config.message}, 当前值: {metric.value} {metric.unit}")
        
        # 执行警报处理（可以扩展为发送通知等）
        await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """处理警报"""
        # 这里可以扩展警报处理逻辑，如：
        # 1. 发送通知（邮件、短信、Webhook等）
        # 2. 执行自动修复操作
        # 3. 记录到外部监控系统
        
        # 目前只记录日志
        logger.info(f"处理警报: {alert['message']} (ID: {alert['id']})")
        
        # 对于严重警报，可以触发自动修复
        if alert["level"] == "critical":
            await self._trigger_auto_remediation(alert)
    
    async def _trigger_auto_remediation(self, alert: Dict[str, Any]):
        """触发自动修复"""
        logger.info(f"为警报 {alert['id']} 触发自动修复")
        
        # 根据警报类型执行不同的修复操作
        if alert["metric_type"] == "memory_usage":
            # 执行内存优化
            gc.collect()
            logger.info("执行内存垃圾回收")
        elif alert["metric_type"] == "cpu_usage":
            # 记录CPU高使用率情况
            logger.warning(f"CPU使用率过高: {alert['value']}%")
        elif alert["metric_type"] == "error_rate":
            # 记录错误率过高情况
            logger.error(f"错误率过高: {alert['value']}%")
    
    def get_metrics_summary(self, metric_type: str, time_range: str = "1h") -> Dict[str, Any]:
        """获取指标摘要"""
        
        with self.system_metrics_lock:
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
            elif time_range == "15m":
                start_time = end_time - timedelta(minutes=15)
            elif time_range == "30m":
                start_time = end_time - timedelta(minutes=30)
            else:
                start_time = end_time - timedelta(hours=1)  # 默认1小时
            
            # 过滤时间范围内的指标
            metrics_in_range = [
                m for m in self.metrics_history[metric_type]
                if start_time <= m.timestamp <= end_time
            ]
        
        if not metrics_in_range:
            return {
                "count": 0, 
                "message": "指定时间范围内无数据",
                "metric_type": metric_type,
                "time_range": time_range
            }
        
        values = [m.value for m in metrics_in_range]
        sources = [m.source for m in metrics_in_range]
        units = set(m.unit for m in metrics_in_range if m.unit)
        
        # 计算统计指标
        summary = {
            "metric_type": metric_type,
            "time_range": time_range,
            "count": len(metrics_in_range),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": statistics.median(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),  # 95%分位数
            "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),  # 99%分位数
            "latest_value": values[-1] if values else 0,
            "trend": self._calculate_trend(values),
            "sources": list(set(sources)),
            "units": list(units),
            "thresholds": self.thresholds.get(metric_type, {})
        }
        
        return summary
    
    def get_component_performance(self, component: str) -> Dict[str, Any]:
        """获取组件性能报告"""
        if component not in self.component_performance:
            return {
                "component": component,
                "message": "组件不存在或无性能数据"
            }
        
        perf_data = self.component_performance[component]
        latency = perf_data["latency"]
        throughput = perf_data["throughput"]
        
        # 计算组件统计
        total_requests = perf_data["success_count"] + perf_data["error_count"]
        success_rate = (perf_data["success_count"] / total_requests * 100) if total_requests > 0 else 100.0
        error_rate = (perf_data["error_count"] / total_requests * 100) if total_requests > 0 else 0.0
        
        component_summary = {
            "component": component,
            "total_requests": total_requests,
            "success_count": perf_data["success_count"],
            "error_count": perf_data["error_count"],
            "success_rate": success_rate,
            "error_rate": error_rate,
            "last_activity": perf_data["last_activity"],
            "latency": {
                "avg": statistics.mean(latency) if latency else 0.0,
                "min": min(latency) if latency else 0.0,
                "max": max(latency) if latency else 0.0,
                "p50": statistics.median(latency) if latency else 0.0,
                "p95": statistics.quantiles(latency, n=20)[18] if len(latency) >= 20 else 0.0,
                "count": len(latency)
            },
            "throughput": {
                "avg": statistics.mean(throughput) if throughput else 0.0,
                "min": min(throughput) if throughput else 0.0,
                "max": max(throughput) if throughput else 0.0,
                "count": len(throughput)
            },
            "health_status": self._calculate_component_health(component, success_rate, error_rate, latency)
        }
        
        return component_summary
    
    def _calculate_component_health(self, component: str, success_rate: float, error_rate: float, latency: deque) -> str:
        """计算组件健康状态"""
        if not latency:
            return "unknown"
        
        avg_latency = statistics.mean(latency)
        
        if success_rate >= 95.0 and error_rate < 2.0 and avg_latency < 500.0:
            return "healthy"
        elif success_rate >= 90.0 and error_rate < 5.0 and avg_latency < 1000.0:
            return "warning"
        else:
            return "unhealthy"
    
    def get_all_components_performance(self) -> Dict[str, Any]:
        """获取所有组件性能报告"""
        components = {}
        healthy_count = 0
        warning_count = 0
        unhealthy_count = 0
        unknown_count = 0
        
        for component in self.component_performance:
            perf = self.get_component_performance(component)
            components[component] = perf
            
            # 更新健康状态计数
            health = perf.get("health_status", "unknown")
            if health == "healthy":
                healthy_count += 1
            elif health == "warning":
                warning_count += 1
            elif health == "unhealthy":
                unhealthy_count += 1
            else:
                unknown_count += 1
        
        return {
            "components": components,
            "summary": {
                "total_components": len(components),
                "healthy_components": healthy_count,
                "warning_components": warning_count,
                "unhealthy_components": unhealthy_count,
                "unknown_components": unknown_count,
                "health_score": healthy_count / len(components) * 100 if components else 0.0
            }
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
        
        with self.system_metrics_lock:
            # 获取所有活跃指标的摘要
            metrics_summaries = {}
            critical_metrics = []
            warning_metrics = []
            healthy_metrics = []
            
            for metric_type in self.thresholds.keys():
                summary = self.get_metrics_summary(metric_type, "1h")
                if "error" not in summary:
                    metrics_summaries[metric_type] = summary
                    
                    # 评估指标健康状态
                    threshold = self.thresholds.get(metric_type, {})
                    if summary["p95"] >= threshold.get("critical", float('inf')):
                        critical_metrics.append(metric_type)
                    elif summary["p95"] >= threshold.get("warning", float('inf')):
                        warning_metrics.append(metric_type)
                    else:
                        healthy_metrics.append(metric_type)
            
            # 计算系统健康状态
            overall_health = "healthy"
            if critical_metrics:
                overall_health = "critical"
            elif warning_metrics:
                overall_health = "warning"
        
        # 生成系统报告
        report = {
            "timestamp": datetime.now(),
            "overall_health": overall_health,
            "metrics_summary": {
                "total_metrics": len(metrics_summaries),
                "healthy_metrics": len(healthy_metrics),
                "warning_metrics": len(warning_metrics),
                "critical_metrics": len(critical_metrics),
                "health_score": len(healthy_metrics) / len(metrics_summaries) * 100 if metrics_summaries else 100.0
            },
            "components": metrics_summaries,
            "component_performance": self.get_all_components_performance(),
            "alerts": {
                "critical": len([a for a in self.alerts if a["level"] == "critical" and not a["acknowledged"]]),
                "warning": len([a for a in self.alerts if a["level"] == "warning" and not a["acknowledged"]]),
                "total": len(self.alerts),
                "acknowledged": len([a for a in self.alerts if a["acknowledged"]])
            },
            "system_resources": self._get_system_resources_summary(),
            "recommendations": self._generate_recommendations(critical_metrics, warning_metrics),
            "monitoring_status": {
                "is_running": self.is_running,
                "interval": self.monitoring_interval,
                "history_size": self.max_history_size
            }
        }
        
        return report
    
    def _get_system_resources_summary(self) -> Dict[str, Any]:
        """获取系统资源摘要"""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "memory_available": memory.available / (1024 * 1024 * 1024),  # GB
                "disk_usage": disk.percent,
                "disk_free": disk.free / (1024 * 1024 * 1024),  # GB
                "network_sent": net_io.bytes_sent / (1024 * 1024),  # MB
                "network_recv": net_io.bytes_recv / (1024 * 1024),  # MB
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"获取系统资源摘要错误: {str(e)}")
            return {}
    
    def _generate_recommendations(self, critical_metrics: List[str], warning_metrics: List[str]) -> List[str]:
        """生成性能改进建议"""
        recommendations = []
        
        # 基于关键指标生成建议
        for metric in critical_metrics:
            if metric == "cpu_usage":
                recommendations.append("CPU使用率过高，建议检查运行中的进程，优化算法或增加CPU资源")
            elif metric == "memory_usage":
                recommendations.append("内存使用率过高，建议优化内存使用，检查内存泄漏或增加内存资源")
            elif metric == "disk_usage":
                recommendations.append("磁盘使用率过高，建议清理磁盘空间或增加存储资源")
            elif metric == "decision_latency":
                recommendations.append("决策延迟过高，建议优化决策算法或增加计算资源")
            elif metric == "error_rate":
                recommendations.append("错误率过高，建议检查系统错误日志，修复bug")
            else:
                recommendations.append(f"{metric} 性能达到临界值，建议进行详细检查和优化")
        
        # 基于警告指标生成建议
        for metric in warning_metrics:
            if metric == "cpu_usage":
                recommendations.append("CPU使用率接近阈值，建议监控CPU密集型任务")
            elif metric == "memory_usage":
                recommendations.append("内存使用率接近阈值，建议优化内存管理")
            elif metric == "disk_usage":
                recommendations.append("磁盘使用率接近阈值，建议考虑扩展存储")
            elif metric == "decision_latency":
                recommendations.append("决策延迟接近阈值，建议优化决策流程")
            else:
                recommendations.append(f"{metric} 性能达到警告值，建议关注并进行优化")
        
        return recommendations
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """确认警报"""
        
        for alert in self.alerts:
            if alert["id"] == alert_id and not alert["acknowledged"]:
                alert["acknowledged"] = True
                alert["acknowledged_time"] = datetime.now()
                alert["acknowledged_by"] = acknowledged_by
                logger.info(f"警报已确认: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def get_active_alerts(self, level: str = None) -> List[Dict[str, Any]]:
        """获取活跃（未确认）警报"""
        alerts = [alert for alert in self.alerts if not alert["acknowledged"]]
        
        if level:
            alerts = [alert for alert in alerts if alert["level"] == level]
        
        return sorted(alerts, key=lambda x: (x["level"] == "critical", x["timestamp"]), reverse=True)
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取警报历史"""
        return sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def clear_old_alerts(self, days: int = 7) -> int:
        """清理旧警报"""
        cutoff_time = datetime.now() - timedelta(days=days)
        original_count = len(self.alerts)
        
        self.alerts = [
            alert for alert in self.alerts 
            if (datetime.now() - alert["timestamp"]).total_seconds() < days * 24 * 3600
        ]
        
        cleared_count = original_count - len(self.alerts)
        logger.info(f"清理了 {cleared_count} 条超过 {days} 天的旧警报")
        return cleared_count
    
    def get_performance_prediction(self, metric_type: str, time_window: int = 60) -> Dict[str, Any]:
        """基于历史数据预测性能趋势"""
        """基于历史数据预测性能趋势（简化实现）"""
        with self.system_metrics_lock:
            if metric_type not in self.metrics_history:
                return {"error": f"未知指标类型: {metric_type}"}
            
            metrics = list(self.metrics_history[metric_type])
            if len(metrics) < 10:  # 数据不足，无法预测
                return {"error": "历史数据不足，无法进行预测"}
        
        # 简化的线性趋势预测
        values = [m.value for m in metrics[-time_window:]]
        if not values:
            return {"error": "没有足够的数据进行预测"}
        
        # 计算趋势斜率
        x = list(range(len(values)))
        if len(x) < 2:
            return {"predicted_value": values[-1], "trend": "stable"}
        
        # 简单线性回归
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x_squared = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x_squared - sum_x ** 2 == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        
        intercept = (sum_y - slope * sum_x) / n
        
        # 预测下一个时间点的值
        predicted_value = slope * (n) + intercept
        
        # 预测趋势
        if abs(slope) < 0.01:  # 斜率接近0，趋势稳定
            trend = "stable"
        elif slope > 0:  # 上升趋势
            trend = "rising"
        else:  # 下降趋势
            trend = "falling"
        
        return {
            "metric_type": metric_type,
            "predicted_value": predicted_value,
            "current_value": values[-1],
            "slope": slope,
            "trend": trend,
            "confidence": min(1.0, len(values) / 100.0)  # 基于数据量的简单置信度
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        report = self.get_system_performance_report()
        alerts = self.get_active_alerts()
        
        return {
            "status": report["overall_health"],
            "critical_alerts": len([a for a in alerts if a["level"] == "critical"]),
            "warning_alerts": len([a for a in alerts if a["level"] == "warning"]),
            "health_score": report["metrics_summary"]["health_score"],
            "component_health": report["component_performance"]["summary"]
        }
    
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