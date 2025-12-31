"""
性能监控API路由
提供迁移学习和边缘计算集成的性能监控功能
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

# 导入性能监控模块
from src.performance.performance_monitor import IntegrationPerformanceMonitor
from src.performance.performance_optimizer import PerformanceOptimizer, AutoOptimizationManager
from src.performance.benchmark_test import BenchmarkTestSuite

router = APIRouter(prefix="/performance", tags=["performance"])

# 创建性能监控器实例
performance_monitor = IntegrationPerformanceMonitor()
performance_optimizer = PerformanceOptimizer(performance_monitor)
auto_optimization_manager = AutoOptimizationManager(performance_monitor)
benchmark_suite = BenchmarkTestSuite()


class PerformanceMetricRequest(BaseModel):
    """性能指标请求"""
    metric_type: str
    value: float
    tags: Optional[Dict[str, Any]] = None


class PerformanceMetricResponse(BaseModel):
    """性能指标响应"""
    metric_type: str
    timestamp: str
    value: float
    tags: Dict[str, Any]


class PerformanceSummaryResponse(BaseModel):
    """性能摘要响应"""
    timestamp: str
    overall_health: str
    components: Dict[str, Any]
    alerts: Dict[str, int]
    recommendations: List[str]


class IntegrationPerformanceRequest(BaseModel):
    """集成性能记录请求"""
    integration_type: str
    operation: str
    duration: float
    success: bool
    additional_tags: Optional[Dict[str, Any]] = None


class MigrationLearningPerformanceRequest(BaseModel):
    """迁移学习性能记录请求"""
    source_domain: str
    target_domain: str
    accuracy: float
    baseline_accuracy: float
    processing_time: float


class EdgeComputingPerformanceRequest(BaseModel):
    """边缘计算性能记录请求"""
    node_id: str
    task_type: str
    edge_latency: float
    cloud_latency: float
    resource_utilization: Dict[str, float]


class OptimizationRecommendationResponse(BaseModel):
    """优化建议响应"""
    timestamp: str
    component: str
    recommendation_type: str
    current_value: float
    target_value: float
    confidence: float
    impact: str


class BenchmarkTestRequest(BaseModel):
    """基准测试请求"""
    test_type: str
    parameters: Dict[str, Any]


class BenchmarkTestResponse(BaseModel):
    """基准测试响应"""
    test_name: str
    timestamp: str
    duration: float
    success_rate: float
    throughput: float
    resource_usage: Dict[str, float]
    metrics: Dict[str, Any]


@router.post("/metrics", response_model=PerformanceMetricResponse)
async def record_performance_metric(request: PerformanceMetricRequest):
    """记录性能指标"""
    try:
        await performance_monitor.record_metric(
            metric_type=request.metric_type,
            value=request.value,
            tags=request.tags or {}
        )
        
        return PerformanceMetricResponse(
            metric_type=request.metric_type,
            timestamp=datetime.now().isoformat(),
            value=request.value,
            tags=request.tags or {}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"记录性能指标失败: {str(e)}"
        )


@router.get("/summary", response_model=PerformanceSummaryResponse)
async def get_performance_summary(time_range: str = Query("1h", regex="^(1h|24h|7d)$")):
    """获取性能摘要"""
    try:
        summary = performance_monitor.get_system_performance_report()
        
        return PerformanceSummaryResponse(
            timestamp=datetime.now().isoformat(),
            overall_health=summary.get("overall_health", "unknown"),
            components=summary.get("components", {}),
            alerts=summary.get("alerts", {"critical": 0, "warning": 0}),
            recommendations=summary.get("recommendations", [])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能摘要失败: {str(e)}"
        )


@router.get("/integration-summary")
async def get_integration_performance_summary():
    """获取集成性能专项摘要"""
    try:
        integration_report = performance_monitor.get_integration_performance_report()
        return integration_report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集成性能摘要失败: {str(e)}"
        )


@router.post("/integration-metrics")
async def record_integration_performance(request: IntegrationPerformanceRequest):
    """记录集成操作性能指标"""
    try:
        await performance_monitor.record_integration_metric(
            integration_type=request.integration_type,
            operation=request.operation,
            duration=request.duration,
            success=request.success,
            additional_tags=request.additional_tags
        )
        
        return {
            "message": "集成性能指标记录成功",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"记录集成性能指标失败: {str(e)}"
        )


@router.post("/migration-learning-metrics")
async def record_migration_learning_performance(request: MigrationLearningPerformanceRequest):
    """记录迁移学习性能指标"""
    try:
        await performance_monitor.record_migration_learning_performance(
            source_domain=request.source_domain,
            target_domain=request.target_domain,
            accuracy=request.accuracy,
            baseline_accuracy=request.baseline_accuracy,
            processing_time=request.processing_time
        )
        
        return {
            "message": "迁移学习性能指标记录成功",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"记录迁移学习性能指标失败: {str(e)}"
        )


@router.post("/edge-computing-metrics")
async def record_edge_computing_performance(request: EdgeComputingPerformanceRequest):
    """记录边缘计算性能指标"""
    try:
        await performance_monitor.record_edge_computing_performance(
            node_id=request.node_id,
            task_type=request.task_type,
            edge_latency=request.edge_latency,
            cloud_latency=request.cloud_latency,
            resource_utilization=request.resource_utilization
        )
        
        return {
            "message": "边缘计算性能指标记录成功",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"记录边缘计算性能指标失败: {str(e)}"
        )


@router.get("/optimization/recommendations", response_model=List[OptimizationRecommendationResponse])
async def get_optimization_recommendations():
    """获取优化建议"""
    try:
        recommendations = await performance_optimizer.analyze_performance_data()
        
        return [
            OptimizationRecommendationResponse(
                timestamp=rec.timestamp.isoformat(),
                component=rec.component,
                recommendation_type=rec.recommendation_type,
                current_value=rec.current_value,
                target_value=rec.target_value,
                confidence=rec.confidence,
                impact=rec.impact
            )
            for rec in recommendations
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取优化建议失败: {str(e)}"
        )


@router.post("/optimization/apply")
async def apply_optimization(component: str, recommendation_type: str):
    """应用优化建议"""
    try:
        # 这里需要实现具体的优化应用逻辑
        # 目前返回模拟结果
        
        return {
            "message": f"已为组件 {component} 应用 {recommendation_type} 优化",
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "optimization_type": recommendation_type,
            "status": "applied"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"应用优化失败: {str(e)}"
        )


@router.get("/optimization/status")
async def get_optimization_status():
    """获取优化状态"""
    try:
        status_info = auto_optimization_manager.get_optimization_status()
        return status_info
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取优化状态失败: {str(e)}"
        )


@router.post("/benchmark/run", response_model=BenchmarkTestResponse)
async def run_benchmark_test(request: BenchmarkTestRequest):
    """运行基准测试"""
    try:
        if request.test_type == "migration_learning":
            # 运行迁移学习基准测试
            test_scenarios = request.parameters.get("scenarios", [])
            result = await benchmark_suite.run_migration_learning_benchmark(test_scenarios)
            
        elif request.test_type == "edge_computing":
            # 运行边缘计算基准测试
            edge_nodes = request.parameters.get("edge_nodes", [])
            tasks = request.parameters.get("tasks", [])
            result = await benchmark_suite.run_edge_computing_benchmark(edge_nodes, tasks)
            
        elif request.test_type == "integration":
            # 运行集成基准测试
            integration_scenarios = request.parameters.get("scenarios", [])
            result = await benchmark_suite.run_integration_benchmark(integration_scenarios)
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的测试类型: {request.test_type}"
            )
        
        return BenchmarkTestResponse(
            test_name=result.test_name,
            timestamp=result.timestamp.isoformat(),
            duration=result.duration,
            success_rate=result.success_rate,
            throughput=result.throughput,
            resource_usage=result.resource_usage,
            metrics=result.metrics
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"运行基准测试失败: {str(e)}"
        )


@router.get("/benchmark/report")
async def get_benchmark_report():
    """获取基准测试报告"""
    try:
        report = benchmark_suite.generate_benchmark_report()
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取基准测试报告失败: {str(e)}"
        )


@router.post("/benchmark/baseline")
async def set_benchmark_baseline(test_name: str):
    """设置基准线"""
    try:
        # 在实际应用中，这里需要获取当前的测试结果并设置为基准线
        # 目前返回模拟结果
        
        return {
            "message": f"已为测试 {test_name} 设置基准线",
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "status": "baseline_set"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置基准线失败: {str(e)}"
        )


@router.get("/alerts")
async def get_performance_alerts():
    """获取性能警报"""
    try:
        alerts = performance_monitor.alerts
        
        # 过滤未确认的警报
        unacknowledged_alerts = [alert for alert in alerts if not alert.get("acknowledged", False)]
        
        return {
            "total_alerts": len(alerts),
            "unacknowledged_alerts": len(unacknowledged_alerts),
            "alerts": unacknowledged_alerts
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能警报失败: {str(e)}"
        )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """确认警报"""
    try:
        success = await performance_monitor.acknowledge_alert(alert_id)
        
        if success:
            return {
                "message": f"警报 {alert_id} 已确认",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到警报: {alert_id}"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"确认警报失败: {str(e)}"
        )


@router.post("/auto-optimization/enable")
async def enable_auto_optimization():
    """启用自动优化"""
    try:
        auto_optimization_manager.enable_auto_optimization()
        
        return {
            "message": "自动优化已启用",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启用自动优化失败: {str(e)}"
        )


@router.post("/auto-optimization/disable")
async def disable_auto_optimization():
    """禁用自动优化"""
    try:
        auto_optimization_manager.disable_auto_optimization()
        
        return {
            "message": "自动优化已禁用",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"禁用自动优化失败: {str(e)}"
        )


@router.post("/auto-optimization/run")
async def run_auto_optimization():
    """手动运行自动优化"""
    try:
        await auto_optimization_manager.start_auto_optimization()
        
        return {
            "message": "自动优化已运行",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"运行自动优化失败: {str(e)}"
        )


@router.get("/metrics/{metric_type}")
async def get_metric_details(metric_type: str, time_range: str = Query("1h", regex="^(1h|24h|7d)$")):
    """获取特定指标的详细数据"""
    try:
        summary = performance_monitor.get_metrics_summary(metric_type, time_range)
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取指标详情失败: {str(e)}"
        )


# 启动时自动启用自动优化
@router.on_event("startup")
async def startup_event():
    """应用启动时自动启用自动优化"""
    try:
        auto_optimization_manager.enable_auto_optimization()
        print("性能监控系统已启动，自动优化已启用")
    except Exception as e:
        print(f"启动性能监控系统失败: {e}")