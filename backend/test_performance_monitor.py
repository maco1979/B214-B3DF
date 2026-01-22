#!/usr/bin/env python3
"""
测试性能监控模块
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入性能监控模块
from src.performance.performance_monitor import PerformanceMonitor

async def test_performance_monitor():
    """测试性能监控模块"""
    print("=== 测试性能监控模块 ===")
    
    # 创建性能监控实例
    monitor = PerformanceMonitor(max_history_size=1000)
    
    print("1. 测试指标记录...")
    # 记录一些测试指标
    for i in range(20):
        await monitor.record_metric(
            metric_type="cpu_usage",
            value=40.0 + i * 2.0,
            unit="%",
            source="test"
        )
        
        await monitor.record_metric(
            metric_type="memory_usage",
            value=60.0 + i * 1.5,
            unit="%",
            source="test"
        )
        
        await monitor.record_metric(
            metric_type="decision_latency",
            value=100.0 + i * 10.0,
            unit="ms",
            source="test",
            component="decision_engine"
        )
    
    print(f"   记录了 {len(monitor.metrics_history['cpu_usage'])} 个CPU使用率指标")
    print(f"   记录了 {len(monitor.metrics_history['memory_usage'])} 个内存使用率指标")
    print(f"   记录了 {len(monitor.metrics_history['decision_latency'])} 个决策延迟指标")
    
    print("\n2. 测试指标摘要...")
    # 获取指标摘要
    cpu_summary = monitor.get_metrics_summary("cpu_usage", "1h")
    print(f"   CPU使用率摘要: 平均值={cpu_summary['avg']:.2f}%, 最大值={cpu_summary['max']:.2f}%, 95%分位数={cpu_summary['p95']:.2f}%")
    
    memory_summary = monitor.get_metrics_summary("memory_usage", "1h")
    print(f"   内存使用率摘要: 平均值={memory_summary['avg']:.2f}%, 最大值={memory_summary['max']:.2f}%, 95%分位数={memory_summary['p95']:.2f}%")
    
    decision_summary = monitor.get_metrics_summary("decision_latency", "1h")
    print(f"   决策延迟摘要: 平均值={decision_summary['avg']:.2f}ms, 最大值={decision_summary['max']:.2f}ms, 95%分位数={decision_summary['p95']:.2f}ms")
    
    print("\n3. 测试组件性能...")
    # 检查组件性能
    component_perf = monitor.get_component_performance("decision_engine")
    print(f"   组件 'decision_engine' 健康状态: {component_perf['health_status']}")
    print(f"   组件 'decision_engine' 总请求数: {component_perf['total_requests']}")
    print(f"   组件 'decision_engine' 成功率: {component_perf['success_rate']:.2f}%")
    print(f"   组件 'decision_engine' 平均延迟: {component_perf['latency']['avg']:.2f}ms")
    
    print("\n4. 测试所有组件性能...")
    all_components = monitor.get_all_components_performance()
    print(f"   总组件数: {all_components['summary']['total_components']}")
    print(f"   健康组件数: {all_components['summary']['healthy_components']}")
    print(f"   警告组件数: {all_components['summary']['warning_components']}")
    print(f"   不健康组件数: {all_components['summary']['unhealthy_components']}")
    
    print("\n5. 测试系统性能报告...")
    # 获取系统性能报告
    system_report = monitor.get_system_performance_report()
    print(f"   系统整体健康状态: {system_report['overall_health']}")
    print(f"   指标总数: {system_report['metrics_summary']['total_metrics']}")
    print(f"   健康指标数: {system_report['metrics_summary']['healthy_metrics']}")
    print(f"   警告指标数: {system_report['metrics_summary']['warning_metrics']}")
    print(f"   关键指标数: {system_report['metrics_summary']['critical_metrics']}")
    print(f"   健康分数: {system_report['metrics_summary']['health_score']:.2f}%")
    
    print("\n6. 测试系统资源摘要...")
    # 获取系统资源摘要
    system_resources = system_report['system_resources']
    print(f"   CPU使用率: {system_resources['cpu_usage']:.2f}%")
    print(f"   内存使用率: {system_resources['memory_usage']:.2f}%")
    print(f"   可用内存: {system_resources['memory_available']:.2f} GB")
    print(f"   磁盘使用率: {system_resources['disk_usage']:.2f}%")
    print(f"   可用磁盘空间: {system_resources['disk_free']:.2f} GB")
    
    print("\n7. 测试警报生成...")
    # 生成一个超过阈值的指标
    await monitor.record_metric(
        metric_type="cpu_usage",
        value=98.0,
        unit="%",
        source="test"
    )
    
    print(f"   生成了 {len(monitor.alerts)} 个警报")
    if monitor.alerts:
        alert = monitor.alerts[0]
        print(f"   警报ID: {alert['id']}")
        print(f"   警报级别: {alert['level']}")
        print(f"   警报指标: {alert['metric_type']}")
        print(f"   警报值: {alert['value']} {alert['unit']}")
        print(f"   警报消息: {alert['message']}")
    
    print("\n8. 测试警报确认...")
    # 确认警报
    if monitor.alerts:
        alert_id = monitor.alerts[0]['id']
        result = await monitor.acknowledge_alert(alert_id, "test_user")
        print(f"   确认警报结果: {'成功' if result else '失败'}")
        
        # 检查警报是否已确认
        alert = next((a for a in monitor.alerts if a['id'] == alert_id), None)
        if alert and alert['acknowledged']:
            print(f"   警报已成功确认，确认者: {alert['acknowledged_by']}")
    
    print("\n9. 测试性能预测...")
    # 测试性能预测
    cpu_prediction = monitor.get_performance_prediction("cpu_usage")
    if "error" not in cpu_prediction:
        print(f"   CPU使用率预测: {cpu_prediction['predicted_value']:.2f}%")
        print(f"   CPU使用率趋势: {cpu_prediction['trend']}")
        print(f"   预测置信度: {cpu_prediction['confidence']:.2f}")
    else:
        print(f"   CPU使用率预测: {cpu_prediction['error']}")
    
    print("\n10. 测试健康状态...")
    # 获取健康状态
    health_status = monitor.get_health_status()
    print(f"   系统健康状态: {health_status['status']}")
    print(f"   健康分数: {health_status['health_score']:.2f}%")
    print(f"   关键警报数: {health_status['critical_alerts']}")
    print(f"   警告警报数: {health_status['warning_alerts']}")
    
    print("\n11. 测试监控启动和停止...")
    # 测试监控启动和停止
    print("   启动性能监控...")
    await monitor.start_monitoring(interval=1.0)
    print(f"   监控状态: {'运行中' if monitor.is_running else '已停止'}")
    
    print("   停止性能监控...")
    await monitor.stop_monitoring()
    print(f"   监控状态: {'运行中' if monitor.is_running else '已停止'}")
    
    print("\n=== 性能监控模块测试完成 ===")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_performance_monitor())
