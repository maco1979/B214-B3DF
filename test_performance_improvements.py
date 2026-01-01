"""
性能测试脚本 - 验证异步缓存和智能体权限改进效果
用于测试新增功能的性能表现
"""

import asyncio
import time
import statistics
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

class PerformanceTestRunner:
    """性能测试运行器"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
    
    def record_result(self, test_name: str, success: bool, 
                     duration_ms: float, details: Dict = None):
        """记录测试结果"""
        self.test_count += 1
        if success:
            self.passed_count += 1
        else:
            self.failed_count += 1
        
        self.results[test_name] = {
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("性能测试结果摘要")
        print("=" * 60)
        print(f"总测试数: {self.test_count}")
        print(f"通过: {self.passed_count}")
        print(f"失败: {self.failed_count}")
        print(f"通过率: {self.passed_count/self.test_count*100:.1f}%")
        print("=" * 60)
        
        for name, result in self.results.items():
            status = "✓" if result["success"] else "✗"
            print(f"{status} {name}: {result['duration_ms']}ms")
            if result["details"]:
                for key, value in result["details"].items():
                    print(f"    {key}: {value}")
        print("=" * 60)


async def test_async_memory_cache():
    """测试异步内存缓存性能"""
    print("\n[测试] 异步内存缓存性能...")
    
    try:
        from backend.src.core.services.async_cache_service import AsyncMemoryCache
        
        cache = AsyncMemoryCache(max_size=1000, default_ttl=60)
        
        # 写入性能测试
        write_times = []
        for i in range(100):
            start = time.perf_counter()
            await cache.set(f"key_{i}", {"data": f"value_{i}", "index": i})
            write_times.append((time.perf_counter() - start) * 1000)
        
        # 读取性能测试
        read_times = []
        for i in range(100):
            start = time.perf_counter()
            value = await cache.get(f"key_{i}")
            read_times.append((time.perf_counter() - start) * 1000)
        
        # 缓存命中率测试
        hits = 0
        misses = 0
        for i in range(200):
            key = f"key_{i % 150}"  # 150个键，但缓存只有100个
            value = await cache.get(key)
            if value:
                hits += 1
            else:
                misses += 1
        
        avg_write = statistics.mean(write_times)
        avg_read = statistics.mean(read_times)
        hit_rate = hits / (hits + misses) * 100
        
        return True, avg_write + avg_read, {
            "avg_write_ms": round(avg_write, 3),
            "avg_read_ms": round(avg_read, 3),
            "hit_rate": f"{hit_rate:.1f}%",
            "write_p99": round(sorted(write_times)[98], 3),
            "read_p99": round(sorted(read_times)[98], 3)
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


async def test_decision_cache():
    """测试决策结果缓存性能"""
    print("\n[测试] 决策结果缓存性能...")
    
    try:
        from backend.src.core.services.async_cache_service import DecisionResultCache
        
        cache = DecisionResultCache()
        
        # 模拟决策结果
        test_decision = {
            "action": "optimize",
            "confidence": 0.95,
            "parameters": {"threshold": 0.8}
        }
        
        # 写入决策
        cache_times = []
        for i in range(50):
            start = time.perf_counter()
            await cache.cache_decision(
                module=f"module_{i % 10}",
                state={"iteration": i},
                objective="performance",
                decision=test_decision
            )
            cache_times.append((time.perf_counter() - start) * 1000)
        
        # 读取决策
        get_times = []
        cache_hits = 0
        for i in range(50):
            start = time.perf_counter()
            result = await cache.get_cached_decision(
                module=f"module_{i % 10}",
                state={"iteration": i},
                objective="performance"
            )
            get_times.append((time.perf_counter() - start) * 1000)
            if result:
                cache_hits += 1
        
        avg_cache = statistics.mean(cache_times)
        avg_get = statistics.mean(get_times)
        
        return True, avg_cache + avg_get, {
            "avg_cache_ms": round(avg_cache, 3),
            "avg_get_ms": round(avg_get, 3),
            "cache_hit_rate": f"{cache_hits/50*100:.1f}%"
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


async def test_agent_permission_manager():
    """测试智能体权限管理器性能"""
    print("\n[测试] 智能体权限管理器性能...")
    
    try:
        from backend.src.core.services.agent_permission_manager import (
            AgentPermissionManager, AgentType
        )
        
        manager = AgentPermissionManager()
        
        # 注册智能体
        register_times = []
        for i in range(20):
            agent_type = list(AgentType)[i % len(AgentType)]
            start = time.perf_counter()
            await manager.register_agent(
                agent_id=f"agent_{i}",
                agent_type=agent_type,
                name=f"Test Agent {i}"
            )
            register_times.append((time.perf_counter() - start) * 1000)
        
        # 权限检查
        check_times = []
        granted_count = 0
        for i in range(100):
            start = time.perf_counter()
            result = await manager.check_permission(
                agent_id=f"agent_{i % 20}",
                permission="decision.read"
            )
            check_times.append((time.perf_counter() - start) * 1000)
            if result:
                granted_count += 1
        
        # 获取审计摘要
        start = time.perf_counter()
        audit_summary = await manager.get_audit_summary(hours=1)
        audit_time = (time.perf_counter() - start) * 1000
        
        avg_register = statistics.mean(register_times)
        avg_check = statistics.mean(check_times)
        
        return True, avg_register + avg_check, {
            "avg_register_ms": round(avg_register, 3),
            "avg_check_ms": round(avg_check, 3),
            "check_p99_ms": round(sorted(check_times)[98], 3),
            "granted_rate": f"{granted_count}%",
            "audit_time_ms": round(audit_time, 3)
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


async def test_audit_monitoring_service():
    """测试审计监控服务性能"""
    print("\n[测试] 审计监控服务性能...")
    
    try:
        from backend.src.core.services.audit_monitoring_service import (
            AuditLogMonitoringService
        )
        
        service = AuditLogMonitoringService()
        
        # 记录智能体动作
        action_times = []
        for i in range(100):
            start = time.perf_counter()
            await service.record_agent_action(
                agent_id=f"agent_{i % 10}",
                agent_type="decision_agent",
                action="process",
                status="success",
                duration_ms=50 + i
            )
            action_times.append((time.perf_counter() - start) * 1000)
        
        # 记录权限检查
        permission_times = []
        for i in range(100):
            start = time.perf_counter()
            await service.record_permission_check(
                agent_id=f"agent_{i % 10}",
                permission="decision.create",
                granted=i % 3 != 0  # 约33%拒绝
            )
            permission_times.append((time.perf_counter() - start) * 1000)
        
        # 导出Prometheus格式
        start = time.perf_counter()
        metrics = await service.get_prometheus_metrics()
        export_time = (time.perf_counter() - start) * 1000
        
        # 获取仪表板数据
        start = time.perf_counter()
        dashboard = await service.get_monitoring_dashboard_data()
        dashboard_time = (time.perf_counter() - start) * 1000
        
        avg_action = statistics.mean(action_times)
        avg_permission = statistics.mean(permission_times)
        
        return True, avg_action + avg_permission, {
            "avg_action_record_ms": round(avg_action, 3),
            "avg_permission_record_ms": round(avg_permission, 3),
            "prometheus_export_ms": round(export_time, 3),
            "dashboard_fetch_ms": round(dashboard_time, 3),
            "metrics_lines": len(metrics.split('\n'))
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


async def test_concurrent_cache_access():
    """测试并发缓存访问性能"""
    print("\n[测试] 并发缓存访问性能...")
    
    try:
        from backend.src.core.services.async_cache_service import AsyncMemoryCache
        
        cache = AsyncMemoryCache(max_size=500, default_ttl=60)
        
        # 预填充缓存
        for i in range(100):
            await cache.set(f"concurrent_key_{i}", {"value": i})
        
        # 并发读写测试
        async def read_task(key_id):
            times = []
            for _ in range(50):
                start = time.perf_counter()
                await cache.get(f"concurrent_key_{key_id % 100}")
                times.append((time.perf_counter() - start) * 1000)
            return times
        
        async def write_task(key_id):
            times = []
            for _ in range(50):
                start = time.perf_counter()
                await cache.set(f"concurrent_new_{key_id}", {"new": key_id})
                times.append((time.perf_counter() - start) * 1000)
            return times
        
        # 10个并发任务
        start = time.perf_counter()
        tasks = []
        for i in range(5):
            tasks.append(read_task(i))
            tasks.append(write_task(i))
        
        results = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start) * 1000
        
        all_times = [t for result in results for t in result]
        
        return True, total_time, {
            "concurrent_tasks": 10,
            "operations_per_task": 50,
            "total_operations": len(all_times),
            "avg_op_time_ms": round(statistics.mean(all_times), 3),
            "total_time_ms": round(total_time, 2),
            "ops_per_second": round(len(all_times) / (total_time / 1000), 0)
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


async def test_rate_limiting():
    """测试速率限制功能"""
    print("\n[测试] 速率限制功能...")
    
    try:
        from backend.src.core.services.agent_permission_manager import (
            AgentPermissionManager, AgentType
        )
        
        manager = AgentPermissionManager()
        
        # 创建一个有速率限制的智能体
        await manager.register_agent(
            agent_id="rate_test_agent",
            agent_type=AgentType.DECISION_AGENT,
            name="Rate Test Agent",
            rate_limit=100  # 每分钟100次
        )
        
        # 快速执行多次权限检查
        times = []
        granted = 0
        denied = 0
        
        for i in range(150):  # 超过限制的请求数
            start = time.perf_counter()
            result = await manager.check_permission(
                agent_id="rate_test_agent",
                permission="decision.read"
            )
            times.append((time.perf_counter() - start) * 1000)
            if result:
                granted += 1
            else:
                denied += 1
        
        return True, statistics.mean(times), {
            "total_requests": 150,
            "rate_limit": 100,
            "granted": granted,
            "denied_by_rate_limit": denied,
            "avg_check_time_ms": round(statistics.mean(times), 3)
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


async def main():
    """运行所有性能测试"""
    print("=" * 60)
    print("性能测试套件 - 验证改进效果")
    print(f"开始时间: {datetime.now().isoformat()}")
    print("=" * 60)
    
    runner = PerformanceTestRunner()
    
    # 运行测试
    tests = [
        ("异步内存缓存", test_async_memory_cache),
        ("决策结果缓存", test_decision_cache),
        ("智能体权限管理器", test_agent_permission_manager),
        ("审计监控服务", test_audit_monitoring_service),
        ("并发缓存访问", test_concurrent_cache_access),
        ("速率限制", test_rate_limiting),
    ]
    
    for test_name, test_func in tests:
        try:
            success, duration, details = await test_func()
            runner.record_result(test_name, success, duration, details)
        except Exception as e:
            runner.record_result(test_name, False, 0, {"error": str(e)})
    
    runner.print_summary()
    
    # 生成JSON报告
    import json
    report = {
        "test_time": datetime.now().isoformat(),
        "summary": {
            "total": runner.test_count,
            "passed": runner.passed_count,
            "failed": runner.failed_count,
            "pass_rate": f"{runner.passed_count/runner.test_count*100:.1f}%"
        },
        "results": runner.results
    }
    
    report_path = "performance_improvement_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试报告已保存至: {report_path}")
    
    return runner.failed_count == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
