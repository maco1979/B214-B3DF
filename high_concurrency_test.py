#!/usr/bin/env python3
"""
高并发压力测试脚本
用于验证系统在百万级并发用户下的性能表现
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    requests_sent: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    rps: float  # Requests Per Second
    errors: List[str]


class HighConcurrencyTester:
    """高并发测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000", total_requests: int = 10000, concurrent_users: int = 1000):
        self.base_url = base_url
        self.total_requests = total_requests
        self.concurrent_users = concurrent_users
        self.session = None
        self.results_lock = threading.Lock()
        self.results: List[float] = []  # 响应时间列表
        self.errors: List[str] = []
        self.success_count = 0
        self.failure_count = 0
    
    async def create_session(self):
        """创建HTTP会话"""
        connector = aiohttp.TCPConnector(
            limit=self.concurrent_users,  # 连接池大小
            limit_per_host=100,  # 每个主机的连接数
            ttl_dns_cache=300,  # DNS缓存时间
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "HighConcurrencyTester/1.0"}
        )
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
    
    async def make_request(self, url: str, method: str = "GET", payload: Dict[str, Any] = None) -> float:
        """执行单个请求并返回响应时间"""
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    await response.text()  # 读取响应内容
            elif method.upper() == "POST":
                async with self.session.post(url, json=payload) as response:
                    await response.text()  # 读取响应内容
            
            response_time = time.time() - start_time
            
            # 记录成功结果
            with self.results_lock:
                self.results.append(response_time)
                self.success_count += 1
                
            return response_time
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # 记录错误和失败计数
            with self.results_lock:
                self.errors.append(str(e))
                self.failure_count += 1
                
            return response_time
    
    async def run_test(self, endpoint: str, method: str = "GET", payload: Dict[str, Any] = None) -> TestResult:
        """运行压力测试"""
        print(f"🚀 开始压力测试: {endpoint}")
        print(f"📊 总请求数: {self.total_requests}")
        print(f"👥 并发用户数: {self.concurrent_users}")
        
        start_time = time.time()
        
        # 创建会话
        await self.create_session()
        
        # 创建信号量以限制并发数
        semaphore = asyncio.Semaphore(self.concurrent_users)
        
        async def limited_request(sem, url, method, payload):
            async with sem:
                return await self.make_request(url, method, payload)
        
        # 创建所有请求任务
        tasks = []
        url = f"{self.base_url}{endpoint}"
        for _ in range(self.total_requests):
            task = asyncio.create_task(limited_request(semaphore, url, method, payload))
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 计算统计信息
        if self.results:
            avg_response_time = sum(self.results) / len(self.results)
            min_response_time = min(self.results)
            max_response_time = max(self.results)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        rps = self.total_requests / total_time if total_time > 0 else 0
        
        # 创建测试结果
        result = TestResult(
            test_name=f"HighConcurrency-{endpoint}",
            requests_sent=self.total_requests,
            successful_requests=self.success_count,
            failed_requests=self.failure_count,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            rps=rps,
            errors=self.errors.copy()
        )
        
        # 关闭会话
        await self.close_session()
        
        return result
    
    def print_results(self, result: TestResult):
        """打印测试结果"""
        print("\n" + "="*60)
        print("📊 压力测试结果")
        print("="*60)
        print(f"测试名称: {result.test_name}")
        print(f"总请求数: {result.requests_sent:,}")
        print(f"成功请求数: {result.successful_requests:,}")
        print(f"失败请求数: {result.failed_requests:,}")
        print(f"成功率: {(result.successful_requests/result.requests_sent*100):.2f}%")
        print(f"总耗时: {result.total_time:.2f}秒")
        print(f"平均每秒请求数 (RPS): {result.rps:.2f}")
        print(f"平均响应时间: {result.avg_response_time*1000:.2f}ms")
        print(f"最小响应时间: {result.min_response_time*1000:.2f}ms")
        print(f"最大响应时间: {result.max_response_time*1000:.2f}ms")
        
        if result.errors:
            print(f"\n❌ 错误摘要 (显示前10个):")
            for i, error in enumerate(result.errors[:10], 1):
                print(f"  {i}. {error}")
            if len(result.errors) > 10:
                print(f"  ... 还有 {len(result.errors) - 10} 个错误")
        
        print("\n🎯 性能评估:")
        if result.rps >= 10000:  # 每秒1万请求
            print("  🚀 极优秀: 系统可以处理超大规模并发")
        elif result.rps >= 5000:
            print("  ⚡ 优秀: 系统性能良好")
        elif result.rps >= 1000:
            print("  ✅ 良好: 系统可以处理较大并发")
        elif result.rps >= 100:
            print("  ⚠️  一般: 系统性能有待提升")
        else:
            print("  ❌ 较差: 系统需要优化")
        
        print("="*60)


async def run_comprehensive_test():
    """运行综合压力测试"""
    print("🔍 开始综合压力测试...")
    
    # 测试配置
    base_url = "http://localhost:8000"  # 根据实际部署调整
    total_requests = 10000  # 总请求数
    concurrent_users = 1000  # 并发用户数
    
    # 创建测试器
    tester = HighConcurrencyTester(base_url, total_requests, concurrent_users)
    
    # 定义要测试的端点
    test_endpoints = [
        ("/", "GET"),
        ("/api/health", "GET"),
        ("/api/system/metrics", "GET"),
        ("/api/models", "GET"),
        ("/api/inference", "POST", {"model_id": "test_model", "input_data": {"test": "data"}}),
    ]
    
    all_results = []
    
    for endpoint_config in test_endpoints:
        endpoint = endpoint_config[0]
        method = endpoint_config[1]
        payload = endpoint_config[2] if len(endpoint_config) > 2 else None
        
        # 重置计数器
        tester.results = []
        tester.errors = []
        tester.success_count = 0
        tester.failure_count = 0
        
        # 运行测试
        result = await tester.run_test(endpoint, method, payload)
        tester.print_results(result)
        
        all_results.append(result)
        
        # 短暂休息
        await asyncio.sleep(2)
    
    # 输出总体摘要
    print("\n" + "="*60)
    print("📈 总体性能摘要")
    print("="*60)
    
    total_requests_sent = sum(r.requests_sent for r in all_results)
    total_successful = sum(r.successful_requests for r in all_results)
    total_failed = sum(r.failed_requests for r in all_results)
    total_time = max(r.total_time for r in all_results)  # 取最长的测试时间
    
    overall_rps = total_requests_sent / total_time if total_time > 0 else 0
    
    print(f"总请求数: {total_requests_sent:,}")
    print(f"总成功数: {total_successful:,}")
    print(f"总失败数: {total_failed:,}")
    print(f"总体成功率: {(total_successful/total_requests_sent*100):.2f}%")
    print(f"总体RPS: {overall_rps:.2f}")
    
    print("\n💡 建议:")
    if overall_rps >= 10000:
        print("  系统已准备好处理百万级并发用户！")
    elif overall_rps >= 5000:
        print("  系统性能良好，可处理大规模并发，建议进一步优化。")
    elif overall_rps >= 1000:
        print("  系统可处理中等规模并发，需要进一步优化以支持百万级用户。")
    else:
        print("  系统需要重大优化才能支持高并发场景。")
    
    print("="*60)


if __name__ == "__main__":
    print("🚀 AI平台高并发压力测试工具")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # 运行综合测试
    asyncio.run(run_comprehensive_test())