"""
AI自主决策系统性能测试脚本
验证系统是否达到秒级实时响应能力
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import httpx
from concurrent.futures import ThreadPoolExecutor

class DecisionPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def test_agriculture_decision(self, iterations: int = 10) -> Dict[str, Any]:
        """测试农业参数优化决策性能"""
        url = f"{self.base_url}/api/v1/decision/agriculture/optimize"
        
        # 模拟农业参数
        payload = {
            "crop_type": "tomato",
            "current_params": {
                "temperature": 25.0,
                "humidity": 0.6,
                "light_intensity": 800,
                "soil_ph": 6.5
            },
            "historical_data": {
                "yield_history": [0.8, 0.9, 0.85, 0.92],
                "quality_scores": [85, 88, 82, 90]
            }
        }
        
        response_times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            try:
                response = await self.client.post(url, json=payload)
                if response.status_code == 200:
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)  # 转换为毫秒
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            # 短暂延迟避免服务器过载
            await asyncio.sleep(0.1)
        
        return {
            "module": "agriculture_decision",
            "response_times": response_times,
            "errors": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "iterations": iterations
        }
    
    async def test_blockchain_decision(self, iterations: int = 10) -> Dict[str, Any]:
        """测试区块链积分分配决策性能"""
        url = f"{self.base_url}/api/v1/decision/blockchain/allocate"
        
        payload = {
            "user_id": "user_123",
            "contribution_score": 85,
            "risk_level": "medium",
            "available_points": 1000,
            "historical_allocations": [
                {"date": "2024-01-01", "points": 100},
                {"date": "2024-01-02", "points": 150}
            ]
        }
        
        response_times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            try:
                response = await self.client.post(url, json=payload)
                if response.status_code == 200:
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            await asyncio.sleep(0.1)
        
        return {
            "module": "blockchain_decision",
            "response_times": response_times,
            "errors": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "iterations": iterations
        }
    
    async def test_model_training_decision(self, iterations: int = 10) -> Dict[str, Any]:
        """测试模型训练决策性能"""
        url = f"{self.base_url}/api/v1/decision/model_training/optimize"
        
        payload = {
            "model_type": "cnn",
            "dataset_size": 10000,
            "current_accuracy": 0.85,
            "available_resources": {
                "gpu_memory": 8,
                "cpu_cores": 4,
                "training_time": 3600
            },
            "performance_history": {
                "accuracy_trend": [0.7, 0.75, 0.8, 0.85],
                "training_times": [1200, 1100, 1000, 950]
            }
        }
        
        response_times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            try:
                response = await self.client.post(url, json=payload)
                if response.status_code == 200:
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            await asyncio.sleep(0.1)
        
        return {
            "module": "model_training_decision",
            "response_times": response_times,
            "errors": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "iterations": iterations
        }
    
    async def test_resource_decision(self, iterations: int = 10) -> Dict[str, Any]:
        """测试资源分配决策性能"""
        url = f"{self.base_url}/api/v1/decision/resource/allocate"
        
        payload = {
            "system_load": {
                "cpu_usage": 0.75,
                "memory_usage": 0.6,
                "network_traffic": 0.8
            },
            "pending_tasks": [
                {"task_type": "training", "priority": "high", "resource_needs": {"cpu": 2, "memory": 4}},
                {"task_type": "inference", "priority": "medium", "resource_needs": {"cpu": 1, "memory": 2}}
            ],
            "available_resources": {
                "total_cpu": 8,
                "total_memory": 16,
                "available_gpu": 2
            }
        }
        
        response_times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            try:
                response = await self.client.post(url, json=payload)
                if response.status_code == 200:
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            await asyncio.sleep(0.1)
        
        return {
            "module": "resource_decision",
            "response_times": response_times,
            "errors": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "iterations": iterations
        }
    
    async def run_comprehensive_test(self, iterations_per_module: int = 20) -> Dict[str, Any]:
        """运行综合性能测试"""
        print("开始AI自主决策系统性能测试...")
        print(f"每个模块测试 {iterations_per_module} 次迭代")
        
        # 并行测试所有模块
        tasks = [
            self.test_agriculture_decision(iterations_per_module),
            self.test_blockchain_decision(iterations_per_module),
            self.test_model_training_decision(iterations_per_module),
            self.test_resource_decision(iterations_per_module)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 分析总体性能
        all_response_times = []
        total_errors = 0
        total_iterations = 0
        
        for result in results:
            all_response_times.extend(result['response_times'])
            total_errors += result['errors']
            total_iterations += result['iterations']
        
        overall_performance = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_iterations": total_iterations,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_iterations) * 100 if total_iterations > 0 else 0,
            "overall_avg_response_time": statistics.mean(all_response_times) if all_response_times else 0,
            "overall_min_response_time": min(all_response_times) if all_response_times else 0,
            "overall_max_response_time": max(all_response_times) if all_response_times else 0,
            "modules": results
        }
        
        return overall_performance
    
    def print_results(self, results: Dict[str, Any]):
        """打印性能测试结果"""
        print("\n" + "="*60)
        print("AI自主决策系统性能测试结果")
        print("="*60)
        
        print(f"测试时间: {results['timestamp']}")
        print(f"总迭代次数: {results['total_iterations']}")
        print(f"总错误数: {results['total_errors']}")
        print(f"错误率: {results['error_rate']:.2f}%")
        print(f"总体平均响应时间: {results['overall_avg_response_time']:.2f}ms")
        print(f"总体最小响应时间: {results['overall_min_response_time']:.2f}ms")
        print(f"总体最大响应时间: {results['overall_max_response_time']:.2f}ms")
        
        # 检查是否达到秒级实时响应要求（<1000ms）
        if results['overall_avg_response_time'] < 1000:
            print("✅ 系统达到秒级实时响应要求!")
        else:
            print("❌ 系统未达到秒级实时响应要求，需要优化")
        
        print("\n各模块详细性能:")
        for module_result in results['modules']:
            print(f"\n{module_result['module']}:")
            print(f"  平均响应时间: {module_result['avg_response_time']:.2f}ms")
            print(f"  最小响应时间: {module_result['min_response_time']:.2f}ms")
            print(f"  最大响应时间: {module_result['max_response_time']:.2f}ms")
            print(f"  错误数: {module_result['errors']}")
    
    async def close(self):
        """关闭客户端连接"""
        await self.client.aclose()

async def main():
    """主测试函数"""
    tester = DecisionPerformanceTester()
    
    try:
        # 运行性能测试
        results = await tester.run_comprehensive_test(iterations_per_module=20)
        
        # 打印结果
        tester.print_results(results)
        
        # 保存结果到文件
        import json
        with open('decision_performance_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n性能测试结果已保存到 decision_performance_results.json")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())