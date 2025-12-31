#!/usr/bin/env python3
"""
迁移学习和边缘计算集成性能基准测试脚本
自动化运行性能基准测试，生成详细的性能报告
"""

import sys
import json
import asyncio
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入性能测试模块
try:
    from backend.src.performance.benchmark_test import BenchmarkTestSuite
    from backend.src.performance.performance_monitor import IntegrationPerformanceMonitor
    from backend.config.migration_edge_integration_config import (
        DeploymentEnvironment, OptimizationStrategy
    )
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保项目依赖已安装并正确配置")
    sys.exit(1)


class PerformanceBenchmarkRunner:
    """性能基准测试运行器"""
    
    def __init__(self, environment: str = "production", strategy: str = "performance") -> None:
        self.environment: DeploymentEnvironment = DeploymentEnvironment(environment)
        self.strategy: OptimizationStrategy = OptimizationStrategy(strategy)
        self.benchmark_suite: BenchmarkTestSuite = BenchmarkTestSuite()
        self.performance_monitor: IntegrationPerformanceMonitor = IntegrationPerformanceMonitor()
        self.results: list[dict[str, Any]] = []
    
    async def run_migration_learning_benchmark(self) -> dict[str, Any]:
        """运行迁移学习基准测试"""
        print("🧠 运行迁移学习基准测试...")
        
        # 定义测试场景
        test_scenarios: list[dict[str, Any]] = [
            {
                "name": "农业图像分类迁移",
                "source_domain": "general_images",
                "target_domain": "agriculture_images",
                "data_size": 5000,
                "complexity": "medium",
                "data_quality": 0.9
            },
            {
                "name": "作物识别迁移",
                "source_domain": "plant_images",
                "target_domain": "crop_images",
                "data_size": 3000,
                "complexity": "high",
                "data_quality": 0.85
            },
            {
                "name": "病虫害检测迁移",
                "source_domain": "disease_images",
                "target_domain": "pest_images",
                "data_size": 2000,
                "complexity": "high",
                "data_quality": 0.8
            }
        ]
        
        result = await self.benchmark_suite.run_migration_learning_benchmark(test_scenarios)
        
        # 记录性能指标
        accuracy: float = float(result.metrics.get("average_accuracy", 0.85))
        await self.performance_monitor.record_migration_learning_performance(
            source_domain="benchmark",
            target_domain="benchmark",
            accuracy=accuracy,
            baseline_accuracy=0.8,
            processing_time=result.duration
        )
        
        return {
            "test_type": "migration_learning",
            "result": result,
            "scenarios": test_scenarios
        }
    
    async def run_edge_computing_benchmark(self) -> dict[str, Any]:
        """运行边缘计算基准测试"""
        print("⚡ 运行边缘计算基准测试...")
        
        # 定义边缘节点
        edge_nodes: list[dict[str, Any]] = [
            {
                "node_id": "edge_node_01",
                "cpu_cores": 4,
                "memory_gb": 8,
                "storage_gb": 64,
                "network_bandwidth": 100
            },
            {
                "node_id": "edge_node_02", 
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 32,
                "network_bandwidth": 50
            }
        ]
        
        # 定义计算任务
        tasks: list[dict[str, Any]] = [
            {
                "task_id": "real_time_inference",
                "computational_intensity": 2,
                "data_size": 10,
                "latency_requirement": 100
            },
            {
                "task_id": "batch_processing",
                "computational_intensity": 5,
                "data_size": 100,
                "latency_requirement": 500
            },
            {
                "task_id": "model_training",
                "computational_intensity": 8,
                "data_size": 1000,
                "latency_requirement": 1000
            }
        ]
        
        result = await self.benchmark_suite.run_edge_computing_benchmark(edge_nodes, tasks)
        
        # 记录性能指标
        edge_latency: float = float(result.metrics.get("average_edge_latency", 0))
        cloud_latency: float = float(result.metrics.get("average_cloud_latency", 0))
        await self.performance_monitor.record_edge_computing_performance(
            node_id="benchmark_node",
            task_type="benchmark",
            edge_latency=edge_latency,
            cloud_latency=cloud_latency,
            resource_utilization={
                "cpu": 0.7,
                "memory": 0.6,
                "storage": 0.3
            }
        )
        
        return {
            "test_type": "edge_computing",
            "result": result,
            "nodes": edge_nodes,
            "tasks": tasks
        }
    
    async def run_integration_benchmark(self) -> dict[str, Any]:
        """运行集成基准测试"""
        print("🔗 运行集成基准测试...")
        
        # 定义集成场景
        integration_scenarios: list[dict[str, Any]] = [
            {
                "name": "迁移学习+边缘推理",
                "integration_complexity": "high",
                "components": ["migration_learning", "edge_computing"],
                "data_flow": "cloud_to_edge"
            },
            {
                "name": "实时决策集成",
                "integration_complexity": "medium", 
                "components": ["decision_engine", "performance_monitor"],
                "data_flow": "edge_to_cloud"
            },
            {
                "name": "端到端工作流",
                "integration_complexity": "high",
                "components": ["migration_learning", "edge_computing", "decision_engine"],
                "data_flow": "hybrid"
            }
        ]
        
        result = await self.benchmark_suite.run_integration_benchmark(integration_scenarios)
        
        # 记录集成性能指标
        await self.performance_monitor.record_integration_metric(
            integration_type="benchmark",
            operation="integration_test",
            duration=result.duration,
            success=True,
            additional_tags={
                "scenarios_count": len(integration_scenarios),
                "success_rate": result.success_rate
            }
        )
        
        return {
            "test_type": "integration",
            "result": result,
            "scenarios": integration_scenarios
        }
    
    async def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """运行综合基准测试"""
        print("🎯 运行综合基准测试...")
        
        start_time = time.time()
        
        # 并行运行所有基准测试
        migration_result, edge_result, integration_result = await asyncio.gather(
            self.run_migration_learning_benchmark(),
            self.run_edge_computing_benchmark(), 
            self.run_integration_benchmark(),
            return_exceptions=True
        )
        
        end_time = time.time()
        
        # 处理结果
        results: list[dict[str, Any]] = []
        for result in [migration_result, edge_result, integration_result]:
            if isinstance(result, Exception):
                print(f"❌ 基准测试失败: {result}")
                results.append({"error": str(result)})
            else:
                results.append(result)
        
        # 计算综合得分
        overall_score = self._calculate_overall_score(results)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment.value,
            "strategy": self.strategy.value,
            "duration": end_time - start_time,
            "overall_score": overall_score,
            "results": results,
            "performance_level": self._get_performance_level(overall_score)
        }
    
    def _calculate_overall_score(self, results: list[dict[str, Any]]) -> float:
        """计算综合得分"""
        if not results:
            return 0.0
        
        total_score: float = 0.0
        valid_results = 0
        
        for result in results:
            if "error" not in result and "result" in result:
                benchmark_result = result["result"]
                
                # 基于成功率和吞吐量计算分数
                success_rate: float = float(benchmark_result.success_rate)
                throughput: float = float(benchmark_result.throughput)
                
                success_score: float = success_rate * 0.6
                throughput_score: float = min(throughput / 10, 1.0) * 0.4
                
                total_score += (success_score + throughput_score)
                valid_results += 1
        
        return total_score / valid_results if valid_results > 0 else 0.0
    
    def _get_performance_level(self, score: float) -> str:
        """获取性能等级"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.7:
            return "良好"
        elif score >= 0.5:
            return "一般"
        else:
            return "需要改进"
    
    def generate_detailed_report(self, benchmark_results: dict[str, Any]) -> dict[str, Any]:
        """生成详细报告"""
        
        report: dict[str, Any] = {
            "benchmark_info": {
                "timestamp": benchmark_results["timestamp"],
                "environment": benchmark_results["environment"],
                "strategy": benchmark_results["strategy"],
                "total_duration": benchmark_results["duration"],
                "overall_score": benchmark_results["overall_score"],
                "performance_level": benchmark_results["performance_level"]
            },
            "test_results": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        # 分析各个测试结果
        for result in benchmark_results["results"]:
            if "error" in result:
                test_type: str = str(result.get("test_type", "unknown"))
                report["test_results"][test_type] = {
                    "status": "failed",
                    "error": result["error"]
                }
            else:
                test_type = str(result["test_type"])
                test_result = result["result"]
                report["test_results"][test_type] = {
                    "status": "completed",
                    "duration": float(test_result.duration),
                    "success_rate": float(test_result.success_rate),
                    "throughput": float(test_result.throughput),
                    "metrics": test_result.metrics
                }
        
        # 性能分析
        report["performance_analysis"] = self._analyze_performance(report["test_results"])
        
        # 生成优化建议
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _analyze_performance(self, test_results: dict[str, Any]) -> dict[str, list[str]]:
        """分析性能数据"""
        
        analysis: dict[str, list[str]] = {
            "strengths": [],
            "weaknesses": [],
            "bottlenecks": [],
            "opportunities": []
        }
        
        for test_type, result in test_results.items():
            if result["status"] != "completed":
                continue
            
            # 分析优势
            success_rate: float = float(result["success_rate"])
            throughput: float = float(result["throughput"])
            
            if success_rate > 0.9:
                analysis["strengths"].append(f"{test_type} 成功率优秀 ({success_rate:.1%})")
            
            if throughput > 5.0:
                analysis["strengths"].append(f"{test_type} 吞吐量良好 ({throughput:.1f} 操作/秒)")
            
            # 分析弱点
            if success_rate < 0.7:
                analysis["weaknesses"].append(f"{test_type} 成功率较低 ({success_rate:.1%})")
            
            if throughput < 1.0:
                analysis["weaknesses"].append(f"{test_type} 吞吐量较低 ({throughput:.1f} 操作/秒)")
            
            # 分析瓶颈
            metrics = result.get("metrics", {})
            if "average_scenario_duration" in metrics and \
               float(metrics["average_scenario_duration"]) > 10.0:
                analysis["bottlenecks"].append(f"{test_type} 场景执行时间较长")
        
        # 分析机会
        if "migration_learning" in test_results and \
           test_results["migration_learning"]["status"] == "completed":
            accuracy: float = float(test_results["migration_learning"]["metrics"].get("average_accuracy", 0))
            if accuracy < 0.85:
                analysis["opportunities"].append("迁移学习精度有提升空间")
        
        if "edge_computing" in test_results and \
           test_results["edge_computing"]["status"] == "completed":
            latency_reduction: float = float(test_results["edge_computing"]["metrics"].get("latency_reduction_percentage", 0))
            if latency_reduction < 20.0:
                analysis["opportunities"].append("边缘计算延迟降低效果可进一步优化")
        
        return analysis
    
    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """生成优化建议"""
        
        recommendations: list[str] = []
        analysis: dict[str, list[str]] = report["performance_analysis"]
        overall_score: float = float(report["benchmark_info"]["overall_score"])
        
        # 基于整体得分
        if overall_score < 0.5:
            recommendations.append("系统性能需要重大优化，建议重新评估架构设计")
        elif overall_score < 0.7:
            recommendations.append("系统性能有较大提升空间，建议进行针对性优化")
        
        # 基于弱点分析
        for weakness in analysis["weaknesses"]:
            if "成功率较低" in weakness:
                recommendations.append("优化错误处理机制，提高操作成功率")
            elif "吞吐量较低" in weakness:
                recommendations.append("优化并发处理能力，提高系统吞吐量")
        
        # 基于瓶颈分析
        for bottleneck in analysis["bottlenecks"]:
            if "场景执行时间较长" in bottleneck:
                recommendations.append("优化算法效率，减少场景执行时间")
        
        # 基于机会分析
        for opportunity in analysis["opportunities"]:
            if "迁移学习精度" in opportunity:
                recommendations.append("优化迁移学习参数，提高模型精度")
            elif "边缘计算延迟" in opportunity:
                recommendations.append("优化边缘资源分配，进一步降低延迟")
        
        # 环境特定建议
        environment: str = str(report["benchmark_info"]["environment"])
        if environment == "production":
            recommendations.append("生产环境建议启用自动优化功能")
        elif environment == "edge":
            recommendations.append("边缘环境建议优化资源使用效率")
        
        if not recommendations:
            recommendations.append("系统性能良好，继续保持当前配置")
        
        return recommendations


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='迁移学习和边缘计算集成性能基准测试')
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'testing', 'staging', 'production', 'edge'],
                       default='production',
                       help='测试环境')
    parser.add_argument('--strategy', '-s',
                       choices=['performance', 'accuracy', 'resource_efficiency', 'latency', 'cost'],
                       default='performance',
                       help='优化策略')
    parser.add_argument('--output', '-o',
                       default='performance_benchmark_report.json',
                       help='输出报告文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 迁移学习和边缘计算集成性能基准测试")
    print("=" * 60)
    print(f"环境: {args.environment}")
    print(f"优化策略: {args.strategy}")
    print(f"输出文件: {args.output}")
    print()
    
    try:
        # 创建测试运行器
        runner = PerformanceBenchmarkRunner(args.environment, args.strategy)
        
        # 运行基准测试
        benchmark_results = await runner.run_comprehensive_benchmark()
        
        # 生成详细报告
        detailed_report = runner.generate_detailed_report(benchmark_results)
        
        # 保存报告
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        
        # 显示摘要
        print("\n📊 基准测试完成!")
        print("=" * 40)
        print(f"综合得分: {detailed_report['benchmark_info']['overall_score']:.3f}")
        print(f"性能等级: {detailed_report['benchmark_info']['performance_level']}")
        print(f"测试耗时: {detailed_report['benchmark_info']['total_duration']:.1f} 秒")
        
        print("\n📈 测试结果:")
        for test_type, result in detailed_report["test_results"].items():
            status_icon = "✅" if result["status"] == "completed" else "❌"
            if result["status"] == "completed":
                print(f"  {status_icon} {test_type}: "
                      f"成功率 {result['success_rate']:.1%}, "
                      f"吞吐量 {result['throughput']:.1f} 操作/秒")
            else:
                print(f"  {status_icon} {test_type}: 失败 - {result.get('error', '未知错误')}")
        
        print("\n💡 主要建议:")
        for i, recommendation in enumerate(detailed_report["recommendations"][:3], 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\n📄 详细报告已保存到: {args.output}")
        
        # 详细模式输出
        if args.verbose:
            print("\n" + "=" * 60)
            print("详细性能分析:")
            print("=" * 60)
            
            analysis = detailed_report["performance_analysis"]
            for category, items in analysis.items():
                if items:
                    print(f"\n{category.upper()}:")
                    for item in items:
                        print(f"  • {item}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 基准测试执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)