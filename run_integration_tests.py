#!/usr/bin/env python3
"""
集成测试脚本 - 测试迁移学习和边缘计算集成功能
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.src.integration.migration_integration import MigrationIntegration
    from backend.src.integration.edge_integration import EdgeIntegration
    from backend.src.integration.decision_integration import DecisionIntegration
    from backend.src.performance.performance_monitor import PerformanceMonitor
    from backend.src.performance.performance_optimizer import PerformanceOptimizer
    from backend.src.migration_learning.risk_control import RiskControlSystem
    from backend.src.edge_computing.deployment_strategy import DeploymentStrategy
    from backend.src.edge_computing.model_lightweight import ModelLightweight
    from backend.src.edge_computing.cloud_edge_sync import CloudEdgeSync
    from backend.src.edge_computing.resource_optimizer import ResourceOptimizer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保后端依赖已安装")
    sys.exit(1)

class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None
        
    def log_test_result(self, test_name, status, message=None, duration=None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_symbol = "✅" if status == "PASS" else "❌"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        print(f"{status_symbol} {test_name}: {status}{duration_str}")
        if message:
            print(f"   {message}")
    
    async def run_migration_learning_tests(self):
        """运行迁移学习集成测试"""
        print("\n🔬 运行迁移学习集成测试...")
        
        try:
            # 测试风险控制系统
            start_time = time.time()
            risk_control = RiskControlSystem()
            
            # 测试数据验证
            test_data = {
                "model_type": "agriculture",
                "source_domain": "wheat_production",
                "target_domain": "corn_production",
                "training_samples": 1000,
                "validation_samples": 200
            }
            
            validation_result = await risk_control.validate_migration_data(test_data)
            duration = time.time() - start_time
            
            if validation_result["valid"]:
                self.log_test_result(
                    "迁移学习数据验证", "PASS", 
                    "数据验证通过", duration
                )
            else:
                self.log_test_result(
                    "迁移学习数据验证", "FAIL", 
                    f"数据验证失败: {validation_result.get('errors', [])}", duration
                )
            
            # 测试规则约束
            rule_test_result = await risk_control.check_rule_constraints(test_data)
            duration = time.time() - start_time
            
            if rule_test_result["compliant"]:
                self.log_test_result(
                    "迁移学习规则约束", "PASS", 
                    "规则约束检查通过", duration
                )
            else:
                self.log_test_result(
                    "迁移学习规则约束", "FAIL", 
                    f"规则约束检查失败: {rule_test_result.get('violations', [])}", duration
                )
            
            # 测试风险预警
            warning_result = await risk_control.assess_risk_level(test_data)
            duration = time.time() - start_time
            
            if warning_result["risk_level"] in ["LOW", "MEDIUM"]:
                self.log_test_result(
                    "迁移学习风险评估", "PASS", 
                    f"风险评估等级: {warning_result['risk_level']}", duration
                )
            else:
                self.log_test_result(
                    "迁移学习风险评估", "WARN", 
                    f"高风险等级: {warning_result['risk_level']}", duration
                )
            
        except Exception as e:
            self.log_test_result(
                "迁移学习集成测试", "ERROR", 
                f"测试执行异常: {str(e)}"
            )
    
    async def run_edge_computing_tests(self):
        """运行边缘计算集成测试"""
        print("\n🌐 运行边缘计算集成测试...")
        
        try:
            # 测试模型轻量化
            start_time = time.time()
            lightweight = ModelLightweight()
            
            # 模拟模型压缩
            model_info = {
                "model_id": "agriculture_model_v1",
                "original_size": 500,  # MB
                "target_size": 50,     # MB
                "accuracy_threshold": 0.85
            }
            
            compression_result = await lightweight.compress_model(model_info)
            duration = time.time() - start_time
            
            if compression_result["success"]:
                self.log_test_result(
                    "模型轻量化压缩", "PASS", 
                    f"压缩率: {compression_result.get('compression_ratio', 0):.2f}", duration
                )
            else:
                self.log_test_result(
                    "模型轻量化压缩", "FAIL", 
                    f"压缩失败: {compression_result.get('error', '未知错误')}", duration
                )
            
            # 测试部署策略
            deployment_strategy = DeploymentStrategy()
            deployment_result = await deployment_strategy.optimize_deployment(
                model_info, 
                edge_nodes=["edge_node_1", "edge_node_2"]
            )
            duration = time.time() - start_time
            
            if deployment_result["optimal"]:
                self.log_test_result(
                    "边缘部署策略优化", "PASS", 
                    f"最优节点: {deployment_result.get('optimal_node', '未知')}", duration
                )
            else:
                self.log_test_result(
                    "边缘部署策略优化", "FAIL", 
                    "部署策略优化失败", duration
                )
            
            # 测试云边协同
            cloud_edge_sync = CloudEdgeSync()
            sync_result = await cloud_edge_sync.sync_model_updates(
                "agriculture_model_v1", 
                "edge_node_1"
            )
            duration = time.time() - start_time
            
            if sync_result["synced"]:
                self.log_test_result(
                    "云边协同同步", "PASS", 
                    "模型同步成功", duration
                )
            else:
                self.log_test_result(
                    "云边协同同步", "FAIL", 
                    f"同步失败: {sync_result.get('error', '未知错误')}", duration
                )
            
        except Exception as e:
            self.log_test_result(
                "边缘计算集成测试", "ERROR", 
                f"测试执行异常: {str(e)}"
            )
    
    async def run_integration_tests(self):
        """运行系统集成测试"""
        print("\n🔗 运行系统集成测试...")
        
        try:
            # 测试迁移学习集成
            migration_integration = MigrationIntegration()
            
            start_time = time.time()
            integration_result = await migration_integration.integrate_with_system()
            duration = time.time() - start_time
            
            if integration_result["integrated"]:
                self.log_test_result(
                    "迁移学习系统集成", "PASS", 
                    "系统集成成功", duration
                )
            else:
                self.log_test_result(
                    "迁移学习系统集成", "FAIL", 
                    f"集成失败: {integration_result.get('error', '未知错误')}", duration
                )
            
            # 测试边缘计算集成
            edge_integration = EdgeIntegration()
            
            start_time = time.time()
            edge_result = await edge_integration.integrate_edge_system()
            duration = time.time() - start_time
            
            if edge_result["integrated"]:
                self.log_test_result(
                    "边缘计算系统集成", "PASS", 
                    "边缘系统集成成功", duration
                )
            else:
                self.log_test_result(
                    "边缘计算系统集成", "FAIL", 
                    f"集成失败: {edge_result.get('error', '未知错误')}", duration
                )
            
            # 测试决策引擎集成
            decision_integration = DecisionIntegration()
            
            start_time = time.time()
            decision_result = await decision_integration.integrate_decision_system()
            duration = time.time() - start_time
            
            if decision_result["integrated"]:
                self.log_test_result(
                    "决策引擎集成", "PASS", 
                    "决策引擎集成成功", duration
                )
            else:
                self.log_test_result(
                    "决策引擎集成", "FAIL", 
                    f"集成失败: {decision_result.get('error', '未知错误')}", duration
                )
            
        except Exception as e:
            self.log_test_result(
                "系统集成测试", "ERROR", 
                f"测试执行异常: {str(e)}"
            )
    
    async def run_performance_tests(self):
        """运行性能测试"""
        print("\n📊 运行性能测试...")
        
        try:
            # 测试性能监控
            performance_monitor = PerformanceMonitor()
            
            start_time = time.time()
            metrics = await performance_monitor.collect_system_metrics()
            duration = time.time() - start_time
            
            if metrics and "cpu_usage" in metrics:
                self.log_test_result(
                    "性能监控数据收集", "PASS", 
                    f"收集到 {len(metrics)} 个指标", duration
                )
            else:
                self.log_test_result(
                    "性能监控数据收集", "FAIL", 
                    "指标收集失败", duration
                )
            
            # 测试性能优化
            performance_optimizer = PerformanceOptimizer()
            
            start_time = time.time()
            optimization_result = await performance_optimizer.analyze_performance()
            duration = time.time() - start_time
            
            if optimization_result["analysis_complete"]:
                recommendations = len(optimization_result.get("recommendations", []))
                self.log_test_result(
                    "性能优化分析", "PASS", 
                    f"生成 {recommendations} 条优化建议", duration
                )
            else:
                self.log_test_result(
                    "性能优化分析", "FAIL", 
                    "性能分析失败", duration
                )
            
        except Exception as e:
            self.log_test_result(
                "性能测试", "ERROR", 
                f"测试执行异常: {str(e)}"
            )
    
    def generate_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        print("\n" + "="*60)
        print("📋 集成测试报告")
        print("="*60)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"错误: {error_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        print(f"测试耗时: {self.end_time - self.start_time:.2f}秒")
        
        # 保存详细报告
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": (passed_tests/total_tests)*100,
                "duration": self.end_time - self.start_time,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": self.test_results
        }
        
        # 保存到文件
        report_file = Path("integration_test_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_file.absolute()}")
        
        return report
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("开始集成测试...")
        self.start_time = time.time()
        
        # 运行各模块测试
        await self.run_migration_learning_tests()
        await self.run_edge_computing_tests()
        await self.run_integration_tests()
        await self.run_performance_tests()
        
        self.end_time = time.time()
        
        # 生成报告
        report = self.generate_report()
        
        # 返回测试结果
        success_rate = report["summary"]["success_rate"]
        if success_rate >= 80:
            print("\n🎉 集成测试通过！系统集成完成。")
            return True
        else:
            print("\n⚠️  集成测试部分失败，请检查详细报告。")
            return False

async def main():
    """主函数"""
    runner = IntegrationTestRunner()
    success = await runner.run_all_tests()
    
    # 根据测试结果退出
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())