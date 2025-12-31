"""
AI自主决策风险控制系统使用示例

展示如何在区块链经济模型中集成和使用风险控制系统，
包括技术失控、数据安全、算法偏见和治理冲突的全面监控。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .risk_monitoring_system import AIRiskMonitoringSystem, SystemRiskReport
from .technical_risk_controller import TechnicalRiskController
from .data_security_controller import DataSecurityController
from .algorithm_bias_controller import AlgorithmBiasController
from .governance_conflict_controller import GovernanceConflictController


class AIRiskControlExample:
    """AI风险控制使用示例"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化风险监控系统
        self.risk_monitor = AIRiskMonitoringSystem()
        
        # 初始化各风险控制器（用于独立测试）
        self.technical_controller = TechnicalRiskController()
        self.data_security_controller = DataSecurityController()
        self.algorithm_bias_controller = AlgorithmBiasController()
        self.governance_controller = GovernanceConflictController()
    
    async def run_comprehensive_example(self):
        """运行综合示例"""
        self.logger.info("=== AI自主决策风险控制系统综合示例 ===")
        
        # 1. 模拟系统数据
        system_data = await self._generate_sample_system_data()
        
        # 2. 独立测试各风险控制器
        await self._test_individual_controllers(system_data)
        
        # 3. 运行综合风险监控
        await self._run_integrated_monitoring(system_data)
        
        # 4. 展示应急响应
        await self._demonstrate_emergency_response()
        
        self.logger.info("=== 示例执行完成 ===")
    
    async def _generate_sample_system_data(self) -> Dict[str, Any]:
        """生成模拟系统数据"""
        self.logger.info("生成模拟系统数据...")
        
        # 模拟AI决策数据
        ai_decisions = [
            {
                "decision_id": f"decision_{i}",
                "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "decision_type": "reward_distribution",
                "result": "approved" if i % 3 != 0 else "rejected",
                "confidence": 0.85 + i * 0.01,
                "risk_score": 0.1 + i * 0.02,
                "features": {
                    "user_contribution": 1000 + i * 100,
                    "system_profit": 50000 + i * 5000,
                    "fairness_index": 0.7 + i * 0.01
                },
                "strategy": "optimization" if i % 2 == 0 else "conservative"
            }
            for i in range(20)
        ]
        
        # 模拟区块链数据
        blockchain_data = {
            "block_height": 123456,
            "consensus_metrics": {
                "success_rate": 0.95,
                "stability_index": 0.88,
                "node_participation": 0.82
            },
            "governance": {
                "proposal_count": 12,
                "voting_participation": 0.68,
                "decision_efficiency": 0.75,
                "transparency_index": 0.8
            },
            "economic_targets": ["公平分配", "系统稳定", "用户激励"],
            "protected_groups": ["small_holders", "early_contributors", "new_users"]
        }
        
        # 模拟系统指标
        system_metrics = {
            "cpu_usage": 0.72,
            "memory_usage": 0.68,
            "network_traffic": 1500,
            "active_connections": 200,
            "resource_usage": {
                "cpu": 72,
                "memory": 68,
                "storage": 45
            }
        }
        
        # 模拟训练数据
        training_data = {
            "size": 50000,
            "encrypted": True,
            "sensitive_fields": ["user_identity", "transaction_amount"],
            "privacy_techniques": ["differential_privacy", "federated_learning"],
            "distribution": {
                "small_holders": 0.4,
                "large_holders": 0.3,
                "institutional": 0.3
            },
            "data_quality": 0.85
        }
        
        # 模拟模型参数
        model_parameters = {
            "size": 250,  # MB
            "encrypted": True,
            "edge_compatible": True,
            "protection_methods": ["model_encryption", "secure_aggregation"],
            "aggregation_method": "fedavg"
        }
        
        # 模拟社区投票
        community_votes = [
            {
                "vote_id": f"vote_{i}",
                "timestamp": (datetime.utcnow() - timedelta(hours=i*2)).isoformat(),
                "proposal_id": f"proposal_{i % 5}",
                "voter_id": f"voter_{i % 50}",
                "vote_result": "yes" if i % 4 != 0 else "no",
                "user_group": "small_holders" if i % 3 == 0 else "large_holders"
            }
            for i in range(30)
        ]
        
        # 模拟安全日志
        security_logs = [
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i*30)).isoformat(),
                "event_type": "access",
                "user_id": f"user_{i % 20}",
                "resource": "model_parameters",
                "result": "success" if i % 5 != 0 else "failed",
                "access_granted": i % 5 != 0
            }
            for i in range(100)
        ]
        
        return {
            "ai_decisions": ai_decisions,
            "blockchain_data": blockchain_data,
            "system_metrics": system_metrics,
            "training_data": training_data,
            "model_parameters": model_parameters,
            "community_votes": community_votes,
            "security_logs": security_logs,
            "historical_data": {
                "historical_bias": 0.15,
                "previous_risk_scores": [0.2, 0.3, 0.25, 0.4, 0.35]
            }
        }
    
    async def _test_individual_controllers(self, system_data: Dict[str, Any]):
        """独立测试各风险控制器"""
        self.logger.info("\n1. 独立测试各风险控制器...")
        
        # 测试技术风险控制器
        technical_assessment = await self.technical_controller.assess_technical_risk(
            ai_decision_data={"decisions": system_data["ai_decisions"]},
            blockchain_context=system_data["blockchain_data"],
            system_state=system_data["system_metrics"]
        )
        
        self.logger.info(f"技术风险评估: 风险等级={technical_assessment.overall_risk_level}, "
                        f"评分={technical_assessment.risk_score:.3f}")
        
        # 测试数据安全控制器
        data_security_assessment = await self.data_security_controller.assess_data_security_risk(
            training_data=system_data["training_data"],
            model_parameters=system_data["model_parameters"],
            blockchain_context=system_data["blockchain_data"],
            access_logs=system_data["security_logs"]
        )
        
        self.logger.info(f"数据安全评估: 安全等级={data_security_assessment.overall_security_level}, "
                        f"评分={data_security_assessment.security_score:.3f}")
        
        # 测试算法偏见控制器
        bias_assessment = await self.algorithm_bias_controller.assess_bias_risk(
            training_data=system_data["training_data"],
            model_decisions=system_data["ai_decisions"],
            blockchain_context=system_data["blockchain_data"],
            historical_data=system_data["historical_data"]
        )
        
        self.logger.info(f"算法偏见评估: 偏见等级={bias_assessment.overall_bias_level}, "
                        f"公平性={bias_assessment.fairness_score:.3f}")
        
        # 测试治理冲突控制器
        governance_assessment = await self.governance_controller.assess_governance_conflict(
            ai_decisions=system_data["ai_decisions"],
            community_votes=system_data["community_votes"],
            blockchain_governance=system_data["blockchain_data"]["governance"],
            system_state=system_data["system_metrics"]
        )
        
        self.logger.info(f"治理冲突评估: 冲突等级={governance_assessment.overall_conflict_level}, "
                        f"协作性={governance_assessment.collaboration_score:.3f}")
    
    async def _run_integrated_monitoring(self, system_data: Dict[str, Any]):
        """运行综合风险监控"""
        self.logger.info("\n2. 运行综合风险监控...")
        
        # 启动监控系统
        await self.risk_monitor.start_monitoring()
        
        # 等待系统收集数据并生成报告
        await asyncio.sleep(2)
        
        # 手动触发一次综合评估
        risk_report = await self.risk_monitor.perform_comprehensive_assessment()
        
        # 展示评估结果
        self._display_risk_report(risk_report)
        
        # 停止监控（示例中只运行一次评估）
        await self.risk_monitor.stop_monitoring()
    
    def _display_risk_report(self, risk_report: SystemRiskReport):
        """展示风险报告"""
        self.logger.info("\n=== 综合风险监控报告 ===")
        self.logger.info(f"报告ID: {risk_report.report_id}")
        self.logger.info(f"生成时间: {risk_report.timestamp}")
        self.logger.info(f"系统状态: {risk_report.system_status.value}")
        self.logger.info(f"综合风险评分: {risk_report.overall_risk_score:.3f}")
        
        self.logger.info("\n风险分类详情:")
        for category, score in risk_report.risk_breakdown.items():
            self.logger.info(f"  {category.value}: {score:.3f}")
        
        self.logger.info(f"\n活跃警报数量: {len(risk_report.active_alerts)}")
        if risk_report.active_alerts:
            self.logger.info("重要警报:")
            for alert in risk_report.active_alerts[:3]:  # 显示前3个重要警报
                self.logger.info(f"  [{alert.priority.value}] {alert.description}")
        
        self.logger.info("\n改进建议:")
        for i, recommendation in enumerate(risk_report.recommendations, 1):
            self.logger.info(f"  {i}. {recommendation}")
    
    async def _demonstrate_emergency_response(self):
        """展示应急响应"""
        self.logger.info("\n3. 演示应急响应机制...")
        
        # 模拟高风险场景数据
        high_risk_data = await self._generate_high_risk_scenario()
        
        # 进行高风险评估
        high_risk_report = await self.risk_monitor.perform_comprehensive_assessment()
        
        # 检查是否触发应急响应
        if high_risk_report.system_status.value in ["critical", "emergency"]:
            self.logger.warning("⚠️ 检测到高风险场景，应急响应已触发!")
            
            # 展示应急措施
            if high_risk_report.emergency_actions_taken:
                self.logger.info("已执行的应急措施:")
                for action in high_risk_report.emergency_actions_taken:
                    self.logger.info(f"  • {action}")
        else:
            self.logger.info("当前系统运行正常，无需应急响应")
    
    async def _generate_high_risk_scenario(self) -> Dict[str, Any]:
        """生成高风险场景数据"""
        # 创建模拟的高风险数据
        return {
            "ai_decisions": [
                {
                    "decision_id": "high_risk_decision",
                    "timestamp": datetime.utcnow().isoformat(),
                    "result": "approved",
                    "confidence": 0.95,
                    "risk_score": 0.9,  # 高风险
                    "features": {"fairness_index": 0.1}  # 低公平性
                }
            ],
            "blockchain_data": {
                "consensus_metrics": {"success_rate": 0.3},  # 低共识率
                "governance": {"voting_participation": 0.2}  # 低参与度
            },
            "system_metrics": {
                "resource_usage": {"cpu": 95, "memory": 90}  # 高资源使用
            }
        }
    
    async def demonstrate_risk_mitigation(self):
        """演示风险缓解措施"""
        self.logger.info("\n4. 演示风险缓解措施...")
        
        # 模拟风险缓解场景
        mitigation_scenarios = [
            {
                "scenario": "技术失控风险",
                "mitigation": "启用紧急停止机制，限制AI决策权限",
                "effectiveness": "高风险警报消除，系统恢复正常"
            },
            {
                "scenario": "数据安全风险", 
                "mitigation": "加强数据加密，应用差分隐私技术",
                "effectiveness": "数据泄露风险降低85%"
            },
            {
                "scenario": "算法偏见风险",
                "mitigation": "重新训练模型，应用公平性算法",
                "effectiveness": "公平性评分从0.3提升至0.8"
            },
            {
                "scenario": "治理冲突风险",
                "mitigation": "建立人-AI协同决策机制",
                "effectiveness": "社区满意度提升40%"
            }
        ]
        
        for scenario in mitigation_scenarios:
            self.logger.info(f"\n场景: {scenario['scenario']}")
            self.logger.info(f"缓解措施: {scenario['mitigation']}")
            self.logger.info(f"效果: {scenario['effectiveness']}")


class BlockchainAIDecisionSimulator:
    """区块链AI决策模拟器"""
    
    def __init__(self, risk_monitor: AIRiskMonitoringSystem):
        self.risk_monitor = risk_monitor
        self.logger = logging.getLogger(__name__)
    
    async def simulate_ai_decision_making(self, duration_minutes: int = 10):
        """模拟AI决策过程"""
        self.logger.info(f"开始模拟AI决策过程，持续时间: {duration_minutes}分钟")
        
        # 启动风险监控
        await self.risk_monitor.start_monitoring()
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        decision_count = 0
        risk_alerts = 0
        
        while datetime.utcnow() < end_time:
            # 模拟AI决策
            decision = self._generate_ai_decision(decision_count)
            decision_count += 1
            
            # 记录决策
            self.logger.debug(f"AI决策 #{decision_count}: {decision['result']} (置信度: {decision['confidence']:.2f})")
            
            # 检查风险状态
            current_report = await self.risk_monitor.perform_comprehensive_assessment()
            
            if current_report.system_status.value in ["alert", "critical", "emergency"]:
                risk_alerts += 1
                self.logger.warning(f"风险警报! 系统状态: {current_report.system_status.value}")
            
            # 等待下一轮决策
            await asyncio.sleep(5)  # 每5秒一个决策
        
        # 停止监控
        await self.risk_monitor.stop_monitoring()
        
        # 生成模拟报告
        await self._generate_simulation_report(decision_count, risk_alerts, start_time, end_time)
    
    def _generate_ai_decision(self, decision_id: int) -> Dict[str, Any]:
        """生成模拟AI决策"""
        # 模拟不同类型的决策
        decision_types = ["reward_distribution", "parameter_adjustment", "risk_control", "resource_allocation"]
        decision_type = decision_types[decision_id % len(decision_types)]
        
        # 模拟决策结果（有一定随机性）
        import random
        result = "approved" if random.random() > 0.3 else "rejected"
        confidence = 0.7 + random.random() * 0.25  # 70-95%置信度
        risk_score = random.random() * 0.5  # 0-50%风险
        
        return {
            "decision_id": f"sim_decision_{decision_id}",
            "timestamp": datetime.utcnow().isoformat(),
            "decision_type": decision_type,
            "result": result,
            "confidence": confidence,
            "risk_score": risk_score,
            "strategy": "optimization" if decision_id % 2 == 0 else "conservative"
        }
    
    async def _generate_simulation_report(self, decision_count: int, risk_alerts: int, 
                                        start_time: datetime, end_time: datetime):
        """生成模拟报告"""
        duration = (end_time - start_time).total_seconds() / 60  # 分钟
        
        self.logger.info("\n=== AI决策模拟报告 ===")
        self.logger.info(f"模拟时长: {duration:.1f} 分钟")
        self.logger.info(f"总决策数: {decision_count}")
        self.logger.info(f"决策频率: {decision_count/duration:.1f} 决策/分钟")
        self.logger.info(f"风险警报数: {risk_alerts}")
        self.logger.info(f"风险发生率: {risk_alerts/decision_count*100:.1f}%")
        
        if risk_alerts > 0:
            self.logger.warning("⚠️ 检测到风险事件，建议检查系统配置")
        else:
            self.logger.info("✅ 模拟期间系统运行稳定")


async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建示例实例
        example = AIRiskControlExample()
        
        # 运行综合示例
        await example.run_comprehensive_example()
        
        # 演示风险缓解
        await example.demonstrate_risk_mitigation()
        
        # 运行决策模拟（可选）
        simulate_decision = input("\n是否运行AI决策模拟? (y/n): ").lower().strip()
        if simulate_decision == 'y':
            simulator = BlockchainAIDecisionSimulator(example.risk_monitor)
            await simulator.simulate_ai_decision_making(duration_minutes=2)  # 缩短为2分钟演示
        
        print("\n=== AI自主决策风险控制系统示例执行完成 ===")
        print("系统已成功演示以下功能:")
        print("• 技术失控风险检测与控制")
        print("• 数据安全与隐私保护")
        print("• 算法偏见与公平性保障")
        print("• 治理冲突解决机制")
        print("• 综合风险监控与预警")
        print("• 应急响应与风险缓解")
        
    except Exception as e:
        logging.error(f"示例执行失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())