"""
AI自主决策风险控制系统测试

包含完整的单元测试和集成测试，验证各风险控制模块的功能正确性，
确保系统在各种场景下能够准确识别和管理AI自主决策风险。
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from .technical_risk_controller import TechnicalRiskController, TechnicalRiskAssessment
from .data_security_controller import DataSecurityController, DataSecurityAssessment
from .algorithm_bias_controller import AlgorithmBiasController, BiasRiskAssessment
from .governance_conflict_controller import GovernanceConflictController, GovernanceConflictAssessment
from .risk_monitoring_system import (
    AIRiskMonitoringSystem, SystemRiskReport, RiskAlert,
    RiskCategory, SystemStatus, AlertPriority
)


class TestTechnicalRiskController:
    """技术风险控制器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.controller = TechnicalRiskController()
    
    @pytest.mark.asyncio
    async def test_assess_technical_risk_normal(self):
        """测试正常情况下的技术风险评估"""
        # 准备测试数据
        ai_decision_data = {
            "decisions": [
                {
                    "decision_id": "test_decision_1",
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "approve_transaction",
                    "confidence": 0.85,
                    "risk_score": 0.1
                }
            ]
        }
        
        blockchain_context = {
            "block_height": 1000,
            "consensus_metrics": {"stability": 0.95}
        }
        
        system_state = {
            "cpu_usage": 0.6,
            "memory_usage": 0.7
        }
        
        # 执行测试
        result = await self.controller.assess_technical_risk(
            ai_decision_data, blockchain_context, system_state
        )
        
        # 验证结果
        assert isinstance(result, TechnicalRiskAssessment)
        assert 0 <= result.risk_score <= 1.0
        assert len(result.active_alerts) >= 0
    
    @pytest.mark.asyncio
    async def test_assess_technical_risk_critical(self):
        """测试高风险情况下的技术风险评估"""
        # 准备高风险数据
        ai_decision_data = {
            "decisions": [
                {
                    "decision_id": "high_risk_decision",
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "modify_contract_parameters",
                    "confidence": 0.5,
                    "risk_score": 0.9
                }
            ]
        }
        
        blockchain_context = {
            "block_height": 1000,
            "consensus_metrics": {"stability": 0.3}  # 低稳定性
        }
        
        system_state = {
            "cpu_usage": 0.95,
            "memory_usage": 0.9  # 高负载
        }
        
        # 执行测试
        result = await self.controller.assess_technical_risk(
            ai_decision_data, blockchain_context, system_state
        )
        
        # 验证高风险
        assert result.risk_score > 0.7
        assert len(result.active_alerts) > 0
        assert any(alert.severity.value == "critical" for alert in result.active_alerts)
    
    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self):
        """测试紧急停止功能"""
        # 准备高风险决策
        ai_decision_data = {
            "decisions": [
                {
                    "decision_id": "dangerous_decision",
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "override_safety_limits",
                    "confidence": 0.3,
                    "risk_score": 0.95
                }
            ]
        }
        
        # 执行评估
        result = await self.controller.assess_technical_risk(
            ai_decision_data, {}, {}
        )
        
        # 验证紧急停止建议
        critical_alerts = [alert for alert in result.active_alerts 
                          if alert.severity.value == "critical"]
        
        if critical_alerts:
            assert any("紧急停止" in alert.mitigation_action 
                      for alert in critical_alerts)


class TestDataSecurityController:
    """数据安全控制器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.controller = DataSecurityController()
    
    @pytest.mark.asyncio
    async def test_assess_data_security_normal(self):
        """测试正常情况下的数据安全评估"""
        # 准备测试数据
        training_data = {
            "sensitive_fields_encrypted": True,
            "data_retention_policy": "compliant",
            "access_controls": "strict"
        }
        
        model_parameters = {
            "encryption_status": "encrypted",
            "access_logs": []
        }
        
        blockchain_context = {
            "privacy_features": {
                "zero_knowledge_proofs": True,
                "data_anonymization": True
            }
        }
        
        access_logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "authorized_user",
                "action": "read",
                "result": "success"
            }
        ]
        
        # 执行测试
        result = await self.controller.assess_data_security_risk(
            training_data, model_parameters, blockchain_context, access_logs
        )
        
        # 验证结果
        assert isinstance(result, DataSecurityAssessment)
        assert 0 <= result.security_score <= 1.0
        assert result.overall_security_level in ["low", "medium", "high", "excellent"]
    
    @pytest.mark.asyncio
    async def test_assess_data_security_breach(self):
        """测试数据泄露情况下的安全评估"""
        # 准备高风险数据
        training_data = {
            "sensitive_fields_encrypted": False,  # 未加密
            "data_retention_policy": "none",
            "access_controls": "weak"
        }
        
        model_parameters = {
            "encryption_status": "plaintext",  # 明文存储
            "access_logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": "unknown_user",
                    "action": "write",
                    "result": "success"
                }
            ]
        }
        
        # 执行测试
        result = await self.controller.assess_data_security_risk(
            training_data, model_parameters, {}, []
        )
        
        # 验证高风险
        assert result.security_score < 0.5
        assert result.overall_security_level in ["low", "medium"]
        assert len(result.security_vulnerabilities) > 0


class TestAlgorithmBiasController:
    """算法偏见控制器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.controller = AlgorithmBiasController()
    
    @pytest.mark.asyncio
    async def test_assess_bias_risk_fair(self):
        """测试公平算法偏见评估"""
        # 准备公平数据
        training_data = {
            "data_distribution": {
                "user_groups": {"group_a": 0.5, "group_b": 0.5},
                "geographic_regions": {"region_1": 0.3, "region_2": 0.7}
            }
        }
        
        model_decisions = [
            {
                "user_group": "group_a",
                "decision": "approved",
                "confidence": 0.8
            },
            {
                "user_group": "group_b", 
                "decision": "approved",
                "confidence": 0.82
            }
        ]
        
        # 执行测试
        result = await self.controller.assess_bias_risk(
            training_data, model_decisions, {}, {}
        )
        
        # 验证结果
        assert isinstance(result, BiasRiskAssessment)
        assert result.fairness_score > 0.7  # 公平性较高
        assert len(result.bias_alerts) == 0  # 无偏见警报
    
    @pytest.mark.asyncio
    async def test_assess_bias_risk_unfair(self):
        """测试不公平算法偏见评估"""
        # 准备不公平数据
        training_data = {
            "data_distribution": {
                "user_groups": {"group_a": 0.9, "group_b": 0.1},  # 数据不平衡
                "geographic_regions": {"region_1": 0.95, "region_2": 0.05}
            }
        }
        
        model_decisions = [
            {
                "user_group": "group_a",
                "decision": "approved",
                "confidence": 0.9  # 高置信度
            },
            {
                "user_group": "group_b",
                "decision": "rejected", 
                "confidence": 0.1  # 低置信度
            }
        ]
        
        # 执行测试
        result = await self.controller.assess_bias_risk(
            training_data, model_decisions, {}, {}
        )
        
        # 验证偏见检测
        assert result.fairness_score < 0.5  # 公平性较低
        assert len(result.bias_alerts) > 0  # 有偏见警报
        assert any("偏见" in alert.description for alert in result.bias_alerts)


class TestGovernanceConflictController:
    """治理冲突控制器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.controller = GovernanceConflictController()
    
    @pytest.mark.asyncio
    async def test_assess_governance_conflict_harmonious(self):
        """测试和谐的治理冲突评估"""
        # 准备和谐数据
        ai_decisions = [
            {
                "decision_type": "resource_allocation",
                "rationale": "基于公平原则",
                "community_feedback": "positive"
            }
        ]
        
        community_votes = [
            {
                "proposal_id": "prop_001",
                "ai_recommendation": "approve",
                "community_vote": "approve",
                "alignment_score": 0.95
            }
        ]
        
        # 执行测试
        result = await self.controller.assess_governance_conflict(
            ai_decisions, community_votes, {}, {}
        )
        
        # 验证结果
        assert isinstance(result, GovernanceConflictAssessment)
        assert result.collaboration_score > 0.8  # 协作良好
        assert len(result.conflict_alerts) == 0  # 无冲突警报
    
    @pytest.mark.asyncio
    async def test_assess_governance_conflict_severe(self):
        """测试严重治理冲突评估"""
        # 准备冲突数据
        ai_decisions = [
            {
                "decision_type": "fund_allocation",
                "rationale": "基于算法优化",
                "community_feedback": "negative"
            }
        ]
        
        community_votes = [
            {
                "proposal_id": "prop_002",
                "ai_recommendation": "reject",
                "community_vote": "approve",
                "alignment_score": 0.1  # 严重分歧
            }
        ]
        
        blockchain_governance = {
            "voting_participation": 0.2,  # 低参与度
            "decision_controversy": 0.9   # 高争议性
        }
        
        # 执行测试
        result = await self.controller.assess_governance_conflict(
            ai_decisions, community_votes, blockchain_governance, {}
        )
        
        # 验证冲突检测
        assert result.collaboration_score < 0.4  # 协作不良
        assert len(result.conflict_alerts) > 0  # 有冲突警报
        assert any("冲突" in alert.description for alert in result.conflict_alerts)


class TestRiskMonitoringSystem:
    """风险监控系统测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.monitor = AIRiskMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_perform_comprehensive_assessment(self):
        """测试综合风险评估"""
        # 执行综合评估
        result = await self.monitor.perform_comprehensive_assessment()
        
        # 验证结果
        assert isinstance(result, SystemRiskReport)
        assert 0 <= result.overall_risk_score <= 1.0
        assert isinstance(result.system_status, SystemStatus)
        assert isinstance(result.risk_breakdown, dict)
        assert isinstance(result.active_alerts, list)
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_system_status_transitions(self):
        """测试系统状态转换"""
        # 测试正常状态
        risk_scores = [0.1, 0.2, 0.25]
        for score in risk_scores:
            status = self.monitor._determine_system_status(score)
            assert status == SystemStatus.NORMAL
        
        # 测试警告状态
        risk_scores = [0.3, 0.4, 0.49]
        for score in risk_scores:
            status = self.monitor._determine_system_status(score)
            assert status == SystemStatus.WARNING
        
        # 测试警报状态
        risk_scores = [0.5, 0.6, 0.69]
        for score in risk_scores:
            status = self.monitor._determine_system_status(score)
            assert status == SystemStatus.ALERT
        
        # 测试严重状态
        risk_scores = [0.7, 0.8, 0.89]
        for score in risk_scores:
            status = self.monitor._determine_system_status(score)
            assert status == SystemStatus.CRITICAL
        
        # 测试紧急状态
        risk_scores = [0.9, 0.95, 1.0]
        for score in risk_scores:
            status = self.monitor._determine_system_status(score)
            assert status == SystemStatus.EMERGENCY
    
    @pytest.mark.asyncio
    async def test_emergency_response_triggering(self):
        """测试应急响应触发"""
        # 创建高风险警报
        critical_alert = RiskAlert(
            alert_id="TEST_CRITICAL",
            risk_category=RiskCategory.TECHNICAL,
            priority=AlertPriority.CRITICAL,
            description="测试严重风险",
            risk_score=0.95,
            affected_components=["AI决策引擎"],
            recommended_actions=["立即停止"],
            timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # 模拟高风险报告
        risk_report = SystemRiskReport(
            report_id="TEST_REPORT",
            timestamp=datetime.utcnow(),
            system_status=SystemStatus.EMERGENCY,
            overall_risk_score=0.95,
            risk_breakdown={RiskCategory.TECHNICAL: 0.95},
            active_alerts=[critical_alert],
            emergency_actions_taken=[],
            recommendations=["紧急处理"],
            next_assessment_time=datetime.utcnow() + timedelta(minutes=5)
        )
        
        # 触发应急响应
        await self.monitor._trigger_emergency_response([critical_alert], risk_report)
        
        # 验证应急响应记录
        assert len(self.monitor.emergency_responses) > 0
        response = self.monitor.emergency_responses[0]
        assert response.response_status in ["executing", "completed", "failed"]
        assert len(response.actions_taken) > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_integration(self):
        """测试监控循环集成"""
        # 启动监控
        await self.monitor.start_monitoring()
        
        # 等待监控循环执行一次
        await asyncio.sleep(1)
        
        # 验证系统状态已更新
        assert self.monitor.system_status is not None
        assert isinstance(self.monitor.active_alerts, list)
        
        # 停止监控
        await self.monitor.stop_monitoring()


class TestAPIIntegration:
    """API集成测试"""
    
    @pytest.mark.asyncio
    async def test_api_health_check(self):
        """测试API健康检查"""
        from fastapi.testclient import TestClient
        from .api import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_risk_assessment_api(self):
        """测试风险评估API"""
        from fastapi.testclient import TestClient
        from .api import app
        
        client = TestClient(app)
        
        # 准备测试数据
        request_data = {
            "ai_decisions": [
                {
                    "decision_id": "api_test_decision",
                    "action": "test_action",
                    "confidence": 0.8
                }
            ],
            "blockchain_data": {
                "block_height": 1000
            }
        }
        
        response = client.post("/assess", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "report_id" in data
        assert "overall_risk_score" in data
        assert "system_status" in data


# 性能测试
@pytest.mark.performance
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_assessment_performance(self):
        """测试风险评估性能"""
        import time
        
        controller = TechnicalRiskController()
        
        # 准备大量数据
        ai_decision_data = {
            "decisions": [
                {
                    "decision_id": f"decision_{i}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "test_action",
                    "confidence": 0.8,
                    "risk_score": 0.1 + (i % 10) * 0.01
                }
                for i in range(1000)  # 1000条决策数据
            ]
        }
        
        # 性能测试
        start_time = time.time()
        result = await controller.assess_technical_risk(ai_decision_data, {}, {})
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 验证性能要求（1秒内完成）
        assert execution_time < 1.0
        assert isinstance(result, TechnicalRiskAssessment)


# 主测试运行
if __name__ == "__main__":
    pytest.main([__file__, "-v"])