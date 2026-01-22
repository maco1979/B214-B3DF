"""
迁移学习集成测试
"""

import pytest
import asyncio
import sys
import os

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.migration_learning.risk_control import MigrationRiskController
from src.migration_learning.data_validation import DataCredibilityValidator
from src.migration_learning.rule_constraints import AgriculturalRuleValidator
from src.migration_learning.warning_system import RiskWarningSystem
from src.integration.migration_integration import MigrationIntegrationManager


class TestMigrationIntegration:
    """迁移学习集成测试类"""
    
    @pytest.fixture
    def mock_decision_engine(self):
        """模拟决策引擎"""
        engine = Mock()
        engine.make_decision = AsyncMock(return_value={"decision": "test_decision", "confidence": 0.9})
        return engine
    
    @pytest.fixture
    def migration_manager(self, mock_decision_engine):
        """迁移学习集成管理器"""
        return MigrationIntegrationManager(mock_decision_engine)
    
    @pytest.mark.asyncio
    async def test_integrate_migration_learning_success(self, migration_manager):
        """测试迁移学习集成成功场景"""
        
        # 准备测试数据
        test_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "rice", "soil_moisture": 0.6},
            "source_domain": "agriculture_general",
            "target_domain": "rice_cultivation"
        }
        
        # 执行集成
        result = await migration_manager.integrate_migration_learning(test_data)
        
        # 验证结果
        assert "migration_applied" in result
        assert "risk_assessment" in result
        assert "decision_result" in result
        assert result["integration_status"] == "success"
    
    @pytest.mark.asyncio
    async def test_integrate_migration_learning_high_risk(self, migration_manager):
        """测试高风险迁移学习场景"""
        
        # 准备高风险测试数据
        test_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "unknown_crop", "soil_moisture": 2.0},  # 异常数据
            "source_domain": "completely_different",
            "target_domain": "agriculture"
        }
        
        # 执行集成
        result = await migration_manager.integrate_migration_learning(test_data)
        
        # 验证高风险处理
        assert result["risk_assessment"]["overall_risk_level"] == "high"
        assert result["migration_applied"] is False  # 高风险应禁用迁移学习
    
    @pytest.mark.asyncio
    async def test_data_validation_integration(self, migration_manager):
        """测试数据验证集成"""
        
        # 准备无效数据
        invalid_data = {
            "task_type": "",  # 空任务类型
            "data": {},
            "source_domain": "agriculture"
        }
        
        # 执行集成（应抛出异常）
        with pytest.raises(ValueError):
            await migration_manager.integrate_migration_learning(invalid_data)
    
    @pytest.mark.asyncio
    async def test_rule_constraints_integration(self, migration_manager):
        """测试规则约束集成"""
        
        # 准备违反约束的数据
        constraint_violation_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "rice"},
            "source_domain": "agriculture",
            "target_domain": "agriculture",  # 同领域迁移，可能违反约束
            "data_privacy_level": "confidential"  # 高隐私级别
        }
        
        # 执行集成
        result = await migration_manager.integrate_migration_learning(constraint_violation_data)
        
        # 验证约束处理
        constraints_check = result["constraints_check"]
        assert "constraints_satisfied" in constraints_check
    
    @pytest.mark.asyncio
    async def test_warning_system_integration(self, migration_manager):
        """测试预警系统集成"""
        
        # 准备触发警告的数据
        warning_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "rice", "temperature": 50},  # 异常高温
            "source_domain": "agriculture",
            "target_domain": "rice_cultivation"
        }
        
        # 执行集成
        result = await migration_manager.integrate_migration_learning(warning_data)
        
        # 验证预警信息
        assert "warnings_triggered" in result
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, migration_manager):
        """测试性能监控"""
        
        test_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "wheat", "soil_ph": 6.5},
            "source_domain": "agriculture",
            "target_domain": "wheat_cultivation"
        }
        
        # 执行多次以测试性能
        start_time = datetime.now()
        
        for i in range(5):
            result = await migration_manager.integrate_migration_learning(test_data)
            assert result["integration_status"] == "success"
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 验证性能（平均响应时间应小于1秒）
        avg_duration = duration / 5
        assert avg_duration < 1.0, f"平均响应时间 {avg_duration}s 超过阈值"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, migration_manager, mock_decision_engine):
        """测试错误处理"""
        
        # 模拟决策引擎失败
        mock_decision_engine.make_decision.side_effect = Exception("Decision engine failure")
        
        test_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "corn"},
            "source_domain": "agriculture",
            "target_domain": "corn_cultivation"
        }
        
        # 执行集成（应正确处理错误）
        result = await migration_manager.integrate_migration_learning(test_data)
        
        # 验证错误处理
        assert result["integration_status"] == "error"
        assert "fallback_decision" in result
    
    @pytest.mark.asyncio
    async def test_configuration_management(self, migration_manager):
        """测试配置管理"""
        
        # 更新配置
        new_config = {
            "risk_threshold": 0.7,
            "enable_advanced_validation": True,
            "max_data_size": 10485760  # 10MB
        }
        
        result = await migration_manager.update_config(new_config)
        
        # 验证配置更新
        assert result["status"] == "success"
        assert result["updated_config"] == new_config
        
        # 验证配置生效
        test_data = {
            "task_type": "test",
            "data": {"test": "data"},
            "source_domain": "test",
            "target_domain": "test"
        }
        
        integration_result = await migration_manager.integrate_migration_learning(test_data)
        assert integration_result["integration_status"] == "success"


class TestMigrationComponentsIntegration:
    """迁移学习组件集成测试"""
    
    @pytest.mark.asyncio
    async def test_risk_control_component(self):
        """测试风险控制组件"""
        
        risk_manager = MigrationRiskController()
        
        # 测试风险评估
        risk_data = {
            "task_type": "agriculture_decision",
            "data_size": 1024,
            "source_domain": "agriculture_general",
            "target_domain": "specific_crop"
        }
        
        assessment = await risk_manager.assess_migration_risk(risk_data)
        
        assert "risk_level" in assessment
        assert "confidence" in assessment
        assert "recommendations" in assessment
    
    @pytest.mark.asyncio
    async def test_data_validation_component(self):
        """测试数据验证组件"""
        
        validation_manager = DataCredibilityValidator()
        
        # 测试有效数据
        valid_data = {
            "task_type": "agriculture_decision",
            "data": {"crop_type": "rice", "temperature": 25},
            "source_domain": "agriculture"
        }
        
        validation_result = await validation_manager.validate_decision_data(valid_data)
        assert validation_result["valid"] is True
        
        # 测试无效数据
        invalid_data = {
            "task_type": "",
            "data": {},
            "source_domain": "agriculture"
        }
        
        validation_result = await validation_manager.validate_decision_data(invalid_data)
        assert validation_result["valid"] is False
        assert "errors" in validation_result
    
    @pytest.mark.asyncio
    async def test_rule_constraints_component(self):
        """测试规则约束组件"""
        
        constraint_manager = AgriculturalRuleValidator()
        
        # 测试约束检查
        constraint_data = {
            "task_type": "agriculture_decision",
            "data_privacy_level": "public",
            "data_retention": "30d"
        }
        
        constraint_result = await constraint_manager.check_migration_constraints(constraint_data)
        
        assert "satisfied" in constraint_result
        assert "violations" in constraint_result
    
    @pytest.mark.asyncio
    async def test_warning_system_component(self):
        """测试预警系统组件"""
        
        warning_system = RiskWarningSystem()
        
        # 测试预警触发
        warning_data = {
            "warning_type": "high_risk_detected",
            "risk_level": "high",
            "context": {"task_type": "agriculture_decision"}
        }
        
        warning_result = await warning_system.trigger_warning(
            "high_risk_detected",
            warning_data,
            {"risk_score": 0.9}
        )
        
        assert "warning_id" in warning_result
        assert "triggered" in warning_result
        
        # 测试获取活跃警告
        active_warnings = await warning_system.get_active_warnings()
        assert isinstance(active_warnings, list)