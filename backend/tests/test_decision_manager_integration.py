"""
DecisionManager集成测试
测试决策管理器的任务类型和风险级别验证功能
"""

import pytest
import sys
import os

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from unittest.mock import Mock, AsyncMock
from src.core.decision.agriculture_decision_engine import AgricultureDecisionEngine, AgricultureState, DecisionObjective
from src.integration.decision_integration import DecisionIntegrationManager
from src.core.decision.decision_manager import DecisionManager


class TestDecisionManagerIntegration:
    """决策管理器集成测试类"""
    
    @pytest.fixture
    def mock_agriculture_engine(self):
        """模拟农业决策引擎"""
        engine = Mock(spec=AgricultureDecisionEngine)
        
        # 创建一个模拟决策结果
        mock_result = Mock(
            action=Mock(value="adjust_temperature"),
            parameters={"temperature_adjustment": 0.5},
            expected_reward=0.85,
            confidence=0.92,
            execution_time=0.123
        )
        
        # 创建一个可调用对象，根据调用上下文返回正确的结果
        def make_decision_mock(*args, **kwargs):
            # 检查是否在异步上下文中调用
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_code = caller_frame.f_code
            caller_func = caller_code.co_name
            
            # 如果是在_handle_integration_failure中调用（失败处理流程）
            # 或者是在异步上下文中调用，返回异步结果
            if caller_func == "_handle_integration_failure" or "async" in caller_code.co_names:
                async def async_result():
                    return mock_result
                return async_result()
            # 否则返回同步结果
            return mock_result
        
        # 设置模拟方法
        engine.make_decision = make_decision_mock
        return engine
    
    @pytest.fixture
    def decision_integration_manager(self, mock_agriculture_engine):
        """决策集成管理器"""
        return DecisionIntegrationManager(mock_agriculture_engine)
    
    @pytest.fixture
    def decision_manager(self):
        """决策管理器"""
        return DecisionManager()
    
    @pytest.mark.asyncio
    async def test_routine_monitoring_with_low_risk(self, decision_integration_manager):
        """测试常规监控任务使用低风险级别"""
        
        # 准备符合条件的请求
        decision_request = {
            "temperature": 25.0,
            "humidity": 60.0,
            "co2_level": 400.0,
            "light_intensity": 1000.0,
            "spectrum_config": {"red": 0.4, "blue": 0.3, "far_red": 0.2},
            "crop_type": "tomato",
            "growth_day": 30,
            "growth_rate": 0.8,
            "health_score": 0.9,
            "yield_potential": 0.85,
            "energy_consumption": 50.0,
            "resource_utilization": 0.7,
            "objective": "maximize_yield",
            "task_type": "routine_monitoring",
            "risk_level": "low",
            "data": {"temperature": 25.0, "humidity": 60.0, "co2_level": 400.0}
        }
        
        # 执行决策
        result = await decision_integration_manager.integrated_decision_making(decision_request)
        
        # 验证结果
        assert "action" in result
        assert "parameters" in result
        assert "risk_assessment" in result
        assert result["risk_assessment"]["overall_risk_level"] == "low"
    
    @pytest.mark.asyncio
    async def test_routine_monitoring_with_invalid_risk(self, decision_integration_manager):
        """测试常规监控任务使用非低风险级别（应该失败）"""
        
        # 准备不符合条件的请求（routine_monitoring使用medium风险）
        decision_request = {
            "temperature": 25.0,
            "humidity": 60.0,
            "co2_level": 400.0,
            "light_intensity": 1000.0,
            "spectrum_config": {"red": 0.4, "blue": 0.3, "far_red": 0.2},
            "crop_type": "tomato",
            "growth_day": 30,
            "growth_rate": 0.8,
            "health_score": 0.9,
            "yield_potential": 0.85,
            "energy_consumption": 50.0,
            "resource_utilization": 0.7,
            "objective": "maximize_yield",
            "task_type": "routine_monitoring",
            "risk_level": "medium",
            "data": {"temperature": 25.0, "humidity": 60.0, "co2_level": 400.0}
        }
        
        # 应该抛出ValueError
        with pytest.raises(ValueError) as excinfo:
            await decision_integration_manager.integrated_decision_making(decision_request)
        
        # 验证错误信息
        assert "任务类型为 'routine_monitoring' 时，风险级别必须为 'low'" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_high_priority_with_any_risk(self, decision_integration_manager):
        """测试高优先级任务可以使用任意风险级别"""
        
        # 测试高优先级任务使用高风险级别
        decision_request = {
            "temperature": 25.0,
            "humidity": 60.0,
            "co2_level": 400.0,
            "light_intensity": 1000.0,
            "spectrum_config": {"red": 0.4, "blue": 0.3, "far_red": 0.2},
            "crop_type": "tomato",
            "growth_day": 30,
            "growth_rate": 0.8,
            "health_score": 0.9,
            "yield_potential": 0.85,
            "energy_consumption": 50.0,
            "resource_utilization": 0.7,
            "objective": "maximize_yield",
            "task_type": "high_priority",
            "risk_level": "high",
            "data": {"temperature": 25.0, "humidity": 60.0, "co2_level": 400.0}
        }
        
        # 执行决策
        result = await decision_integration_manager.integrated_decision_making(decision_request)
        
        # 验证结果
        assert "action" in result
        assert "parameters" in result
        assert result["risk_assessment"]["overall_risk_level"] == "high"
    
    @pytest.mark.asyncio
    async def test_missing_task_type(self, decision_integration_manager):
        """测试缺少task_type的请求（应该失败）"""
        
        # 准备缺少task_type的请求
        decision_request = {
            "temperature": 25.0,
            "humidity": 60.0,
            "co2_level": 400.0,
            "light_intensity": 1000.0,
            "spectrum_config": {"red": 0.4, "blue": 0.3, "far_red": 0.2},
            "crop_type": "tomato",
            "growth_day": 30,
            "growth_rate": 0.8,
            "health_score": 0.9,
            "yield_potential": 0.85,
            "energy_consumption": 50.0,
            "resource_utilization": 0.7,
            "objective": "maximize_yield",
            # 缺少task_type
            "risk_level": "medium",
            "data": {"temperature": 25.0, "humidity": 60.0, "co2_level": 400.0}
        }
        
        # 应该抛出ValueError
        with pytest.raises(ValueError) as excinfo:
            await decision_integration_manager.integrated_decision_making(decision_request)
        
        # 验证错误信息
        assert "缺少必要字段: task_type" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_unknown_task_type(self, decision_manager):
        """测试未知任务类型（应该失败）"""
        
        # 准备未知任务类型的请求
        decision_request = {
            "temperature": 25.0,
            "humidity": 60.0,
            "task_type": "unknown_task",
            "risk_level": "medium",
            "data": {"temperature": 25.0, "humidity": 60.0}
        }
        
        # 应该抛出ValueError
        with pytest.raises(ValueError) as excinfo:
            await decision_manager.integrated_decision_making(decision_request)
        
        # 验证错误信息
        assert "未知的任务类型: unknown_task" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_default_risk_level(self, decision_manager):
        """测试默认风险级别"""
        
        # 准备缺少risk_level的请求
        decision_request = {
            "temperature": 25.0,
            "humidity": 60.0,
            "task_type": "high_priority",
            # 缺少risk_level
            "data": {"temperature": 25.0, "humidity": 60.0}
        }
        
        # 执行决策管理器验证
        result = await decision_manager.integrated_decision_making(decision_request)
        
        # 验证默认风险级别为medium
        assert result["risk"] == "medium"
    
    @pytest.mark.asyncio
    async def test_critical_decision_with_high_risk(self, decision_integration_manager):
        """测试关键决策任务"""
        
        # 准备关键决策请求
        decision_request = {
            "temperature": 35.0,  # 高温
            "humidity": 80.0,    # 高湿度
            "co2_level": 600.0,
            "light_intensity": 1500.0,
            "spectrum_config": {"red": 0.4, "blue": 0.3, "far_red": 0.2},
            "crop_type": "tomato",
            "growth_day": 45,
            "growth_rate": 0.7,
            "health_score": 0.75,
            "yield_potential": 0.78,
            "energy_consumption": 75.0,
            "resource_utilization": 0.9,
            "objective": "enhance_resistance",
            "task_type": "critical_decision",
            "risk_level": "high",
            "data": {"temperature": 35.0, "humidity": 80.0, "co2_level": 600.0}
        }
        
        # 执行决策
        result = await decision_integration_manager.integrated_decision_making(decision_request)
        
        # 验证结果
        assert "action" in result
        assert "parameters" in result
        assert result["risk_assessment"]["overall_risk_level"] == "high"
        assert result["processing_mode"]["risk_control"] == "strict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
