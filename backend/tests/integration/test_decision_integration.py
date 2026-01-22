"""
决策引擎集成测试
"""

import pytest
import asyncio
import sys
import os

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.core.decision_engine import DecisionEngine
from src.integration.decision_integration import DecisionIntegrationManager
from src.integration.migration_integration import MigrationLearningIntegration as MigrationIntegrationManager
from src.integration.edge_integration import EdgeIntegrationManager


class TestDecisionIntegration:
    """决策引擎集成测试类"""
    
    @pytest.fixture
    def mock_decision_engine(self):
        """模拟决策引擎"""
        engine = Mock()
        engine.make_decision = AsyncMock(return_value={
            "decision": "integrated_decision",
            "confidence": 0.92,
            "reasoning": "基于多源数据综合分析"
        })
        # 添加异步方法模拟
        engine.enable_conservative_mode = AsyncMock(return_value={
            "status": "conservative_mode_enabled"
        })
        engine.update_migration_parameters = AsyncMock(return_value={
            "status": "parameters_updated"
        })
        return engine
    
    @pytest.fixture
    def decision_manager(self, mock_decision_engine):
        """决策集成管理器"""
        return DecisionIntegrationManager(mock_decision_engine)
    
    @pytest.mark.asyncio
    async def test_integrated_decision_making_success(self, decision_manager):
        """测试集成化决策成功场景"""
        
        # 准备标准决策请求
        decision_request = {
            "task_type": "agriculture_optimization",
            "data": {
                "crop_type": "rice",
                "soil_moisture": 0.65,
                "temperature": 28.5,
                "humidity": 0.75
            },
            "requirements": {
                "accuracy": 0.85,
                "latency": 1000
            },
            "constraints": {
                "privacy_level": "medium",
                "data_retention": "30d"
            }
        }
        
        # 执行集成化决策
        result = await decision_manager.integrated_decision_making(decision_request)
        
        # 验证结果结构
        assert "decision" in result
        assert "confidence" in result
        assert "risk_assessment" in result
        assert "processing_mode" in result
        assert "integration_timestamp" in result
        
        # 验证集成功能
        assert result["risk_assessment"]["overall_risk_level"] in ["low", "medium", "high"]
        assert result["processing_mode"]["risk_control"] in ["standard", "enhanced", "strict"]
    
    @pytest.mark.asyncio
    async def test_risk_based_processing_mode_selection(self, decision_manager):
        """测试基于风险的处理模式选择"""
        
        # 测试低风险场景
        low_risk_request = {
            "task_type": "routine_monitoring",
            "data": {"metric": "temperature", "value": 25.0}
        }
        
        low_risk_result = await decision_manager.integrated_decision_making(low_risk_request)
        low_risk_mode = low_risk_result["processing_mode"]
        
        # 调试信息
        print(f"低风险请求: {low_risk_request}")
        print(f"处理模式: {low_risk_mode}")
        print(f"完整结果: {low_risk_result}")
        
        # 低风险应启用迁移学习和边缘计算
        assert low_risk_mode["migration_learning"] is True
        assert low_risk_mode["edge_computing"] is True
        assert low_risk_mode["risk_control"] == "standard"
        
        # 测试高风险场景
        high_risk_request = {
            "task_type": "critical_decision",
            "data": {
                "sensor_readings": [100, 200, 300],  # 异常值
                "system_status": "degraded"
            }
        }
        
        high_risk_result = await decision_manager.integrated_decision_making(high_risk_request)
        high_risk_mode = high_risk_result["processing_mode"]
        
        # 高风险应禁用高级功能，使用严格风险控制
        assert high_risk_mode["migration_learning"] is False
        assert high_risk_mode["edge_computing"] is False
        assert high_risk_mode["risk_control"] == "strict"
    
    @pytest.mark.asyncio
    async def test_data_validation_integration(self, decision_manager):
        """测试数据验证集成"""
        
        # 测试无效数据
        invalid_request = {
            "task_type": "",  # 空任务类型
            "data": {},
            "requirements": {}
        }
        
        # 应返回包含集成失败标记的结果
        result = await decision_manager.integrated_decision_making(invalid_request)
        assert result.get("integration_failed") is True
        assert "字段 task_type 的值不能为空" in result.get("fallback_reason", "")
    
    @pytest.mark.asyncio
    async def test_constraint_violation_handling(self, decision_manager):
        """测试约束违反处理"""
        
        # 准备违反约束的请求
        constraint_violation_request = {
            "task_type": "sensitive_analysis",
            "data": {
                "personal_info": "confidential",
                "analysis_depth": "deep"
            },
            "constraints": {
                "privacy_level": "public"  # 冲突的隐私级别
            }
        }
        
        result = await decision_manager.integrated_decision_making(constraint_violation_request)
        
        # 验证约束违反处理
        processing_mode = result["processing_mode"]
        assert processing_mode.get("constraint_violation") is True
        assert processing_mode["risk_control"] == "strict"
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self, decision_manager, mock_decision_engine):
        """测试组件失败处理"""
        
        # 模拟迁移学习组件失败
        with patch.object(MigrationIntegrationManager, 'integrate_migration_learning') as mock_migration:
            mock_migration.side_effect = Exception("Migration learning failed")
            
            # 准备正常请求
            normal_request = {
                "task_type": "agriculture_decision",
                "data": {"crop": "wheat"}
            }
            
            # 执行决策（应正确处理组件失败）
            result = await decision_manager.integrated_decision_making(normal_request)
            
            # 验证系统继续运行
            assert "decision" in result
            assert "integration_failed" not in result  # 不应标记为集成失败
    
    @pytest.mark.asyncio
    async def test_parallel_component_execution(self, decision_manager):
        """测试并行组件执行"""
        
        # 准备并发测试数据
        test_request = {
            "task_type": "complex_analysis",
            "data": {"parameters": [1, 2, 3, 4, 5]},
            "requirements": {"parallel_processing": True}
        }
        
        # 执行集成决策
        start_time = datetime.now()
        result = await decision_manager.integrated_decision_making(test_request)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # 验证并行执行（总时间应小于各组件串行时间之和）
        # 假设各组件平均执行时间200ms，串行需要600ms，并行应显著更快
        assert duration < 0.6, f"执行时间 {duration}s 可能未并行执行"
        
        # 验证结果完整性
        assert "decision" in result
        assert "risk_assessment" in result
    
    @pytest.mark.asyncio
    async def test_decision_result_integration(self, decision_manager):
        """测试决策结果整合"""
        
        test_request = {
            "task_type": "multi_source_decision",
            "data": {
                "sensor_data": [25.5, 26.0, 24.8],
                "weather_forecast": {"temp": 26, "humidity": 0.7},
                "historical_patterns": ["pattern_a", "pattern_b"]
            }
        }
        
        result = await decision_manager.integrated_decision_making(test_request)
        
        # 验证结果整合
        assert "decision" in result
        assert "confidence" in result
        
        # 验证迁移学习增强（如果应用）
        if result.get("migration_enhanced"):
            assert "enhancement_factor" in result
            assert result["enhancement_factor"] > 1.0
        
        # 验证边缘处理信息（如果应用）
        if result.get("edge_processing_info"):
            edge_info = result["edge_processing_info"]
            assert "selected_node" in edge_info
            assert "processing_mode" in edge_info
    
    @pytest.mark.asyncio
    async def test_post_processing_and_monitoring(self, decision_manager):
        """测试后处理和监控"""
        
        test_request = {
            "task_type": "monitored_decision",
            "data": {"value": 42, "threshold": 50}
        }
        
        result = await decision_manager.integrated_decision_making(test_request)
        
        # 验证后处理
        assert "monitoring_active" in result
        assert result["monitoring_active"] is True
        assert "monitoring_task_id" in result
        
        # 验证风险后处理
        assert "risk_control" in result
        risk_control_info = result["risk_control"]
        assert "post_processed" in risk_control_info
    
    @pytest.mark.asyncio
    async def test_integration_status_management(self, decision_manager):
        """测试集成状态管理"""
        
        # 获取集成状态
        status = await decision_manager.get_integration_status()
        
        # 验证状态结构
        assert "migration_learning_enabled" in status
        assert "edge_computing_enabled" in status
        assert "risk_control_enabled" in status
        assert "last_update" in status
        
        # 验证组件状态
        assert "migration_learning_status" in status
        assert "edge_computing_status" in status
        assert "risk_control_status" in status
        assert "warning_system_status" in status
    
    @pytest.mark.asyncio
    async def test_configuration_management(self, decision_manager):
        """测试配置管理"""
        
        # 准备配置更新
        config_updates = {
            "migration_learning": {
                "risk_threshold": 0.7,
                "enable_advanced_features": True
            },
            "risk_control": {
                "strict_mode_threshold": 0.8,
                "monitoring_duration": 600  # 10分钟
            }
        }
        
        # 更新配置
        update_result = await decision_manager.update_integration_config(config_updates)
        
        # 验证配置更新
        assert update_result["status"] == "success"
        assert "updated_config" in update_result
        
        # 验证配置生效
        status = await decision_manager.get_integration_status()
        assert status["last_config_update"] is not None


class TestDecisionIntegrationEdgeCases:
    """决策集成边界情况测试"""
    
    @pytest.mark.asyncio
    async def test_empty_request_handling(self, decision_manager):
        """测试空请求处理"""
        
        empty_request = {}
        
        # 应正确处理空请求
        with pytest.raises(ValueError):
            await decision_manager.integrated_decision_making(empty_request)
    
    @pytest.mark.asyncio
    async def test_large_data_handling(self, decision_manager):
        """测试大数据处理"""
        
        # 准备大数据请求
        large_data_request = {
            "task_type": "big_data_analysis",
            "data": {
                "large_dataset": list(range(10000)),  # 大数据集
                "metadata": {"size": "large", "compression": "enabled"}
            }
        }
        
        # 执行决策
        result = await decision_manager.integrated_decision_making(large_data_request)
        
        # 验证大数据处理
        assert "decision" in result
        
        # 验证可能的数据压缩或分块处理
        processing_mode = result["processing_mode"]
        if processing_mode.get("edge_computing") is False:
            # 大数据可能不适合边缘计算
            assert "data_size" in result.get("edge_processing_info", {}).get("reason", "")
    
    @pytest.mark.asyncio
    async def test_high_concurrency_scenario(self, decision_manager):
        """测试高并发场景"""
        
        # 准备多个并发请求
        concurrent_requests = [
            {
                "task_type": f"concurrent_task_{i}",
                "data": {"value": i, "timestamp": datetime.now().timestamp()}
            } for i in range(10)
        ]
        
        # 并发执行
        tasks = [
            decision_manager.integrated_decision_making(req)
            for req in concurrent_requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有请求处理成功
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(concurrent_requests)
        
        # 验证每个结果的结构
        for result in successful_results:
            assert "decision" in result
            assert "integration_timestamp" in result
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_failure(self, decision_manager, mock_decision_engine):
        """测试系统故障恢复"""
        
        # 模拟系统故障
        mock_decision_engine.make_decision.side_effect = Exception("System failure")
        
        test_request = {
            "task_type": "recovery_test",
            "data": {"test": "data"}
        }
        
        # 第一次执行应失败
        with pytest.raises(Exception):
            await decision_manager.integrated_decision_making(test_request)
        
        # 恢复系统
        mock_decision_engine.make_decision.side_effect = None
        mock_decision_engine.make_decision.return_value = {"decision": "recovered"}
        
        # 第二次执行应成功
        result = await decision_manager.integrated_decision_making(test_request)
        assert result["decision"] == "recovered"


class TestDecisionIntegrationPerformance:
    """决策集成性能测试"""
    
    @pytest.mark.asyncio
    async def test_decision_latency_benchmark(self, decision_manager):
        """测试决策延迟基准"""
        
        test_request = {
            "task_type": "performance_test",
            "data": {"benchmark": True}
        }
        
        # 测试多次执行的平均延迟
        execution_times = []
        
        for i in range(20):
            start_time = datetime.now()
            result = await decision_manager.integrated_decision_making(test_request)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            execution_times.append(duration)
            
            # 验证每次执行成功
            assert "decision" in result
        
        # 计算统计信息
        avg_latency = sum(execution_times) / len(execution_times)
        max_latency = max(execution_times)
        
        # 验证性能要求（平均延迟 < 500ms，最大延迟 < 1s）
        assert avg_latency < 0.5, f"平均延迟 {avg_latency}s 超过阈值"
        assert max_latency < 1.0, f"最大延迟 {max_latency}s 超过阈值"
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, decision_manager):
        """测试内存使用监控"""
        
        # 准备内存密集型请求
        memory_intensive_request = {
            "task_type": "memory_intensive",
            "data": {
                "large_array": list(range(1000)),
                "nested_objects": [{"data": "x" * 100} for _ in range(100)]
            }
        }
        
        # 执行多次以检测内存泄漏
        for i in range(5):
            result = await decision_manager.integrated_decision_making(memory_intensive_request)
            assert "decision" in result
        
        # 这里可以添加实际的内存监控逻辑
        # 目前主要是验证功能正常，不出现内存错误
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, decision_manager):
        """测试吞吐量测量"""
        
        # 准备批量请求
        batch_requests = [
            {
                "task_type": f"throughput_test_{i}",
                "data": {"index": i, "data": "test"}
            } for i in range(50)
        ]
        
        # 测量批量处理时间
        start_time = datetime.now()
        
        tasks = [
            decision_manager.integrated_decision_making(req)
            for req in batch_requests
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()
        
        total_duration = (end_time - start_time).total_seconds()
        throughput = len(batch_requests) / total_duration  # 请求/秒
        
        # 验证吞吐量要求（> 10 请求/秒）
        assert throughput > 10, f"吞吐量 {throughput} 请求/秒 低于阈值"
        
        # 验证所有请求成功处理
        assert len(results) == len(batch_requests)
        for result in results:
            try:
                assert "decision" in result
            except Exception:
                raise