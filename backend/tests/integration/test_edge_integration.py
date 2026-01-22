"""
边缘计算集成测试
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.edge_computing.deployment_strategy import EdgeDeploymentStrategy
from src.edge_computing.model_lightweight import ModelLightweightProcessor as ModelLightweightManager
from src.edge_computing.resource_optimizer import EdgeResourceOptimizer
from src.edge_computing.cloud_edge_sync import CloudEdgeSyncManager
from src.integration.edge_integration import EdgeIntegrationManager


class TestEdgeIntegration:
    """边缘计算集成测试类"""
    
    @pytest.fixture
    def mock_decision_engine(self):
        """模拟决策引擎"""
        engine = Mock()
        engine.make_decision = AsyncMock(return_value={"decision": "edge_decision", "confidence": 0.85})
        return engine
    
    @pytest.fixture
    def edge_manager(self, mock_decision_engine):
        """边缘计算集成管理器"""
        return EdgeIntegrationManager(mock_decision_engine)
    
    @pytest.mark.asyncio
    async def test_integrate_edge_computing_success(self, edge_manager):
        """测试边缘计算集成成功场景"""
        
        # 准备适合边缘计算的测试数据
        test_data = {
            "task_type": "real_time_analysis",
            "data_size": 1024 * 1024,  # 1MB
            "latency_requirement": 500,  # 500ms
            "privacy_requirement": "high",
            "compute_requirement": "medium"
        }
        
        # 执行集成
        result = await edge_manager.integrate_edge_computing(test_data)
        
        # 验证结果
        assert "edge_enabled" in result
        assert "selected_node" in result
        assert "processing_mode" in result
        assert result["edge_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_edge_suitability_analysis(self, edge_manager):
        """测试边缘计算适用性分析"""
        
        # 测试适合边缘计算的场景
        suitable_data = {
            "task_type": "real_time_analysis",
            "data_size": 5 * 1024 * 1024,  # 5MB
            "latency_requirement": 200,
            "privacy_requirement": "high"
        }
        
        suitability = await edge_manager._analyze_edge_suitability(suitable_data)
        assert suitability["suitable"] is True
        
        # 测试不适合边缘计算的场景（数据过大）
        unsuitable_data = {
            "task_type": "real_time_analysis", 
            "data_size": 20 * 1024 * 1024,  # 20MB
            "latency_requirement": 200,
            "privacy_requirement": "high"
        }
        
        suitability = await edge_manager._analyze_edge_suitability(unsuitable_data)
        assert suitability["suitable"] is False
        assert "数据大小" in suitability["reason"]
    
    @pytest.mark.asyncio
    async def test_edge_node_selection(self, edge_manager):
        """测试边缘节点选择"""
        
        test_data = {
            "task_type": "local_processing",
            "compute_requirement": "high",
            "location_preference": "region_a"
        }
        
        # 测试节点选择
        selected_node = await edge_manager._select_edge_node(test_data)
        
        assert selected_node is not None
        assert "node_id" in selected_node
        assert "compute_capacity" in selected_node
        assert "memory_available" in selected_node
    
    @pytest.mark.asyncio
    async def test_model_lightweight_processing(self, edge_manager):
        """测试模型轻量化处理"""
        
        test_data = {
            "model_type": "agriculture_model",
            "accuracy_requirement": 0.8
        }
        
        # 测试轻量化模型准备
        lightweight_model = await edge_manager._prepare_lightweight_model(test_data)
        
        assert lightweight_model is not None
        assert "compressed_size_mb" in lightweight_model
        assert "accuracy" in lightweight_model
        assert "compression_ratio" in lightweight_model
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self, edge_manager):
        """测试资源优化分配"""
        
        # 模拟节点数据
        selected_node = {
            "node_id": "edge_node_1",
            "compute_capacity": "high",
            "memory_available": 8192,
            "network_latency": 50
        }
        
        # 模拟轻量化模型
        lightweight_model = {
            "model_size": 256,
            "memory_usage": 512,
            "compute_requirements": "medium"
        }
        
        test_data = {
            "compute_intensity": "high",
            "network_bandwidth": 100
        }
        
        # 测试资源分配
        allocation = await edge_manager._optimize_resource_allocation(
            test_data, selected_node, lightweight_model
        )
        
        assert allocation is not None
        assert "cpu_allocation" in allocation
        assert "memory_allocation" in allocation
        assert "network_allocation" in allocation
    
    @pytest.mark.asyncio
    async def test_edge_deployment(self, edge_manager):
        """测试边缘部署"""
        
        # 准备部署数据
        test_data = {
            "task_type": "real_time_analysis",
            "parameters": {"analysis_depth": "high"}
        }
        
        selected_node = {
            "node_id": "edge_node_1",
            "compute_capacity": "high"
        }
        
        lightweight_model = {
            "model_config": {"version": "1.0"},
            "deployment_requirements": {"runtime": "python3.8"}
        }
        
        resource_allocation = {
            "cpu_allocation": 2,
            "memory_allocation": 2048
        }
        
        # 测试部署
        deployment_result = await edge_manager._deploy_to_edge(
            test_data, selected_node, lightweight_model, resource_allocation
        )
        
        assert deployment_result is not None
        assert "deployment_id" in deployment_result
        assert "status" in deployment_result
    
    @pytest.mark.asyncio
    async def test_cloud_edge_synchronization(self, edge_manager):
        """测试云边协同"""
        
        deployment_result = {
            "deployment_id": "test_deployment_123",
            "sync_mode": "real_time",
            "data_retention": "7d"
        }
        
        # 测试同步启动
        sync_result = await edge_manager._start_cloud_edge_sync(deployment_result)
        
        assert sync_result is not None
        assert "sync_id" in sync_result
        assert "status" in sync_result
    
    @pytest.mark.asyncio
    async def test_edge_performance_monitoring(self, edge_manager):
        """测试边缘性能监控"""
        
        # 测试性能监控
        performance_data = await edge_manager.monitor_edge_performance("test_deployment_123")
        
        assert performance_data is not None
        assert "latency" in performance_data
        assert "throughput" in performance_data
        assert "resource_utilization" in performance_data
    
    @pytest.mark.asyncio
    async def test_edge_fallback_scenario(self, edge_manager):
        """测试边缘计算回退场景"""
        
        # 准备不适合边缘计算的数据
        fallback_data = {
            "task_type": "batch_processing",  # 不适合实时处理
            "data_size": 50 * 1024 * 1024,   # 50MB，超过限制
            "latency_requirement": 5000       # 5秒，要求不高
        }
        
        # 执行集成（应回退到云端）
        result = await edge_manager.integrate_edge_computing(fallback_data)
        
        # 验证回退处理
        assert result["edge_enabled"] is False
        assert result["processing_mode"] == "cloud"
        assert "reason" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_in_edge_integration(self, edge_manager):
        """测试边缘集成错误处理"""
        
        # 准备异常数据
        error_data = {
            "task_type": "real_time_analysis",
            "data_size": -100,  # 无效数据大小
            "latency_requirement": 200
        }
        
        # 执行集成（应正确处理错误）
        result = await edge_manager.integrate_edge_computing(error_data)
        
        # 验证错误处理
        assert "error" in result
        assert result["edge_enabled"] is False
        assert result["processing_mode"] == "cloud"


class TestEdgeComponentsIntegration:
    """边缘计算组件集成测试"""
    
    @pytest.mark.asyncio
    async def test_deployment_strategy_component(self):
        """测试部署策略组件"""
        
        deployment_strategy = EdgeDeploymentStrategy()
        
        # 测试节点选择
        available_nodes = [
            {
                "node_id": "node1",
                "compute_capacity": "high",
                "memory_available": 8192,
                "network_latency": 50
            },
            {
                "node_id": "node2",
                "compute_capacity": "medium", 
                "memory_available": 4096,
                "network_latency": 30
            }
        ]
        
        task_requirements = {
            "compute_requirement": "high",
            "memory_requirement": "medium",
            "network_requirement": "low"
        }
        
        selected_node = await deployment_strategy.select_optimal_node(
            available_nodes, task_requirements
        )
        
        assert selected_node is not None
        assert selected_node["node_id"] == "node1"  # 应选择计算能力强的节点
    
    @pytest.mark.asyncio
    async def test_model_lightweight_component(self):
        """测试模型轻量化组件"""
        
        lightweight_manager = ModelLightweightManager()
        
        # 测试策略选择
        strategy = await lightweight_manager.select_strategy(
            "agriculture_model", 0.85
        )
        
        assert strategy is not None
        assert "compression_method" in strategy
        assert "target_size" in strategy
        
        # 测试轻量化应用
        lightweight_model = await lightweight_manager.apply_lightweight(
            "agriculture_model", strategy
        )
        
        assert lightweight_model is not None
        assert "original_size" in lightweight_model
        assert "compressed_size" in lightweight_model
        assert "accuracy_loss" in lightweight_model
    
    @pytest.mark.asyncio
    async def test_resource_optimizer_component(self):
        """测试资源优化器组件"""
        
        resource_optimizer = EdgeResourceOptimizer()
        
        # 测试资源分配优化
        resource_requirements = {
            "compute_intensity": "high",
            "memory_usage": 1024,
            "network_bandwidth": 50
        }
        
        node_capacity = {
            "compute_capacity": "high",
            "memory_available": 4096,
            "network_capacity": 100
        }
        
        allocation = await resource_optimizer.optimize_allocation(
            resource_requirements, node_capacity
        )
        
        assert allocation is not None
        assert "cpu_cores" in allocation
        assert "memory_mb" in allocation
        assert "bandwidth_mbps" in allocation
    
    @pytest.mark.asyncio
    async def test_cloud_edge_sync_component(self):
        """测试云边协同组件"""
        
        sync_manager = CloudEdgeSyncManager()
        
        # 测试同步启动
        sync_config = {
            "deployment_id": "test_deployment",
            "sync_mode": "real_time",
            "data_retention": "7d"
        }
        
        sync_result = await sync_manager.start_sync(sync_config)
        
        assert sync_result is not None
        assert "sync_id" in sync_result
        assert "status" in sync_result
        
        # 测试性能指标获取
        performance_metrics = await sync_manager.get_performance_metrics("test_deployment")
        
        assert performance_metrics is not None
        assert "avg_latency" in performance_metrics
        assert "throughput" in performance_metrics


class TestEdgeIntegrationPerformance:
    """边缘计算集成性能测试"""
    
    @pytest.mark.asyncio
    async def test_edge_integration_latency(self, edge_manager):
        """测试边缘集成延迟"""
        
        test_data = {
            "task_type": "real_time_analysis",
            "data_size": 2 * 1024 * 1024,
            "latency_requirement": 300
        }
        
        # 测试响应时间
        start_time = datetime.now()
        
        for i in range(10):
            result = await edge_manager.integrate_edge_computing(test_data)
            assert "edge_enabled" in result
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        avg_latency = duration / 10
        
        # 验证平均延迟小于500ms
        assert avg_latency < 0.5, f"平均延迟 {avg_latency}s 超过阈值"
    
    @pytest.mark.asyncio
    async def test_edge_resource_utilization(self, edge_manager):
        """测试边缘资源利用率"""
        
        # 模拟并发请求
        test_data_list = [
            {
                "task_type": "real_time_analysis",
                "data_size": 1 * 1024 * 1024,
                "latency_requirement": 200
            } for _ in range(5)
        ]
        
        # 并发执行
        tasks = [edge_manager.integrate_edge_computing(data) for data in test_data_list]
        results = await asyncio.gather(*tasks)
        
        # 验证所有请求成功处理
        for result in results:
            assert "edge_enabled" in result
            assert "processing_mode" in result
        
        # 验证资源分配合理性
        edge_enabled_count = sum(1 for r in results if r["edge_enabled"])
        cloud_fallback_count = sum(1 for r in results if not r["edge_enabled"])
        
        # 应有合理的边缘/云端分配比例
        assert edge_enabled_count >= cloud_fallback_count, "边缘计算使用率过低"
    
    @pytest.mark.asyncio
    async def test_edge_scalability(self, edge_manager):
        """测试边缘计算可扩展性"""
        
        # 测试不同规模的任务
        task_scales = [
            {"data_size": 0.5 * 1024 * 1024, "expected_edge": True},   # 小任务
            {"data_size": 5 * 1024 * 1024, "expected_edge": True},     # 中等任务
            {"data_size": 15 * 1024 * 1024, "expected_edge": False}    # 大任务
        ]
        
        for scale in task_scales:
            test_data = {
                "task_type": "real_time_analysis",
                "data_size": scale["data_size"],
                "latency_requirement": 300
            }
            
            result = await edge_manager.integrate_edge_computing(test_data)
            
            # 验证预期行为
            if scale["expected_edge"]:
                assert result["edge_enabled"] is True, f"数据大小 {scale['data_size']} 应使用边缘计算"
            else:
                assert result["edge_enabled"] is False, f"数据大小 {scale['data_size']} 应回退到云端"