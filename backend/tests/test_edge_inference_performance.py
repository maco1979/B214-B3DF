"""
边缘推理服务性能测试

该测试脚本用于验证边缘推理服务的推理延迟是否满足<100ms的要求。
"""

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock
import numpy as np

from src.distributed_dcnn.federated_edge import FederatedDCNNCoordinator, FederatedDCNNConfig, EdgeInferenceService
from src.distributed_dcnn.core import DCNNConfig
from src.edge_computing.edge_manager import EdgeManager
from src.distributed_dcnn.blockchain_rewards import ContributionCalculator, ContributionMetrics, ContributionType


class TestEdgeInferencePerformance:
    """边缘推理服务性能测试类"""
    
    @pytest.fixture
    def mock_edge_manager(self):
        """模拟边缘管理器"""
        edge_manager = Mock(spec=EdgeManager)
        
        # 创建模拟边缘节点
        mock_node = Mock()
        mock_node.status = "idle"
        mock_node.capabilities = {
            "compute_power": 2.0,  # 计算能力
            "memory_available": 4096  # 内存可用量（MB）
        }
        mock_node.node_id = "edge_node_1"
        
        # 设置边缘管理器的边缘节点
        edge_manager.edge_nodes = {
            "edge_node_1": mock_node,
            "edge_node_2": Mock(
                status="idle",
                capabilities={"compute_power": 1.5, "memory_available": 2048},
                node_id="edge_node_2"
            )
        }
        
        # 模拟推理请求方法，返回模拟结果并模拟不同的延迟
        async def mock_inference_request(node_id, model_type, input_data):
            # 模拟不同的推理延迟（50ms-150ms）
            latency = np.random.uniform(0.05, 0.15)
            await asyncio.sleep(latency)
            return {
                "predictions": [0.9, 0.1],
                "confidence": 0.9,
                "latency": latency
            }
        
        edge_manager.inference_request = AsyncMock(side_effect=mock_inference_request)
        edge_manager.get_system_overview = Mock(return_value={
            "total_nodes": 2,
            "active_nodes": 2,
            "avg_utilization": 0.3
        })
        
        return edge_manager
    
    @pytest.fixture
    def federated_config(self):
        """创建联邦DCNN配置"""
        dcnn_config = DCNNConfig(
            input_shape=(3, 224, 224),
            num_classes=2,
            depth=50,
            base_filters=64,
            activation="relu",
            pooling="max",
            use_batch_norm=True
        )
        
        return FederatedDCNNConfig(
            dcnn_config=dcnn_config,
            training_mode="federated",
            differential_privacy=True,
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0,
            edge_node_selection="optimal",
            model_compression=True,
            compression_ratio=0.5
        )
    
    @pytest.fixture
    def reward_manager(self):
        """模拟奖励管理器"""
        manager = Mock()
        manager.record_contribution = AsyncMock(return_value={
            "success": True,
            "reward_points": 10.5
        })
        return manager
    
    @pytest.fixture
    async def inference_service(self, mock_edge_manager, federated_config, reward_manager):
        """创建边缘推理服务实例"""
        # 创建联邦协调器
        coordinator = FederatedDCNNCoordinator(federated_config)
        coordinator.edge_manager = mock_edge_manager
        
        # 初始化全局模型参数
        rng = np.random.RandomState(42)
        dummy_params = {
            "conv1": np.random.randn(64, 3, 7, 7),
            "fc1": np.random.randn(2, 1024)
        }
        coordinator.global_model_params = dummy_params
        
        # 创建推理服务
        service = EdgeInferenceService(coordinator, reward_manager)
        
        return service
    
    @pytest.mark.asyncio
    async def test_inference_latency_requirement(self, inference_service, mock_edge_manager):
        """测试推理延迟是否满足<100ms的要求"""
        
        # 准备测试数据
        input_data = np.random.randn(1, 3, 224, 224).tolist()  # 转换为可序列化的格式
        
        # 测试参数
        num_requests = 100  # 执行100次请求
        max_allowed_latency = 0.1  # 100ms
        
        print(f"开始执行 {num_requests} 次推理请求...")
        
        # 存储每次请求的延迟
        latencies = []
        successful_requests = 0
        failed_requests = 0
        
        # 执行多次推理请求
        for i in range(num_requests):
            try:
                start_time = time.time()
                result = await inference_service.inference_request(input_data)
                end_time = time.time()
                
                # 计算实际延迟
                actual_latency = end_time - start_time
                latencies.append(actual_latency)
                
                # 验证结果
                assert result["success"] is True
                assert "predictions" in result
                assert "inference_time" in result
                assert "delay_threshold_met" in result
                
                # 检查是否满足延迟要求
                if result["delay_threshold_met"]:
                    successful_requests += 1
                else:
                    failed_requests += 1
                
                # 打印进度
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i + 1}/{num_requests} 次请求...")
                    
            except Exception as e:
                failed_requests += 1
                print(f"请求 {i + 1} 失败: {e}")
        
        # 计算统计数据
        if latencies:
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            # 满足延迟要求的比例
            success_ratio = successful_requests / num_requests
        else:
            avg_latency = float('inf')
            min_latency = float('inf')
            max_latency = float('inf')
            p95_latency = float('inf')
            p99_latency = float('inf')
            success_ratio = 0.0
        
        # 打印测试结果
        print("\n===== 边缘推理服务性能测试结果 =====")
        print(f"总请求数: {num_requests}")
        print(f"成功请求数: {successful_requests}")
        print(f"失败请求数: {failed_requests}")
        print(f"平均延迟: {avg_latency * 1000:.2f} ms")
        print(f"最小延迟: {min_latency * 1000:.2f} ms")
        print(f"最大延迟: {max_latency * 1000:.2f} ms")
        print(f"95%延迟: {p95_latency * 1000:.2f} ms")
        print(f"99%延迟: {p99_latency * 1000:.2f} ms")
        print(f"满足延迟要求(<100ms)的比例: {success_ratio * 100:.2f}%")
        
        # 验证性能要求
        assert avg_latency < max_allowed_latency, f"平均延迟 {avg_latency * 1000:.2f} ms 超过 {max_allowed_latency * 1000} ms 的阈值"
        assert p95_latency < max_allowed_latency, f"95%延迟 {p95_latency * 1000:.2f} ms 超过 {max_allowed_latency * 1000} ms 的阈值"
        assert success_ratio > 0.9, f"满足延迟要求的比例 {success_ratio * 100:.2f}% 低于90%的阈值"
    
    @pytest.mark.asyncio
    async def test_node_selection_strategy(self, inference_service, mock_edge_manager):
        """测试节点选择策略的有效性"""
        
        # 准备测试数据
        input_data = np.random.randn(1, 3, 224, 224).tolist()
        
        # 执行多次推理请求，记录选择的节点
        node_selection_count = {}
        num_requests = 50
        
        for i in range(num_requests):
            result = await inference_service.inference_request(input_data)
            assert result["success"] is True
            
            selected_node = result["edge_node"]
            node_selection_count[selected_node] = node_selection_count.get(selected_node, 0) + 1
        
        # 打印节点选择结果
        print("\n===== 节点选择策略测试结果 =====")
        for node_id, count in node_selection_count.items():
            percentage = (count / num_requests) * 100
            print(f"节点 {node_id}: {count} 次选择 ({percentage:.2f}%)")
        
        # 验证节点选择的合理性：计算能力强的节点应该被选择更多次
        assert node_selection_count.get("edge_node_1", 0) > node_selection_count.get("edge_node_2", 0), \
            "计算能力强的节点应该被选择更多次"
    
    @pytest.mark.asyncio
    async def test_inference_throughput(self, inference_service, mock_edge_manager):
        """测试边缘推理服务的吞吐量"""
        
        # 准备测试数据
        input_data = np.random.randn(1, 3, 224, 224).tolist()
        
        # 测试并发请求
        num_concurrent_requests = 10
        num_rounds = 5
        
        print(f"开始测试 {num_concurrent_requests} 个并发请求，共 {num_rounds} 轮...")
        
        total_requests = num_concurrent_requests * num_rounds
        start_time = time.time()
        
        for round_idx in range(num_rounds):
            # 创建并发任务
            tasks = [
                inference_service.inference_request(input_data) 
                for _ in range(num_concurrent_requests)
            ]
            
            # 执行并发任务
            results = await asyncio.gather(*tasks)
            
            # 验证结果
            for result in results:
                assert result["success"] is True
            
            print(f"第 {round_idx + 1}/{num_rounds} 轮并发请求完成")
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = total_requests / total_time
        
        # 打印吞吐量测试结果
        print(f"\n===== 吞吐量测试结果 =====")
        print(f"总请求数: {total_requests}")
        print(f"总耗时: {total_time:.2f} s")
        print(f"吞吐量: {throughput:.2f} 请求/秒")
        
        # 验证吞吐量要求
        assert throughput > 50.0, f"吞吐量 {throughput:.2f} 请求/秒 低于50请求/秒的阈值"


if __name__ == "__main__":
    """直接运行测试"""
    # 创建测试实例
    test = TestEdgeInferencePerformance()
    
    # 运行延迟测试
    print("\n" + "="*60)
    print("运行边缘推理服务延迟测试")
    print("="*60)
    
    # 由于测试需要asyncio环境，我们需要创建一个事件循环来运行测试
    import asyncio
    
    async def run_test():
        # 创建模拟对象
        mock_edge_manager = test.mock_edge_manager()
        federated_config = test.federated_config()
        reward_manager = test.reward_manager()
        
        # 创建推理服务
        inference_service = await test.inference_service(mock_edge_manager, federated_config, reward_manager)
        
        # 运行延迟测试
        await test.test_inference_latency_requirement(inference_service, mock_edge_manager)
        
        # 运行节点选择策略测试
        print("\n" + "="*60)
        print("运行节点选择策略测试")
        print("="*60)
        await test.test_node_selection_strategy(inference_service, mock_edge_manager)
        
        # 运行吞吐量测试
        print("\n" + "="*60)
        print("运行吞吐量测试")
        print("="*60)
        await test.test_inference_throughput(inference_service, mock_edge_manager)
    
    # 运行测试
    try:
        asyncio.run(run_test())
        print("\n" + "="*60)
        print("所有测试通过！")
        print("="*60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("测试失败！")
        print("="*60)
