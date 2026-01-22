#!/usr/bin/env python3
"""
简单的合规性测试脚本，不依赖Flax库
验证数据本地化存储、差分隐私保护和交易溯源功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from src.edge_computing.cloud_edge_sync import CloudEdgeSyncManager
from src.privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from src.distributed_dcnn.blockchain_rewards import ContributionMetrics, ContributionType, BlockchainRewardManager

class MockFabricClient:
    """模拟区块链客户端"""
    def __init__(self):
        self.operations = []
    
    async def invoke_contract(self, contract_name, function_name, *args, **kwargs):
        self.operations.append((contract_name, function_name, args, kwargs))
        return {'success': True, 'transaction_id': 'mock_tx_123'}

def test_data_localization():
    """测试数据本地化存储功能"""
    print("\n=== 测试数据本地化存储功能 ===")
    
    # 创建云边同步管理器
    config = {
        "localization": {
            "enabled": True,
            "allowed_regions": ["CN"]
        }
    }
    sync_manager = CloudEdgeSyncManager(config)
    
    # 注册边缘节点
    sync_manager.register_edge_node("node_cn_1", {"region": "CN"})
    sync_manager.register_edge_node("node_us_1", {"region": "US"})
    sync_manager.register_edge_node("node_cn_2", {"region": "CN"})
    
    # 测试节点位置验证
    assert sync_manager._is_valid_node_location("node_cn_1") == True
    assert sync_manager._is_valid_node_location("node_us_1") == False
    assert sync_manager._is_valid_node_location("node_cn_2") == True
    
    # 测试敏感数据识别
    sensitive_keys = {"user_id", "device_id", "location", "health_data"}
    for key in sensitive_keys:
        assert sync_manager._is_sensitive_data(key) == True
    
    assert sync_manager._is_sensitive_data("temperature") == False
    assert sync_manager._is_sensitive_data("humidity") == False
    
    print("✅ 数据本地化存储功能测试通过")
    return True

def test_differential_privacy():
    """测试差分隐私保护功能"""
    print("\n=== 测试差分隐私保护功能 ===")
    
    # 创建差分隐私实例，验证epsilon=1.0
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    assert dp.epsilon == 1.0
    
    # 测试梯度裁剪
    import numpy as np
    gradients = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    clipped_gradients = dp.clip_gradients(gradients, clip_norm=1.0)
    
    # 验证梯度被裁剪
    total_norm = 0.0
    for grad in clipped_gradients:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    assert total_norm <= 1.0
    
    print("✅ 差分隐私保护功能测试通过")
    return True

def test_transaction_traceability():
    """测试交易溯源功能"""
    print("\n=== 测试交易溯源功能 ===")
    
    # 创建模拟区块链客户端
    mock_client = MockFabricClient()
    
    # 创建奖励管理器
    reward_manager = BlockchainRewardManager(mock_client)
    
    # 记录贡献
    metrics = ContributionMetrics(
        participant_id="node_cn_1",
        contribution_type=ContributionType.COMPUTE_CONTRIBUTION,
        compute_time=0.1,  # 100ms以下
        compute_efficiency=1.0
    )
    
    # 模拟异步调用
    import asyncio
    asyncio.run(reward_manager.record_contribution(metrics))
    
    # 验证奖励历史和交易哈希
    assert len(reward_manager.reward_history) == 1
    reward = reward_manager.reward_history[0]
    assert reward.transaction_hash is not None
    assert len(reward.transaction_hash) == 64  # SHA256哈希长度
    
    print(f"✅ 交易溯源功能测试通过，生成交易哈希: {reward.transaction_hash}")
    return True

def test_edge_inference_latency():
    """测试边缘推理延迟"""
    print("\n=== 测试边缘推理延迟 ===")
    
    # 这里我们只验证边缘节点选择策略的实现
    # 实际延迟测试需要运行完整的推理流程
    
    # 验证节点选择策略权重配置
    from src.distributed_dcnn.federated_edge import EdgeInferenceService
    
    config = {
        "model_id": "test_dcnn",
        "max_inference_latency": 100,  # 100ms
        "node_selection": {
            "compute_power_weight": 0.4,
            "memory_weight": 0.3,
            "load_weight": 0.2,
            "latency_weight": 0.1
        }
    }
    
    # 创建边缘推理服务实例
    inference_service = EdgeInferenceService(config)
    
    # 验证配置
    assert inference_service.max_inference_latency == 100
    assert inference_service.node_selection_weights == {
        'compute_power': 0.4,
        'memory': 0.3,
        'load': 0.2,
        'latency': 0.1
    }
    
    print("✅ 边缘推理延迟配置测试通过")
    return True

def main():
    """主测试函数"""
    print("开始执行合规性测试...")
    
    try:
        # 执行所有测试
        results = [
            test_data_localization(),
            test_differential_privacy(),
            test_transaction_traceability(),
            test_edge_inference_latency()
        ]
        
        # 检查测试结果
        if all(results):
            print("\n🎉 所有合规性测试通过！")
            return 0
        else:
            print("\n❌ 部分测试失败！")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
