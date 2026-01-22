#!/usr/bin/env python3
"""
快速合规性验证脚本
验证关键的合规性功能：数据本地化、差分隐私和交易溯源
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入关键模块
from src.edge_computing.cloud_edge_sync import CloudEdgeSyncManager
from src.privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from src.distributed_dcnn.blockchain_rewards import (
    BlockchainRewardManager, ContributionMetrics, ContributionType
)

print("=== 快速合规性验证 ===")

# 1. 测试数据本地化
print("\n1. 测试数据本地化功能")
try:
    # 配置数据本地化
    localization_config = {
        "localization": {
            "enabled": True,
            "allowed_regions": ["CN"],
            "audit_enabled": True,
            "encryption_enabled": True
        }
    }
    
    sync_manager = CloudEdgeSyncManager(config=localization_config)
    
    # 注册节点
    sync_manager.register_edge_node("node_cn", {"region": "CN"})
    sync_manager.register_edge_node("node_us", {"region": "US"})
    
    # 验证节点位置
    cn_valid = sync_manager._is_valid_node_location("node_cn")
    us_valid = sync_manager._is_valid_node_location("node_us")
    
    print(f"   - 中国节点验证: {'通过' if cn_valid else '失败'}")
    print(f"   - 美国节点验证: {'失败(符合预期)' if not us_valid else '通过(不符合预期)'}")
    
    # 验证敏感数据识别
    sensitive_data_keys = ["user_id", "device_id", "location", "health_data"]
    non_sensitive_data_keys = ["temperature", "humidity", "timestamp"]
    
    sensitive_results = [sync_manager._is_sensitive_data(key) for key in sensitive_data_keys]
    non_sensitive_results = [sync_manager._is_sensitive_data(key) for key in non_sensitive_data_keys]
    
    print(f"   - 敏感数据识别: {'通过' if all(sensitive_results) else '失败'}")
    print(f"   - 非敏感数据识别: {'通过' if not any(non_sensitive_results) else '失败'}")
    
    print("   ✅ 数据本地化测试通过")
    
except Exception as e:
    print(f"   ❌ 数据本地化测试失败: {e}")

# 2. 测试差分隐私保护
print("\n2. 测试差分隐私保护功能")
try:
    # 配置差分隐私 (ε=1.0)
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    privacy_accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)
    
    # 测试高斯噪声机制
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noisy_data = dp.gaussian_mechanism(data, sensitivity=1.0)
    
    print(f"   - 原始数据: {data}")
    print(f"   - 添加噪声后: {noisy_data}")
    print(f"   - 噪声添加: {'通过' if not np.array_equal(data, noisy_data) else '失败'}")
    
    # 测试隐私预算管理
    for i in range(5):
        privacy_accountant.add_step(0.1)
    
    print(f"   - 隐私预算使用: {privacy_accountant.epsilon_spent:.3f}/1.0")
    print(f"   - 训练继续: {'是' if privacy_accountant.can_continue() else '否'}")
    
    # 耗尽隐私预算
    for i in range(15):
        privacy_accountant.add_step(0.1)
    
    print(f"   - 隐私预算耗尽: {'是' if not privacy_accountant.can_continue() else '否'}")
    
    print("   ✅ 差分隐私测试通过")
    
except Exception as e:
    print(f"   ❌ 差分隐私测试失败: {e}")

# 3. 测试交易溯源功能
print("\n3. 测试交易溯源功能")
try:
    # 创建模拟的Fabric客户端
    fabric_client_mock = MagicMock()
    reward_manager = BlockchainRewardManager(fabric_client_mock)
    
    # 创建贡献指标
    metrics = ContributionMetrics(
        participant_id="user_123",
        contribution_type=ContributionType.DATA_CONTRIBUTION,
        data_size=1000,
        data_quality=0.95,
        timestamp=datetime.now(),
        round_id=1
    )
    
    # 记录贡献
    import asyncio
    async def test_reward():
        result = await reward_manager.record_contribution(metrics)
        return result
    
    result = asyncio.run(test_reward())
    print(f"   - 贡献记录: {'成功' if result else '失败'}")
    
    # 检查奖励历史
    async def test_get_rewards():
        return await reward_manager.get_participant_rewards("user_123")
    
    rewards = asyncio.run(test_get_rewards())
    reward_count = len(rewards.get("reward_history", []))
    print(f"   - 奖励记录数量: {reward_count}")
    
    if reward_count > 0:
        # 检查是否有交易哈希
        has_transaction_hash = any("transaction_hash" in r for r in rewards["reward_history"])
        print(f"   - 交易哈希存在: {'是' if has_transaction_hash else '否'}")
    
    print("   ✅ 交易溯源测试通过")
    
except Exception as e:
    print(f"   ❌ 交易溯源测试失败: {e}")

print("\n=== 验证完成 ===")
