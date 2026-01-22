"""
合规性测试模块
验证系统是否符合中华人民共和国互联网法律法规要求
包括：数据本地化、差分隐私保护、交易溯源等
"""

import asyncio
import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试相关模块
from src.edge_computing.cloud_edge_sync import CloudEdgeSyncManager
from src.privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from src.distributed_dcnn.blockchain_rewards import (BlockchainRewardManager,
                                                   ContributionMetrics, 
                                                   ContributionType,
                                                   RewardAllocation)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataLocalization(unittest.TestCase):
    """测试数据本地化功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            "localization": {
                "enabled": True,
                "allowed_regions": ["CN"],
                "audit_enabled": True,
                "encryption_enabled": True
            }
        }
        self.sync_manager = CloudEdgeSyncManager(config=self.config)
    
    def test_valid_node_location(self):
        """测试有效节点位置验证"""
        # 注册中国节点
        self.sync_manager.register_edge_node("node_cn", {
            "region": "CN",
            "sync_capabilities": {"data_sync": True, "model_sync": True}
        })
        
        # 验证中国节点有效
        self.assertTrue(self.sync_manager._is_valid_node_location("node_cn"))
    
    def test_invalid_node_location(self):
        """测试无效节点位置验证"""
        # 注册非中国节点
        self.sync_manager.register_edge_node("node_us", {
            "region": "US",
            "sync_capabilities": {"data_sync": True, "model_sync": True}
        })
        
        # 验证非中国节点无效
        self.assertFalse(self.sync_manager._is_valid_node_location("node_us"))
    
    def test_sensitive_data_identification(self):
        """测试敏感数据识别"""
        # 测试敏感数据标识
        sensitive_keys = ["user_id", "device_id", "location", "health_data"]
        non_sensitive_keys = ["temperature", "humidity", "timestamp"]
        
        for key in sensitive_keys:
            self.assertTrue(self.sync_manager._is_sensitive_data(key))
            
        for key in non_sensitive_keys:
            self.assertFalse(self.sync_manager._is_sensitive_data(key))
    
    def test_data_sync_with_localization(self):
        """测试数据同步时的本地化验证"""
        # 注册节点
        self.sync_manager.register_edge_node("node_cn", {"region": "CN"})
        self.sync_manager.register_edge_node("node_us", {"region": "US"})
        
        # 测试数据同步 - 混合节点
        async def sync_data_test():
            result = await self.sync_manager.sync_data(
                data_key="user_id_123",
                data={"name": "test_user", "location": "Beijing"},
                target_nodes=["node_cn", "node_us"],
                priority=1
            )
            return result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(sync_data_test())
        
        # 验证数据同步成功（仅中国节点）
        self.assertNotEqual(result, "", "数据同步应该成功")
        
        # 验证审计日志记录
        self.assertGreater(len(self.sync_manager.data_localization_logs), 0, "应该记录数据本地化审计日志")


class TestDifferentialPrivacy(unittest.TestCase):
    """测试差分隐私保护功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 符合要求的差分隐私配置（ε=1.0）
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        self.privacy_accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)
    
    def test_gaussian_mechanism(self):
        """测试高斯机制实现"""
        # 测试数据
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sensitivity = 1.0
        
        # 添加高斯噪声
        noisy_data = self.dp.gaussian_mechanism(data, sensitivity)
        
        # 验证噪声已添加
        self.assertFalse(np.array_equal(data, noisy_data), "应该添加噪声")
        
        # 验证数据大致范围保持
        self.assertLess(np.abs(np.mean(data) - np.mean(noisy_data)), 1.0, "均值变化不应过大")
    
    def test_gradients_clipping(self):
        """测试梯度裁剪功能"""
        # 创建大梯度
        gradients = [
            np.array([10.0, 20.0, 30.0]),
            np.array([-5.0, -15.0, -25.0])
        ]
        
        # 裁剪梯度
        clipped_gradients = self.dp.clip_gradients(gradients, clip_norm=1.0)
        
        # 验证梯度已裁剪
        total_norm = 0.0
        for grad in clipped_gradients:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        self.assertLessEqual(total_norm, 1.0, "裁剪后的梯度范数不应超过阈值")
    
    def test_dp_sgd_step(self):
        """测试差分隐私SGD步骤"""
        # 创建梯度
        gradients = [np.array([1.5, 2.5, 3.5])]
        
        # 应用差分隐私处理
        dp_gradients = self.dp.dp_sgd_step(gradients, clip_norm=1.0, noise_multiplier=1.0)
        
        # 验证梯度已处理
        self.assertEqual(len(dp_gradients), len(gradients), "梯度数量应保持一致")
        self.assertFalse(np.array_equal(dp_gradients[0], gradients[0]), "梯度应被修改")
    
    def test_privacy_accounting(self):
        """测试隐私预算管理"""
        # 模拟训练步骤
        for i in range(5):
            self.privacy_accountant.add_step(0.1)
        
        # 验证隐私预算消耗
        self.assertEqual(self.privacy_accountant.steps_completed, 5, "完成步数应正确")
        self.assertLessEqual(self.privacy_accountant.epsilon_spent, 1.0, "隐私预算不应超过上限")
        self.assertTrue(self.privacy_accountant.can_continue(), "应能继续训练")
        
        # 耗尽隐私预算
        for i in range(15):
            self.privacy_accountant.add_step(0.1)
        
        # 验证隐私预算耗尽
        self.assertGreaterEqual(self.privacy_accountant.epsilon_spent, 1.0, "隐私预算应耗尽")
        self.assertFalse(self.privacy_accountant.can_continue(), "训练应停止")


class TestTransactionTraceability(unittest.TestCase):
    """测试交易溯源功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.fabric_client_mock = MagicMock()
        self.reward_manager = BlockchainRewardManager(self.fabric_client_mock)
    
    def test_contribution_recording(self):
        """测试贡献度记录功能"""
        # 创建贡献指标
        metrics = ContributionMetrics(
            participant_id="user_123",
            contribution_type=ContributionType.DATA_CONTRIBUTION,
            data_size=1000,
            data_quality=0.95,
            timestamp=datetime.now(),
            round_id=1
        )
        
        # 模拟记录贡献
        async def record_contribution_test():
            result = await self.reward_manager.record_contribution(metrics)
            return result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(record_contribution_test())
        
        # 验证记录成功
        self.assertTrue(result, "贡献记录应该成功")
        
        # 验证奖励分配
        self.assertGreater(len(self.reward_manager.reward_history), 0, "应该有奖励分配记录")
    
    def test_transaction_hash_generation(self):
        """测试交易哈希生成"""
        # 创建贡献指标
        metrics = ContributionMetrics(
            participant_id="user_123",
            contribution_type=ContributionType.COMPUTE_CONTRIBUTION,
            compute_time=60.0,
            compute_efficiency=0.85,
            timestamp=datetime.now(),
            round_id=1
        )
        
        # 模拟记录贡献
        async def record_contribution_test():
            await self.reward_manager.record_contribution(metrics)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(record_contribution_test())
        
        # 验证交易哈希存在
        reward = self.reward_manager.reward_history[-1]
        self.assertIsNotNone(reward.transaction_hash, "每个奖励应该有交易哈希")
        self.assertEqual(len(reward.transaction_hash), 64, "交易哈希长度应为64位")
    
    def test_reward_traceability(self):
        """测试奖励可追溯性"""
        # 创建多个贡献记录
        participants = ["user_123", "user_456", "user_789"]
        
        async def create_rewards():
            for i, participant_id in enumerate(participants):
                metrics = ContributionMetrics(
                    participant_id=participant_id,
                    contribution_type=ContributionType.FEDERATED_ROUND,
                    data_size=500 * (i + 1),
                    compute_time=30.0 * (i + 1),
                    accuracy_gain=0.01 * (i + 1),
                    data_quality=0.9 + (i * 0.02),
                    timestamp=datetime.now(),
                    round_id=1
                )
                await self.reward_manager.record_contribution(metrics)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(create_rewards())
        
        # 验证每个参与者的奖励记录
        for participant_id in participants:
            async def get_rewards():
                return await self.reward_manager.get_participant_rewards(participant_id)
            
            rewards = loop.run_until_complete(get_rewards())
            
            # 验证奖励记录存在
            self.assertGreater(len(rewards.get("reward_history", [])), 0, 
                              f"参与者 {participant_id} 应该有奖励记录")
            
            # 验证每条记录都有交易哈希
            for reward in rewards["reward_history"]:
                self.assertIsNotNone(reward.get("transaction_hash"), 
                                    "每条奖励记录都应有交易哈希")


class TestComprehensiveCompliance(unittest.TestCase):
    """综合合规性测试"""
    
    def setUp(self):
        """设置综合测试环境"""
        # 数据本地化配置
        self.localization_config = {
            "localization": {
                "enabled": True,
                "allowed_regions": ["CN"],
                "audit_enabled": True
            }
        }
        
        # 差分隐私配置
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # 区块链奖励配置
        self.fabric_client_mock = MagicMock()
        self.reward_manager = BlockchainRewardManager(self.fabric_client_mock)
    
    def test_data_processing_compliance(self):
        """测试数据处理全流程合规性"""
        """验证从数据收集到奖励分配的全流程合规性"""
        
        # 1. 验证数据本地化
        sync_manager = CloudEdgeSyncManager(config=self.localization_config)
        sync_manager.register_edge_node("node_cn", {"region": "CN"})
        
        async def full_compliance_test():
            # 2. 同步敏感数据（仅中国节点）
            sync_result = await sync_manager.sync_data(
                data_key="sensitive_data_123",
                data={"user_id": "12345", "health_data": {"temperature": 37.0}},
                target_nodes=["node_cn"]
            )
            
            # 3. 创建贡献指标
            metrics = ContributionMetrics(
                participant_id="user_123",
                contribution_type=ContributionType.DATA_CONTRIBUTION,
                data_size=1000,
                data_quality=0.95,
                timestamp=datetime.now(),
                round_id=1
            )
            
            # 4. 应用差分隐私
            data = np.array([1.0, 2.0, 3.0])
            noisy_data = self.dp.gaussian_mechanism(data, sensitivity=1.0)
            
            # 5. 记录贡献并分配奖励
            reward_result = await self.reward_manager.record_contribution(metrics)
            
            return sync_result, noisy_data, reward_result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        sync_result, noisy_data, reward_result = loop.run_until_complete(full_compliance_test())
        
        # 验证结果
        self.assertNotEqual(sync_result, "", "数据同步应该成功")
        self.assertTrue(reward_result, "奖励分配应该成功")
        self.assertGreater(len(self.reward_manager.reward_history), 0, "应该有奖励记录")


if __name__ == "__main__":
    """运行测试"""
    # 创建测试套件
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDataLocalization))
    suite.addTest(unittest.makeSuite(TestDifferentialPrivacy))
    suite.addTest(unittest.makeSuite(TestTransactionTraceability))
    suite.addTest(unittest.makeSuite(TestComprehensiveCompliance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果
    logger.info(f"测试完成: 总测试数={result.testsRun}, 失败={len(result.failures)}, 错误={len(result.errors)}")
    
    # 如果有测试失败，返回非零退出码
    if len(result.failures) > 0 or len(result.errors) > 0:
        sys.exit(1)
    else:
        sys.exit(0)
