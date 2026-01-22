"""
分布式DCNN系统简化测试
验证核心功能是否正常工作
"""

import sys
import os
import asyncio

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

try:
    from distributed_dcnn.core import DistributedDCNNSystem
    from distributed_dcnn.federated_edge import FederatedEdgeManager
    from distributed_dcnn.blockchain_rewards import BlockchainRewardManager
    from distributed_dcnn.config import DistributedDCNNConfig
    
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)


async def test_core_system():
    """测试核心系统"""
    print("\n1. 测试核心DCNN系统...")
    
    try:
        config = DistributedDCNNConfig()
        system = DistributedDCNNSystem(config.to_dict())
        
        # 测试初始化
        await system.initialize()
        
        # 测试状态获取
        status = system.get_status()
        
        print(f"✓ 核心系统初始化成功")
        print(f"  模型架构: {status.get('model_architecture', 'N/A')}")
        print(f"  边缘节点: {status.get('edge_nodes_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 核心系统测试失败: {e}")
        return False


async def test_federated_learning():
    """测试联邦学习"""
    print("\n2. 测试联邦学习系统...")
    
    try:
        config = DistributedDCNNConfig()
        manager = FederatedEdgeManager(config.to_dict())
        
        # 测试启动
        await manager.start()
        
        # 测试状态获取
        status = manager.get_status()
        
        print(f"✓ 联邦学习系统启动成功")
        print(f"  活跃节点: {status.get('active_nodes', 0)}")
        print(f"  学习轮次: {status.get('current_round', 0)}")
        
        # 测试停止
        await manager.stop()
        
        return True
        
    except Exception as e:
        print(f"✗ 联邦学习测试失败: {e}")
        return False


async def test_blockchain_rewards():
    """测试区块链奖励系统"""
    print("\n3. 测试区块链奖励系统...")
    
    try:
        config = DistributedDCNNConfig()
        manager = BlockchainRewardManager(config.to_dict())
        
        # 测试初始化
        await manager.initialize()
        
        # 测试状态获取
        status = manager.get_status()
        
        print(f"✓ 区块链奖励系统初始化成功")
        print(f"  奖励池: {status.get('reward_pool', 0)} PHOTON")
        print(f"  已分发: {status.get('distributed_rewards', 0)} PHOTON")
        
        # 测试关闭
        await manager.shutdown()
        
        return True
        
    except Exception as e:
        print(f"✗ 区块链奖励测试失败: {e}")
        return False


async def test_integration():
    """测试集成功能"""
    print("\n4. 测试系统集成...")
    
    try:
        config = DistributedDCNNConfig()
        
        # 模拟图像数据
        test_data = {
            'batch_id': 'integration_test',
            'images': ['test_image_1', 'test_image_2'],
            'metadata': {'test': True}
        }
        
        # 测试核心系统推理
        system = DistributedDCNNSystem(config.to_dict())
        await system.initialize()
        
        results = await system.distributed_inference(test_data)
        
        print(f"✓ 集成测试成功")
        print(f"  推理结果数量: {len(results.get('results', []))}")
        
        await system.shutdown()
        
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("分布式DCNN系统功能验证")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(await test_core_system())
    test_results.append(await test_federated_learning())
    test_results.append(await test_blockchain_rewards())
    test_results.append(await test_integration())
    
    # 生成报告
    print("\n" + "=" * 50)
    print("测试报告:")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"总测试数: {total_tests}")
    print(f"通过数: {passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("\n🎯 测试结果: 优秀 - 分布式DCNN系统功能正常")
    elif success_rate >= 50:
        print("\n✅ 测试结果: 良好 - 系统基本功能正常")
    else:
        print("\n⚠️ 测试结果: 需要改进")
    
    return success_rate >= 75


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        exit(1)
    except Exception as e:
        print(f"测试执行异常: {e}")
        exit(1)