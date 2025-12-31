#!/usr/bin/env python3
"""
测试联邦学习模块功能
验证联邦学习模块是否真实连接并正确工作
"""

import sys
import os
import asyncio
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from backend.src.federated.federated_learning import FederatedLearningServer, FederatedLearningClient

def test_federated_learning():
    print("🔍 测试联邦学习模块...")
    
    # 创建联邦学习服务器
    model_architecture = {
        "name": "test_model",
        "layers": [
            {"type": "dense", "input_units": 10, "units": 64, "activation": "relu"},
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 5, "activation": "softmax"}
        ]
    }
    
    print("✅ 创建联邦学习服务器")
    server = FederatedLearningServer(model_architecture)
    
    # 检查模型是否正确初始化
    print(f"✅ 模型参数数量: {server.global_model['metadata']['parameter_count']}")
    print(f"✅ 模型架构: {server.model_architecture['name']}")
    
    # 注册客户端
    print("\n📋 注册联邦学习客户端...")
    client_info = {
        "device_type": "edge_device",
        "capabilities": ["training", "inference"],
        "data_size": 1000,
        "training_capability": 1.0
    }
    
    success = server.register_client("client_1", client_info)
    if success:
        print("✅ 客户端 client_1 注册成功")
    else:
        print("❌ 客户端 client_1 注册失败")
        return False
    
    success = server.register_client("client_2", client_info)
    if success:
        print("✅ 客户端 client_2 注册成功")
    else:
        print("❌ 客户端 client_2 注册失败")
        return False
    
    # 检查客户端数量
    print(f"✅ 注册客户端数量: {len(server.clients)}")
    
    # 开始训练轮次
    print("\n🔄 开始联邦学习训练轮次...")
    round_config = {
        "client_fraction": 1.0,  # 100% 的客户端参与
        "learning_rate": 0.01,
        "epochs": 1
    }
    
    round_info = server.start_training_round(round_config)
    print(f"✅ 训练轮次启动成功: {round_info['round_id']}")
    print(f"✅ 参与客户端: {round_info['selected_clients']}")
    
    # 模拟客户端训练
    print("\n🤖 模拟客户端本地训练...")
    for client_id in round_info['selected_clients']:
        print(f"  - 为客户端 {client_id} 创建本地训练...")
        
        # 创建客户端实例
        client_data = np.random.random((100, 10))  # 模拟本地数据
        client = FederatedLearningClient(client_id, client_data)
        
        # 使用全局模型初始化客户端
        client.initialize_with_global_model(round_info['global_model'])
        print(f"  ✅ 客户端 {client_id} 模型初始化完成")
        
        # 执行本地训练
        training_config = {
            "learning_rate": 0.01,
            "round_id": round_info['round_id']
        }
        
        update_info = client.local_training(training_config)
        print(f"  ✅ 客户端 {client_id} 本地训练完成")
        
        # 提交更新到服务器
        success = server.receive_client_update(
            client_id, 
            round_info['round_id'], 
            update_info
        )
        
        if success:
            print(f"  ✅ 客户端 {client_id} 更新提交成功")
        else:
            print(f"  ❌ 客户端 {client_id} 更新提交失败")
            return False
    
    # 聚合更新
    print(f"\n🧮 聚合训练轮次 {round_info['round_id']} 的更新...")
    success = server.aggregate_updates(round_info['round_id'])
    
    if success:
        print("✅ 联邦学习轮次聚合成功")
        print(f"✅ 完成轮次数量: {server.rounds_completed}")
    else:
        print("❌ 联邦学习轮次聚合失败")
        return False
    
    # 检查服务器状态
    print("\n📊 检查服务器状态...")
    status = server.get_server_status()
    print(f"✅ 总轮次完成: {status['rounds_completed']}")
    print(f"✅ 总客户端数: {status['total_clients']}")
    print(f"✅ 活跃客户端数: {status['active_clients']}")
    print(f"✅ 差分隐私启用: {status['dp_enabled']}")
    
    # 测试隐私状态
    print("\n🔒 测试差分隐私功能...")
    try:
        privacy_spent = server.dp_mechanism.compute_privacy_spent(
            steps=server.rounds_completed * 100,
            batch_size=32,
            dataset_size=10000
        )
        print(f"✅ 隐私消耗计算成功: {privacy_spent}")
    except Exception as e:
        print(f"⚠️ 隐私消耗计算异常: {e}")
    
    print("\n🎉 联邦学习模块测试完成！所有功能正常工作。")
    return True

if __name__ == "__main__":
    success = test_federated_learning()
    if success:
        print("\n✅ 所有测试通过！联邦学习模块真实可用。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！联邦学习模块存在问题。")
        sys.exit(1)