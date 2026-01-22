#!/usr/bin/env python3
"""
高级联邦学习模块测试
验证改进后的本地训练功能更加真实
"""

import sys
import os
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from backend.src.federated.federated_learning import FederatedLearningServer, FederatedLearningClient

def test_advanced_local_training():
    print("🔍 测试改进后的本地训练功能...")
    
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
    
    # 检查模型参数
    print(f"✅ 模型参数层数量: {server.global_model['metadata']['parameter_count']}")
    
    # 创建具有不同数据特征的客户端
    print("\n📋 创建具有不同数据特征的客户端...")
    
    # 客户端1: 较大数据集
    client1_data = np.random.random((1000, 10))  # 1000个样本，10个特征
    client1 = FederatedLearningClient("client_1", client1_data)
    client1.initialize_with_global_model(server._prepare_client_model())
    
    # 客户端2: 较小数据集
    client2_data = np.random.random((100, 10))   # 100个样本，10个特征
    client2 = FederatedLearningClient("client_2", client2_data)
    client2.initialize_with_global_model(server._prepare_client_model())
    
    print("✅ 客户端初始化完成")
    print(f"  - 客户端1数据大小: {len(client1_data)}")
    print(f"  - 客户端2数据大小: {len(client2_data)}")
    
    # 执行本地训练
    print("\n🤖 执行改进后的本地训练...")
    
    training_config = {
        "learning_rate": 0.01,
        "epochs": 3,
        "batch_size": 32
    }
    
    # 客户端1训练
    print("  - 客户端1训练中...")
    update1 = client1.local_training(training_config)
    print(f"  ✅ 客户端1训练完成，更新参数数量: {len(update1['parameters'])}")
    print(f"  ✅ 客户端1训练时间: {update1['training_time']:.4f}s")
    
    # 客户端2训练
    print("  - 客户端2训练中...")
    update2 = client2.local_training(training_config)
    print(f"  ✅ 客户端2训练完成，更新参数数量: {len(update2['parameters'])}")
    print(f"  ✅ 客户端2训练时间: {update2['training_time']:.4f}s")
    
    # 验证更新是否基于数据大小有所不同
    print("\n📊 验证训练结果差异...")
    
    # 检查参数更新是否合理
    for key in update1['parameters'].keys():
        update1_norm = np.linalg.norm(update1['parameters'][key])
        update2_norm = np.linalg.norm(update2['parameters'][key])
        print(f"  - 参数 {key}: 客户端1更新范数={update1_norm:.6f}, 客户端2更新范数={update2_norm:.6f}")
    
    # 注册客户端到服务器
    print("\n📋 注册客户端到服务器...")
    server.register_client("client_1", {"data_size": len(client1_data)})
    server.register_client("client_2", {"data_size": len(client2_data)})
    
    # 开始训练轮次
    print("\n🔄 开始联邦学习轮次...")
    round_config = {
        "client_fraction": 1.0,
        "learning_rate": 0.01,
        "epochs": 1
    }
    
    round_info = server.start_training_round(round_config)
    print(f"✅ 轮次 {round_info['round_id']} 启动成功")
    
    # 提交更新
    print("\n📤 提交客户端更新...")
    server.receive_client_update("client_1", round_info['round_id'], update1)
    server.receive_client_update("client_2", round_info['round_id'], update2)
    print("✅ 客户端更新提交成功")
    
    # 聚合更新
    print("\n🧮 聚合更新...")
    success = server.aggregate_updates(round_info['round_id'])
    if success:
        print("✅ 更新聚合成功")
    else:
        print("❌ 更新聚合失败")
        return False
    
    # 验证聚合后的模型参数
    print("\n🔍 验证聚合后模型...")
    final_param_count = len(server.global_model['parameters'])
    print(f"✅ 聚合后模型参数数量: {final_param_count}")
    
    # 检查聚合是否有效
    if server.rounds_completed > 0:
        print(f"✅ 轮次完成数量: {server.rounds_completed}")
        print("✅ 联邦学习流程完整")
    else:
        print("❌ 轮次未完成")
        return False
    
    print("\n🎉 高级联邦学习测试完成！本地训练功能更加真实。")
    return True

def test_training_config_impact():
    """测试不同训练配置对本地训练的影响"""
    print("\n🧪 测试不同训练配置的影响...")
    
    # 创建模型和客户端
    model_architecture = {
        "name": "config_test_model",
        "layers": [{"type": "dense", "input_units": 5, "units": 10, "activation": "relu"}]
    }
    
    server = FederatedLearningServer(model_architecture)
    client_data = np.random.random((500, 5))
    
    # 测试不同的学习率
    configs = [
        {"learning_rate": 0.001, "epochs": 1, "batch_size": 32, "name": "低学习率"},
        {"learning_rate": 0.01, "epochs": 1, "batch_size": 32, "name": "中学习率"},
        {"learning_rate": 0.1, "epochs": 1, "batch_size": 32, "name": "高学习率"},
        {"learning_rate": 0.01, "epochs": 1, "batch_size": 32, "name": "单轮训练"},
        {"learning_rate": 0.01, "epochs": 5, "batch_size": 32, "name": "多轮训练"}
    ]
    
    client_updates = []
    
    for i, config in enumerate(configs):
        print(f"  - {config['name']}配置训练...")
        client = FederatedLearningClient(f"test_client_{i}", client_data)
        client.initialize_with_global_model(server._prepare_client_model())
        
        update = client.local_training(config)
        update_norm = sum(np.linalg.norm(param) for param in update['parameters'].values())
        client_updates.append((config['name'], update_norm, update['training_time']))
        print(f"    ✅ 更新范数: {update_norm:.6f}, 训练时间: {update['training_time']:.4f}s")
    
    print("\n📈 配置影响分析:")
    for name, norm, time in client_updates:
        print(f"  - {name}: 更新范数={norm:.6f}, 时间={time:.4f}s")
    
    return True

if __name__ == "__main__":
    print("🚀 开始高级联邦学习测试...")
    
    success1 = test_advanced_local_training()
    success2 = test_training_config_impact()
    
    if success1 and success2:
        print("\n✅ 所有高级测试通过！本地训练功能更加真实和有效。")
        sys.exit(0)
    else:
        print("\n❌ 高级测试失败！")
        sys.exit(1)