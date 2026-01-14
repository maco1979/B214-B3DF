#!/usr/bin/env python3
"""
测试联邦学习聚合功能
"""

import requests
import json
import time

# API基础URL
BASE_URL = "http://localhost:8001/api"

# 测试步骤
print("=== 测试联邦学习聚合功能 ===\n")

# 1. 注册客户端
print("1. 注册客户端...")
clients = []
for i in range(2):
    client_data = {
        "client_id": f"test_client_{i}",
        "info": {
            "training_capability": 1.0,
            "data_size": 1000
        }
    }
    response = requests.post(f"{BASE_URL}/federated/clients/register", json=client_data)
    print(f"  客户端 {i} 注册: {response.status_code} - {response.json().get('message')}")
    clients.append(f"test_client_{i}")

print()

# 2. 开始训练轮次
print("2. 开始训练轮次...")
round_config = {
    "client_fraction": 1.0,
    "learning_rate": 0.01,
    "epochs": 1
}
response = requests.post(f"{BASE_URL}/federated/rounds/start", json=round_config)
round_info = response.json()
round_id = round_info.get("round_info", {}).get("round_id")
print(f"  训练轮次已启动: {round_id}")
print(f"  状态: {response.status_code} - {response.json().get('message')}")

print()

# 3. 提交客户端更新
print("3. 提交客户端更新...")
for client_id in clients:
    # 简单的参数更新示例
    update_data = {
        "client_id": client_id,
        "update": {
            "parameters": {
                "dense_0_weight": [[0.01, 0.02], [0.03, 0.04]],  # 简化的示例参数
                "dense_0_bias": [0.01, 0.02],
                "dense_1_weight": [[0.05, 0.06], [0.07, 0.08]],
                "dense_1_bias": [0.03, 0.04],
                "dense_2_weight": [[0.09, 0.10], [0.11, 0.12]],
                "dense_2_bias": [0.05, 0.06]
            },
            "data_size": 1000,
            "training_time": 10.5
        }
    }
    response = requests.post(f"{BASE_URL}/federated/rounds/{round_id}/updates", json=update_data)
    print(f"  客户端 {client_id} 更新: {response.status_code} - {response.json().get('message')}")

print()

# 4. 调用聚合API
print(f"4. 调用聚合API (轮次: {round_id})...")
response = requests.post(f"{BASE_URL}/federated/rounds/{round_id}/aggregate")
print(f"  聚合结果: {response.status_code}")
print(f"  响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print("\n=== 测试完成 ===")
