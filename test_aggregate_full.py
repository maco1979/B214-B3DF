#!/usr/bin/env python3
"""
完整测试联邦学习聚合功能
"""

import requests
import json
import numpy as np

# API基础URL
BASE_URL = "http://localhost:8001/api"

# 测试步骤
print("=== 完整测试联邦学习聚合功能 ===\n")

# 1. 注册客户端
print("1. 注册客户端...")
client_id = "test_client"
client_data = {
    "client_id": client_id,
    "info": {
        "training_capability": 1.0,
        "data_size": 1000
    }
}
response = requests.post(f"{BASE_URL}/federated/clients/register", json=client_data)
print(f"  客户端注册: {response.status_code} - {response.json().get('message')}")

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

# 3. 获取全局模型，了解参数形状
print("3. 获取全局模型...")
response = requests.get(f"{BASE_URL}/federated/model")
model_data = response.json()
print(f"  状态: {response.status_code}")

# 打印模型参数形状
print("  模型参数形状:")
for key, param in model_data.get("model", {}).get("parameters", {}).items():
    if isinstance(param, list):
        # 计算形状
        shape = []
        current = param
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0] if current else None
        print(f"    {key}: {shape}")

print()

# 4. 提交客户端更新（使用与模型匹配的参数形状）
print("4. 提交客户端更新...")
# 使用模型中的实际参数形状来创建更新
parameters = model_data.get("model", {}).get("parameters", {})
update_params = {}

for key, param in parameters.items():
    if isinstance(param, list):
        # 创建与模型形状匹配的随机更新
        # 将列表转换为NumPy数组
        np_param = np.array(param)
        # 创建相同形状的随机更新
        np_update = np.random.normal(0, 0.01, np_param.shape)
        # 转换回列表
        update_params[key] = np_update.tolist()

update_data = {
    "client_id": client_id,
    "update": {
        "parameters": update_params,
        "data_size": 1000,
        "training_time": 10.5
    }
}

response = requests.post(f"{BASE_URL}/federated/rounds/{round_id}/updates", json=update_data)
print(f"  客户端更新: {response.status_code} - {response.json().get('message')}")

print()

# 5. 调用聚合API
print(f"5. 调用聚合API (轮次: {round_id})...")
response = requests.post(f"{BASE_URL}/federated/rounds/{round_id}/aggregate")
print(f"  聚合结果: {response.status_code}")
print(f"  响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print("\n=== 测试完成 ===")
