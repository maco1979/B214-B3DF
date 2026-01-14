#!/usr/bin/env python3
"""
测试没有客户端参与时的联邦学习聚合功能
"""

import requests
import json

# API基础URL
BASE_URL = "http://localhost:8001/api"

# 测试步骤
print("=== 测试没有客户端参与时的聚合功能 ===\n")

# 1. 开始训练轮次（不注册客户端）
print("1. 开始训练轮次...")
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

# 2. 调用聚合API（没有客户端参与）
print(f"2. 调用聚合API (轮次: {round_id})...")
response = requests.post(f"{BASE_URL}/federated/rounds/{round_id}/aggregate")
print(f"  聚合结果: {response.status_code}")
print(f"  响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print("\n=== 测试完成 ===")
