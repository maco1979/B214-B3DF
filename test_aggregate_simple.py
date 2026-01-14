#!/usr/bin/env python3
"""
简单测试联邦学习聚合功能
"""

import requests
import json

# API基础URL
BASE_URL = "http://localhost:8001/api"

# 测试步骤
print("=== 简单测试联邦学习聚合功能 ===\n")

# 1. 获取训练轮次
print("1. 获取训练轮次...")
response = requests.get(f"{BASE_URL}/federated/rounds")
rounds_data = response.json()
print(f"  状态: {response.status_code}")
print(f"  轮次数量: {len(rounds_data.get('data', []))}")

if rounds_data.get('data'):
    # 2. 调用聚合API（使用最近的轮次）
    latest_round = rounds_data['data'][-1]
    round_id = latest_round['round_id']
    print(f"\n2. 调用聚合API (轮次: {round_id})...")
    response = requests.post(f"{BASE_URL}/federated/rounds/{round_id}/aggregate")
    print(f"  聚合结果: {response.status_code}")
    print(f"  响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print("\n=== 测试完成 ===")
