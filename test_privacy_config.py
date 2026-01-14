#!/usr/bin/env python3
"""
测试隐私保护配置API端点，验证PUT方法是否能正常工作
"""

import requests
import json

# API基础URL
BASE_URL = "http://localhost:8001/api"

# 测试步骤
print("=== 测试隐私保护配置API ===\n")

# 1. 获取当前隐私配置
print("1. 获取当前隐私配置...")
response = requests.get(f"{BASE_URL}/federated/privacy/status")
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print()

# 2. 使用PUT方法更新隐私配置
print("2. 使用PUT方法更新隐私配置...")
privacy_config = {
    "epsilon": 0.5,  # 值越小，隐私保护越强
    "delta": 0.00001,  # 很小的失败概率
    "enabled": True  # 启用隐私保护
}

response = requests.put(f"{BASE_URL}/federated/privacy/config", json=privacy_config)
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print()

# 3. 再次获取隐私配置，验证更新是否成功
print("3. 再次获取隐私配置，验证更新是否成功...")
response = requests.get(f"{BASE_URL}/federated/privacy/status")
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

# 检查更新是否成功
if response.status_code == 200:
    data = response.json().get("data", {})
    if data.get("epsilon") == privacy_config["epsilon"] and \
       data.get("delta") == privacy_config["delta"] and \
       data.get("enabled") == privacy_config["enabled"]:
        print("\n✅ 隐私配置更新成功！")
    else:
        print("\n❌ 隐私配置更新失败！")
        print(f"  期望: {privacy_config}")
        print(f"  实际: {data}")
else:
    print("\n❌ 获取隐私配置失败！")

print("\n=== 测试完成 ===")
