#!/usr/bin/env python3
"""
测试完整的联邦学习客户端注册流程
包括权限配置和协议签署等环节
"""

import requests
import json

# API基础URL
BASE_URL = "http://localhost:8001/api"

# 测试步骤
print("=== 测试完整的联邦客户端注册流程 ===\n")

# 1. 提交注册申请
print("1. 提交注册申请...")
client_id = "test_org_client_full"
registration_data = {
    "client_id": client_id,
    "entity_info": {
        "type": "organization",
        "name": "测试机构",
        "unified_social_credit_code": "91110108MA01234567",
        "registered_address": "北京市海淀区",
        "contact_person": "张三",
        "contact_phone": "13800138000"
    },
    "capability": {
        "hardware": {
            "cpu_model": "Intel Xeon E5-2690",
            "gpu_model": "NVIDIA Tesla V100",
            "memory_gb": 64,
            "storage_gb": 1024
        },
        "network": {
            "upload_bandwidth_mbps": 100,
            "download_bandwidth_mbps": 1000,
            "latency_ms": 50
        },
        "data": {
            "type": "medical_images",
            "size": 10000,
            "quality_score": 0.95
        },
        "software": {
            "federated_framework": "FedML",
            "framework_version": "1.0.0"
        }
    },
    "role": "training_node",
    "allowed_tasks": ["medical_image_classification"]
}

response = requests.post(f"{BASE_URL}/federated/clients/register", json=registration_data)
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print()

# 2. 核验客户端身份
print("2. 核验客户端身份...")
response = requests.post(f"{BASE_URL}/federated/registrations/{client_id}/verify")
print(f"  状态: {response.status_code}")

print()

# 3. 评估客户端能力
print("3. 评估客户端能力...")
response = requests.post(f"{BASE_URL}/federated/registrations/{client_id}/assess")
print(f"  状态: {response.status_code}")

print()

# 4. 配置客户端权限
print("4. 配置客户端权限...")
permission_config = {
    "role": "training_node",
    "allowed_tasks": ["medical_image_classification", "disease_prediction"],
    "permission_level": "advanced"
}
response = requests.post(f"{BASE_URL}/federated/registrations/{client_id}/configure_permissions", json=permission_config)
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print()

# 5. 签署联邦协作协议
print("5. 签署联邦协作协议...")
agreement_info = {
    "type": "federated_collaboration",
    "agreement_id": "fed_agreement_2026_001",
    "signed_by": "张三",
    "signed_at": "2026-01-08T08:00:00Z"
}
response = requests.post(f"{BASE_URL}/federated/registrations/{client_id}/sign_agreement", json=agreement_info)
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print()

# 6. 审核通过注册申请
print("6. 审核通过注册申请...")
reviewer_info = {
    "reviewer_id": "admin_001",
    "comments": "身份核验通过，能力评估达标，权限配置完成，协议已签署，批准注册"
}
response = requests.post(f"{BASE_URL}/federated/registrations/{client_id}/approve", json=reviewer_info)
print(f"  状态: {response.status_code}")
print(f"  响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

print()

# 7. 验证客户端已激活
print("7. 验证客户端已激活...")
response = requests.get(f"{BASE_URL}/federated/clients")
clients = response.json()
print(f"  状态: {response.status_code}")
print(f"  已激活客户端数量: {clients.get('total_count')}")

# 查找测试客户端
for client in clients.get("data", []):
    if client.get("client_id") == client_id:
        print(f"  测试客户端 {client_id} 已成功激活")
        print(f"  客户端状态: {client.get('status')}")
        print(f"  客户端角色: {client.get('permissions', {}).get('role')}")
        print(f"  允许的任务: {client.get('permissions', {}).get('allowed_tasks')}")
        print(f"  权限级别: {client.get('permissions', {}).get('permission_level')}")
        break
else:
    print(f"  测试客户端 {client_id} 未找到，激活失败")

print("\n=== 测试完成 ===")
