#!/usr/bin/env python3
"""
测试API密钥认证功能
验证推理端点是否能够正确验证API密钥
"""

import requests
import json

base_url = "http://localhost:8000"
valid_api_key = "sk-2a8aa989f4864f32a3131201bcc04ad2"
invalid_api_key = "invalid-api-key-123"

# 创建一个测试模型
print("=== 创建测试模型 ===")
model_data = {
    "name": "test-model",
    "model_type": "ai",
    "description": "测试文本生成模型",
    "framework": "transformer",
    "hyperparameters": {
        "vocab_size": 1000,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6
    },
    "tags": ["test", "api-key-auth"]
}

create_model_response = requests.post(
    f"{base_url}/api/models/",
    json=model_data
)

if create_model_response.status_code in [200, 201]:
    model_id = create_model_response.json().get("id")
    print(f"✅ 成功创建测试模型，ID: {model_id}")
else:
    print(f"❌ 创建测试模型失败: {create_model_response.status_code} {create_model_response.text}")
    exit(1)

# 测试文本生成推理
print("\n=== 测试文本生成推理 ===")

text_generation_data = {
    "model_id": model_id,
    "prompt": "Hello, world!",
    "max_length": 50,
    "temperature": 0.7
}

# 1. 测试没有API密钥的情况
print("\n1. 测试没有API密钥的情况:")
response = requests.post(
    f"{base_url}/api/inference/text/generation",
    json=text_generation_data
)
print(f"状态码: {response.status_code}")
print(f"响应: {response.text}")

# 2. 测试使用无效API密钥的情况
print("\n2. 测试使用无效API密钥的情况:")
response = requests.post(
    f"{base_url}/api/inference/text/generation",
    json=text_generation_data,
    headers={"api-key": invalid_api_key}
)
print(f"状态码: {response.status_code}")
print(f"响应: {response.text}")

# 3. 测试使用有效API密钥的情况
print("\n3. 测试使用有效API密钥的情况:")
response = requests.post(
    f"{base_url}/api/inference/text/generation",
    json=text_generation_data,
    headers={"api-key": valid_api_key}
)
print(f"状态码: {response.status_code}")

if response.status_code == 200:
    print(f"✅ 成功! 推理结果: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
else:
    print(f"❌ 失败: {response.text}")

# 测试图像分类推理
print("\n=== 测试图像分类推理 ===")

image_classification_data = {
    "model_id": model_id,
    "image_data": [[[0.5 for _ in range(3)] for _ in range(32)] for _ in range(32)],  # 32x32x3的图像数据
    "top_k": 3
}

# 使用有效API密钥测试图像分类
response = requests.post(
    f"{base_url}/api/inference/image/classification",
    json=image_classification_data,
    headers={"api-key": valid_api_key}
)
print(f"状态码: {response.status_code}")

if response.status_code == 200:
    print(f"✅ 成功! 图像分类结果: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
else:
    print(f"❌ 失败: {response.text}")

# 测试获取推理统计信息
print("\n=== 测试获取推理统计信息 ===")

response = requests.get(
    f"{base_url}/api/inference/stats/{model_id}",
    headers={"api-key": valid_api_key}
)
print(f"状态码: {response.status_code}")

if response.status_code == 200:
    print(f"✅ 成功! 推理统计信息: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
else:
    print(f"❌ 失败: {response.text}")

print("\n=== 测试完成 ===")
