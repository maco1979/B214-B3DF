#!/usr/bin/env python3
"""
测试推理模型大脑API
验证API密钥认证和推理功能是否正常工作
"""

import requests
import json

def test_api_key_auth():
    """测试API密钥认证"""
    print("=== 测试API密钥认证 ===")
    
    # API端点
    url = "http://localhost:8000/api/inference/text-generation"
    
    # 无效的API密钥
    print("测试无效API密钥...")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "invalid-key"
    }
    payload = {
        "model_id": "test-model_v1",
        "prompt": "Hello, world!",
        "parameters": {
            "max_length": 50,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 401:
            print("✅ 无效API密钥测试通过: 返回401 Unauthorized")
        else:
            print(f"❌ 无效API密钥测试失败: 返回状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 有效的API密钥
    print("测试有效API密钥...")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "sk-2a8aa989f4864f32a3131201bcc04ad2"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            print("✅ 有效API密钥测试通过: 返回200 OK")
            print(f"响应内容: {response.json()}")
        elif response.status_code == 404:
            print("⚠️  模型不存在，需要先创建模型")
        else:
            print(f"❌ 有效API密钥测试失败: 返回状态码 {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    # API端点
    url = "http://localhost:8000/api/models"
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "sk-2a8aa989f4864f32a3131201bcc04ad2"
    }
    
    # 创建transformer模型（用于文本生成）
    print("创建文本生成模型...")
    payload = {
        "name": "test-model",
        "model_type": "transformer",
        "description": "测试文本生成模型",
        "hyperparameters": {
            "vocab_size": 30000,
            "max_seq_len": 2048,
            "d_model": 2048,
            "num_heads": 16,
            "num_layers": 24,
            "d_ff": 8192
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 201:
            print("✅ 文本生成模型创建成功")
            model_info = response.json()
            print(f"模型ID: {model_info['model_id']}")
            return model_info['model_id']
        else:
            print(f"❌ 文本生成模型创建失败: 返回状态码 {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 创建vision模型（用于图像分类）
    print("创建图像分类模型...")
    payload = {
        "name": "test-vision-model",
        "model_type": "vision",
        "description": "测试图像分类模型",
        "hyperparameters": {
            "image_size": 224,
            "image_channels": 3,
            "num_classes": 10,
            "base_channels": 64
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 201:
            print("✅ 图像分类模型创建成功")
            model_info = response.json()
            print(f"模型ID: {model_info['model_id']}")
        else:
            print(f"❌ 图像分类模型创建失败: 返回状态码 {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_text_generation(model_id):
    """测试文本生成功能"""
    print("\n=== 测试文本生成功能 ===")
    
    url = "http://localhost:8000/api/inference/text-generation"
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "sk-2a8aa989f4864f32a3131201bcc04ad2"
    }
    
    payload = {
        "model_id": model_id,
        "prompt": "Hello, world!",
        "parameters": {
            "max_length": 50,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            print("✅ 文本生成测试通过")
            result = response.json()
            print(f"生成的文本: {result['generated_text']}")
            print(f"生成的标记数量: {result['generated_tokens']}")
        else:
            print(f"❌ 文本生成测试失败: 返回状态码 {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_list_models():
    """测试模型列表功能"""
    print("\n=== 测试模型列表功能 ===")
    
    url = "http://localhost:8000/api/models"
    
    headers = {
        "X-API-Key": "sk-2a8aa989f4864f32a3131201bcc04ad2"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ 模型列表测试通过")
            models = response.json()
            print(f"共有 {len(models)} 个模型")
            for model in models:
                print(f"- 模型ID: {model['model_id']}, 类型: {model['model_type']}, 名称: {model['name']}")
        else:
            print(f"❌ 模型列表测试失败: 返回状态码 {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

if __name__ == "__main__":
    print("启动推理模型大脑API测试...")
    print("注意: 请确保后端服务已经启动 (python -m uvicorn src.api.main:app --reload)")
    
    # 测试API密钥认证
    test_api_key_auth()
    
    # 创建测试模型
    model_id = test_model_creation()
    
    # 测试模型列表
    test_list_models()
    
    # 如果模型创建成功，测试文本生成
    if model_id:
        test_text_generation(model_id)
    
    print("\n=== 测试完成 ===")
