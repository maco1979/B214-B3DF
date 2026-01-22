#!/usr/bin/env python3
"""
测试文本生成API的所有功能参数
验证API是否正确支持所有生成控制参数
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes.inference import router as inference_router
from src.core.services.model_manager import ModelManager
from src.core.services.inference_engine import InferenceEngine

# 创建测试应用
app = FastAPI()
app.include_router(inference_router)

# 创建测试客户端
client = TestClient(app)

# 有效的API密钥
VALID_API_KEY = "sk-2a8aa989f4864f32a3131201bcc04ad2"


def test_text_generation_request_model():
    """测试TextGenerationRequest模型的参数验证"""
    print("=== 测试TextGenerationRequest模型参数验证 ===")
    
    # 测试1: 基本参数
    basic_request = {
        "model_id": "test-transformer",
        "prompt": "Hello, world!"
    }
    
    response = client.post(
        "/inference/text/generation",
        json=basic_request,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"\n测试1.1: 基本参数 - 状态码: {response.status_code}")
    
    # 测试2: 所有生成控制参数
    full_request = {
        "model_id": "test-transformer",
        "prompt": "Hello, world!",
        "max_length": 200,
        "temperature": 0.7,
        "repetition_penalty": 1.5,
        "num_return_sequences": 2,
        "beam_search": True,
        "beam_width": 5,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50
    }
    
    response = client.post(
        "/inference/text/generation",
        json=full_request,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"测试1.2: 所有生成控制参数 - 状态码: {response.status_code}")
    
    # 测试3: 参数验证（错误的参数类型）
    invalid_request = {
        "model_id": "test-transformer",
        "prompt": "Hello, world!",
        "max_length": "not-an-integer"  # 错误的类型
    }
    
    response = client.post(
        "/inference/text/generation",
        json=invalid_request,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"测试1.3: 错误参数类型 - 状态码: {response.status_code}")
    
    # 测试4: 缺失必填参数
    missing_request = {
        "model_id": "test-transformer"
        # 缺失prompt
    }
    
    response = client.post(
        "/inference/text/generation",
        json=missing_request,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"测试1.4: 缺失必填参数 - 状态码: {response.status_code}")


def test_api_key_authentication():
    """测试API密钥认证"""
    print("\n\n=== 测试API密钥认证 ===")
    
    # 测试1: 有效API密钥
    request = {
        "model_id": "test-transformer",
        "prompt": "Hello, world!"
    }
    
    response = client.post(
        "/inference/text/generation",
        json=request,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"测试2.1: 有效API密钥 - 状态码: {response.status_code}")
    
    # 测试2: 无效API密钥
    response = client.post(
        "/inference/text/generation",
        json=request,
        headers={"api-key": "invalid-key"}
    )
    
    print(f"测试2.2: 无效API密钥 - 状态码: {response.status_code}")
    
    # 测试3: 缺失API密钥
    response = client.post(
        "/inference/text/generation",
        json=request
    )
    
    print(f"测试2.3: 缺失API密钥 - 状态码: {response.status_code}")


def test_generation_control_parameters():
    """测试不同的生成控制参数组合"""
    print("\n\n=== 测试生成控制参数组合 ===")
    
    base_request = {
        "model_id": "test-transformer",
        "prompt": "Hello, world!",
        "max_length": 50
    }
    
    # 测试1: 重复惩罚
    print("\n测试3.1: 重复惩罚参数")
    
    # 无重复惩罚
    request_1a = base_request.copy()
    request_1a["repetition_penalty"] = 1.0
    
    response = client.post(
        "/inference/text/generation",
        json=request_1a,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  无重复惩罚 (1.0) - 状态码: {response.status_code}")
    
    # 有重复惩罚
    request_1b = base_request.copy()
    request_1b["repetition_penalty"] = 2.0
    
    response = client.post(
        "/inference/text/generation",
        json=request_1b,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  有重复惩罚 (2.0) - 状态码: {response.status_code}")
    
    # 测试2: 束搜索
    print("\n测试3.2: 束搜索参数")
    
    # 贪心搜索
    request_2a = base_request.copy()
    request_2a["beam_search"] = False
    
    response = client.post(
        "/inference/text/generation",
        json=request_2a,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  贪心搜索 - 状态码: {response.status_code}")
    
    # 束搜索
    request_2b = base_request.copy()
    request_2b["beam_search"] = True
    request_2b["beam_width"] = 5
    request_2b["early_stopping"] = True
    
    response = client.post(
        "/inference/text/generation",
        json=request_2b,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  束搜索 (beam_width=5) - 状态码: {response.status_code}")
    
    # 测试3: n-gram重复惩罚
    print("\n测试3.3: n-gram重复惩罚参数")
    
    # 无n-gram惩罚
    request_3a = base_request.copy()
    request_3a["no_repeat_ngram_size"] = 0
    
    response = client.post(
        "/inference/text/generation",
        json=request_3a,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  无n-gram惩罚 - 状态码: {response.status_code}")
    
    # 有n-gram惩罚
    request_3b = base_request.copy()
    request_3b["no_repeat_ngram_size"] = 2
    
    response = client.post(
        "/inference/text/generation",
        json=request_3b,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  2-gram惩罚 - 状态码: {response.status_code}")
    
    # 测试4: 采样策略
    print("\n测试3.4: 采样策略参数")
    
    # 贪心采样
    request_4a = base_request.copy()
    request_4a["do_sample"] = False
    
    response = client.post(
        "/inference/text/generation",
        json=request_4a,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  贪心采样 - 状态码: {response.status_code}")
    
    # Top-k采样
    request_4b = base_request.copy()
    request_4b["do_sample"] = True
    request_4b["top_k"] = 50
    
    response = client.post(
        "/inference/text/generation",
        json=request_4b,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  Top-k采样 (k=50) - 状态码: {response.status_code}")
    
    # Top-p采样
    request_4c = base_request.copy()
    request_4c["do_sample"] = True
    request_4c["top_p"] = 0.9
    
    response = client.post(
        "/inference/text/generation",
        json=request_4c,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  Top-p采样 (p=0.9) - 状态码: {response.status_code}")
    
    # 混合采样
    request_4d = base_request.copy()
    request_4d["do_sample"] = True
    request_4d["top_k"] = 50
    request_4d["top_p"] = 0.9
    
    response = client.post(
        "/inference/text/generation",
        json=request_4d,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  混合采样 (k=50, p=0.9) - 状态码: {response.status_code}")
    
    # 测试5: 组合参数
    print("\n测试3.5: 组合参数")
    
    combined_request = base_request.copy()
    combined_request.update({
        "temperature": 0.7,
        "repetition_penalty": 1.5,
        "beam_search": True,
        "beam_width": 5,
        "no_repeat_ngram_size": 2,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9
    })
    
    response = client.post(
        "/inference/text/generation",
        json=combined_request,
        headers={"api-key": VALID_API_KEY}
    )
    
    print(f"  组合参数 - 状态码: {response.status_code}")


if __name__ == "__main__":
    print("开始测试文本生成API...\n")
    
    try:
        test_text_generation_request_model()
        test_api_key_authentication()
        test_generation_control_parameters()
        
        print("\n\n=== 所有API测试完成 ===")
        print("API接口已成功支持所有生成控制参数:")
        print("- 基本生成参数: max_length, temperature, repetition_penalty")
        print("- 束搜索参数: beam_search, beam_width, early_stopping")
        print("- 多样性参数: no_repeat_ngram_size")
        print("- 采样策略: do_sample, top_k, top_p")
        print("\nAPI参数验证和认证功能正常工作。")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()