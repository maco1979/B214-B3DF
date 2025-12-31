#!/usr/bin/env python3
"""
量化API测试脚本
直接通过HTTP请求测试后端的量化功能
"""

import requests
import json
import time

# 后端服务地址
BASE_URL = "http://localhost:8000/api"

def test_quantization_flow():
    """测试完整的量化流程：创建模型 -> 量化模型 -> 量化推理"""
    print("=== 开始测试量化功能 ===")
    
    # 1. 创建一个Transformer模型
    print("\n1. 创建Transformer模型...")
    create_model_data = {
        "name": "test-quant-model",
        "description": "用于测试量化功能的Transformer模型",
        "model_type": "transformer",
        "hyperparameters": {
            "vocab_size": 30000,
            "max_seq_len": 2048,
            "d_model": 2048,
            "num_heads": 16,
            "num_layers": 24,
            "d_ff": 8192
        }
    }
    
    response = requests.post(f"{BASE_URL}/models/", json=create_model_data)
    print(f"   创建模型响应状态码: {response.status_code}")
    if response.status_code != 201:
        print(f"   创建模型失败: {response.text}")
        return False
    
    model_data = response.json()
    model_id = model_data["model_id"]
    print(f"   创建模型成功，模型ID: {model_id}")
    
    # 2. 查看模型列表，确认模型创建成功
    print(f"\n2. 查看模型列表...")
    response = requests.get(f"{BASE_URL}/models/")
    if response.status_code == 200:
        models = response.json()
        print(f"   获取到 {len(models)} 个模型")
        for model in models:
            if model["model_id"] == model_id:
                print(f"   ✅ 找到创建的模型: {model['name']} (ID: {model_id})")
                break
    
    # 3. 量化模型
    print(f"\n3. 量化模型 (ID: {model_id})...")
    quantize_data = {
        "quantization_type": "int8"
    }
    
    response = requests.post(f"{BASE_URL}/models/{model_id}/quantize", json=quantize_data)
    print(f"   量化请求响应状态码: {response.status_code}")
    if response.status_code != 200:
        print(f"   量化失败: {response.text}")
        return False
    
    quantize_result = response.json()
    print(f"   量化结果: {json.dumps(quantize_result, indent=2, ensure_ascii=False)}")
    
    if quantize_result.get("success"):
        quantized_model_id = quantize_result["model_id"]
        print(f"   ✅ 模型量化成功，量化模型ID: {quantized_model_id}")
        
        # 4. 查看量化模型的详细信息
        print(f"\n4. 获取量化模型详情...")
        response = requests.get(f"{BASE_URL}/models/{quantized_model_id}")
        if response.status_code == 200:
            quantized_model_info = response.json()
            print(f"   量化模型信息: {json.dumps(quantized_model_info, indent=2, ensure_ascii=False)}")
            
            # 检查量化信息
            if quantized_model_info.get("quantization", {}).get("enabled"):
                print(f"   ✅ 量化信息正确: {quantized_model_info['quantization']}")
        
        # 5. 使用量化模型进行推理
        print(f"\n5. 使用量化模型进行推理...")
        inference_data = {
            "input_data": {
                "text": "Hello, this is a test for quantized model inference.",
                "max_tokens": 50
            },
            "quantization_type": "int8"
        }
        
        response = requests.post(f"{BASE_URL}/models/{quantized_model_id}/inference/quantized", json=inference_data)
        print(f"   推理请求响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            inference_result = response.json()
            print(f"   推理结果: {json.dumps(inference_result, indent=2, ensure_ascii=False)}")
            
            if inference_result.get("success"):
                print(f"   ✅ 量化推理成功！")
                return True
            else:
                print(f"   ❌ 量化推理失败: {inference_result.get('error')}")
                return False
        else:
            print(f"   ❌ 推理请求失败: {response.text}")
            return False
    else:
        print(f"   ❌ 模型量化失败: {quantize_result.get('error')}")
        return False

def test_quantization_different_types():
    """测试不同类型的量化"""
    print("\n=== 测试不同类型的量化 ===")
    
    # 创建一个测试模型
    create_model_data = {
        "name": "test-quant-types",
        "description": "用于测试不同量化类型的模型",
        "model_type": "transformer"
    }
    
    response = requests.post(f"{BASE_URL}/models/", json=create_model_data)
    if response.status_code != 201:
        print(f"创建测试模型失败: {response.text}")
        return False
    
    model_id = response.json()["model_id"]
    
    # 测试不同的量化类型
    quantization_types = ["int8", "float16"]
    
    for quant_type in quantization_types:
        print(f"\n测试{quant_type}量化...")
        quantize_data = {
            "quantization_type": quant_type
        }
        
        response = requests.post(f"{BASE_URL}/models/{model_id}/quantize", json=quantize_data)
        print(f"   {quant_type}量化响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ {quant_type}量化成功！")
                print(f"   量化模型ID: {result['model_id']}")
            else:
                print(f"   ❌ {quant_type}量化失败: {result.get('error')}")
        else:
            print(f"   ❌ {quant_type}量化请求失败: {response.text}")
    
    return True

if __name__ == "__main__":
    try:
        # 测试完整的量化流程
        success1 = test_quantization_flow()
        
        # 测试不同量化类型
        success2 = test_quantization_different_types()
        
        # 总结测试结果
        print("\n=== 测试总结 ===")
        if success1 and success2:
            print("✅ 所有量化功能测试通过！")
        else:
            print("❌ 部分测试失败，需要进一步检查。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
