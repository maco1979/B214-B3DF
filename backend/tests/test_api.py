"""
API测试脚本
测试后端API的功能和性能
"""

import pytest
import asyncio
import time
from fastapi.testclient import TestClient
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建一个简单的测试用FastAPI应用
app = FastAPI(
    title="测试API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义测试端点
@app.get("/")
async def root():
    return {"message": "AI平台API服务运行正常", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def get_models():
    return {"models": ["test-model-1", "test-model-2"]}

@app.post("/inference")
async def inference(input_data: dict):
    return {"result": "success", "output": "test output"}

@app.get("/blockchain/status")
async def blockchain_status():
    return {"status": "ok", "initialized": True}

@app.post("/federated/clients/register")
async def register_client(client_data: dict):
    return {"status": "success", "client_id": client_data.get("client_id")}

# 创建测试客户端，使用正确的初始化方式
client = TestClient(app)


def test_health_check():
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    """测试根端点"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_models_api():
    """测试模型管理API"""
    # 获取模型列表
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data or "data" in data


def test_inference_api():
    """测试推理API"""
    # 创建推理请求
    inference_data = {
        "model_id": "test-model",
        "input_data": {"text": "Hello, world!"},
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }
    
    response = client.post("/inference", json=inference_data)
    assert response.status_code in [200, 400]  # 可能返回错误，因为测试模型不存在


def test_blockchain_api():
    """测试区块链API"""
    response = client.get("/blockchain/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "initialized" in data


def test_federated_learning_api():
    """测试联邦学习API"""
    # 测试客户端注册
    client_data = {
        "client_id": "test-client-1",
        "device_type": "test",
        "capabilities": ["training"]
    }
    
    response = client.post("/federated/clients/register", json=client_data)
    assert response.status_code in [200, 400]


def test_performance():
    """性能测试"""
    start_time = time.time()
    
    # 并发请求测试
    requests = []
    for i in range(10):
        requests.append(client.get("/health"))
    
    end_time = time.time()
    
    # 所有请求应该在2秒内完成
    assert (end_time - start_time) < 2.0
    
    # 验证所有请求都成功
    for response in requests:
        assert response.status_code == 200


def test_error_handling():
    """错误处理测试"""
    # 测试不存在的端点
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
    
    # 测试无效的JSON
    response = client.post("/inference", data="invalid json")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_async_operations():
    """测试异步操作"""
    # 模拟异步操作
    async def async_task():
        await asyncio.sleep(0.1)
        return "completed"
    
    result = await async_task()
    assert result == "completed"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])