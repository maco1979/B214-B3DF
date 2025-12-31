#!/usr/bin/env python3
"""
快速启动AI平台API文档服务
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 创建FastAPI应用
app = FastAPI(
    title="AI农业平台API",
    description="AI决策和模型管理平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "AI农业平台API服务运行中",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "1.0.0"
    }

@app.get("/api/models")
async def get_models():
    """获取模型列表"""
    return {
        "models": [
            {
                "id": "model-1",
                "name": "农业图像识别模型",
                "type": "CNN",
                "status": "active",
                "accuracy": 0.95
            },
            {
                "id": "model-2", 
                "name": "作物生长预测模型",
                "type": "LSTM",
                "status": "training",
                "accuracy": 0.87
            }
        ]
    }

@app.post("/api/inference")
async def run_inference(data: dict):
    """运行模型推理"""
    return {
        "prediction": "healthy_crop",
        "confidence": 0.92,
        "model_used": "agriculture-cnn-v1"
    }

@app.get("/api/blockchain/rewards")
async def get_rewards():
    """获取区块链奖励信息"""
    return {
        "total_rewards": 1500.5,
        "pending_rewards": 250.0,
        "transactions": [
            {"id": "tx1", "amount": 100.0, "type": "model_training"},
            {"id": "tx2", "amount": 50.5, "type": "data_contribution"}
        ]
    }

@app.post("/api/auth/login")
async def login(credentials: dict):
    """用户登录"""
    return {
        "access_token": "demo_token_12345",
        "user_info": {
            "id": "user-1",
            "email": "user@example.com",
            "role": "admin"
        }
    }

if __name__ == "__main__":
    print("🚀 启动AI平台API文档服务...")
    print("🌍 端口: 8000") 
    print("📚 文档地址: http://localhost:8000/docs")
    print("🔌 健康检查: http://localhost:8000/health")
    print("")
    print("服务启动中...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")