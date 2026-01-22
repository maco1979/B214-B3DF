#!/usr/bin/env python3
"""
简单的API测试脚本
用于检查FastAPI应用是否能正常启动
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI项目API服务",
    description="基于JAX+Flax的最先进AI项目API服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 根路径
@app.get("/")
async def root():
    return {
        "message": "AI项目API服务",
        "version": "1.0.0",
        "status": "ok"
    }

# 简单的健康检查端点
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2026-01-13"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)