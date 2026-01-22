"""
AI平台主API应用
集成所有服务模块，提供统一的API接口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# 导入核心服务
from src.core.services import model_manager, inference_engine

# 导入API路由
from src.api.routes import models, ai_control, blockchain, model_manager as model_manager_router, auth

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="AI平台API",
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
    
    # 注册API路由
    app.include_router(models.router, prefix="/api")
    app.include_router(ai_control.router, prefix="/api")
    app.include_router(blockchain.router, prefix="/api")
    app.include_router(model_manager_router.router, prefix="/api")
    app.include_router(auth.router, prefix="/api")
    
    # 添加健康检查端点
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "version": "1.0.0",
            "services": {
                "model_manager": "active",
                "inference_engine": "active"
            }
        }
    
    @app.get("/")
    async def root():
        return {
            "message": "AI平台API服务运行中",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    # 应用启动事件
    @app.on_event("startup")
    async def startup_event():
        """应用启动时初始化服务"""
        print("正在初始化模型管理器...")
        init_result = await model_manager.initialize()
        if init_result["success"]:
            print("✅ 模型管理器初始化成功")
        else:
            print(f"❌ 模型管理器初始化失败: {init_result['error']}")
        
        print("AI平台API启动完成")
    
    # 应用关闭事件
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭时清理资源"""
        print("正在关闭模型管理器...")
        await model_manager.close()
        print("AI平台API已关闭")
    
    return app