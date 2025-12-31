"""
API模块
包含FastAPI应用和所有路由
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import models_router, inference_router, training_router, system_router, edge_router, federated_router, agriculture_router, decision_router, model_training_decision_router, resource_decision_router, decision_monitoring_router, camera_router, performance_router, blockchain_router, ai_control_router, auth_router, jepa_dtmpc_router, community_router


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    import asyncio
    # 应用Flax兼容性补丁
    from src.core.utils.flax_patch import apply_flax_patch
    apply_flax_patch()
    from src.core.services import model_manager
    
    app = FastAPI(
        title="AI项目API服务",
        description="基于JAX+Flax的最先进AI项目API服务",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 初始化模型管理器
    @app.on_event("startup")
    async def startup_event():
        await model_manager.initialize()
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应限制来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由到/api前缀下
    app.include_router(models_router, prefix="/api")
    app.include_router(inference_router, prefix="/api")
    app.include_router(training_router, prefix="/api")
    app.include_router(system_router, prefix="/api")
    app.include_router(edge_router, prefix="/api")
    
    # 仅在区块链路由可用时注册
    if blockchain_router:
        app.include_router(blockchain_router, prefix="/api")
        
    app.include_router(federated_router, prefix="/api")
    app.include_router(agriculture_router, prefix="/api")
    app.include_router(decision_router, prefix="/api")
    app.include_router(model_training_decision_router, prefix="/api")
    app.include_router(resource_decision_router, prefix="/api")
    app.include_router(decision_monitoring_router, prefix="/api")
    app.include_router(camera_router, prefix="/api")
    app.include_router(performance_router, prefix="/api")
    app.include_router(ai_control_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(jepa_dtmpc_router, prefix="/api")
    app.include_router(community_router, prefix="/api")
    
    # 根路径
    @app.get("/")
    async def root():
        return {
            "message": "AI项目API服务",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    return app


# 创建 FastAPI 应用实例
app = create_app()