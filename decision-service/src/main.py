"""
决策服务主应用入口
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.config import settings
from .api.routes.decision import router as decision_router

# 配置日志
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("决策服务启动中...")
    
    # 初始化资源
    await initialize_services()
    
    yield
    
    # 关闭时执行
    logger.info("决策服务关闭中...")
    await cleanup_services()


async def initialize_services():
    """初始化服务"""
    logger.info("初始化决策服务...")
    # 这里可以初始化数据库连接、Redis连接等
    logger.info("决策服务初始化完成")


async def cleanup_services():
    """清理服务资源"""
    logger.info("清理决策服务资源...")
    # 这里可以关闭数据库连接、Redis连接等
    logger.info("资源清理完成")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    description="AI农业平台决策服务微服务",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 添加中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加请求处理时间头"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    logger.info(f"请求: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"响应: {response.status_code}")
    return response


# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"全局异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "message": str(exc)
        }
    )


# 注册路由
app.include_router(decision_router)


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"启动决策服务: {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )