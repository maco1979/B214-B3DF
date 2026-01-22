"""配置管理模块
使用Pydantic BaseSettings统一管理应用配置
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    # API服务配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_PREFIX: str = "/api"
    
    # WebSocket配置
    WS_PORT: int = 8001
    WS_PATH: str = "/ws/camera"
    HEARTBEAT_INTERVAL: int = 5  # 心跳间隔（秒）
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]  # 开发环境允许所有域名
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # 应用配置
    APP_NAME: str = "AI平台API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 数据库配置
    DATABASE_URL: str = "sqlite:///./app.db"
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # 超时配置
    REQUEST_TIMEOUT: int = 30  # 请求超时时间（秒）
    
    # 安全配置
    SECRET_KEY: str = ""  # 从环境变量加载
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"  # 从.env文件加载配置
        case_sensitive = True  # 环境变量区分大小写
        extra = "allow"  # 允许额外的字段


# 创建全局配置实例
settings = Settings()
