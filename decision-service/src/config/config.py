"""
决策服务配置
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    app_name: str = "AI决策服务"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8001
    
    # 数据库配置
    database_url: str = "postgresql://user:password@localhost:5432/ai_platform"
    
    # Redis配置
    redis_url: str = "redis://localhost:6379"
    
    # 模型配置
    model_cache_size: int = 100
    model_update_interval: int = 3600
    
    # CORS配置
    cors_origins: list[str] = ["*"]
    
    # 决策服务配置
    decision_timeout: int = 30
    batch_size_limit: int = 100
    metrics_enabled: bool = True
    api_key: str = "your-api-key-here"
    
    # 决策算法参数
    decision_threshold: float = 0.7
    max_decision_time: int = 30
    
    # 监控配置
    prometheus_port: int = 9090
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }
# 全局配置实例（在导入环境可能包含额外 env 时仍能安全构造）
try:
    settings = Settings()
except Exception:
    # 若校验失败，退回到未验证的实例以避免在导入时抛出异常
    settings = Settings.model_construct()