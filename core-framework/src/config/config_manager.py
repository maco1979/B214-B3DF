"""
配置管理模块
提供统一的配置管理接口，支持环境变量、配置文件等多种配置源
"""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from pydantic import BaseSettings, validator


class Config(BaseSettings):
    """配置管理器"""
    
    # 应用配置
    app_name: str = "ai-agriculture-service"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # 数据库配置
    database_url: str = "postgresql://user:password@localhost/ai_agriculture"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 10
    
    # 服务配置
    service_port: int = 8000
    service_host: str = "0.0.0.0"
    service_timeout: int = 30
    
    # 安全配置
    secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    
    # 外部服务配置
    blockchain_service_url: str = "http://localhost:8001"
    model_training_service_url: str = "http://localhost:8002"
    edge_computing_service_url: str = "http://localhost:8003"
    data_service_url: str = "http://localhost:8004"
    
    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = True
    
    # 性能配置
    max_workers: int = 10
    queue_size: int = 100
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"环境必须是: {', '.join(valid_envs)}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是: {', '.join(valid_levels)}")
        return v.upper()
    
    def __init__(self, **kwargs):
        # 支持从YAML文件加载配置
        config_file = kwargs.pop("config_file", None)
        if config_file:
            self._load_from_file(config_file)
        
        super().__init__(**kwargs)
    
    def _load_from_file(self, config_file: Union[str, Path]) -> None:
        """从YAML文件加载配置"""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            
            # 更新配置
            for key, value in file_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
    
    def reload(self) -> None:
        """重新加载配置"""
        # 重新加载环境变量
        self.__init__()
    
    def is_production(self) -> bool:
        """检查是否为生产环境"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """检查是否为开发环境"""
        return self.environment == "development"
    
    def get_service_url(self, service_name: str) -> str:
        """获取服务URL"""
        service_url_mapping = {
            "blockchain": self.blockchain_service_url,
            "model_training": self.model_training_service_url,
            "edge_computing": self.edge_computing_service_url,
            "data": self.data_service_url,
        }
        
        if service_name not in service_url_mapping:
            raise ValueError(f"未知的服务: {service_name}")
        
        return service_url_mapping[service_name]


# 全局配置实例
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def set_config(config: Config) -> None:
    """设置全局配置实例"""
    global _config_instance
    _config_instance = config