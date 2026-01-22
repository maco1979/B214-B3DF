"""
边缘计算配置
管理边缘计算相关的配置参数
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

class EdgeEnvironment(Enum):
    """边缘计算环境"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class EdgeConfig:
    """边缘计算配置"""
    
    # 环境配置
    environment: EdgeEnvironment = EdgeEnvironment.DEVELOPMENT
    debug: bool = True
    
    # 边缘节点配置
    max_nodes: int = 100
    heartbeat_interval: int = 30  # 秒
    heartbeat_timeout: int = 300   # 秒
    
    # WASM运行时配置
    wasm_memory_limit: int = 128 * 1024 * 1024  # 128MB
    wasm_stack_size: int = 64 * 1024            # 64KB
    wasm_enable_simd: bool = True
    wasm_enable_threads: bool = True
    
    # 联邦学习配置
    fl_min_clients: int = 3
    fl_max_clients: int = 20
    fl_round_timeout: int = 3600  # 秒
    fl_aggregation_method: str = "fedavg"  # fedavg, fedprox, etc.
    
    # 差分隐私配置
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    
    # 负载均衡配置
    lb_weight_compute: float = 0.4
    lb_weight_network: float = 0.3
    lb_weight_load: float = 0.3
    
    # 监控配置
    metrics_retention_days: int = 30
    log_level: str = "INFO"
    
    # 网络配置
    coordinator_address: str = "localhost:8000"
    edge_node_port_range: List[int] = field(default_factory=lambda: [8001, 8100])
    
    # 存储配置
    model_cache_dir: str = "./cache/models"
    data_cache_dir: str = "./cache/data"
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保缓存目录存在
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.data_cache_dir, exist_ok=True)
        
        # 根据环境调整配置
        if self.environment == EdgeEnvironment.PRODUCTION:
            self.debug = False
            self.log_level = "WARNING"
            self.wasm_memory_limit = 256 * 1024 * 1024  # 256MB
            self.fl_min_clients = 10
            self.fl_max_clients = 50
        elif self.environment == EdgeEnvironment.STAGING:
            self.debug = False
            self.log_level = "INFO"
            self.wasm_memory_limit = 192 * 1024 * 1024  # 192MB
    
    @classmethod
    def from_env(cls) -> 'EdgeConfig':
        """从环境变量创建配置"""
        config = cls()
        
        # 环境变量覆盖
        env = os.getenv("EDGE_ENVIRONMENT", "development").lower()
        if env == "production":
            config.environment = EdgeEnvironment.PRODUCTION
        elif env == "staging":
            config.environment = EdgeEnvironment.STAGING
        
        config.debug = os.getenv("EDGE_DEBUG", "true").lower() == "true"
        config.log_level = os.getenv("EDGE_LOG_LEVEL", "INFO")
        
        # 数值配置
        try:
            config.max_nodes = int(os.getenv("EDGE_MAX_NODES", "100"))
            config.heartbeat_interval = int(os.getenv("EDGE_HEARTBEAT_INTERVAL", "30"))
            config.heartbeat_timeout = int(os.getenv("EDGE_HEARTBEAT_TIMEOUT", "300"))
            config.wasm_memory_limit = int(os.getenv("EDGE_WASM_MEMORY_LIMIT", str(config.wasm_memory_limit)))
            config.fl_min_clients = int(os.getenv("EDGE_FL_MIN_CLIENTS", "3"))
            config.fl_max_clients = int(os.getenv("EDGE_FL_MAX_CLIENTS", "20"))
            config.dp_epsilon = float(os.getenv("EDGE_DP_EPSILON", "1.0"))
        except (ValueError, TypeError):
            pass  # 使用默认值
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "max_nodes": self.max_nodes,
            "heartbeat_interval": self.heartbeat_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "wasm_memory_limit": self.wasm_memory_limit,
            "wasm_stack_size": self.wasm_stack_size,
            "wasm_enable_simd": self.wasm_enable_simd,
            "wasm_enable_threads": self.wasm_enable_threads,
            "fl_min_clients": self.fl_min_clients,
            "fl_max_clients": self.fl_max_clients,
            "fl_round_timeout": self.fl_round_timeout,
            "fl_aggregation_method": self.fl_aggregation_method,
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
            "dp_clip_norm": self.dp_clip_norm,
            "lb_weight_compute": self.lb_weight_compute,
            "lb_weight_network": self.lb_weight_network,
            "lb_weight_load": self.lb_weight_load,
            "metrics_retention_days": self.metrics_retention_days,
            "log_level": self.log_level,
            "coordinator_address": self.coordinator_address,
            "edge_node_port_range": self.edge_node_port_range,
            "model_cache_dir": self.model_cache_dir,
            "data_cache_dir": self.data_cache_dir
        }

# 全局配置实例
edge_config = EdgeConfig.from_env()