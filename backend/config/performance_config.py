"""
AI自主决策系统性能优化配置
确保系统实现秒级实时响应能力
"""

import os
from typing import Dict, Any

class PerformanceConfig:
    """性能优化配置类"""
    
    # 决策引擎性能配置
    DECISION_ENGINE_CONFIG = {
        "max_concurrent_decisions": 10,  # 最大并发决策数
        "decision_timeout": 2.0,  # 决策超时时间（秒）
        "cache_size": 1000,  # 决策缓存大小
        "cache_ttl": 300,  # 缓存生存时间（秒）
        "batch_size": 5,  # 批量决策大小
        "optimization_level": "high"  # 优化级别
    }
    
    # 强化学习模型配置
    REINFORCEMENT_LEARNING_CONFIG = {
        "model_update_interval": 3600,  # 模型更新间隔（秒）
        "experience_buffer_size": 10000,  # 经验缓冲区大小
        "batch_training_size": 32,  # 批量训练大小
        "learning_rate": 0.001,  # 学习率
        "discount_factor": 0.99,  # 折扣因子
        "exploration_rate": 0.1,  # 探索率
        "target_update_frequency": 1000  # 目标网络更新频率
    }
    
    # API响应优化配置
    API_PERFORMANCE_CONFIG = {
        "max_request_size": 1024 * 1024,  # 最大请求大小（1MB）
        "response_timeout": 5.0,  # 响应超时时间（秒）
        "rate_limit_per_minute": 100,  # 每分钟请求限制
        "compression_enabled": True,  # 启用响应压缩
        "cors_enabled": True,  # 启用CORS
        "gzip_level": 6  # Gzip压缩级别
    }
    
    # 数据库性能配置
    DATABASE_CONFIG = {
        "pool_size": 20,  # 连接池大小
        "max_overflow": 10,  # 最大溢出连接数
        "pool_recycle": 3600,  # 连接回收时间（秒）
        "pool_pre_ping": True,  # 预检查连接
        "echo": False,  # 是否输出SQL日志
        "query_timeout": 10.0  # 查询超时时间（秒）
    }
    
    # 缓存配置
    CACHE_CONFIG = {
        "backend": "redis",  # 缓存后端
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB", 0)),
        "password": os.getenv("REDIS_PASSWORD", ""),
        "key_prefix": "ai_decision:",  # 键前缀
        "default_timeout": 300,  # 默认超时时间（秒）
        "serializer": "json"  # 序列化器
    }
    
    # 监控和日志配置
    MONITORING_CONFIG = {
        "metrics_enabled": True,  # 启用指标收集
        "metrics_port": 9090,  # 指标端口
        "log_level": "INFO",  # 日志级别
        "log_format": "json",  # 日志格式
        "performance_thresholds": {
            "max_response_time": 1000,  # 最大响应时间阈值（ms）
            "error_rate_threshold": 0.01,  # 错误率阈值
            "throughput_threshold": 100  # 吞吐量阈值（req/s）
        }
    }
    
    # 实时优化配置
    REAL_TIME_OPTIMIZATION = {
        "async_processing": True,  # 启用异步处理
        "thread_pool_size": 10,  # 线程池大小
        "process_pool_size": 4,  # 进程池大小
        "memory_optimization": True,  # 内存优化
        "gpu_acceleration": True,  # GPU加速
        "precomputation_enabled": True  # 启用预计算
    }
    
    @classmethod
    def get_optimized_config(cls, environment: str = "production") -> Dict[str, Any]:
        """获取针对特定环境优化的配置"""
        config = {
            "decision_engine": cls.DECISION_ENGINE_CONFIG.copy(),
            "reinforcement_learning": cls.REINFORCEMENT_LEARNING_CONFIG.copy(),
            "api_performance": cls.API_PERFORMANCE_CONFIG.copy(),
            "database": cls.DATABASE_CONFIG.copy(),
            "cache": cls.CACHE_CONFIG.copy(),
            "monitoring": cls.MONITORING_CONFIG.copy(),
            "real_time_optimization": cls.REAL_TIME_OPTIMIZATION.copy()
        }
        
        # 根据环境调整配置
        if environment == "development":
            config["decision_engine"]["max_concurrent_decisions"] = 5
            config["decision_engine"]["cache_size"] = 100
            config["api_performance"]["rate_limit_per_minute"] = 1000
            config["database"]["pool_size"] = 10
            config["monitoring"]["log_level"] = "DEBUG"
            
        elif environment == "testing":
            config["decision_engine"]["max_concurrent_decisions"] = 2
            config["decision_engine"]["decision_timeout"] = 5.0
            config["cache"]["backend"] = "memory"
            config["real_time_optimization"]["thread_pool_size"] = 2
            
        elif environment == "production":
            # 生产环境使用最严格的优化配置
            config["decision_engine"]["optimization_level"] = "extreme"
            config["api_performance"]["response_timeout"] = 3.0
            config["database"]["pool_size"] = 50
            config["real_time_optimization"]["thread_pool_size"] = 20
            config["monitoring"]["performance_thresholds"]["max_response_time"] = 500
        
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """验证配置是否合理"""
        try:
            # 检查决策引擎配置
            decision_config = config["decision_engine"]
            assert decision_config["max_concurrent_decisions"] > 0
            assert decision_config["decision_timeout"] > 0
            assert decision_config["cache_size"] >= 0
            
            # 检查API配置
            api_config = config["api_performance"]
            assert api_config["max_request_size"] > 0
            assert api_config["response_timeout"] > 0
            
            # 检查数据库配置
            db_config = config["database"]
            assert db_config["pool_size"] > 0
            assert db_config["query_timeout"] > 0
            
            return True
            
        except (KeyError, AssertionError):
            return False
    
    @classmethod
    def generate_performance_report(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能配置报告"""
        return {
            "config_summary": {
                "total_sections": len(config),
                "optimization_level": config.get("decision_engine", {}).get("optimization_level", "unknown"),
                "cache_enabled": config.get("cache", {}).get("backend") != "none",
                "async_processing": config.get("real_time_optimization", {}).get("async_processing", False)
            },
            "performance_indicators": {
                "expected_max_concurrency": config.get("decision_engine", {}).get("max_concurrent_decisions", 0),
                "expected_response_time": config.get("api_performance", {}).get("response_timeout", 0) * 1000,
                "database_connections": config.get("database", {}).get("pool_size", 0)
            },
            "recommendations": cls._generate_recommendations(config)
        }
    
    @classmethod
    def _generate_recommendations(cls, config: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        decision_config = config.get("decision_engine", {})
        api_config = config.get("api_performance", {})
        
        if decision_config.get("cache_size", 0) < 100:
            recommendations.append("考虑增加决策缓存大小以提高响应速度")
        
        if api_config.get("response_timeout", 0) > 5.0:
            recommendations.append("建议降低API响应超时时间以提高实时性")
        
        if not config.get("real_time_optimization", {}).get("async_processing", False):
            recommendations.append("建议启用异步处理以提高并发性能")
        
        if len(recommendations) == 0:
            recommendations.append("当前配置已达到最优性能水平")
        
        return recommendations

# 导出默认配置
DEFAULT_CONFIG = PerformanceConfig.get_optimized_config("production")
DEVELOPMENT_CONFIG = PerformanceConfig.get_optimized_config("development")
TESTING_CONFIG = PerformanceConfig.get_optimized_config("testing")