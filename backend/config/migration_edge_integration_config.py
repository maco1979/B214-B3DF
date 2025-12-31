"""
迁移学习和边缘计算集成配置管理
提供集成系统的高性能配置和部署管理
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class DeploymentEnvironment(Enum):
    """部署环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"


class OptimizationStrategy(Enum):
    """优化策略类型"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    LATENCY = "latency"
    COST = "cost"


@dataclass
class MigrationLearningConfig:
    """迁移学习配置"""
    enabled: bool = True
    source_domain_adaption: bool = True
    target_domain_validation: bool = True
    risk_control_threshold: float = 0.8
    max_accuracy_loss: float = 0.1
    batch_processing_size: int = 32
    model_compression_rate: float = 0.5
    
    # 性能优化参数
    learning_rate: float = 0.001
    training_epochs: int = 10
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    
    # 资源限制
    max_memory_usage: int = 4096  # MB
    max_processing_time: float = 300.0  # seconds
    
    def validate(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        
        if self.risk_control_threshold < 0 or self.risk_control_threshold > 1:
            errors.append("风险控制阈值必须在0-1之间")
            
        if self.max_accuracy_loss < 0 or self.max_accuracy_loss > 1:
            errors.append("最大精度损失必须在0-1之间")
            
        if self.batch_processing_size <= 0:
            errors.append("批处理大小必须大于0")
            
        if self.model_compression_rate <= 0 or self.model_compression_rate > 1:
            errors.append("模型压缩率必须在0-1之间")
            
        return errors


@dataclass
class EdgeComputingConfig:
    """边缘计算配置"""
    enabled: bool = True
    cloud_edge_sync: bool = True
    dynamic_resource_allocation: bool = True
    task_offloading_enabled: bool = True
    
    # 性能参数
    edge_latency_threshold: float = 100.0  # ms
    cloud_latency_threshold: float = 300.0  # ms
    sync_interval: int = 60  # seconds
    
    # 资源参数
    edge_node_cpu_limit: float = 0.8  # 80%
    edge_node_memory_limit: float = 0.8  # 80%
    edge_node_storage_limit: float = 0.9  # 90%
    
    # 网络参数
    bandwidth_threshold: float = 10.0  # Mbps
    network_stability_threshold: float = 0.95  # 95%
    
    def validate(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        
        if self.edge_latency_threshold <= 0:
            errors.append("边缘延迟阈值必须大于0")
            
        if self.cloud_latency_threshold <= 0:
            errors.append("云延迟阈值必须大于0")
            
        if self.sync_interval <= 0:
            errors.append("同步间隔必须大于0")
            
        if any(limit <= 0 or limit > 1 for limit in [
            self.edge_node_cpu_limit, 
            self.edge_node_memory_limit, 
            self.edge_node_storage_limit
        ]):
            errors.append("资源限制必须在0-1之间")
            
        return errors


@dataclass
class IntegrationConfig:
    """集成系统配置"""
    # 基础配置
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.PERFORMANCE
    
    # 组件配置
    migration_learning: MigrationLearningConfig = None
    edge_computing: EdgeComputingConfig = None
    
    # 性能监控配置
    performance_monitoring_enabled: bool = True
    auto_optimization_enabled: bool = True
    benchmark_testing_enabled: bool = True
    
    # 安全配置
    data_encryption_enabled: bool = True
    access_control_enabled: bool = True
    audit_logging_enabled: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if self.migration_learning is None:
            self.migration_learning = MigrationLearningConfig()
        if self.edge_computing is None:
            self.edge_computing = EdgeComputingConfig()
    
    def validate(self) -> List[str]:
        """验证整个配置"""
        errors = []
        errors.extend(self.migration_learning.validate())
        errors.extend(self.edge_computing.validate())
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = asdict(self)
        config_dict["environment"] = self.environment.value
        config_dict["optimization_strategy"] = self.optimization_strategy.value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntegrationConfig':
        """从字典创建配置"""
        # 处理枚举类型
        if "environment" in config_dict:
            config_dict["environment"] = DeploymentEnvironment(config_dict["environment"])
        if "optimization_strategy" in config_dict:
            config_dict["optimization_strategy"] = OptimizationStrategy(config_dict["optimization_strategy"])
        
        # 处理嵌套配置
        if "migration_learning" in config_dict:
            config_dict["migration_learning"] = MigrationLearningConfig(**config_dict["migration_learning"])
        if "edge_computing" in config_dict:
            config_dict["edge_computing"] = EdgeComputingConfig(**config_dict["edge_computing"])
        
        return cls(**config_dict)


class IntegrationConfigManager:
    """集成配置管理器"""
    
    def __init__(self, config_file: str = "integration_config.json"):
        self.config_file = config_file
        self.current_config: Optional[IntegrationConfig] = None
        self.config_history: List[Dict[str, Any]] = []
    
    def load_config(self, environment: Optional[DeploymentEnvironment] = None) -> IntegrationConfig:
        """加载配置"""
        
        # 如果指定了环境，使用环境特定的配置
        if environment:
            config = self._get_environment_config(environment)
        else:
            # 尝试从文件加载
            config = self._load_from_file()
        
        self.current_config = config
        self._save_to_history(config)
        
        return config
    
    def save_config(self, config: IntegrationConfig) -> bool:
        """保存配置"""
        try:
            # 验证配置
            errors = config.validate()
            if errors:
                raise ValueError(f"配置验证失败: {', '.join(errors)}")
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.current_config = config
            self._save_to_history(config)
            
            return True
            
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        if not self.current_config:
            print("没有当前配置，请先加载配置")
            return False
        
        try:
            # 创建新配置
            current_dict = self.current_config.to_dict()
            current_dict.update(updates)
            
            new_config = IntegrationConfig.from_dict(current_dict)
            
            return self.save_config(new_config)
            
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    
    def get_optimized_config(self, 
                           environment: DeploymentEnvironment,
                           strategy: OptimizationStrategy) -> IntegrationConfig:
        """获取优化配置"""
        
        base_config = self._get_environment_config(environment)
        
        # 根据优化策略调整配置
        if strategy == OptimizationStrategy.PERFORMANCE:
            base_config.migration_learning.batch_processing_size = 64
            base_config.edge_computing.edge_latency_threshold = 50.0
            base_config.auto_optimization_enabled = True
            
        elif strategy == OptimizationStrategy.ACCURACY:
            base_config.migration_learning.max_accuracy_loss = 0.05
            base_config.migration_learning.training_epochs = 20
            base_config.edge_computing.task_offloading_enabled = False
            
        elif strategy == OptimizationStrategy.RESOURCE_EFFICIENCY:
            base_config.migration_learning.model_compression_rate = 0.7
            base_config.edge_computing.edge_node_cpu_limit = 0.6
            base_config.performance_monitoring_enabled = True
            
        elif strategy == OptimizationStrategy.LATENCY:
            base_config.migration_learning.max_processing_time = 60.0
            base_config.edge_computing.edge_latency_threshold = 30.0
            base_config.edge_computing.sync_interval = 30
            
        elif strategy == OptimizationStrategy.COST:
            base_config.migration_learning.enabled = False  # 禁用迁移学习以节省成本
            base_config.edge_computing.dynamic_resource_allocation = True
            base_config.benchmark_testing_enabled = False
        
        return base_config
    
    def generate_deployment_script(self, config: IntegrationConfig) -> str:
        """生成部署脚本"""
        
        script = f"""#!/bin/bash

# 迁移学习和边缘计算集成部署脚本
# 环境: {config.environment.value}
# 优化策略: {config.optimization_strategy.value}

set -e

echo "开始部署集成系统..."

# 环境变量设置
export DEPLOYMENT_ENV={config.environment.value}
export OPTIMIZATION_STRATEGY={config.optimization_strategy.value}

# 迁移学习配置
export MIGRATION_LEARNING_ENABLED={str(config.migration_learning.enabled).lower()}
export MIGRATION_RISK_THRESHOLD={config.migration_learning.risk_control_threshold}
export MIGRATION_MAX_ACCURACY_LOSS={config.migration_learning.max_accuracy_loss}

# 边缘计算配置
export EDGE_COMPUTING_ENABLED={str(config.edge_computing.enabled).lower()}
export EDGE_LATENCY_THRESHOLD={config.edge_computing.edge_latency_threshold}
export CLOUD_LATENCY_THRESHOLD={config.edge_computing.cloud_latency_threshold}

# 性能监控
export PERFORMANCE_MONITORING={str(config.performance_monitoring_enabled).lower()}
export AUTO_OPTIMIZATION={str(config.auto_optimization_enabled).lower()}

# 安全配置
export DATA_ENCRYPTION={str(config.data_encryption_enabled).lower()}
export ACCESS_CONTROL={str(config.access_control_enabled).lower()}

echo "环境变量设置完成"

# 启动服务
echo "启动集成服务..."
python -m src.integration.api_integration &

# 启动性能监控
if [ "$PERFORMANCE_MONITORING" = "true" ]; then
    echo "启动性能监控服务..."
    python -m src.performance.performance_monitor &
fi

# 启动边缘计算网关
if [ "$EDGE_COMPUTING_ENABLED" = "true" ]; then
    echo "启动边缘计算网关..."
    python -m src.edge_computing.cloud_edge_sync &
fi

echo "部署完成!"
echo "集成系统已启动，环境: $DEPLOYMENT_ENV"
echo "优化策略: $OPTIMIZATION_STRATEGY"
"""
        
        return script
    
    def validate_deployment(self, config: IntegrationConfig) -> Dict[str, Any]:
        """验证部署配置"""
        
        validation_result = {
            "environment": config.environment.value,
            "strategy": config.optimization_strategy.value,
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # 检查配置有效性
        config_errors = config.validate()
        if config_errors:
            validation_result["valid"] = False
            validation_result["errors"].extend(config_errors)
        
        # 环境特定检查
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.data_encryption_enabled:
                validation_result["warnings"].append("生产环境建议启用数据加密")
            
            if not config.performance_monitoring_enabled:
                validation_result["warnings"].append("生产环境建议启用性能监控")
        
        # 策略特定建议
        if config.optimization_strategy == OptimizationStrategy.PERFORMANCE:
            if config.migration_learning.batch_processing_size < 32:
                validation_result["recommendations"].append(
                    "性能优化策略建议增加批处理大小"
                )
        
        elif config.optimization_strategy == OptimizationStrategy.ACCURACY:
            if config.migration_learning.max_accuracy_loss > 0.05:
                validation_result["recommendations"].append(
                    "精度优化策略建议降低最大精度损失"
                )
        
        return validation_result
    
    def _load_from_file(self) -> IntegrationConfig:
        """从文件加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                return IntegrationConfig.from_dict(config_dict)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
        
        # 返回默认配置
        return self._get_environment_config(DeploymentEnvironment.PRODUCTION)
    
    def _get_environment_config(self, environment: DeploymentEnvironment) -> IntegrationConfig:
        """获取环境特定配置"""
        
        base_config = IntegrationConfig(environment=environment)
        
        if environment == DeploymentEnvironment.DEVELOPMENT:
            base_config.migration_learning.batch_processing_size = 16
            base_config.edge_computing.edge_latency_threshold = 200.0
            base_config.performance_monitoring_enabled = True
            base_config.auto_optimization_enabled = False
            
        elif environment == DeploymentEnvironment.TESTING:
            base_config.migration_learning.max_processing_time = 600.0
            base_config.edge_computing.sync_interval = 120
            base_config.benchmark_testing_enabled = True
            
        elif environment == DeploymentEnvironment.STAGING:
            base_config.migration_learning.risk_control_threshold = 0.9
            base_config.edge_computing.dynamic_resource_allocation = True
            base_config.data_encryption_enabled = True
            
        elif environment == DeploymentEnvironment.PRODUCTION:
            base_config.migration_learning.risk_control_threshold = 0.95
            base_config.edge_computing.edge_latency_threshold = 50.0
            base_config.auto_optimization_enabled = True
            base_config.audit_logging_enabled = True
            
        elif environment == DeploymentEnvironment.EDGE:
            base_config.migration_learning.model_compression_rate = 0.8
            base_config.edge_computing.edge_node_cpu_limit = 0.7
            base_config.edge_computing.edge_node_memory_limit = 0.7
            base_config.performance_monitoring_enabled = True
        
        return base_config
    
    def _save_to_history(self, config: IntegrationConfig):
        """保存配置历史"""
        history_entry = {
            "timestamp": self._get_timestamp(),
            "config": config.to_dict(),
            "environment": config.environment.value,
            "strategy": config.optimization_strategy.value
        }
        
        self.config_history.append(history_entry)
        
        # 只保留最近10个历史记录
        if len(self.config_history) > 10:
            self.config_history = self.config_history[-10:]
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


# 默认配置管理器实例
config_manager = IntegrationConfigManager()

# 预定义配置
def get_default_config(environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION) -> IntegrationConfig:
    """获取默认配置"""
    return config_manager._get_environment_config(environment)

def get_optimized_config(environment: DeploymentEnvironment, strategy: OptimizationStrategy) -> IntegrationConfig:
    """获取优化配置"""
    return config_manager.get_optimized_config(environment, strategy)

# 工具函数
def validate_config_file(config_file: str) -> Dict[str, Any]:
    """验证配置文件"""
    manager = IntegrationConfigManager(config_file)
    config = manager.load_config()
    return manager.validate_deployment(config)

def generate_deployment_script_from_file(config_file: str) -> str:
    """从配置文件生成部署脚本"""
    manager = IntegrationConfigManager(config_file)
    config = manager.load_config()
    return manager.generate_deployment_script(config)