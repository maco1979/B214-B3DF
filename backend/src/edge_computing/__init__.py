"""
边缘计算部署策略模块

本模块提供农业AI模型在边缘计算环境中的部署策略，包括：
- 模型轻量化处理
- 分层部署策略
- 云边协同机制
- 资源优化调度
"""

__version__ = "1.0.0"
__author__ = "AI农业平台团队"

from .model_lightweight import ModelLightweightProcessor, ModelLightweight
from .deployment_strategy import EdgeDeploymentStrategy
from .cloud_edge_sync import CloudEdgeSyncManager
from .resource_optimizer import EdgeResourceOptimizer

__all__ = [
    "ModelLightweightProcessor",
    "ModelLightweight",
    "EdgeDeploymentStrategy", 
    "CloudEdgeSyncManager",
    "EdgeResourceOptimizer"
]