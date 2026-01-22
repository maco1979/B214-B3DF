"""
AI农业平台集成模块

本模块负责将迁移学习风险控制、边缘计算部署策略与现有AI自主决策系统集成
"""

__version__ = "1.0.0"
__author__ = "AI农业平台团队"

from .migration_integration import MigrationLearningIntegration
from .edge_integration import EdgeIntegrationManager as EdgeComputingIntegration
from .decision_integration import DecisionIntegrationManager as DecisionSystemIntegration

__all__ = [
    "MigrationLearningIntegration",
    "EdgeComputingIntegration", 
    "DecisionSystemIntegration"
]