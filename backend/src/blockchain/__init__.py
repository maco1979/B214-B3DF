"""
Hyperledger Fabric区块链集成模块
提供AI模型训练数据溯源、模型版本管理和联邦学习协调的区块链服务
"""

from .fabric_client import FabricClient
from .smart_contracts import ModelRegistryContract, DataProvenanceContract
from .blockchain_manager import BlockchainManager
from .config import BlockchainConfig

__all__ = [
    "FabricClient",
    "ModelRegistryContract", 
    "DataProvenanceContract",
    "BlockchainManager",
    "BlockchainConfig"
]