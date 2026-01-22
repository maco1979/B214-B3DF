"""
区块链配置模块
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional, ClassVar
import os


class BlockchainConfig(BaseSettings):
    """区块链配置类"""
    
    # 网络配置
    network_type: str = "development"
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 1337
    
    # 智能合约配置
    contract_addresses: Dict[str, str] = {
        "model_registry": "0x0000000000000000000000000000000000000000",
        "data_provenance": "0x0000000000000000000000000000000000000001",
        "federated_learning": "0x0000000000000000000000000000000000000002"
    }
    
    # Hyperledger Fabric配置
    fabric_network_config: Optional[Dict[str, Any]] = None
    
    # 账户配置
    private_key: str = "0x0000000000000000000000000000000000000000000000000000000000000001"
    account_address: str = "0x0000000000000000000000000000000000000000"
    
    # 交易配置
    gas_limit: int = 3000000
    gas_price: int = 20000000000
    
    # 智能合约ABI定义
    SMART_CONTRACTS: ClassVar[Dict[str, Any]] = {
        "model_registry": {
            "name": "modelregistry",
            "version": "1.0",
            "functions": ["registerModel", "updateModel", "verifyModel", "getModelHistory"]
        },
        "data_provenance": {
            "name": "dataprovenance", 
            "version": "1.0",
            "functions": ["recordUsage", "getProvenance", "verifyDataIntegrity"]
        },
        "federated_learning": {
            "name": "federatedlearning",
            "version": "1.0", 
            "functions": ["startRound", "submitUpdate", "completeRound", "getRoundStatus"]
        }
    }
    
    class Config:
        env_file = ".env"
        env_prefix = "BLOCKCHAIN_"
        extra = "ignore"
    
    @classmethod
    def get_config(cls) -> 'BlockchainConfig':
        """获取配置实例"""
        return cls()
    
    def get_contract_address(self, contract_name: str) -> str:
        """获取智能合约地址"""
        return self.contract_addresses.get(contract_name, "")
    
    def is_development_mode(self) -> bool:
        """检查是否为开发模式"""
        return self.network_type == "development"
    
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        return {
            "network_type": self.network_type,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "is_development": self.is_development_mode()
        }