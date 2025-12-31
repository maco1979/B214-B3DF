"""
分布式DCNN系统配置
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ComputeCapacity(Enum):
    """计算能力等级"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    GPU = "gpu"


class ModelArchitecture(Enum):
    """模型架构类型"""
    RESNET50 = "resnet50"
    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNET_B0 = "efficientnet_b0"
    CUSTOM_DCNN = "custom_dcnn"


@dataclass
class EdgeNodeConfig:
    """边缘节点配置"""
    id: str
    location: str
    compute_capacity: ComputeCapacity
    ip_address: str
    port: int
    available_memory: int  # MB
    available_storage: int  # GB
    network_bandwidth: float  # Mbps


@dataclass
class ModelConfig:
    """模型配置"""
    architecture: ModelArchitecture
    input_size: tuple
    num_classes: int
    pretrained_weights: str
    quantization: bool = True
    pruning: bool = False


@dataclass
class FederatedLearningConfig:
    """联邦学习配置"""
    rounds: int
    batch_size: int
    learning_rate: float
    aggregation_algorithm: str = "fedavg"
    privacy_budget: float = 1.0
    differential_privacy: bool = True


@dataclass
class BlockchainConfig:
    """区块链配置"""
    network: str
    reward_token: str
    reward_pool: int
    contract_address: str
    gas_limit: int = 300000
    confirmations: int = 3


class DistributedDCNNConfig:
    """分布式DCNN系统完整配置"""
    
    def __init__(self):
        # 边缘节点配置
        self.edge_nodes: List[EdgeNodeConfig] = [
            EdgeNodeConfig(
                id="edge_beijing_001",
                location="北京",
                compute_capacity=ComputeCapacity.HIGH,
                ip_address="192.168.1.101",
                port=8080,
                available_memory=8192,
                available_storage=500,
                network_bandwidth=1000.0
            ),
            EdgeNodeConfig(
                id="edge_shanghai_001", 
                location="上海",
                compute_capacity=ComputeCapacity.MEDIUM,
                ip_address="192.168.1.102",
                port=8080,
                available_memory=4096,
                available_storage=250,
                network_bandwidth=500.0
            ),
            EdgeNodeConfig(
                id="edge_shenzhen_001",
                location="深圳", 
                compute_capacity=ComputeCapacity.GPU,
                ip_address="192.168.1.103",
                port=8080,
                available_memory=16384,
                available_storage=1000,
                network_bandwidth=2000.0
            )
        ]
        
        # 模型配置
        self.model_config = ModelConfig(
            architecture=ModelArchitecture.RESNET50,
            input_size=(224, 224),
            num_classes=1000,
            pretrained_weights="imagenet",
            quantization=True,
            pruning=True
        )
        
        # 联邦学习配置
        self.federated_learning = FederatedLearningConfig(
            rounds=100,
            batch_size=32,
            learning_rate=0.001,
            aggregation_algorithm="fedavg",
            privacy_budget=1.0,
            differential_privacy=True
        )
        
        # 区块链配置
        self.blockchain = BlockchainConfig(
            network="ethereum",
            reward_token="PHOTON",
            reward_pool=1000000,
            contract_address="0xPHOTON_DCNN_REWARDS",
            gas_limit=300000,
            confirmations=3
        )
        
        # 系统配置
        self.system = {
            'log_level': 'INFO',
            'max_concurrent_tasks': 10,
            'health_check_interval': 30,
            'model_update_interval': 3600,
            'backup_enabled': True
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'edge_nodes': [
                {
                    'id': node.id,
                    'location': node.location,
                    'compute_capacity': node.compute_capacity.value,
                    'ip_address': node.ip_address,
                    'port': node.port,
                    'available_memory': node.available_memory,
                    'available_storage': node.available_storage,
                    'network_bandwidth': node.network_bandwidth
                }
                for node in self.edge_nodes
            ],
            'model_config': {
                'architecture': self.model_config.architecture.value,
                'input_size': self.model_config.input_size,
                'num_classes': self.model_config.num_classes,
                'pretrained_weights': self.model_config.pretrained_weights,
                'quantization': self.model_config.quantization,
                'pruning': self.model_config.pruning
            },
            'federated_learning': {
                'rounds': self.federated_learning.rounds,
                'batch_size': self.federated_learning.batch_size,
                'learning_rate': self.federated_learning.learning_rate,
                'aggregation_algorithm': self.federated_learning.aggregation_algorithm,
                'privacy_budget': self.federated_learning.privacy_budget,
                'differential_privacy': self.federated_learning.differential_privacy
            },
            'blockchain': {
                'network': self.blockchain.network,
                'reward_token': self.blockchain.reward_token,
                'reward_pool': self.blockchain.reward_pool,
                'contract_address': self.blockchain.contract_address,
                'gas_limit': self.blockchain.gas_limit,
                'confirmations': self.blockchain.confirmations
            },
            'system': self.system
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistributedDCNNConfig':
        """从字典创建配置对象"""
        config = cls()
        
        # 更新边缘节点配置
        if 'edge_nodes' in config_dict:
            config.edge_nodes = [
                EdgeNodeConfig(
                    id=node_data['id'],
                    location=node_data['location'],
                    compute_capacity=ComputeCapacity(node_data['compute_capacity']),
                    ip_address=node_data['ip_address'],
                    port=node_data['port'],
                    available_memory=node_data['available_memory'],
                    available_storage=node_data['available_storage'],
                    network_bandwidth=node_data['network_bandwidth']
                )
                for node_data in config_dict['edge_nodes']
            ]
        
        # 更新模型配置
        if 'model_config' in config_dict:
            model_data = config_dict['model_config']
            config.model_config = ModelConfig(
                architecture=ModelArchitecture(model_data['architecture']),
                input_size=tuple(model_data['input_size']),
                num_classes=model_data['num_classes'],
                pretrained_weights=model_data['pretrained_weights'],
                quantization=model_data.get('quantization', True),
                pruning=model_data.get('pruning', False)
            )
        
        # 更新联邦学习配置
        if 'federated_learning' in config_dict:
            fl_data = config_dict['federated_learning']
            config.federated_learning = FederatedLearningConfig(
                rounds=fl_data['rounds'],
                batch_size=fl_data['batch_size'],
                learning_rate=fl_data['learning_rate'],
                aggregation_algorithm=fl_data.get('aggregation_algorithm', 'fedavg'),
                privacy_budget=fl_data.get('privacy_budget', 1.0),
                differential_privacy=fl_data.get('differential_privacy', True)
            )
        
        # 更新区块链配置
        if 'blockchain' in config_dict:
            bc_data = config_dict['blockchain']
            config.blockchain = BlockchainConfig(
                network=bc_data['network'],
                reward_token=bc_data['reward_token'],
                reward_pool=bc_data['reward_pool'],
                contract_address=bc_data['contract_address'],
                gas_limit=bc_data.get('gas_limit', 300000),
                confirmations=bc_data.get('confirmations', 3)
            )
        
        # 更新系统配置
        if 'system' in config_dict:
            config.system.update(config_dict['system'])
        
        return config


# 默认配置实例
DEFAULT_CONFIG = DistributedDCNNConfig()