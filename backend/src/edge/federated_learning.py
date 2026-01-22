"""
联邦学习框架
支持去中心化的模型训练和参数聚合
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import secrets

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np

logger = logging.getLogger(__name__)

class FLState(Enum):
    """联邦学习状态"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class FLClient:
    """联邦学习客户端"""
    client_id: str
    address: str
    is_online: bool = False
    last_seen: Optional[datetime] = None
    data_size: int = 0
    training_capability: float = 1.0  # 训练能力系数
    
@dataclass
class FLRound:
    """联邦学习轮次"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    global_model_hash: str = ""
    
@dataclass
class FLModelUpdate:
    """模型更新"""
    client_id: str
    round_id: int
    model_updates: Dict[str, jnp.ndarray]
    data_size: int
    timestamp: datetime
    signature: str = ""

class FederatedLearning:
    """联邦学习协调器"""
    
    def __init__(self, coordinator_address: str):
        self.coordinator_address = coordinator_address
        self.clients: Dict[str, FLClient] = {}
        self.rounds: Dict[int, FLRound] = {}
        self.current_round: Optional[FLRound] = None
        self.state = FLState.INITIALIZING
        
        # 模型参数
        self.global_model: Optional[Dict[str, jnp.ndarray]] = None
        self.model_updates: List[FLModelUpdate] = []
        
    async def initialize_round(self, global_model: Dict[str, jnp.ndarray], 
                             target_clients: int = 10) -> bool:
        """初始化新一轮联邦学习"""
        try:
            if self.state != FLState.INITIALIZING:
                logger.warning("只能在初始化状态下开始新轮次")
                return False
                
            # 生成轮次ID
            round_id = len(self.rounds) + 1
            
            # 选择参与客户端
            available_clients = [c for c in self.clients.values() if c.is_online]
            selected_clients = self._select_clients(available_clients, target_clients)
            
            # 创建轮次记录
            self.current_round = FLRound(
                round_id=round_id,
                start_time=datetime.now(),
                participants=[c.client_id for c in selected_clients]
            )
            
            # 计算全局模型哈希
            self.global_model = global_model
            self.current_round.global_model_hash = self._hash_model(global_model)
            
            # 发送训练请求给选中的客户端
            await self._notify_clients(selected_clients, "start_training")
            
            self.state = FLState.TRAINING
            logger.info(f"联邦学习轮次 {round_id} 初始化成功，参与客户端: {len(selected_clients)}")
            return True
            
        except Exception as e:
            logger.error(f"联邦学习轮次初始化失败: {e}")
            self.state = FLState.ERROR
            return False
    
    def _select_clients(self, clients: List[FLClient], target_count: int) -> List[FLClient]:
        """选择参与训练的客户端，优化大模型训练场景"""
        if len(clients) <= target_count:
            return clients
            
        # 基于训练能力、数据质量和模型兼容性进行选择
        weighted_clients = []
        for client in clients:
            # 计算权重：训练能力 * 数据规模 * 在线稳定性
            weight = (client.training_capability * 
                     min(client.data_size / 1000, 1.0) *
                     self._calculate_online_stability(client))
            
            # 对于大模型训练，增加内存和计算资源权重
            # 这里假设客户端信息中包含内存和计算资源信息
            # 如果客户端支持大模型训练，增加额外权重
            weighted_clients.append((weight, client))
            
        # 按权重排序并选择
        weighted_clients.sort(key=lambda x: x[0], reverse=True)
        selected = [client for _, client in weighted_clients[:target_count]]
        
        return selected
    
    def _calculate_online_stability(self, client: FLClient) -> float:
        """计算客户端在线稳定性"""
        if not client.last_seen:
            return 0.5
            
        # 基于最后在线时间和历史稳定性计算
        time_diff = datetime.now() - client.last_seen
        if time_diff < timedelta(minutes=5):
            return 1.0
        elif time_diff < timedelta(hours=1):
            return 0.8
        else:
            return 0.3
    
    async def receive_model_update(self, update: FLModelUpdate) -> bool:
        """接收客户端模型更新"""
        try:
            if self.state != FLState.TRAINING:
                logger.warning("当前不在训练状态，无法接收模型更新")
                return False
                
            # 验证更新有效性
            if not self._validate_update(update):
                logger.warning(f"客户端 {update.client_id} 的模型更新验证失败")
                return False
                
            # 存储更新
            self.model_updates.append(update)
            
            # 检查是否所有客户端都已提交更新
            if len(self.model_updates) >= len(self.current_round.participants):
                await self._aggregate_updates()
                
            logger.info(f"收到客户端 {update.client_id} 的模型更新")
            return True
            
        except Exception as e:
            logger.error(f"接收模型更新失败: {e}")
            return False
    
    def _validate_update(self, update: FLModelUpdate) -> bool:
        """验证模型更新有效性"""
        # 检查客户端是否在参与列表中
        if update.client_id not in self.current_round.participants:
            return False
            
        # 检查轮次ID是否匹配
        if update.round_id != self.current_round.round_id:
            return False
            
        # 检查时间戳是否合理
        time_diff = datetime.now() - update.timestamp
        if time_diff > timedelta(hours=24):
            return False
            
        # 检查数据规模是否合理
        if update.data_size <= 0:
            return False
            
        return True
    
    async def _aggregate_updates(self):
        """聚合所有客户端的模型更新"""
        try:
            self.state = FLState.AGGREGATING
            
            if not self.model_updates:
                logger.warning("没有可聚合的模型更新")
                return
                
            # 计算总数据量
            total_data_size = sum(update.data_size for update in self.model_updates)
            
            # 联邦平均算法 (FedAvg)
            aggregated_updates = {}
            
            # 初始化聚合参数
            first_update = self.model_updates[0].model_updates
            for key in first_update.keys():
                aggregated_updates[key] = jnp.zeros_like(first_update[key])
            
            # 加权平均
            for update in self.model_updates:
                weight = update.data_size / total_data_size
                
                for key, param_update in update.model_updates.items():
                    aggregated_updates[key] += param_update * weight
            
            # 更新全局模型
            if self.global_model:
                for key in self.global_model.keys():
                    if key in aggregated_updates:
                        self.global_model[key] += aggregated_updates[key]
            
            # 记录轮次完成
            self.current_round.end_time = datetime.now()
            self.rounds[self.current_round.round_id] = self.current_round
            
            # 清理本轮更新
            self.model_updates.clear()
            
            self.state = FLState.COMPLETED
            logger.info(f"联邦学习轮次 {self.current_round.round_id} 聚合完成")
            
            # 通知客户端新模型可用
            await self._notify_clients(
                [self.clients[cid] for cid in self.current_round.participants],
                "model_updated"
            )
            
        except Exception as e:
            logger.error(f"模型更新聚合失败: {e}")
            self.state = FLState.ERROR
    
    async def _notify_clients(self, clients: List[FLClient], action: str):
        """通知客户端执行操作"""
        # 这里应该是实际的网络通信
        # 由于环境限制，这里模拟通知过程
        
        for client in clients:
            logger.info(f"通知客户端 {client.client_id} 执行操作: {action}")
            
            # 模拟网络延迟
            await asyncio.sleep(0.1)
    
    def _hash_model(self, model: Dict[str, jnp.ndarray]) -> str:
        """计算模型参数哈希"""
        # 将模型参数序列化为字节
        model_bytes = b""
        for key in sorted(model.keys()):
            param_bytes = model[key].tobytes()
            model_bytes += key.encode() + param_bytes
            
        return hashlib.sha256(model_bytes).hexdigest()
    
    def register_client(self, client_id: str, address: str, data_size: int = 0) -> bool:
        """注册联邦学习客户端"""
        try:
            if client_id in self.clients:
                logger.warning(f"客户端 {client_id} 已存在")
                return False
                
            client = FLClient(
                client_id=client_id,
                address=address,
                is_online=True,
                last_seen=datetime.now(),
                data_size=data_size
            )
            
            self.clients[client_id] = client
            logger.info(f"客户端 {client_id} 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"客户端注册失败: {e}")
            return False
    
    def update_client_status(self, client_id: str, is_online: bool = True) -> bool:
        """更新客户端状态"""
        if client_id not in self.clients:
            logger.warning(f"客户端 {client_id} 不存在")
            return False
            
        self.clients[client_id].is_online = is_online
        self.clients[client_id].last_seen = datetime.now()
        
        return True
    
    def get_round_statistics(self, round_id: int) -> Optional[Dict[str, Any]]:
        """获取轮次统计信息"""
        if round_id not in self.rounds:
            return None
            
        round_info = self.rounds[round_id]
        
        return {
            "round_id": round_info.round_id,
            "start_time": round_info.start_time.isoformat(),
            "end_time": round_info.end_time.isoformat() if round_info.end_time else None,
            "participants": len(round_info.participants),
            "duration": (round_info.end_time - round_info.start_time).total_seconds() if round_info.end_time else None,
            "model_hash": round_info.global_model_hash
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        online_clients = sum(1 for c in self.clients.values() if c.is_online)
        
        return {
            "state": self.state.value,
            "total_clients": len(self.clients),
            "online_clients": online_clients,
            "completed_rounds": len([r for r in self.rounds.values() if r.end_time]),
            "current_round": self.current_round.round_id if self.current_round else None
        }

class DifferentialPrivacy:
    """差分隐私保护"""
    
    @staticmethod
    def add_laplace_noise(data: jnp.ndarray, epsilon: float, sensitivity: float) -> jnp.ndarray:
        """添加拉普拉斯噪声"""
        scale = sensitivity / epsilon
        noise = jax.random.laplace(jax.random.PRNGKey(42), data.shape) * scale
        return data + noise
    
    @staticmethod
    def add_gaussian_noise(data: jnp.ndarray, epsilon: float, delta: float, 
                          sensitivity: float) -> jnp.ndarray:
        """添加高斯噪声"""
        sigma = sensitivity * jnp.sqrt(2 * jnp.log(1.25 / delta)) / epsilon
        noise = jax.random.normal(jax.random.PRNGKey(42), data.shape) * sigma
        return data + noise
    
    @staticmethod
    def clip_gradients(gradients: Dict[str, jnp.ndarray], clip_norm: float) -> Dict[str, jnp.ndarray]:
        """梯度裁剪"""
        clipped_grads = {}
        
        for key, grad in gradients.items():
            norm = jnp.linalg.norm(grad)
            if norm > clip_norm:
                clipped_grads[key] = grad * (clip_norm / norm)
            else:
                clipped_grads[key] = grad
                
        return clipped_grads