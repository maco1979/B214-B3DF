"""
分布式DCNN联邦学习与边缘计算集成
实现隐私保护的分布式训练和边缘推理
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from .blockchain_rewards import ContributionCalculator, ContributionMetrics, ContributionType

from .core import DistributedDCNN, DCNNConfig, DistributedDCNNTrainer
from ..federated.federated_learning import FederatedLearningServer, FederatedLearningClient
from ..edge.edge_manager import EdgeManager, EdgeNode
from ..privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant


logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """训练模式"""
    FEDERATED = "federated"      # 联邦学习
    EDGE_ONLY = "edge_only"      # 仅边缘训练
    HYBRID = "hybrid"            # 混合模式


@dataclass
class FederatedDCNNConfig:
    """联邦DCNN配置"""
    dcnn_config: DCNNConfig
    training_mode: TrainingMode = TrainingMode.FEDERATED
    
    # 联邦学习参数
    client_fraction: float = 0.1          # 客户端选择比例
    rounds: int = 100                     # 训练轮次
    local_epochs: int = 5                # 本地训练轮次
    batch_size: int = 32                 # 批次大小
    
    # 隐私保护参数
    differential_privacy: bool = True    # 启用差分隐私
    epsilon: float = 1.0                 # 隐私预算
    delta: float = 1e-5                  # 隐私参数
    clip_norm: float = 1.0               # 梯度裁剪范数
    
    # 边缘计算参数
    edge_node_selection: str = "optimal"  # 边缘节点选择策略
    model_compression: bool = True        # 模型压缩
    compression_ratio: float = 0.5      # 压缩比例


@dataclass
class EdgeTrainingResult:
    """边缘训练结果"""
    node_id: str
    model_updates: Dict[str, jnp.ndarray]
    data_size: int
    training_time: float
    accuracy: float
    loss: float
    timestamp: datetime


@dataclass
class FederatedRound:
    """联邦学习轮次"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    global_model_hash: str = ""
    aggregation_metrics: Dict[str, Any] = field(default_factory=dict)


class FederatedDCNNCoordinator:
    """联邦DCNN协调器"""
    
    def __init__(self, config: FederatedDCNNConfig):
        self.config = config
        
        # 初始化全局模型
        self.global_model = DistributedDCNN(config.dcnn_config)
        self.global_trainer = DistributedDCNNTrainer(config.dcnn_config)
        
        # 联邦学习组件
        self.federated_server = FederatedLearningServer({
            'model_architecture': 'distributed_dcnn',
            'config': config.dcnn_config
        })
        
        # 边缘计算组件
        self.edge_manager = EdgeManager()
        
        # 训练状态
        self.current_round: Optional[FederatedRound] = None
        self.training_history: List[FederatedRound] = []
        self.edge_results: Dict[str, EdgeTrainingResult] = {}
        
        # 差分隐私保护
        self.differential_privacy = DifferentialPrivacy(
            epsilon=self.config.epsilon,
            delta=self.config.delta
        )
        self.privacy_accountant = PrivacyAccountant(
            target_epsilon=self.config.epsilon * 10,  # 为多轮训练预留足够的隐私预算
            target_delta=self.config.delta
        )

        # 性能监控
        self.metrics = {
            'total_rounds': 0,
            'successful_rounds': 0,
            'avg_training_time': 0.0,
            'model_accuracy_history': [],
            'privacy_budget_used': 0.0
        }
    
    async def initialize_training(self) -> bool:
        """初始化训练"""
        try:
            logger.info("初始化联邦DCNN训练")
            
            # 注册边缘节点为联邦学习客户端
            await self._register_edge_clients()
            
            # 初始化全局模型参数
            self._initialize_global_model()
            
            logger.info("联邦DCNN训练初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"训练初始化失败: {e}")
            return False
    
    async def _register_edge_clients(self):
        """注册边缘节点为联邦学习客户端"""
        for node_id, node in self.edge_manager.edge_nodes.items():
            client_info = {
                'node_type': 'edge',
                'capabilities': node.capabilities,
                'data_size': node.capabilities.get('data_size', 0),
                'compute_power': node.capabilities.get('compute_power', 1.0)
            }
            
            self.federated_server.register_client(node_id, client_info)
            logger.info(f"注册边缘节点 {node_id} 为联邦学习客户端")
    
    def _initialize_global_model(self):
        """初始化全局模型"""
        # 使用预训练权重或随机初始化
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, *self.config.dcnn_config.input_shape))
        
        self.global_model_params = self.global_model.init(rng, dummy_input)
        logger.info("全局模型初始化完成")
    
    async def start_training_round(self) -> bool:
        """开始新的训练轮次"""
        try:
            round_id = len(self.training_history) + 1
            
            # 创建轮次记录
            self.current_round = FederatedRound(
                round_id=round_id,
                start_time=datetime.now()
            )
            
            # 选择参与本轮训练的客户端
            selected_clients = self._select_clients_for_round()
            
            if not selected_clients:
                logger.warning("没有可用的客户端参与训练")
                return False
            
            # 准备发送给客户端的模型
            client_model = self._prepare_client_model()
            
            # 启动边缘训练
            training_tasks = []
            for client_id in selected_clients:
                task = self._start_edge_training(client_id, client_model)
                training_tasks.append(task)
            
            # 等待所有训练完成
            await asyncio.gather(*training_tasks)
            
            # 聚合模型更新
            success = await self._aggregate_model_updates()
            
            if success:
                self.current_round.end_time = datetime.now()
                self.training_history.append(self.current_round)
                
                self.metrics['total_rounds'] += 1
                self.metrics['successful_rounds'] += 1
                
                logger.info(f"联邦学习轮次 {round_id} 完成")
            
            return success
            
        except Exception as e:
            logger.error(f"训练轮次启动失败: {e}")
            return False
    
    def _select_clients_for_round(self) -> List[str]:
        """选择参与训练的客户端"""
        # 基于多种因素选择客户端
        available_clients = [
            cid for cid, info in self.federated_server.clients.items()
            if info['status'] == 'active'
        ]
        
        if not available_clients:
            return []
        
        # 计算客户端权重
        client_weights = []
        for client_id in available_clients:
            weight = self._calculate_client_weight(client_id)
            client_weights.append((weight, client_id))
        
        # 按权重排序
        client_weights.sort(key=lambda x: x[0], reverse=True)
        
        # 选择前N个客户端
        num_selected = max(1, int(len(available_clients) * self.config.client_fraction))
        selected = [client_id for _, client_id in client_weights[:num_selected]]
        
        return selected
    
    def _calculate_client_weight(self, client_id: str) -> float:
        """计算客户端权重"""
        client_info = self.federated_server.clients[client_id]
        
        # 基于数据量、计算能力、历史表现等计算权重
        data_weight = min(client_info['info'].get('data_size', 0) / 1000, 1.0)
        compute_weight = client_info['info'].get('compute_power', 1.0)
        
        # 历史表现权重（如果有）
        history_weight = 1.0
        if client_id in self.edge_results:
            recent_results = [
                r for r in self.edge_results.values() 
                if r.node_id == client_id
            ]
            if recent_results:
                avg_accuracy = np.mean([r.accuracy for r in recent_results])
                history_weight = avg_accuracy
        
        total_weight = data_weight * compute_weight * history_weight
        
        return total_weight
    
    def _prepare_client_model(self) -> Dict[str, Any]:
        """准备发送给客户端的模型"""
        model_data = {
            'parameters': self.global_model_params,
            'config': self.config.dcnn_config,
            'round_id': self.current_round.round_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # 如果需要模型压缩
        if self.config.model_compression:
            model_data = self._compress_model_for_edge(model_data)
        
        return model_data
    
    def _compress_model_for_edge(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """为边缘设备压缩模型"""
        # 简化实现：实际应用中需要更复杂的压缩算法
        compressed_params = {}
        
        for key, param in model_data['parameters'].items():
            # 简单的参数裁剪
            if hasattr(param, 'shape') and len(param.shape) > 1:
                # 对权重矩阵进行低秩近似
                compressed_param = self._low_rank_approximation(param)
                compressed_params[key] = compressed_param
            else:
                compressed_params[key] = param
        
        model_data['parameters'] = compressed_params
        model_data['compression_info'] = {
            'compression_ratio': self.config.compression_ratio,
            'compressed_at': datetime.now().isoformat()
        }
        
        return model_data
    
    def _low_rank_approximation(self, matrix: jnp.ndarray, rank: int = None) -> jnp.ndarray:
        """低秩近似"""
        if rank is None:
            rank = max(1, int(matrix.shape[-1] * self.config.compression_ratio))
        
        # SVD分解
        U, s, Vt = jnp.linalg.svd(matrix, full_matrices=False)
        
        # 保留前rank个奇异值
        U_approx = U[:, :rank]
        s_approx = s[:rank]
        Vt_approx = Vt[:rank, :]
        
        # 重建近似矩阵
        approx_matrix = U_approx @ jnp.diag(s_approx) @ Vt_approx
        
        return approx_matrix
    
    async def _start_edge_training(self, client_id: str, model_data: Dict[str, Any]):
        """启动边缘训练"""
        try:
            # 发送模型到边缘节点
            edge_node = self.edge_manager.edge_nodes.get(client_id)
            if not edge_node:
                logger.warning(f"边缘节点 {client_id} 不存在")
                return
            
            # 模拟边缘训练过程
            start_time = time.time()
            
            # 在实际系统中，这里应该通过网络通信发送训练任务
            training_result = await self._simulate_edge_training(client_id, model_data)
            
            training_time = time.time() - start_time
            
            # 存储训练结果
            result = EdgeTrainingResult(
                node_id=client_id,
                model_updates=training_result['updates'],
                data_size=training_result['data_size'],
                training_time=training_time,
                accuracy=training_result['accuracy'],
                loss=training_result['loss'],
                timestamp=datetime.now()
            )
            
            self.edge_results[f"{client_id}_round_{self.current_round.round_id}"] = result
            
            # 记录参与客户端
            self.current_round.participants.append(client_id)
            
            logger.info(f"边缘节点 {client_id} 训练完成，准确率: {training_result['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"边缘节点 {client_id} 训练失败: {e}")
    
    async def _simulate_edge_training(self, client_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟边缘训练"""
        # 在实际系统中，这里应该是真实的训练逻辑
        
        # 模拟训练延迟
        await asyncio.sleep(1.0)
        
        # 生成模拟的模型更新
        updates = {}
        for key, param in model_data['parameters'].items():
            if hasattr(param, 'shape'):
                # 生成随机梯度更新
                gradient = jax.random.normal(
                    jax.random.PRNGKey(int(time.time())), 
                    param.shape
                ) * 0.01
                updates[key] = gradient
        
        # 模拟训练结果
        return {
            'updates': updates,
            'data_size': 1000,  # 模拟数据量
            'accuracy': 0.85 + jax.random.uniform(jax.random.PRNGKey(42), ()) * 0.1,
            'loss': 0.2 + jax.random.uniform(jax.random.PRNGKey(43), ()) * 0.1
        }
    
    async def _aggregate_model_updates(self) -> bool:
        """聚合模型更新"""
        try:
            # 收集本轮所有更新
            updates = []
            data_sizes = []
            
            for client_id in self.current_round.participants:
                result_key = f"{client_id}_round_{self.current_round.round_id}"
                if result_key in self.edge_results:
                    result = self.edge_results[result_key]
                    updates.append(result.model_updates)
                    data_sizes.append(result.data_size)
            
            if not updates:
                logger.warning("没有可聚合的模型更新")
                return False
            
            # 应用联邦平均算法
            aggregated_update = self.global_trainer.federated_aggregation(updates, data_sizes)
            
            # 应用差分隐私（如果启用）
            if self.config.differential_privacy:
                # 对每个参数进行裁剪
                clipped_update = {}
                total_norm = 0.0
                
                # 计算总范数
                for key, update in aggregated_update.items():
                    total_norm += jnp.sum(update ** 2)
                total_norm = jnp.sqrt(total_norm)
                
                # 裁剪梯度
                clip_norm = self.config.clip_norm
                if total_norm > clip_norm:
                    scale = clip_norm / total_norm
                    for key, update in aggregated_update.items():
                        clipped_update[key] = update * scale
                else:
                    clipped_update = aggregated_update
                
                # 应用差分隐私
                aggregated_update = self._apply_differential_privacy(clipped_update)
            
            # 更新全局模型
            self._update_global_model(aggregated_update)
            
            # 记录聚合指标
            self.current_round.aggregation_metrics = {
                'participants_count': len(updates),
                'total_data_size': sum(data_sizes),
                'avg_accuracy': np.mean([r.accuracy for r in 
                    [self.edge_results[f"{cid}_round_{self.current_round.round_id}"] 
                     for cid in self.current_round.participants]]),
                'privacy_budget_used': self.metrics['privacy_budget_used'],
                'privacy_remaining': self.privacy_accountant.target_epsilon - self.privacy_accountant.epsilon_spent
            }
            
            return True
            
        except Exception as e:
            logger.error(f"模型更新聚合失败: {e}")
            return False
    
    def _apply_differential_privacy(self, updates: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """应用差分隐私保护"""
        noisy_updates = {}
        
        for key, update in updates.items():
            # 转换为numpy数组以便处理
            update_np = np.array(update)
            
            # 计算敏感度
            sensitivity = self.config.clip_norm
            
            # 使用Gaussian机制添加噪声
            noisy_update_np = self.differential_privacy.gaussian_mechanism(
                data=update_np,
                sensitivity=sensitivity
            )
            
            # 转换回jnp数组
            noisy_updates[key] = jnp.array(noisy_update_np)
            
            # 更新隐私预算使用
            self.privacy_accountant.add_step(self.config.epsilon)
            self.metrics['privacy_budget_used'] = self.privacy_accountant.epsilon_spent
        
        return noisy_updates
    
    def _update_global_model(self, aggregated_update: Dict[str, jnp.ndarray]):
        """更新全局模型"""
        # 应用更新到全局模型参数
        for key, update in aggregated_update.items():
            if key in self.global_model_params:
                self.global_model_params[key] += update
    
    async def run_training(self, total_rounds: int = None):
        """运行完整训练流程"""
        if total_rounds is None:
            total_rounds = self.config.rounds
        
        logger.info(f"开始联邦DCNN训练，总轮次: {total_rounds}")
        
        for round_num in range(1, total_rounds + 1):
            success = await self.start_training_round()
            
            if success:
                # 记录模型性能
                current_accuracy = self.current_round.aggregation_metrics['avg_accuracy']
                self.metrics['model_accuracy_history'].append(current_accuracy)
                
                logger.info(f"轮次 {round_num}/{total_rounds} 完成，准确率: {current_accuracy:.3f}")
            else:
                logger.warning(f"轮次 {round_num} 失败")
            
            # 轮次间延迟
            await asyncio.sleep(5.0)
        
        logger.info("联邦DCNN训练完成")
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'current_round': self.current_round.round_id if self.current_round else None,
            'total_rounds': len(self.training_history),
            'metrics': self.metrics,
            'edge_nodes_status': self.edge_manager.get_system_overview(),
            'federated_status': self.federated_server.get_server_status()
        }


class EdgeInferenceService:
    """边缘推理服务"""
    
    def __init__(self, coordinator: FederatedDCNNCoordinator, reward_manager=None):
        self.coordinator = coordinator
        self.edge_manager = coordinator.edge_manager
        self.reward_manager = reward_manager
        self.contribution_calculator = ContributionCalculator()
    
    async def inference_request(self, 
                              input_data: Any, 
                              model_id: str = "global",
                              edge_node: str = None) -> Dict[str, Any]:
        """推理请求"""
        try:
            # 选择边缘节点
            if edge_node is None:
                edge_node = await self._select_inference_node()
            
            if edge_node is None:
                return {
                    'success': False,
                    'error': '没有可用的边缘节点'
                }
            
            # 获取模型
            if model_id == "global":
                model = self.coordinator.global_model
                params = self.coordinator.global_model_params
            else:
                # 加载特定模型（简化实现）
                model, params = self._load_model(model_id)
            
            # 执行推理
            start_time = time.time()
            
            # 使用边缘节点的WASM运行时进行推理
            result = await self.edge_manager.inference_request(
                edge_node, "distributed_dcnn", input_data
            )
            
            inference_time = time.time() - start_time
            
            # 更新节点的平均响应时间统计
            selected_node = self.edge_manager.edge_nodes.get(edge_node)
            if selected_node:
                # 计算新的平均响应时间（指数加权移动平均）
                if hasattr(selected_node, 'avg_response_time'):
                    old_avg = selected_node.avg_response_time
                    new_avg = old_avg * 0.7 + inference_time * 0.3  # 30%权重给新测量值
                else:
                    new_avg = inference_time
                
                setattr(selected_node, 'avg_response_time', new_avg)
                
                # 记录推理时间是否满足<100ms要求
                if inference_time > 0.1:
                    logger.warning(f"推理延迟超出目标: {inference_time:.3f}s > 0.1s")
            
            # 计算并记录贡献度
            contribution_info = None
            if self.reward_manager:
                try:
                    # 计算输入数据大小（字节数）
                    data_size = len(str(input_data))  # 简化实现，实际应根据数据类型计算
                    
                    # 创建贡献指标
                    contribution_metrics = ContributionMetrics(
                        participant_id=edge_node,
                        contribution_type=ContributionType.COMPUTE_CONTRIBUTION,
                        compute_time=inference_time,
                        compute_efficiency=1.0 / max(inference_time, 0.001),  # 效率与时间成反比
                        timestamp=datetime.now()
                    )
                    
                    # 计算贡献度分数
                    contribution_score = self.contribution_calculator.calculate_contribution_score(
                        contribution_metrics
                    )
                    
                    # 记录贡献信息
                    contribution_info = {
                        'contribution_score': contribution_score,
                        'contribution_type': contribution_metrics.contribution_type.value,
                        'compute_time': inference_time,
                        'compute_efficiency': contribution_metrics.compute_efficiency
                    }
                    
                    logger.info(f"边缘节点 {edge_node} 贡献度: {contribution_score:.2f}")
                    
                    # 调用区块链奖励管理器记录贡献
                    await self.reward_manager.record_contribution(contribution_metrics)
                    
                except Exception as e:
                    logger.error(f"计算贡献度失败: {e}")
            
            response = {
                'success': True,
                'predictions': result,
                'inference_time': inference_time,
                'edge_node': edge_node,
                'model_id': model_id,
                'delay_threshold_met': inference_time < 0.1  # 标记是否满足延迟要求
            }
            
            # 如果计算了贡献度，添加到响应中
            if contribution_info:
                response['contribution_info'] = contribution_info
            
            return response
            
        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _select_inference_node(self) -> Optional[str]:
        """选择推理节点
        
        基于负载、延迟和计算能力选择最优节点，确保推理延迟<100ms
        """
        # 获取所有可用节点
        available_nodes = [
            n for n in self.edge_manager.edge_nodes.values()
            if n.status in ['idle', 'busy']
        ]
        
        if not available_nodes:
            return None
        
        # 收集节点性能指标
        node_metrics = []
        for node in available_nodes:
            # 获取节点能力
            capabilities = node.capabilities
            compute_power = capabilities.get('compute_power', 1.0)
            memory_available = capabilities.get('memory_available', 1024)
            
            # 获取节点负载
            load = node.status == 'busy'  # 简化实现，实际应获取CPU/内存使用率
            load_score = 0.5 if load else 1.0
            
            # 获取历史延迟数据（实际应从监控系统获取）
            # 简化实现，使用节点的平均响应时间
            avg_response_time = getattr(node, 'avg_response_time', 0.1)  # 默认100ms
            
            # 计算节点评分
            # 权重：计算能力(40%) + 内存可用性(30%) + 负载状态(20%) + 延迟(10%)
            score = (
                compute_power * 0.4 +
                (memory_available / 1024) * 0.3 +
                load_score * 0.2 +
                (1 / max(avg_response_time, 0.001)) * 0.1  # 延迟越低，得分越高
            )
            
            node_metrics.append((score, node.node_id, node))
        
        # 按评分排序，选择最优节点
        node_metrics.sort(key=lambda x: x[0], reverse=True)
        
        # 更新节点的使用统计（用于后续优化）
        selected_node = node_metrics[0][2]
        setattr(selected_node, 'last_used', datetime.now())
        
        return node_metrics[0][1]
    
    def _load_model(self, model_id: str):
        """加载模型（简化实现）"""
        # 在实际系统中，这里应该从模型存储加载
        return self.coordinator.global_model, self.coordinator.global_model_params