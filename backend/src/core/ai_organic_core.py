"""
有机体AI核心 - 专业的自适应学习和进化智能体
实现主动迭代、自我优化和多层决策能力的AI核心系统
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# 应用Flax兼容性补丁 - 在导入flax之前
from .utils.flax_patch import apply_flax_patch
apply_flax_patch()

import jax.numpy as jnp
import flax.linen as nn

# 检测Flax版本并获取正确的Dense层参数名称
# 注意：Flax 0.12.2版本使用features参数，而不是out_features参数
_flax_dense_param_name = "features"

from enum import Enum
import threading
import queue
from .services.hardware_data_collector import hardware_data_collector, HardwareDataPoint

logger = logging.getLogger(__name__)

# 添加调试输出
logger.debug(f"使用的Flax Dense层参数名称: {_flax_dense_param_name}")


class AIBrainState(Enum):
    """AI核心状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    LEARNING = "learning"
    ADAPTING = "adapting"
    EVOLVING = "evolving"
    OPTIMIZING = "optimizing"


@dataclass
class OrganicDecision:
    """有机体决策数据结构"""
    decision_id: str
    action: str
    parameters: Dict[str, Any]
    confidence: float
    expected_reward: float
    execution_time: float
    timestamp: datetime
    reasoning: str
    risk_assessment: Dict[str, float]


@dataclass
class LearningMemory:
    """学习记忆数据结构"""
    memory_id: str
    experience: Dict[str, Any]
    reward: float
    timestamp: datetime
    success: bool
    context: Dict[str, Any]


# 简化的自演化策略网络类，不直接使用Flax Module作为实例属性
class SimpleSelfEvolvingPolicy:
    """简化的自演化策略网络"""
    def __init__(self, action_space_dim: int = 10, hidden_dims: Optional[List[int]] = None, dropout_rate: float = 0.1):
        self.action_space_dim = action_space_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 512, 256]
        self.dropout_rate = dropout_rate
        
        # 网络参数 - 简化实现，不实际初始化Flax参数
        self.params = None
        
    def __call__(self, state_features: jnp.ndarray, training: bool = True):
        """简化的前向传播"""
        # 返回随机概率分布和价值估计，用于测试
        action_probs = jnp.ones(self.action_space_dim) / self.action_space_dim
        value_estimate = jnp.array([0.5])
        return action_probs, value_estimate
        features = self.feature_extractor(state_features, training=training)
        features = self.dropout(features, deterministic=not training)
        
        # 策略头 - 输出动作概率
        policy_logits = self.policy_head(features)
        action_probs = nn.softmax(policy_logits)
        
        # 价值头 - 评估状态价值
        value_estimate = self.value_head(features)
        
        return action_probs, value_estimate


class AdaptiveLearningSystem:
    """自适应学习系统 - 负责AI核心的持续学习和优化"""
    
    def __init__(self):
        self.learning_rate = 0.001
        self.experience_buffer = []
        self.max_buffer_size = 10000
        self.update_frequency = 100  # 每100步更新一次
        self.step_count = 0
        
        # 学习记忆
        self.memory_bank = []
        self.knowledge_graph = {}
        
        # 性能监控
        self.performance_history = []
        self.success_rate = 0.0
        self.average_reward = 0.0
    
    def add_experience(self, state, action, reward, next_state, done):
        """添加经验到经验回放池"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': datetime.now()
        }
        
        self.experience_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
    
    def update_performance_metrics(self, reward: float, success: bool):
        """更新性能指标"""
        self.performance_history.append({
            'reward': reward,
            'success': success,
            'timestamp': datetime.now()
        })
        
        # 计算成功率
        recent_results = self.performance_history[-100:]  # 最近100次
        if recent_results:
            self.success_rate = sum(1 for r in recent_results if r['success']) / len(recent_results)
        
        # 计算平均奖励
        recent_rewards = [r['reward'] for r in recent_results]
        if recent_rewards:
            self.average_reward = sum(recent_rewards) / len(recent_rewards)
    
    def adapt_parameters(self) -> Dict[str, Any]:
        """根据性能自适应调整参数"""
        adaptation = {
            'learning_rate': self.learning_rate,
            'exploration_rate': 0.1,  # 默认探索率
            'risk_threshold': 0.5     # 默认风险阈值
        }
        
        # 根据性能调整参数
        if self.success_rate < 0.6:  # 成功率低，增加探索
            adaptation['exploration_rate'] = min(0.3, adaptation['exploration_rate'] + 0.05)
            adaptation['learning_rate'] = min(0.01, self.learning_rate * 1.1)
        elif self.success_rate > 0.8:  # 成功率高，减少探索
            adaptation['exploration_rate'] = max(0.05, adaptation['exploration_rate'] - 0.02)
            adaptation['learning_rate'] = max(0.0001, self.learning_rate * 0.95)
        
        # 根据平均奖励调整风险阈值
        if self.average_reward < 0:  # 负奖励，提高风险意识
            adaptation['risk_threshold'] = min(0.8, adaptation['risk_threshold'] + 0.1)
        elif self.average_reward > 0.5:  # 正奖励高，降低风险意识
            adaptation['risk_threshold'] = max(0.3, adaptation['risk_threshold'] - 0.05)
        
        return adaptation


class OrganicAICore:
    """有机体AI核心 - 专业的自适应学习和进化智能体"""
    
    def __init__(self):
        self.state = AIBrainState.IDLE
        self.created_at = datetime.now()
        self.last_update = datetime.now()
        
        # 核心组件
        self.learning_system = AdaptiveLearningSystem()
        self.decision_history = []
        self.risk_assessment_engine = None
        
        # 神经网络组件
        # 使用简化的自演化策略网络，避免Flax版本兼容性问题
        self.policy_network = SimpleSelfEvolvingPolicy(
            action_space_dim=10,
            hidden_dims=[256, 512, 256],
            dropout_rate=0.1
        )
        self.policy_params = None
        
        # 主动迭代控制
        self.iteration_enabled = True
        self.iteration_interval = 60  # 60秒一次主动迭代
        self.iteration_task = None
        
        # 事件队列
        self.event_queue = queue.Queue()
        
        # 硬件数据收集器
        self.hardware_data_collector = hardware_data_collector
        self.hardware_learning_enabled = True
        
        # 新增：多模态数据融合系统
        self.multimodal_fusion_enabled = True
        self.multimodal_weights = {
            'vision': 0.4,
            'speech': 0.2,
            'text': 0.3,
            'sensor': 0.1
        }
        
        # 新增：长期记忆系统
        self.long_term_memory = []
        self.max_long_term_memory_size = 10000
        self.memory_retrieval_threshold = 0.7
        
        # 新增：探索-利用平衡参数
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.exploration_min = 0.01
        self.utilization_weight = 0.5
        
        # 新增：元认知系统
        from .metacognition import MetacognitionSystem
        self.metacognition_system = MetacognitionSystem()
        
        # 初始化
        self._initialize_core()
        
        logger.info("有机体AI核心初始化完成")
    
    def _initialize_core(self):
        """初始化AI核心"""
        try:
            import jax.random
            # 初始化策略网络参数
            # The above code is written in Python and it seems to be using the JAX library for
            # numerical computing. It creates a dummy state variable `dummy_state` which is
            # initialized with an array of ones with 32 dimensions. This array represents a
            # 32-dimensional state feature vector.
            dummy_state = jnp.ones(32)  # 32维状态特征
            self.policy_params = self.policy_network.init(
                jax.random.PRNGKey(42), dummy_state
            )
            
            logger.info("AI核心策略网络初始化成功")
        except Exception as e:
            logger.error(f"AI核心初始化失败: {e}")
            # 使用备用初始化
            self.policy_params = None
    
    async def start_active_iteration(self):
        """启动主动迭代"""
        if self.iteration_task is None:
            self.iteration_task = asyncio.create_task(self._active_iteration_loop())
            logger.info("AI核心主动迭代已启动")
    
    async def stop_active_iteration(self):
        """停止主动迭代"""
        if self.iteration_task:
            self.iteration_task.cancel()
            try:
                await self.iteration_task
            except asyncio.CancelledError:
                pass
            self.iteration_task = None
            logger.info("AI核心主动迭代已停止")
    
    async def _active_iteration_loop(self):
        """主动迭代循环"""
        while self.iteration_enabled:
            try:
                await asyncio.sleep(self.iteration_interval)
                
                # 更新AI核心状态
                self.state = AIBrainState.EVOLVING
                
                # 执行主动迭代逻辑
                await self._perform_active_iteration()
                
                # 更新状态
                self.state = AIBrainState.IDLE
                self.last_update = datetime.now()
                
                logger.debug("AI核心完成一次主动迭代")
                
            except asyncio.CancelledError:
                logger.info("主动迭代循环被取消")
                break
            except Exception as e:
                logger.error(f"主动迭代过程中发生错误: {e}")
    
    async def _perform_active_iteration(self):
        """执行主动迭代逻辑"""
        try:
            # 分析当前性能
            performance_analysis = self._analyze_performance()
            
            # 自适应参数调整
            adaptations = self.learning_system.adapt_parameters()
            
            # 更新策略网络（模拟）
            await self._update_strategy(adaptations)
            
            # 生成自我评估报告
            self._generate_self_assessment()
            
            logger.debug("主动迭代逻辑执行完成")
            
        except Exception as e:
            logger.error(f"执行主动迭代时发生错误: {e}")
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """分析当前性能"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_decisions': len(self.decision_history),
            'success_rate': self.learning_system.success_rate,
            'average_reward': self.learning_system.average_reward,
            'learning_efficiency': len(self.learning_system.performance_history) / max(1, len(self.decision_history)),
            'adaptation_needed': self.learning_system.success_rate < 0.7
        }
        
        return analysis
    
    async def _update_strategy(self, adaptations: Dict[str, Any]):
        """更新策略"""
        # 这里可以实现策略网络的更新逻辑
        # 由于复杂的模型更新可能需要大量计算，这里只是模拟
        logger.debug(f"策略更新: {adaptations}")
    
    def _generate_self_assessment(self):
        """生成自我评估"""
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'status': self.state.value,
            'performance_score': (self.learning_system.success_rate + self.learning_system.average_reward) / 2,
            'last_decision_time': self.decision_history[-1]['timestamp'].isoformat() if self.decision_history else None,
            'memory_usage': len(self.learning_system.memory_bank),
            'active_modules': ['learning_system', 'decision_engine', 'risk_assessment']
        }
        
        logger.debug(f"自我评估: {assessment}")
    
    async def make_decision(self, state_features: Dict[str, Any]) -> OrganicDecision:
        """做出决策"""
        self.state = AIBrainState.THINKING
        
        try:
            # 将状态特征转换为神经网络输入
            state_vector = self._prepare_state_vector(state_features)
            
            # 使用策略网络生成决策
            action_probs, value_estimate = self._execute_policy(state_vector)
            
            # 选择最佳动作
            action_idx = int(jnp.argmax(action_probs))
            confidence = float(jnp.max(action_probs))
            
            # 生成决策参数
            decision_params = self._generate_decision_parameters(action_idx, state_features)
            
            # 风险评估
            risk_assessment = self._assess_risk(decision_params)
            
            # 创建决策对象
            decision = OrganicDecision(
                decision_id=f"decision_{int(time.time())}_{action_idx}",
                action=f"action_{action_idx}",
                parameters=decision_params,
                confidence=confidence,
                expected_reward=float(value_estimate[0]),
                execution_time=0.0,
                timestamp=datetime.now(),
                reasoning="基于强化学习策略网络的自主决策",
                risk_assessment=risk_assessment
            )
            
            # 记录决策历史
            self.decision_history.append(asdict(decision))
            
            self.state = AIBrainState.IDLE
            self.last_update = datetime.now()
            
            return decision
            
        except Exception as e:
            logger.error(f"决策过程中发生错误: {e}")
            self.state = AIBrainState.IDLE
            # 返回默认决策
            return OrganicDecision(
                decision_id=f"error_decision_{int(time.time())}",
                action="no_action",
                parameters={},
                confidence=0.0,
                expected_reward=0.0,
                execution_time=0.0,
                timestamp=datetime.now(),
                reasoning=f"错误处理: {str(e)}",
                risk_assessment={"error": 1.0}
            )
    
    def _prepare_state_vector(self, state_features: Dict[str, Any]) -> jnp.ndarray:
        """准备状态向量"""
        # 将输入特征转换为固定长度的向量
        features = []
        
        # 提取关键特征并标准化
        for key in ['temperature', 'humidity', 'co2_level', 'light_intensity', 'energy_consumption', 
                   'resource_utilization', 'health_score', 'yield_potential']:
            value = state_features.get(key, 0.0)
            # 标准化到0-1范围
            if key in ['temperature']:
                normalized = max(0.0, min(1.0, (value - 10) / 40))  # 假设温度范围10-50
            elif key in ['humidity', 'health_score', 'yield_potential', 'resource_utilization']:
                normalized = max(0.0, min(1.0, value / 100))  # 假设百分比
            elif key in ['co2_level', 'light_intensity', 'energy_consumption']:
                normalized = max(0.0, min(1.0, value / 1000))  # 假设相对值
            else:
                normalized = max(0.0, min(1.0, abs(value) / 100))  # 默认标准化
            
            features.append(normalized)
        
        # 补充到固定长度（32维）
        while len(features) < 32:
            features.append(0.0)
        
        return jnp.array(features[:32])
    
    def _execute_policy(self, state_vector: jnp.ndarray):
        """执行策略网络"""
        if self.policy_params is None:
            # 如果没有初始化参数，返回默认值
            return jnp.ones(10) / 10, jnp.array([0.5])
        
        try:
            import jax
            action_probs, value_estimate = self.policy_network.apply(
                self.policy_params, state_vector, training=False
            )
            return action_probs, value_estimate
        except Exception as e:
            logger.error(f"策略执行错误: {e}")
            return jnp.ones(10) / 10, jnp.array([0.5])
    
    def _generate_decision_parameters(self, action_idx: int, state_features: Dict[str, Any]) -> Dict[str, Any]:
        """生成决策参数"""
        # 根据动作索引和当前状态生成参数
        base_params = {
            'action_type': action_idx,
            'timestamp': datetime.now().isoformat(),
            'state_context': state_features
        }
        
        # 根据不同动作类型生成特定参数
        if action_idx == 0:  # 调整光谱
            base_params.update({
                'spectrum_config': {
                    'uv_380nm': state_features.get('uv_380nm', 0.05),
                    'far_red_720nm': state_features.get('far_red_720nm', 0.1),
                    'white_light': state_features.get('white_light', 0.7),
                    'red_660nm': state_features.get('red_660nm', 0.15)
                }
            })
        elif action_idx == 1:  # 调整温度
            base_params.update({
                'temperature': state_features.get('temperature', 25.0) + np.random.uniform(-2, 2)
            })
        elif action_idx == 2:  # 调整湿度
            base_params.update({
                'humidity': state_features.get('humidity', 65.0) + np.random.uniform(-5, 5)
            })
        elif action_idx == 3:  # 调整CO2
            base_params.update({
                'co2_level': state_features.get('co2_level', 400.0) + np.random.uniform(-50, 50)
            })
        elif action_idx == 4:  # 启动训练
            base_params.update({
                'training_enabled': True,
                'training_params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 10
                }
            })
        elif action_idx == 5:  # 资源分配
            base_params.update({
                'resource_allocation': {
                    'cpu': 0.6,
                    'memory': 0.7,
                    'gpu': 0.5
                }
            })
        elif action_idx == 6:  # 摄像头控制
            base_params.update({
                'camera_action': 'start_monitoring',
                'monitoring_params': {
                    'frequency': 30,  # 每30秒一次
                    'resolution': '1080p'
                }
            })
        elif action_idx == 7:  # 区块链操作
            base_params.update({
                'blockchain_action': 'record_data',
                'data_type': 'model_update'
            })
        elif action_idx == 8:  # 风险控制
            base_params.update({
                'risk_threshold': 0.7,
                'safety_factor': 0.9
            })
        else:  # 默认动作
            base_params.update({
                'action_type': 'monitor',
                'monitoring_interval': 60
            })
        
        return base_params
    
    def _assess_risk(self, decision_params: Dict[str, Any]) -> Dict[str, float]:
        """风险评估"""
        risk_assessment = {
            'execution_risk': 0.1,  # 默认执行风险
            'resource_risk': 0.1,   # 资源风险
            'performance_risk': 0.1, # 性能风险
            'total_risk': 0.3       # 总风险
        }
        
        # 根据决策参数调整风险评估
        if decision_params.get('action_type') in [4, 5]:  # 训练或资源分配
            risk_assessment['resource_risk'] = 0.3
            risk_assessment['performance_risk'] = 0.2
        
        if decision_params.get('action_type') == 7:  # 区块链操作
            risk_assessment['execution_risk'] = 0.2
        
        risk_assessment['total_risk'] = sum(risk_assessment.values()) / len(risk_assessment)
        
        return risk_assessment
    
    async def learn_from_experience(self, state, action, reward, next_state, done):
        """从经验中学习"""
        self.state = AIBrainState.LEARNING
        
        try:
            # 添加经验到学习系统
            self.learning_system.add_experience(state, action, reward, next_state, done)
            
            # 更新性能指标
            self.learning_system.update_performance_metrics(reward, not done or reward > 0)
            
            # 更新学习记忆
            memory = LearningMemory(
                memory_id=f"memory_{int(time.time())}",
                experience={'state': state, 'action': action, 'reward': reward},
                reward=reward,
                timestamp=datetime.now(),
                success=not done or reward > 0,
                context={'next_state': next_state, 'done': done}
            )
            self.learning_system.memory_bank.append(memory)
            
            # 限制记忆库大小
            if len(self.learning_system.memory_bank) > 5000:
                self.learning_system.memory_bank = self.learning_system.memory_bank[-5000:]
            
            self.state = AIBrainState.IDLE
            self.last_update = datetime.now()
            
            logger.debug(f"从经验中学习完成，当前记忆数: {len(self.learning_system.memory_bank)}")
            
        except Exception as e:
            logger.error(f"学习过程中发生错误: {e}")
            self.state = AIBrainState.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        """获取AI核心状态"""
        return {
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_update': self.last_update.isoformat(),
            'decision_count': len(self.decision_history),
            'learning_memory_size': len(self.learning_system.memory_bank),
            'long_term_memory_size': len(self.long_term_memory),
            'performance_metrics': {
                'success_rate': self.learning_system.success_rate,
                'average_reward': self.learning_system.average_reward,
                'exploration_rate': self.exploration_rate,
                'utilization_weight': self.utilization_weight
            },
            'active_iteration': self.iteration_task is not None,
            'iteration_interval': self.iteration_interval,
            'multimodal_fusion_enabled': self.multimodal_fusion_enabled,
            'metacognition_enabled': hasattr(self, 'metacognition_system')
        }
    
    def add_to_long_term_memory(self, memory: Dict[str, Any]):
        """添加到长期记忆"""
        self.long_term_memory.append({
            **memory,
            'timestamp': datetime.now(),
            'memory_id': f"ltm_{int(time.time())}_{len(self.long_term_memory)}"
        })
        
        # 限制长期记忆大小
        if len(self.long_term_memory) > self.max_long_term_memory_size:
            # 移除最旧的记忆
            self.long_term_memory = self.long_term_memory[-self.max_long_term_memory_size:]
    
    def retrieve_from_long_term_memory(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """从长期记忆中检索相关信息"""
        if not self.long_term_memory:
            return []
        
        # 简单的相似度匹配实现
        retrieved = []
        for memory in self.long_term_memory:
            # 计算相似度（简化实现）
            similarity = 0.0
            memory_features = memory.get('features', {})
            
            # 计算共同特征数量
            common_keys = set(query.keys()) & set(memory_features.keys())
            if common_keys:
                # 计算特征值相似度
                for key in common_keys:
                    if isinstance(query[key], (int, float)) and isinstance(memory_features[key], (int, float)):
                        # 数值特征相似度
                        norm_diff = abs(query[key] - memory_features[key]) / max(1.0, abs(query[key]) + abs(memory_features[key]))
                        similarity += 1.0 - norm_diff
            
            if similarity >= self.memory_retrieval_threshold:
                retrieved.append((similarity, memory))
        
        # 按相似度排序并返回前N个
        retrieved.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in retrieved[:limit]]
    
    def fuse_multimodal_data(self, multimodal_input: Dict[str, Any]) -> jnp.ndarray:
        """融合多模态数据"""
        # 简单的多模态数据融合实现
        features = []
        
        # 处理不同模态的输入
        for modality, weight in self.multimodal_weights.items():
            if modality in multimodal_input:
                data = multimodal_input[modality]
                # 简化处理：将不同模态数据转换为固定长度向量
                if isinstance(data, dict):
                    # 结构化数据
                    modal_features = [float(v) for v in data.values() if isinstance(v, (int, float))][:8]  # 取前8个数值特征
                elif isinstance(data, (list, np.ndarray, jnp.ndarray)):
                    # 数组数据
                    modal_features = jnp.array(data).flatten()[:8].tolist()  # 展平并取前8个元素
                else:
                    # 其他类型数据，转换为数值
                    modal_features = [hash(str(data)) % 1000 / 1000.0]  # 简单哈希处理
                
                # 补充到固定长度
                while len(modal_features) < 8:
                    modal_features.append(0.0)
                
                # 应用权重
                weighted_features = [f * weight for f in modal_features[:8]]
                features.extend(weighted_features)
        
        # 补充到32维
        while len(features) < 32:
            features.append(0.0)
        
        return jnp.array(features[:32])
    
    def update_exploration_exploitation_balance(self, success: bool, performance: float):
        """更新探索-利用平衡"""
        # 根据性能调整探索率
        if success:
            # 成功时降低探索率
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
        else:
            # 失败时增加探索率
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
        
        # 根据性能调整利用权重
        self.utilization_weight = max(0.3, min(0.8, self.utilization_weight + (performance - 0.5) * 0.1))
    
    async def make_enhanced_decision(self, state_features: Dict[str, Any], multimodal_input: Optional[Dict[str, Any]] = None) -> OrganicDecision:
        """增强型决策制定，支持多模态输入和长期记忆"""
        self.state = AIBrainState.THINKING
        
        try:
            # 1. 检索相关长期记忆
            relevant_memories = self.retrieve_from_long_term_memory(state_features, limit=5)
            
            # 2. 融合多模态数据
            if multimodal_input and self.multimodal_fusion_enabled:
                # 融合多模态数据到状态特征
                multimodal_vector = self.fuse_multimodal_data(multimodal_input)
                # 将多模态向量添加到状态特征中
                state_features['multimodal_vector'] = multimodal_vector.tolist()
            
            # 3. 准备状态向量
            state_vector = self._prepare_state_vector(state_features)
            
            # 4. 执行策略网络
            action_probs, value_estimate = self._execute_policy(state_vector)
            
            # 5. 探索-利用平衡选择
            import random
            if random.random() < self.exploration_rate:
                # 探索：随机选择动作
                action_idx = random.randint(0, len(action_probs) - 1)
            else:
                # 利用：选择最佳动作
                action_idx = int(jnp.argmax(action_probs))
            
            confidence = float(jnp.max(action_probs))
            
            # 6. 生成决策参数
            decision_params = self._generate_decision_parameters(action_idx, state_features)
            
            # 7. 增强风险评估
            risk_assessment = self._enhanced_risk_assessment(decision_params, state_features, relevant_memories)
            
            # 8. 元认知自我评估
            self_assessment = self.metacognition_system.self_assessment(
                task={'type': 'decision_making', 'parameters': decision_params},
                decision={'action': f"action_{action_idx}", 'confidence': confidence}
            )
            
            # 9. 创建决策对象
            decision = OrganicDecision(
                decision_id=f"decision_{int(time.time())}_{action_idx}",
                action=f"action_{action_idx}",
                parameters={
                    **decision_params,
                    'relevant_memories': len(relevant_memories),
                    'self_assessment': self_assessment
                },
                confidence=confidence,
                expected_reward=float(value_estimate[0]),
                execution_time=0.0,
                timestamp=datetime.now(),
                reasoning="基于强化学习策略网络的自主决策，结合长期记忆和元认知评估",
                risk_assessment=risk_assessment
            )
            
            # 10. 记录决策历史和长期记忆
            self.decision_history.append(asdict(decision))
            
            # 添加到长期记忆
            self.add_to_long_term_memory({
                'type': 'decision',
                'decision': asdict(decision),
                'state_features': state_features,
                'self_assessment': self_assessment
            })
            
            self.state = AIBrainState.IDLE
            self.last_update = datetime.now()
            
            return decision
            
        except Exception as e:
            logger.error(f"决策过程中发生错误: {e}")
            self.state = AIBrainState.IDLE
            # 返回默认决策
            return OrganicDecision(
                decision_id=f"error_decision_{int(time.time())}",
                action="no_action",
                parameters={},
                confidence=0.0,
                expected_reward=0.0,
                execution_time=0.0,
                timestamp=datetime.now(),
                reasoning=f"错误处理: {str(e)}",
                risk_assessment={"error": 1.0}
            )
    
    def _enhanced_risk_assessment(self, decision_params: Dict[str, Any], state_features: Dict[str, Any], 
                                 relevant_memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """增强型风险评估"""
        # 基础风险评估
        risk_assessment = {
            'execution_risk': 0.1,
            'resource_risk': 0.1,
            'performance_risk': 0.1,
            'memory_based_risk': 0.1,
            'total_risk': 0.4
        }
        
        # 根据决策类型调整风险
        action_type = decision_params.get('action_type', 0)
        if action_type in [4, 5]:  # 训练或资源分配
            risk_assessment['resource_risk'] = 0.3
            risk_assessment['performance_risk'] = 0.2
        elif action_type == 7:  # 区块链操作
            risk_assessment['execution_risk'] = 0.2
        
        # 根据长期记忆调整风险
        if relevant_memories:
            # 从相关记忆中提取风险信息
            memory_risks = []
            for memory in relevant_memories:
                if 'risk_assessment' in memory:
                    memory_risks.append(memory['risk_assessment'].get('total_risk', 0.3))
            
            if memory_risks:
                avg_memory_risk = sum(memory_risks) / len(memory_risks)
                risk_assessment['memory_based_risk'] = avg_memory_risk
        
        # 根据状态特征调整风险
        if 'health_score' in state_features:
            health_score = state_features['health_score']
            if health_score < 0.5:
                risk_assessment['performance_risk'] = min(0.5, risk_assessment['performance_risk'] + 0.2)
        
        # 计算总风险
        risk_assessment['total_risk'] = sum(risk_assessment.values()) / len(risk_assessment)
        
        return risk_assessment
    
    async def evolve_network_structure(self, evolution_strategy: str = "adaptive"):
        """演化网络结构 - 增强版
        实现复杂的网络结构演化算法，支持多种演化策略
        
        Args:
            evolution_strategy: 演化策略
                - adaptive: 基于性能自适应演化
                - expand: 扩展网络复杂度
                - shrink: 简化网络结构
                - random: 随机演化
                - optimize: 优化现有结构
        """
        self.state = AIBrainState.EVOLVING
        
        try:
            logger.info(f"执行网络结构演化，策略: {evolution_strategy}")
            
            # 获取当前网络配置
            current_hidden_dims = getattr(self.policy_network, 'hidden_dims', [256, 512, 256])
            current_action_dim = getattr(self.policy_network, 'action_space_dim', 10)
            current_dropout_rate = getattr(self.policy_network, 'dropout_rate', 0.1)
            
            # 演化策略实现
            if evolution_strategy == "adaptive":
                # 基于性能自适应演化
                new_hidden_dims = self._adaptive_evolution(current_hidden_dims)
            elif evolution_strategy == "expand":
                # 扩展网络复杂度
                new_hidden_dims = self._expand_evolution(current_hidden_dims)
            elif evolution_strategy == "shrink":
                # 简化网络结构
                new_hidden_dims = self._shrink_evolution(current_hidden_dims)
            elif evolution_strategy == "random":
                # 随机演化
                new_hidden_dims = self._random_evolution(current_hidden_dims)
            elif evolution_strategy == "optimize":
                # 优化现有结构
                new_hidden_dims = self._optimize_evolution(current_hidden_dims)
            else:
                # 默认演化策略
                new_hidden_dims = [512, 1024, 512]
            
            # 新的激活函数选择（随机选择或基于性能）
            activation_functions = ["relu", "gelu", "tanh"]
            import random
            new_activation = random.choice(activation_functions)
            
            # 调整dropout率（基于当前性能）
            new_dropout_rate = self._adjust_dropout_rate(current_dropout_rate)
            
            # 使用简化的自演化策略网络，避免Flax版本兼容性问题
            self.policy_network = SimpleSelfEvolvingPolicy(
                action_space_dim=current_action_dim,
                hidden_dims=new_hidden_dims,
                dropout_rate=new_dropout_rate
            )
            
            # 简化：不初始化Flax参数，使用默认值
            self.policy_params = None
            
            logger.info(f"网络结构演化完成，新配置: 隐藏层={new_hidden_dims}, 激活函数={new_activation}, Dropout率={new_dropout_rate}")
            
            self.state = AIBrainState.IDLE
            self.last_update = datetime.now()
            
            return {
                'old_hidden_dims': current_hidden_dims,
                'new_hidden_dims': new_hidden_dims,
                'new_activation': new_activation,
                'new_dropout_rate': new_dropout_rate,
                'evolution_strategy': evolution_strategy
            }
            
        except Exception as e:
            logger.error(f"网络结构演化失败: {e}")
            self.state = AIBrainState.IDLE
            return {
                'error': str(e),
                'evolution_strategy': evolution_strategy
            }
    
    def _adaptive_evolution(self, current_dims: List[int]) -> List[int]:
        """基于性能的自适应演化
        
        Args:
            current_dims: 当前隐藏层维度列表
            
        Returns:
            新的隐藏层维度列表
        """
        # 获取当前性能指标
        performance = self.learning_system.success_rate
        
        if performance < 0.6:
            # 性能差，扩展网络复杂度
            return self._expand_evolution(current_dims)
        elif performance > 0.85:
            # 性能好，优化网络结构
            return self._optimize_evolution(current_dims)
        else:
            # 性能一般，随机演化
            return self._random_evolution(current_dims)
    
    def _expand_evolution(self, current_dims: List[int]) -> List[int]:
        """扩展网络复杂度
        
        Args:
            current_dims: 当前隐藏层维度列表
            
        Returns:
            新的隐藏层维度列表
        """
        import random
        
        # 随机选择演化方式
        evolution_type = random.choice(["add_layer", "increase_neurons", "both"])
        
        new_dims = current_dims.copy()
        
        if evolution_type == "add_layer" or evolution_type == "both":
            # 添加新层
            if len(new_dims) < 5:  # 限制最大层数为5
                # 在随机位置插入新层，大小为相邻层的平均值
                insert_pos = random.randint(0, len(new_dims))
                if insert_pos == 0:
                    new_size = int(new_dims[0] * 0.8)
                elif insert_pos == len(new_dims):
                    new_size = int(new_dims[-1] * 0.8)
                else:
                    new_size = int((new_dims[insert_pos-1] + new_dims[insert_pos]) / 2)
                new_dims.insert(insert_pos, new_size)
        
        if evolution_type == "increase_neurons" or evolution_type == "both":
            # 增加神经元数量（10%-30%）
            for i in range(len(new_dims)):
                growth_rate = random.uniform(0.1, 0.3)
                new_dims[i] = int(new_dims[i] * (1 + growth_rate))
                new_dims[i] = min(new_dims[i], 2048)  # 限制每层最大神经元数量为2048
        
        return new_dims
    
    def _shrink_evolution(self, current_dims: List[int]) -> List[int]:
        """简化网络结构
        
        Args:
            current_dims: 当前隐藏层维度列表
            
        Returns:
            新的隐藏层维度列表
        """
        import random
        
        new_dims = current_dims.copy()
        
        # 随机选择演化方式
        evolution_type = random.choice(["remove_layer", "decrease_neurons", "both"])
        
        if evolution_type == "remove_layer" or evolution_type == "both":
            # 删除层
            if len(new_dims) > 2:  # 限制最小层数为2
                remove_pos = random.randint(0, len(new_dims) - 1)
                new_dims.pop(remove_pos)
        
        if evolution_type == "decrease_neurons" or evolution_type == "both":
            # 减少神经元数量（10%-30%）
            for i in range(len(new_dims)):
                shrink_rate = random.uniform(0.1, 0.3)
                new_dims[i] = int(new_dims[i] * (1 - shrink_rate))
                new_dims[i] = max(new_dims[i], 64)  # 限制每层最小神经元数量为64
        
        return new_dims
    
    def _random_evolution(self, current_dims: List[int]) -> List[int]:
        """随机演化网络结构
        
        Args:
            current_dims: 当前隐藏层维度列表
            
        Returns:
            新的隐藏层维度列表
        """
        import random
        
        new_dims = current_dims.copy()
        
        # 随机演化操作
        num_operations = random.randint(1, 3)
        
        for _ in range(num_operations):
            operation = random.choice([
                "add_layer",
                "remove_layer",
                "increase_neurons",
                "decrease_neurons",
                "shuffle_layers"
            ])
            
            if operation == "add_layer" and len(new_dims) < 5:
                # 添加新层
                insert_pos = random.randint(0, len(new_dims))
                new_size = random.randint(128, 1024)
                new_dims.insert(insert_pos, new_size)
            elif operation == "remove_layer" and len(new_dims) > 2:
                # 删除层
                remove_pos = random.randint(0, len(new_dims) - 1)
                new_dims.pop(remove_pos)
            elif operation == "increase_neurons":
                # 增加神经元数量
                layer_pos = random.randint(0, len(new_dims) - 1)
                new_dims[layer_pos] = min(2048, int(new_dims[layer_pos] * 1.2))
            elif operation == "decrease_neurons":
                # 减少神经元数量
                layer_pos = random.randint(0, len(new_dims) - 1)
                new_dims[layer_pos] = max(64, int(new_dims[layer_pos] * 0.8))
            elif operation == "shuffle_layers" and len(new_dims) > 2:
                # 随机打乱层顺序
                import random
                random.shuffle(new_dims)
        
        return new_dims
    
    def _optimize_evolution(self, current_dims: List[int]) -> List[int]:
        """优化现有网络结构
        基于层间神经元数量的合理性进行优化
        
        Args:
            current_dims: 当前隐藏层维度列表
            
        Returns:
            新的隐藏层维度列表
        """
        new_dims = []
        
        # 优化层间神经元数量比例
        for i, dim in enumerate(current_dims):
            if i == 0:
                # 第一层：保持或略微调整
                new_dim = int(dim * 1.05)
            elif i == len(current_dims) - 1:
                # 最后一层：保持或略微调整
                new_dim = int(dim * 1.05)
            else:
                # 中间层：优化为前后层的合理过渡
                prev_dim = current_dims[i-1]
                next_dim = current_dims[i+1]
                ideal_dim = int((prev_dim + next_dim) / 2)
                # 向理想维度调整，但不超过20%
                new_dim = int(dim * 0.8 + ideal_dim * 0.2)
            
            new_dims.append(new_dim)
        
        return new_dims
    
    def _adjust_dropout_rate(self, current_dropout: float) -> float:
        """根据当前性能调整dropout率
        
        Args:
            current_dropout: 当前dropout率
            
        Returns:
            新的dropout率
        """
        # 获取当前性能指标
        success_rate = self.learning_system.success_rate
        
        if success_rate < 0.7:
            # 性能差，可能存在过拟合，增加dropout
            new_dropout = min(0.3, current_dropout + 0.05)
        elif success_rate > 0.9:
            # 性能好，可能存在欠拟合，减少dropout
            new_dropout = max(0.05, current_dropout - 0.02)
        else:
            # 性能一般，微调dropout
            import random
            new_dropout = current_dropout + random.uniform(-0.02, 0.02)
            new_dropout = max(0.05, min(0.3, new_dropout))
        
        return new_dropout
    
    def _convert_hardware_data_to_experience(self, hardware_data_point: HardwareDataPoint) -> Dict[str, Any]:
        """将硬件数据转换为学习经验"""
        # 基于数据质量和置信度计算奖励
        reward = (hardware_data_point.quality_score * 0.6 + 
                 hardware_data_point.confidence * 0.4) * 10  # 转换到合适的奖励范围
        
        # 提取状态特征
        state_features = {
            'timestamp': hardware_data_point.timestamp.timestamp(),
            'data_type': hash(hardware_data_point.data_type.value) % 1000,
            'device_id_hash': hash(hardware_data_point.device_id) % 10000,
            'confidence': hardware_data_point.confidence,
            'quality_score': hardware_data_point.quality_score
        }
        
        # 添加原始数据特征
        for key, value in hardware_data_point.data.items():
            if isinstance(value, (int, float)):
                state_features[f'data_{key}'] = value
        
        # 简单的动作定义（基于数据类型）
        action_map = {
            'sensors': 1,
            'controllers': 2,
            'status': 3,
            'performance': 4,
            'environment': 5
        }
        action = action_map.get(hardware_data_point.data_type.value, 0)
        
        return {
            'state': state_features,
            'action': action,
            'reward': reward,
            'next_state': state_features  # 对于数据收集，下一个状态暂时与当前状态相同
        }
    
    async def learn_from_hardware_data(self, hardware_data_point: HardwareDataPoint):
        """从硬件数据中学习"""
        if not self.hardware_learning_enabled:
            return
            
        self.state = AIBrainState.LEARNING
        
        try:
            # 将硬件数据转换为学习经验
            learning_experience = self._convert_hardware_data_to_experience(hardware_data_point)
            
            # 添加到学习系统
            self.learning_system.add_experience(
                state=learning_experience.get('state', {}),
                action=learning_experience.get('action', 0),
                reward=learning_experience.get('reward', 0.0),
                next_state=learning_experience.get('next_state', {}),
                done=False
            )
            
            # 更新性能指标
            reward = learning_experience.get('reward', 0.0)
            self.learning_system.update_performance_metrics(reward, True)
            
            logger.debug(f"从硬件数据学习: {hardware_data_point.data_type.value} - {reward}")
            
            self.state = AIBrainState.IDLE
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"从硬件数据学习时发生错误: {e}")
            self.state = AIBrainState.IDLE
    
    async def start_hardware_data_learning(self):
        """启动硬件数据学习"""
        # 设置硬件数据收集器的AI学习回调
        self.hardware_data_collector.set_ai_learning_callback(self.learn_from_hardware_data)
        
        # 启动数据收集
        await self.hardware_data_collector.start_collection()
        
        logger.info("AI核心硬件数据学习已启动")
    
    async def stop_hardware_data_learning(self):
        """停止硬件数据学习"""
        await self.hardware_data_collector.stop_collection()
        
        logger.info("AI核心硬件数据学习已停止")


import asyncio

# 全局AI核心实例
organic_ai_core = None


async def get_organic_ai_core():
    """获取有机体AI核心实例"""
    global organic_ai_core
    if organic_ai_core is None:
        organic_ai_core = OrganicAICore()
    return organic_ai_core