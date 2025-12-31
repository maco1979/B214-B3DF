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
import jax.numpy as jnp
import flax.linen as nn
from enum import Enum
import threading
import queue
from .services.hardware_data_collector import hardware_data_collector, HardwareDataPoint

logger = logging.getLogger(__name__)


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


class OrganicNeuralNetwork(nn.Module):
    """有机神经网络 - 支持动态结构变化的神经网络"""
    hidden_dims: Optional[List[int]] = None
    output_dim: int = 64
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    def setup(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 256]
        
        self.layers = []
        for dim in self.hidden_dims:
            self.layers.append(nn.Dense(features=dim))
        
        self.output_layer = nn.Dense(features=self.output_dim)
        self.dropout = nn.Dropout(rate=0.1)
    
    def __call__(self, x, training: bool = True):
        for layer in self.layers:
            x = layer(x)
            if self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "gelu":
                x = nn.gelu(x)
            x = self.dropout(x, deterministic=not training)
        
        x = self.output_layer(x)
        return x


class SelfEvolvingPolicy(nn.Module):
    """自演化策略网络"""
    action_space_dim: int = 10  # 默认值
    hidden_dims: Optional[List[int]] = None
    dropout_rate: float = 0.1
    
    def setup(self):
        # 使用默认值初始化隐藏层维度
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 256]
        
        self.feature_extractor = OrganicNeuralNetwork(
            hidden_dims=self.hidden_dims[:-1],
            output_dim=self.hidden_dims[-1],
            dropout_rate=0.1
        )
        self.policy_head = nn.Dense(features=self.action_space_dim)
        self.value_head = nn.Dense(features=1)
        self.dropout = nn.Dropout(rate=0.1)
     
    def __call__(self, state_features: jnp.ndarray, training: bool = True):
        """执行策略网络前向传播"""
        # 特征提取
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
        # 使用默认参数初始化，然后在需要时再初始化参数
        # 先创建实例，然后设置参数
        self.policy_network = SelfEvolvingPolicy()
        # 通过设置属性来配置网络
        self.policy_network.action_space_dim = 10
        self.policy_network.hidden_dims = [256, 512, 256]
        self.policy_network.dropout_rate = 0.1
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
        
        # 初始化
        self._initialize_core()
        
        logger.info("有机体AI核心初始化完成")
    
    def _initialize_core(self):
        """初始化AI核心"""
        try:
            import jax.random
            # 初始化策略网络参数
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
            'performance_metrics': {
                'success_rate': self.learning_system.success_rate,
                'average_reward': self.learning_system.average_reward
            },
            'active_iteration': self.iteration_task is not None,
            'iteration_interval': self.iteration_interval
        }
    
    async def evolve_network_structure(self):
        """演化网络结构"""
        self.state = AIBrainState.EVOLVING
        
        try:
            # 这里可以实现网络结构的演化逻辑
            # 例如：增加/删除层、调整神经元数量、改变激活函数等
            logger.info("执行网络结构演化")
            
            # 模拟结构演化
            new_hidden_dims = [512, 1024, 512]  # 增加网络复杂度
            self.policy_network = SelfEvolvingPolicy(
                action_space_dim=self.policy_network.action_space_dim,
                hidden_dims=new_hidden_dims,
                dropout_rate=getattr(self.policy_network, 'dropout_rate', 0.1)
            )
            
            # 重新初始化参数
            import jax.random
            dummy_state = jnp.ones(32)
            self.policy_params = self.policy_network.init(
                jax.random.PRNGKey(int(time.time())), dummy_state
            )
            
            logger.info("网络结构演化完成")
            
            self.state = AIBrainState.IDLE
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"网络结构演化失败: {e}")
            self.state = AIBrainState.IDLE
    
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


# 全局AI核心实例
organic_ai_core = OrganicAICore()


async def get_organic_ai_core():
    """获取有机体AI核心实例"""
    return organic_ai_core