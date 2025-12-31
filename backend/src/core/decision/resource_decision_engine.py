"""
系统资源动态分配决策引擎 - 基于强化学习的自主资源分配系统
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class ResourceAction(Enum):
    """资源分配决策动作枚举"""
    ALLOCATE_CPU = "allocate_cpu"
    ALLOCATE_MEMORY = "allocate_memory"
    ALLOCATE_GPU = "allocate_gpu"
    ALLOCATE_STORAGE = "allocate_storage"
    BALANCE_LOAD = "balance_load"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ResourceObjective(Enum):
    """资源分配决策目标枚举"""
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_COST = "minimize_cost"
    OPTIMIZE_EFFICIENCY = "optimize_efficiency"
    BALANCE_FAIRNESS = "balance_fairness"


@dataclass
class ResourceState:
    """系统资源状态"""
    # CPU资源
    cpu_total_cores: int  # CPU总核心数
    cpu_utilization: float  # CPU利用率
    cpu_available_cores: int  # 可用CPU核心数
    cpu_load_average: float  # CPU负载平均值
    
    # 内存资源
    memory_total: float  # 总内存（GB）
    memory_used: float  # 已用内存（GB）
    memory_available: float  # 可用内存（GB）
    memory_utilization: float  # 内存利用率
    
    # GPU资源
    gpu_total_memory: float  # GPU总显存（GB）
    gpu_used_memory: float  # GPU已用显存（GB）
    gpu_utilization: float  # GPU利用率
    gpu_temperature: float  # GPU温度
    
    # 存储资源
    storage_total: float  # 总存储（GB）
    storage_used: float  # 已用存储（GB）
    storage_available: float  # 可用存储（GB）
    storage_io_utilization: float  # 存储IO利用率
    
    # 网络资源
    network_bandwidth: float  # 网络带宽（Mbps）
    network_utilization: float  # 网络利用率
    network_latency: float  # 网络延迟（ms）
    
    # 系统负载
    system_load: float  # 系统负载
    active_processes: int  # 活跃进程数
    system_uptime: float  # 系统运行时间（小时）
    
    # 应用需求
    ai_training_demand: float  # AI训练需求强度
    inference_demand: float  # 推理需求强度
    blockchain_demand: float  # 区块链需求强度
    agriculture_demand: float  # 农业应用需求强度


@dataclass
class ResourceDecisionResult:
    """资源分配决策结果"""
    action: ResourceAction
    parameters: Dict[str, float]
    expected_reward: float
    confidence: float
    execution_time: float
    risk_assessment: Dict[str, float]


class ResourceRLPolicy(nn.Module):
    """资源分配强化学习策略网络"""
    hidden_dim: int = 256
    num_actions: int = 8
    
    @nn.compact
    def __call__(self, state_features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        输入状态特征，输出动作概率和价值估计
        """
        # 特征编码层
        x = nn.Dense(features=self.hidden_dim)(state_features)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        
        # 策略头
        policy_logits = nn.Dense(features=self.num_actions)(x)
        action_probs = nn.softmax(policy_logits)
        
        # 价值头
        value_estimate = nn.Dense(features=1)(x)
        
        return action_probs, value_estimate


class ResourceDecisionEngine:
    """系统资源动态分配决策引擎"""
    
    def __init__(self):
        self.rl_policy = ResourceRLPolicy()
        
        # 决策历史记录
        self.decision_history = []
        self.reward_history = []
        
        # 资源阈值参数
        self.resource_thresholds = {
            "cpu_critical": 0.9,  # CPU临界阈值
            "memory_critical": 0.85,  # 内存临界阈值
            "gpu_critical": 0.8,  # GPU临界阈值
            "storage_critical": 0.9,  # 存储临界阈值
            "load_balancing_threshold": 0.7,  # 负载均衡阈值
            "scaling_threshold": 0.75  # 扩缩容阈值
        }
        
        # 初始化模型参数
        self._initialize_policy()
    
    def _initialize_policy(self):
        """初始化策略网络参数"""
        # 使用随机初始化
        dummy_state = jnp.ones(20)  # 20维状态特征
        self.policy_params = self.rl_policy.init(
            jax.random.PRNGKey(42), dummy_state
        )
    
    def make_decision(self, 
                     current_state: ResourceState,
                     objective: ResourceObjective) -> ResourceDecisionResult:
        """
        基于当前状态和目标做出资源分配决策
        """
        # 风险评估
        risk_assessment = self._assess_risk(current_state)
        
        # 提取状态特征
        state_features = self._extract_state_features(current_state, objective)
        
        # 使用策略网络选择动作
        action_probs, value_estimate = self.rl_policy.apply(
            self.policy_params, state_features
        )
        
        # 基于风险评估调整动作选择
        adjusted_probs = self._adjust_probs_by_risk(action_probs, risk_assessment)
        
        # 选择动作
        action_idx = jnp.argmax(adjusted_probs)
        action = list(ResourceAction)[action_idx]
        
        # 生成动作参数
        parameters = self._generate_action_parameters(
            action, current_state, objective, risk_assessment
        )
        
        # 计算预期奖励
        expected_reward = self._calculate_expected_reward(
            current_state, action, parameters, objective, risk_assessment
        )
        
        return ResourceDecisionResult(
            action=action,
            parameters=parameters,
            expected_reward=float(expected_reward),
            confidence=float(adjusted_probs[action_idx]),
            execution_time=0.05,  # 毫秒级响应
            risk_assessment=risk_assessment
        )
    
    def _assess_risk(self, state: ResourceState) -> Dict[str, float]:
        """风险评估"""
        risk_scores = {}
        
        # CPU风险
        cpu_risk = max(0, state.cpu_utilization - self.resource_thresholds["cpu_critical"])
        risk_scores["cpu_risk"] = cpu_risk
        
        # 内存风险
        memory_risk = max(0, state.memory_utilization - self.resource_thresholds["memory_critical"])
        risk_scores["memory_risk"] = memory_risk
        
        # GPU风险
        gpu_risk = max(0, state.gpu_utilization - self.resource_thresholds["gpu_critical"])
        risk_scores["gpu_risk"] = gpu_risk
        
        # 存储风险
        storage_risk = max(0, state.storage_io_utilization - self.resource_thresholds["storage_critical"])
        risk_scores["storage_risk"] = storage_risk
        
        # 负载风险
        load_risk = max(0, state.system_load - 1.0)  # 负载超过1.0表示有排队进程
        risk_scores["load_risk"] = load_risk
        
        # 综合风险
        total_risk = (cpu_risk * 0.25 + memory_risk * 0.25 + 
                     gpu_risk * 0.2 + storage_risk * 0.2 + load_risk * 0.1)
        risk_scores["total_risk"] = total_risk
        
        # 风险等级
        if total_risk > 0.7:
            risk_scores["risk_level"] = 3.0  # 高风险
        elif total_risk > 0.4:
            risk_scores["risk_level"] = 2.0  # 中风险
        else:
            risk_scores["risk_level"] = 1.0  # 低风险
        
        return risk_scores
    
    def _extract_state_features(self, 
                               state: ResourceState,
                               objective: ResourceObjective) -> jnp.ndarray:
        """提取状态特征向量"""
        features = [
            # CPU资源
            state.cpu_utilization,
            min(state.cpu_available_cores / state.cpu_total_cores, 1.0),
            state.cpu_load_average,
            
            # 内存资源
            state.memory_utilization,
            min(state.memory_available / state.memory_total, 1.0),
            
            # GPU资源
            state.gpu_utilization,
            min(state.gpu_used_memory / state.gpu_total_memory, 1.0),
            min(state.gpu_temperature / 85.0, 1.0),  # 温度归一化，85度为上限
            
            # 存储资源
            state.storage_io_utilization,
            min(state.storage_available / state.storage_total, 1.0),
            
            # 网络资源
            state.network_utilization,
            min(state.network_latency / 100.0, 1.0),  # 延迟归一化，100ms为上限
            
            # 系统负载
            state.system_load,
            min(state.active_processes / 100.0, 1.0),  # 进程数归一化
            min(state.system_uptime / 720.0, 1.0),  # 运行时间归一化，30天为上限
            
            # 应用需求
            state.ai_training_demand,
            state.inference_demand,
            state.blockchain_demand,
            state.agriculture_demand,
            
            # 目标编码
            self._encode_objective(objective)
        ]
        
        return jnp.array(features, dtype=jnp.float32)
    
    def _encode_objective(self, objective: ResourceObjective) -> float:
        """编码决策目标"""
        objective_encoding = {
            ResourceObjective.MAXIMIZE_PERFORMANCE: 0.0,
            ResourceObjective.MINIMIZE_COST: 0.25,
            ResourceObjective.OPTIMIZE_EFFICIENCY: 0.5,
            ResourceObjective.BALANCE_FAIRNESS: 0.75
        }
        return objective_encoding.get(objective, 0.0)
    
    def _adjust_probs_by_risk(self, action_probs: jnp.ndarray, risk_assessment: Dict[str, float]) -> jnp.ndarray:
        """基于风险评估调整动作概率"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        # 高风险时倾向于负载均衡和扩缩容动作
        if risk_level >= 3.0:
            # 增加负载均衡和扩缩容动作的概率
            balance_idx = list(ResourceAction).index(ResourceAction.BALANCE_LOAD)
            scale_up_idx = list(ResourceAction).index(ResourceAction.SCALE_UP)
            
            adjusted_probs = jnp.array(action_probs)
            adjusted_probs = adjusted_probs.at[balance_idx].multiply(1.8)
            adjusted_probs = adjusted_probs.at[scale_up_idx].multiply(1.5)
            
            # 重新归一化
            return adjusted_probs / jnp.sum(adjusted_probs)
        
        return action_probs
    
    def _generate_action_parameters(self,
                                  action: ResourceAction,
                                  state: ResourceState,
                                  objective: ResourceObjective,
                                  risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成动作具体参数"""
        if action == ResourceAction.ALLOCATE_CPU:
            return self._generate_cpu_allocation_parameters(state, objective, risk_assessment)
        elif action == ResourceAction.ALLOCATE_MEMORY:
            return self._generate_memory_allocation_parameters(state, objective, risk_assessment)
        elif action == ResourceAction.ALLOCATE_GPU:
            return self._generate_gpu_allocation_parameters(state, objective, risk_assessment)
        elif action == ResourceAction.ALLOCATE_STORAGE:
            return self._generate_storage_allocation_parameters(state, objective, risk_assessment)
        elif action == ResourceAction.BALANCE_LOAD:
            return self._generate_load_balancing_parameters(state, objective, risk_assessment)
        elif action == ResourceAction.SCALE_UP:
            return self._generate_scale_up_parameters(state, objective, risk_assessment)
        elif action == ResourceAction.SCALE_DOWN:
            return self._generate_scale_down_parameters(state, objective, risk_assessment)
        else:
            return {}
    
    def _generate_cpu_allocation_parameters(self, 
                                           state: ResourceState,
                                           objective: ResourceObjective,
                                           risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成CPU分配参数"""
        # 基础分配策略
        base_cores = 2.0
        
        # 基于目标调整
        if objective == ResourceObjective.MAXIMIZE_PERFORMANCE:
            core_multiplier = 2.0
        elif objective == ResourceObjective.MINIMIZE_COST:
            core_multiplier = 0.5
        else:
            core_multiplier = 1.0
        
        # 基于需求调整
        demand_factor = max(state.ai_training_demand, state.inference_demand)
        core_multiplier *= (1.0 + demand_factor * 0.5)
        
        # 基于风险调整
        risk_factor = 1.0 - risk_assessment.get("cpu_risk", 0.0)
        
        total_cores = base_cores * core_multiplier * risk_factor
        
        return {
            "cpu_cores": max(1.0, min(total_cores, state.cpu_available_cores)),
            "priority_level": 8.0 if objective == ResourceObjective.MAXIMIZE_PERFORMANCE else 5.0,
            "cpu_affinity": 1.0,  # 允许CPU亲和性
            "cpu_quota": 100.0  # CPU配额百分比
        }
    
    def _generate_memory_allocation_parameters(self,
                                              state: ResourceState,
                                              objective: ResourceObjective,
                                              risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成内存分配参数"""
        base_memory = 4.0  # GB
        
        # 基于目标调整
        if objective == ResourceObjective.MAXIMIZE_PERFORMANCE:
            memory_multiplier = 2.5
        elif objective == ResourceObjective.MINIMIZE_COST:
            memory_multiplier = 0.7
        else:
            memory_multiplier = 1.5
        
        # 基于AI训练需求调整
        if state.ai_training_demand > 0.7:
            memory_multiplier *= 1.8
        
        # 基于风险调整
        risk_factor = 1.0 - risk_assessment.get("memory_risk", 0.0)
        
        total_memory = base_memory * memory_multiplier * risk_factor
        
        return {
            "memory_gb": max(1.0, min(total_memory, state.memory_available)),
            "memory_swap": 2.0,  # 交换空间倍数
            "memory_limit_hard": 1.0,  # 硬限制
            "memory_reservation": 0.8  # 预留比例
        }
    
    def _generate_gpu_allocation_parameters(self,
                                          state: ResourceState,
                                          objective: ResourceObjective,
                                          risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成GPU分配参数"""
        base_memory = 2.0  # GB
        
        # 基于目标调整
        if objective == ResourceObjective.MAXIMIZE_PERFORMANCE:
            gpu_multiplier = 3.0
        elif objective == ResourceObjective.MINIMIZE_COST:
            gpu_multiplier = 0.5
        else:
            gpu_multiplier = 1.5
        
        # 基于AI训练需求调整
        if state.ai_training_demand > 0.5:
            gpu_multiplier *= 2.0
        
        # 基于温度调整
        temperature_factor = max(0, 1.0 - state.gpu_temperature / 85.0)
        
        total_memory = base_memory * gpu_multiplier * temperature_factor
        
        return {
            "gpu_memory_gb": max(0.5, min(total_memory, state.gpu_total_memory - state.gpu_used_memory)),
            "gpu_utilization_limit": 0.9,
            "gpu_temperature_threshold": 80.0,
            "gpu_power_limit": 200.0  # 功率限制（W）
        }
    
    def _generate_storage_allocation_parameters(self,
                                              state: ResourceState,
                                              objective: ResourceObjective,
                                              risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成存储分配参数"""
        base_storage = 50.0  # GB
        
        # 基于目标调整
        if objective == ResourceObjective.MAXIMIZE_PERFORMANCE:
            storage_multiplier = 2.0
            io_priority = "high"
        elif objective == ResourceObjective.MINIMIZE_COST:
            storage_multiplier = 0.8
            io_priority = "low"
        else:
            storage_multiplier = 1.2
            io_priority = "medium"
        
        # 基于区块链需求调整
        if state.blockchain_demand > 0.6:
            storage_multiplier *= 1.5
        
        total_storage = base_storage * storage_multiplier
        
        return {
            "storage_gb": max(10.0, min(total_storage, state.storage_available)),
            "io_priority": io_priority,
            "read_iops": 1000.0,
            "write_iops": 500.0,
            "backup_frequency": 24.0  # 备份频率（小时）
        }
    
    def _generate_load_balancing_parameters(self,
                                           state: ResourceState,
                                           objective: ResourceObjective,
                                           risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成负载均衡参数"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        if risk_level >= 3.0:
            return {
                "migration_threshold": 0.3,  # 低迁移阈值
                "rebalance_aggressiveness": 0.9,
                "priority_redistribution": 1.0,
                "emergency_mode": 1.0
            }
        elif risk_level >= 2.0:
            return {
                "migration_threshold": 0.5,
                "rebalance_aggressiveness": 0.7,
                "priority_redistribution": 0.8,
                "emergency_mode": 0.0
            }
        else:
            return {
                "migration_threshold": 0.7,
                "rebalance_aggressiveness": 0.5,
                "priority_redistribution": 0.6,
                "emergency_mode": 0.0
            }
    
    def _generate_scale_up_parameters(self,
                                    state: ResourceState,
                                    objective: ResourceObjective,
                                    risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成扩容参数"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        if risk_level >= 3.0:
            scale_factor = 2.0  # 紧急扩容
            instance_type = "high_performance"
        elif risk_level >= 2.0:
            scale_factor = 1.5  # 中度扩容
            instance_type = "balanced"
        else:
            scale_factor = 1.2  # 轻度扩容
            instance_type = "cost_optimized"
        
        return {
            "scale_factor": scale_factor,
            "instance_type": instance_type,
            "min_instances": 1.0,
            "max_instances": 10.0,
            "cool_down_period": 300.0  # 冷却期（秒）
        }
    
    def _generate_scale_down_parameters(self,
                                      state: ResourceState,
                                      objective: ResourceObjective,
                                      risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成缩容参数"""
        # 缩容决策基于资源利用率和风险
        utilization_score = (state.cpu_utilization + state.memory_utilization + 
                           state.gpu_utilization) / 3.0
        
        if utilization_score < 0.3:
            scale_factor = 0.5  # 大幅缩容
        elif utilization_score < 0.5:
            scale_factor = 0.7  # 中度缩容
        else:
            scale_factor = 0.9  # 轻度缩容
        
        return {
            "scale_factor": scale_factor,
            "preserve_capacity": 0.3,  # 保留容量比例
            "drain_timeout": 600.0,  # 排空超时（秒）
            "instance_termination_delay": 60.0  # 实例终止延迟（秒）
        }
    
    def _calculate_expected_reward(self,
                                  state: ResourceState,
                                  action: ResourceAction,
                                  parameters: Dict[str, float],
                                  objective: ResourceObjective,
                                  risk_assessment: Dict[str, float]) -> float:
        """计算预期奖励"""
        base_reward = 0.0
        
        # 基于目标的奖励计算
        if objective == ResourceObjective.MAXIMIZE_PERFORMANCE:
            base_reward = (1.0 - state.cpu_utilization) * 40.0 + \
                         (1.0 - state.memory_utilization) * 30.0 + \
                         (1.0 - state.gpu_utilization) * 30.0
        elif objective == ResourceObjective.MINIMIZE_COST:
            # 成本与资源使用成反比
            base_reward = (state.cpu_available_cores / state.cpu_total_cores) * 50.0 + \
                         (state.memory_available / state.memory_total) * 30.0 + \
                         ((state.gpu_total_memory - state.gpu_used_memory) / state.gpu_total_memory) * 20.0
        elif objective == ResourceObjective.OPTIMIZE_EFFICIENCY:
            # 效率与资源利用率成正比，但避免过载
            efficiency_score = min(state.cpu_utilization, 0.8) * 30.0 + \
                              min(state.memory_utilization, 0.8) * 30.0 + \
                              min(state.gpu_utilization, 0.8) * 40.0
            base_reward = efficiency_score
        else:  # BALANCE_FAIRNESS
            # 公平性奖励，考虑资源分配的均衡性
            resource_balance = 1.0 - abs(state.cpu_utilization - state.memory_utilization) - \
                            abs(state.memory_utilization - state.gpu_utilization)
            base_reward = resource_balance * 60.0 + (1.0 - state.system_load) * 40.0
        
        # 动作有效性奖励
        action_effectiveness = self._evaluate_action_effectiveness(action, state, risk_assessment)
        base_reward *= action_effectiveness
        
        # 参数合理性奖励
        parameter_quality = self._evaluate_parameter_quality(parameters, state)
        base_reward *= parameter_quality
        
        return float(base_reward)
    
    def _evaluate_action_effectiveness(self, 
                                      action: ResourceAction, 
                                      state: ResourceState,
                                      risk_assessment: Dict[str, float]) -> float:
        """评估动作有效性"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        if action in [ResourceAction.BALANCE_LOAD, ResourceAction.SCALE_UP]:
            # 高风险时负载均衡和扩容更有效
            return max(0.3, risk_level / 3.0)
        elif action in [ResourceAction.SCALE_DOWN, ResourceAction.NO_ACTION]:
            # 低风险时缩容和无动作更有效
            return max(0.5, 1.0 - risk_level / 3.0)
        else:
            return 0.7  # 默认有效性
    
    def _evaluate_parameter_quality(self, parameters: Dict[str, float], state: ResourceState) -> float:
        """评估参数质量"""
        quality_score = 1.0
        
        # 检查参数是否在合理范围内
        for param, value in parameters.items():
            if "cpu_cores" in param and value > state.cpu_available_cores:
                quality_score *= 0.8
            elif "memory_gb" in param and value > state.memory_available:
                quality_score *= 0.7
            elif "gpu_memory" in param and value > (state.gpu_total_memory - state.gpu_used_memory):
                quality_score *= 0.6
            elif "storage_gb" in param and value > state.storage_available:
                quality_score *= 0.5
        
        return quality_score
    
    def update_policy(self, state: ResourceState, action: ResourceAction, 
                     reward: float, next_state: ResourceState):
        """更新策略网络（简化实现）"""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        }
        
        self.decision_history.append(experience)
        self.reward_history.append(reward)
        
        # 保持历史记录长度
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)
            self.reward_history.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取决策性能指标"""
        if not self.reward_history:
            return {"average_reward": 0.0, "decision_count": 0}
        
        return {
            "average_reward": float(np.mean(self.reward_history)),
            "decision_count": len(self.reward_history),
            "recent_success_rate": float(np.mean(self.reward_history[-10:]) > 0.5),
            "resource_efficiency": self._calculate_resource_efficiency()
        }
    
    def _calculate_resource_efficiency(self) -> float:
        """计算资源效率"""
        if len(self.decision_history) < 5:
            return 0.5
        
        # 分析最近的资源分配决策效果
        resource_decisions = [
            exp for exp in self.decision_history[-20:]
            if exp["action"] in [ResourceAction.ALLOCATE_CPU, ResourceAction.ALLOCATE_MEMORY, 
                               ResourceAction.ALLOCATE_GPU, ResourceAction.BALANCE_LOAD]
        ]
        
        if not resource_decisions:
            return 0.5
        
        avg_reward = np.mean([exp["reward"] for exp in resource_decisions])
        return float(avg_reward / 100.0)  # 归一化