"""
强化学习决策引擎 - 基于PPO算法的多目标自主决策系统
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import json
import time

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """决策类型枚举"""
    AGRICULTURE = "agriculture"
    BLOCKCHAIN = "blockchain"
    MODEL_TRAINING = "model_training"
    RESOURCE_ALLOCATION = "resource_allocation"


@dataclass
class DecisionState:
    """决策状态数据类"""
    decision_type: DecisionType
    state_vector: np.ndarray
    timestamp: float
    context: Optional[Dict[str, Any]] = None


@dataclass
class DecisionAction:
    """决策动作数据类"""
    decision_type: DecisionType
    action_vector: np.ndarray
    confidence: float
    reasoning: Optional[str] = None
    timestamp: float = None


@dataclass
class Experience:
    """经验回放数据类"""
    state: DecisionState
    action: DecisionAction
    reward: float
    next_state: DecisionState
    done: bool
    timestamp: float


class PolicyNetwork(nn.Module):
    """策略网络 - 生成决策动作"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络架构
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # 均值和标准差
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回动作的均值和标准差"""
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """价值网络 - 评估状态价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值"""
        return self.network(state)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Experience) -> None:
        """添加经验到缓冲区"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """从缓冲区采样经验"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """返回缓冲区大小"""
        return len(self.buffer)


class RLDecisionEngine:
    """强化学习决策引擎"""
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 64,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        # 网络参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络和价值网络
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_network = ValueNetwork(state_dim, hidden_dim)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate
        )
        
        # 强化学习参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer()
        
        # 决策模块注册表
        self.decision_modules: Dict[DecisionType, Any] = {}
        
        # 性能统计
        self.decision_count = 0
        self.average_decision_time = 0.0
        self.total_reward = 0.0
        
        logger.info("RL决策引擎初始化完成")
    
    def register_module(self, decision_type: DecisionType, module: Any) -> None:
        """注册决策模块"""
        self.decision_modules[decision_type] = module
        logger.info(f"注册决策模块: {decision_type.value}")
    
    def make_decision(self, state: DecisionState) -> DecisionAction:
        """基于当前状态生成决策"""
        start_time = time.time()
        
        # 转换为张量
        state_tensor = torch.FloatTensor(state.state_vector).unsqueeze(0)
        
        # 策略网络前向传播
        with torch.no_grad():
            mean, std = self.policy_network(state_tensor)
            
            # 采样动作
            normal = torch.distributions.Normal(mean, std)
            action_sample = normal.sample()
            
            # 计算动作概率
            action_prob = normal.log_prob(action_sample).sum(dim=-1)
            
            # 计算价值
            state_value = self.value_network(state_tensor)
        
        # 转换为numpy数组
        action_vector = action_sample.squeeze(0).numpy()
        confidence = torch.exp(action_prob).item()
        
        # 生成决策动作
        action = DecisionAction(
            decision_type=state.decision_type,
            action_vector=action_vector,
            confidence=confidence,
            timestamp=time.time()
        )
        
        # 更新性能统计
        decision_time = time.time() - start_time
        self.decision_count += 1
        self.average_decision_time = (
            (self.average_decision_time * (self.decision_count - 1) + decision_time) / 
            self.decision_count
        )
        
        logger.debug(f"决策生成完成: {state.decision_type.value}, 耗时: {decision_time:.4f}s")
        
        return action
    
    def update_policy(self, experiences: List[Experience]) -> Dict[str, float]:
        """使用经验数据更新策略"""
        if not experiences:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        # 准备训练数据
        states = torch.FloatTensor([exp.state.state_vector for exp in experiences])
        actions = torch.FloatTensor([exp.action.action_vector for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state.state_vector for exp in experiences])
        dones = torch.BoolTensor([exp.done for exp in experiences])
        
        # 计算目标价值
        with torch.no_grad():
            next_values = self.value_network(next_states).squeeze(-1)
            target_values = rewards + self.gamma * next_values * (~dones).float()
        
        # 计算当前价值
        current_values = self.value_network(states).squeeze(-1)
        
        # 计算优势函数
        advantages = target_values - current_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算旧策略的概率
        with torch.no_grad():
            old_means, old_stds = self.policy_network(states)
            old_normal = torch.distributions.Normal(old_means, old_stds)
            old_log_probs = old_normal.log_prob(actions).sum(dim=-1)
        
        # 计算新策略的概率
        new_means, new_stds = self.policy_network(states)
        new_normal = torch.distributions.Normal(new_means, new_stds)
        new_log_probs = new_normal.log_prob(actions).sum(dim=-1)
        
        # 计算策略比率
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪损失
        policy_loss1 = ratios * advantages
        policy_loss2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        # 价值损失
        value_loss = nn.MSELoss()(current_values, target_values)
        
        # 熵奖励
        entropy = new_normal.entropy().mean()
        
        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.optimizer.step()
        
        # 记录损失
        loss_info = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
        
        logger.info(f"策略更新完成: {loss_info}")
        
        return loss_info
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验到回放缓冲区"""
        self.replay_buffer.add(experience)
        self.total_reward += experience.reward
        
        # 定期训练
        if self.replay_buffer.size() >= 1000:
            experiences = self.replay_buffer.sample(256)
            self.update_policy(experiences)
    
    def save_model(self, filepath: str) -> None:
        """保存模型参数"""
        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'decision_count': self.decision_count,
            'total_reward': self.total_reward
        }
        torch.save(checkpoint, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载模型参数"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.decision_count = checkpoint.get('decision_count', 0)
        self.total_reward = checkpoint.get('total_reward', 0.0)
        logger.info(f"模型已从 {filepath} 加载")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            "decision_count": self.decision_count,
            "average_decision_time": self.average_decision_time,
            "total_reward": self.total_reward,
            "buffer_size": self.replay_buffer.size(),
            "registered_modules": list(self.decision_modules.keys())
        }


class MultiObjectiveOptimizer:
    """多目标优化器 - 处理多个冲突的优化目标"""
    
    def __init__(self, objectives: List[str], weights: Optional[Dict[str, float]] = None):
        self.objectives = objectives
        self.weights = weights or {obj: 1.0 / len(objectives) for obj in objectives}
        
    def weighted_sum(self, objective_values: Dict[str, float]) -> float:
        """加权和法计算综合得分"""
        return sum(self.weights[obj] * objective_values[obj] for obj in self.objectives)
    
    def pareto_dominance(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """判断帕累托支配关系"""
        # solution1 支配 solution2 当且仅当在所有目标上都不差，且至少在一个目标上更好
        not_worse = all(solution1[obj] >= solution2[obj] for obj in self.objectives)
        strictly_better = any(solution1[obj] > solution2[obj] for obj in self.objectives)
        return not_worse and strictly_better


if __name__ == "__main__":
    # 测试代码
    engine = RLDecisionEngine()
    
    # 创建测试状态
    test_state = DecisionState(
        decision_type=DecisionType.AGRICULTURE,
        state_vector=np.random.randn(128),
        timestamp=time.time()
    )
    
    # 生成决策
    action = engine.make_decision(test_state)
    print(f"生成的决策: {action.action_vector}")
    print(f"决策置信度: {action.confidence}")