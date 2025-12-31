"""
避免局部最优的奖励函数实现
针对生成式模型和强化学习环境设计
"""

from typing import List, Dict, Any, Optional
import jax.numpy as jnp
import numpy as np
from collections import defaultdict


class RewardFunction:
    """奖励函数基类"""
    
    def compute_reward(self, state: Any, action: Any, next_state: Any, **kwargs) -> float:
        """
        计算奖励值
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            **kwargs: 额外参数
            
        Returns:
            奖励值
        """
        raise NotImplementedError


class ExplorationReward(RewardFunction):
    """探索奖励函数
    
    鼓励模型探索未访问过的状态，避免陷入局部最优
    使用访问计数和距离度量来计算探索奖励
    """
    
    def __init__(self, exploration_weight: float = 0.1, distance_weight: float = 0.05, decay_rate: float = 0.999):
        self.exploration_weight = exploration_weight
        self.distance_weight = distance_weight
        self.decay_rate = decay_rate
        self.state_visits = defaultdict(int)  # 状态访问计数
    
    def compute_reward(self, state: Any, action: Any, next_state: Any, **kwargs) -> float:
        # 状态哈希化，用于计数
        state_key = self._hash_state(state)
        next_state_key = self._hash_state(next_state)
        
        # 更新访问计数
        self.state_visits[next_state_key] += 1
        
        # 基础奖励（可以从kwargs中获取或设为0）
        base_reward = kwargs.get('base_reward', 0.0)
        
        # 探索奖励：访问次数越少，奖励越高
        visit_count = self.state_visits[next_state_key]
        exploration_reward = self.exploration_weight / (1 + np.log1p(visit_count))
        
        # 距离奖励：与历史状态距离越远，奖励越高
        if state_key in self.state_visits:
            distance = self._compute_state_distance(state, next_state)
            distance_reward = self.distance_weight * distance
        else:
            distance_reward = 0.0
        
        # 总奖励
        total_reward = base_reward + exploration_reward + distance_reward
        
        # 更新所有状态的访问计数衰减
        for key in self.state_visits:
            self.state_visits[key] *= self.decay_rate
        
        return float(total_reward)
    
    def _hash_state(self, state: Any) -> str:
        """将状态哈希化为字符串"""
        if isinstance(state, np.ndarray) or isinstance(state, jnp.ndarray):
            return str(state.tolist())
        elif isinstance(state, list):
            return str(state)
        else:
            return str(state)
    
    def _compute_state_distance(self, state1: Any, state2: Any) -> float:
        """计算两个状态之间的距离"""
        if isinstance(state1, (np.ndarray, jnp.ndarray)) and isinstance(state2, (np.ndarray, jnp.ndarray)):
            # 使用余弦相似度作为距离度量
            if state1.ndim > 1:
                state1 = state1.flatten()
            if state2.ndim > 1:
                state2 = state2.flatten()
            
            dot_product = np.dot(state1, state2)
            norm1 = np.linalg.norm(state1)
            norm2 = np.linalg.norm(state2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            return 1.0 - cosine_similarity  # 转换为距离
        
        return 0.0


class DiversityReward(RewardFunction):
    """多样性奖励函数
    
    鼓励生成多样化的输出，避免重复模式
    可以用于文本生成、动作选择等场景
    """
    
    def __init__(self, diversity_weight: float = 0.2, ngram_size: int = 2):
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self.generated_ngrams = set()  # 已生成的n-gram集合
    
    def compute_reward(self, state: Any, action: Any, next_state: Any, **kwargs) -> float:
        # 基础奖励
        base_reward = kwargs.get('base_reward', 0.0)
        
        # 多样性奖励
        diversity_reward = 0.0
        
        # 如果是文本生成，从kwargs获取生成的序列
        generated_sequence = kwargs.get('generated_sequence', None)
        if generated_sequence:
            # 计算n-gram多样性
            ngrams = self._extract_ngrams(generated_sequence)
            
            # 统计新的n-gram数量
            new_ngrams = ngrams - self.generated_ngrams
            if ngrams:
                diversity_score = len(new_ngrams) / len(ngrams)
                diversity_reward = self.diversity_weight * diversity_score
            
            # 更新已生成的n-gram集合
            self.generated_ngrams.update(ngrams)
        
        return base_reward + diversity_reward
    
    def _extract_ngrams(self, sequence: List[int], n: Optional[int] = None) -> set:
        """从序列中提取n-gram"""
        if n is None:
            n = self.ngram_size
        
        if len(sequence) < n:
            return set()
        
        ngrams = set()
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i+n])
            ngrams.add(ngram)
        
        return ngrams


class EntropyReward(RewardFunction):
    """基于熵的奖励函数
    
    鼓励模型在不确定的情况下进行探索
    使用动作概率分布的熵作为奖励
    """
    
    def __init__(self, entropy_weight: float = 0.1):
        self.entropy_weight = entropy_weight
    
    def compute_reward(self, state: Any, action: Any, next_state: Any, **kwargs) -> float:
        # 基础奖励
        base_reward = kwargs.get('base_reward', 0.0)
        
        # 熵奖励
        entropy_reward = 0.0
        
        # 从kwargs获取动作概率分布
        action_probs = kwargs.get('action_probs', None)
        if action_probs is not None:
            # 计算熵
            entropy = self._compute_entropy(action_probs)
            entropy_reward = self.entropy_weight * entropy
        
        return base_reward + entropy_reward
    
    def _compute_entropy(self, probs: jnp.ndarray) -> float:
        """计算概率分布的熵"""
        # 避免log(0)
        probs = jnp.clip(probs, 1e-10, 1.0)
        entropy = -jnp.sum(probs * jnp.log(probs))
        return float(entropy)


class CombinedReward(RewardFunction):
    """组合奖励函数
    
    将多种奖励函数组合起来使用
    """
    
    def __init__(self, reward_functions: List[Dict[str, Any]]):
        """
        Args:
            reward_functions: 奖励函数列表，每个元素是一个字典，包含:
                - 'function': 奖励函数实例
                - 'weight': 权重
        """
        self.reward_functions = reward_functions
    
    def compute_reward(self, state: Any, action: Any, next_state: Any, **kwargs) -> float:
        total_reward = 0.0
        
        for rf_config in self.reward_functions:
            rf = rf_config['function']
            weight = rf_config['weight']
            reward = rf.compute_reward(state, action, next_state, **kwargs)
            total_reward += weight * reward
        
        return total_reward


class LocalOptimaAvoidanceReward(RewardFunction):
    """局部最优避免奖励函数
    
    专门设计用于避免模型陷入局部最优解
    结合了探索、多样性和历史性能的考量
    """
    
    def __init__(self, 
                 exploration_weight: float = 0.1, 
                 diversity_weight: float = 0.15, 
                 improvement_weight: float = 0.2, 
                 stagnation_penalty: float = 0.1):
        self.exploration_weight = exploration_weight
        self.diversity_weight = diversity_weight
        self.improvement_weight = improvement_weight
        self.stagnation_penalty = stagnation_penalty
        
        self.state_visits = defaultdict(int)
        self.ngram_history = set()
        self.reward_history = []
        self.best_reward = -float('inf')
    
    def compute_reward(self, state: Any, action: Any, next_state: Any, **kwargs) -> float:
        # 基础奖励
        base_reward = kwargs.get('base_reward', 0.0)
        generated_sequence = kwargs.get('generated_sequence', None)
        
        # 1. 探索奖励
        state_key = self._hash_state(state)
        next_state_key = self._hash_state(next_state)
        self.state_visits[next_state_key] += 1
        
        visit_count = self.state_visits[next_state_key]
        exploration_reward = self.exploration_weight / (1 + np.log1p(visit_count))
        
        # 2. 多样性奖励
        diversity_reward = 0.0
        if generated_sequence:
            ngrams = self._extract_ngrams(generated_sequence)
            if ngrams:
                new_ngrams = ngrams - self.ngram_history
                diversity_score = len(new_ngrams) / len(ngrams)
                diversity_reward = self.diversity_weight * diversity_score
                self.ngram_history.update(ngrams)
        
        # 3. 改进奖励
        improvement_reward = 0.0
        if base_reward > self.best_reward:
            improvement_reward = self.improvement_weight * (base_reward - self.best_reward)
            self.best_reward = base_reward
        
        # 4. 停滞惩罚
        stagnation_penalty = 0.0
        self.reward_history.append(base_reward)
        if len(self.reward_history) > 10:  # 检查最近10步的性能
            recent_rewards = self.reward_history[-10:]
            avg_recent = np.mean(recent_rewards)
            
            if len(self.reward_history) > 20:
                previous_rewards = self.reward_history[-20:-10]
                avg_previous = np.mean(previous_rewards)
                
                # 如果最近的平均奖励没有提高，施加惩罚
                if avg_recent <= avg_previous:
                    stagnation_penalty = -self.stagnation_penalty * (avg_previous - avg_recent + 1e-6)
        
        # 总奖励
        total_reward = base_reward + exploration_reward + diversity_reward + improvement_reward + stagnation_penalty
        
        return float(total_reward)
    
    def _hash_state(self, state: Any) -> str:
        """将状态哈希化为字符串"""
        if isinstance(state, (np.ndarray, jnp.ndarray)):
            return str(state.tolist())
        elif isinstance(state, list):
            return str(state)
        else:
            return str(state)
    
    def _extract_ngrams(self, sequence: List[int], n: int = 2) -> set:
        """从序列中提取n-gram"""
        if len(sequence) < n:
            return set()
        
        ngrams = set()
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i+n])
            ngrams.add(ngram)
        
        return ngrams


# 工具函数：创建默认的奖励函数组合
def create_default_reward_function() -> CombinedReward:
    """
    创建一个默认的奖励函数组合，包含:
    - 探索奖励
    - 多样性奖励
    - 熵奖励
    """
    exploration_rf = ExplorationReward(exploration_weight=0.15, distance_weight=0.1)
    diversity_rf = DiversityReward(diversity_weight=0.2, ngram_size=2)
    entropy_rf = EntropyReward(entropy_weight=0.1)
    
    return CombinedReward([
        {'function': exploration_rf, 'weight': 1.0},
        {'function': diversity_rf, 'weight': 1.0},
        {'function': entropy_rf, 'weight': 1.0}
    ])


# 工具函数：创建局部最优避免奖励函数
def create_local_optima_avoidance_reward() -> LocalOptimaAvoidanceReward:
    """
    创建一个专门用于避免局部最优的奖励函数
    """
    return LocalOptimaAvoidanceReward(
        exploration_weight=0.1,
        diversity_weight=0.15,
        improvement_weight=0.2,
        stagnation_penalty=0.1
    )
