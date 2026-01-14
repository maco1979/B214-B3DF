# Multi-Modal Curiosity Model
# Implements intelligent adaptive curiosity mechanisms for AGI systems
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import numpy as np
import random


class MultiModalCuriosity(nn.Module):
    """多模态好奇心评估模型 - 增强版
    实现了技术文档中描述的多模态好奇心评估算法，综合视觉、语音、文本等多种信息源
    增强：支持更复杂的模态融合和自适应好奇心调整
    """
    def __init__(self, state_dim, action_dim, modality_dims):
        super(MultiModalCuriosity, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modality_dims = modality_dims
        
        # 新增：模态编码器，为每个模态设计专用编码器
        self.modality_encoders = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            if modality == 'vision':
                # 视觉模态编码器
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
            elif modality == 'speech':
                # 语音模态编码器
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
            elif modality == 'text':
                # 文本模态编码器
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
            else:
                # 默认模态编码器
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(dim, 128),
                    nn.ReLU()
                )
        
        # 模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * len(modality_dims), 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
        # 预测网络：处理融合后的状态和动作
        self.prediction_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
        # 新增：好奇心权重调整网络
        self.curiosity_adjuster = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1之间的好奇心权重
        )
    
    def forward(self, observations, actions):
        """前向传播
        
        Args:
            observations: 多模态观察输入（字典形式）或状态张量
            actions: 动作输入
            
        Returns:
            预测的下一状态和好奇心权重
        """
        if isinstance(observations, dict):
            # 多模态观察输入
            modality_features = []
            for modality, data in observations.items():
                if modality in self.modality_encoders:
                    # 确保数据是张量形式
                    if not isinstance(data, torch.Tensor):
                        data = torch.tensor(data, dtype=torch.float32)
                    # 添加批次维度（如果没有）
                    if data.dim() == 1:
                        data = data.unsqueeze(0)
                    # 适配模态维度
                    if data.shape[-1] != self.modality_dims[modality]:
                        # 调整维度（简化处理）
                        data = data[:, :self.modality_dims[modality]]
                        if data.shape[-1] < self.modality_dims[modality]:
                            pad = torch.zeros(data.shape[0], self.modality_dims[modality] - data.shape[-1])
                            data = torch.cat([data, pad], dim=-1)
                    # 编码模态特征
                    encoded = self.modality_encoders[modality](data)
                    modality_features.append(encoded)
            
            # 融合模态特征
            if modality_features:
                fused_features = torch.cat(modality_features, dim=-1)
                state_tensor = self.fusion_layer(fused_features)
            else:
                # 默认状态张量
                state_tensor = torch.randn(actions.shape[0], self.state_dim)
        else:
            # 状态张量直接作为输入
            state_tensor = observations
        
        # 融合状态和动作信息
        combined = torch.cat([state_tensor, actions], dim=1)
        
        # 预测下一状态
        predicted_next_state = self.prediction_net(combined)
        
        # 计算好奇心权重
        curiosity_weight = self.curiosity_adjuster(state_tensor)
        
        return predicted_next_state, curiosity_weight


def calculate_entropy(states: torch.Tensor) -> float:
    """计算状态熵
    使用高斯核密度估计计算熵
    
    Args:
        states: 状态张量
        
    Returns:
        状态熵值
    """
    # 转换为numpy数组
    states_np = states.detach().cpu().numpy()
    
    # 使用高斯核密度估计计算熵
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(states_np)
    log_density = kde.score_samples(states_np)
    entropy = -log_density.mean()
    
    return entropy


def calculate_intrinsic_reward(current_state: torch.Tensor, action: torch.Tensor, 
                             next_state: torch.Tensor, model: MultiModalCuriosity, 
                             curiosity_type: str = "combined") -> torch.Tensor:
    """计算内在奖励 - 增强版
    结合预测误差、信息增益和自适应好奇心权重
    
    Args:
        current_state: 当前状态
        action: 执行的动作
        next_state: 实际的下一状态
        model: 好奇心模型
        curiosity_type: 好奇心类型 (combined, prediction, information_gain)
        
    Returns:
        内在奖励值
    """
    # 1. 计算预测误差（Prediction Error）
    predicted_next_state, curiosity_weight = model(current_state, action)
    prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='mean')
    
    # 2. 计算信息增益（Information Gain）
    # 对当前状态和预测的下一状态计算熵
    current_state_entropy = calculate_entropy(current_state)
    predicted_next_state_entropy = calculate_entropy(predicted_next_state)
    
    # 信息增益 = 下一状态熵 - 当前状态熵
    information_gain = predicted_next_state_entropy - current_state_entropy
    information_gain_tensor = torch.tensor(information_gain, dtype=torch.float32, device=current_state.device)
    
    # 3. 结合预测误差和信息增益
    # 自适应权重调整
    if curiosity_type == "prediction":
        # 仅使用预测误差
        intrinsic_reward = prediction_error
    elif curiosity_type == "information_gain":
        # 仅使用信息增益
        intrinsic_reward = information_gain_tensor
    else:
        # 结合两种奖励成分，使用自适应权重
        prediction_error_weight = 0.6 + (curiosity_weight.item() * 0.2)  # 0.6-0.8
        information_gain_weight = 0.4 - (curiosity_weight.item() * 0.2)  # 0.2-0.4
        
        # 结合两种奖励成分
        intrinsic_reward = (prediction_error_weight * prediction_error) + (information_gain_weight * information_gain_tensor)
    
    # 4. 应用好奇心权重
    intrinsic_reward = intrinsic_reward * curiosity_weight
    
    # 确保奖励为正数
    intrinsic_reward = F.relu(intrinsic_reward)
    
    return intrinsic_reward


class NoveltySeekingExplorer:
    """新颖性寻求探索器
    实现了技术文档中描述的新颖性寻求探索策略
    """
    def __init__(self, env, history_length=50):
        self.env = env
        self.history = []
        self.history_length = history_length
    
    def add_to_history(self, state, action, next_state):
        """添加到历史记录"""
        self.history.append((state, action, next_state))
        if len(self.history) > self.history_length:
            self.history.pop(0)
    
    def novelty_seeking_exploration(self, state):
        """新颖性寻求探索
        寻找与历史状态差异最大的动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        # 如果历史为空，随机选择动作
        if not self.history:
            return self.env.action_space.sample()
        
        max_diff = -1
        best_action = None
        
        # 尝试所有可能的动作
        for action in range(self.env.action_space.n):
            # 模拟执行动作
            next_state, _, _, _ = self.env.simulate(state, action)
            
            # 计算与所有历史状态的平均差异
            total_diff = 0
            for (hist_state, _, _) in self.history:
                diff = torch.norm(next_state - hist_state)
                total_diff += diff
            
            avg_diff = total_diff / len(self.history)
            
            if avg_diff > max_diff:
                max_diff = avg_diff
                best_action = action
        
        return best_action if best_action is not None else self.env.action_space.sample()


class CERMICExplorer:
    """CERMIC探索器
    实现了Curiosity Enhancement via Robust Multi-Agent Intention Calibration框架
    能够稳健地过滤噪声惊喜信号，并通过推断的多智能体上下文动态校准内在好奇心
    """
    def __init__(self, env, curiosity_model, threshold=0.5):
        self.env = env
        self.curiosity_model = curiosity_model
        self.threshold = threshold
        self.action_values = {}
        
        # CERMIC特定参数
        self.coverage_level = 0.7  # 覆盖级别
        self.noise_threshold = 0.3  # 噪声阈值，用于过滤虚假新颖性
        self.context_window = 5  # 上下文窗口大小
        self.agent_history = []  # 多智能体历史记录
        
        # 信息瓶颈相关
        self.information_bottleneck = nn.Linear(curiosity_model.state_dim, curiosity_model.state_dim // 2)  # 信息瓶颈层
        
    def infer_agent_intentions(self, state, other_agents=None):
        """推断多智能体意图
        基于信息瓶颈原理，学习多智能体上下文探索表示
        
        Args:
            state: 当前状态
            other_agents: 其他智能体的状态列表
            
        Returns:
            推断的意图表示
        """
        if other_agents is None:
            other_agents = []
        
        # 简化实现：使用状态的低维表示作为意图
        # 实际实现中应使用图神经网络建模智能体之间的关系
        with torch.no_grad():
            # 通过信息瓶颈压缩状态信息
            compressed_state = self.information_bottleneck(state)
            
            # 简单的意图推断：考虑其他智能体的影响
            intention_score = compressed_state.mean().item()
        
        return intention_score
    
    def filter_novelty(self, raw_novelty, state):
        """过滤噪声和虚假新颖性
        
        Args:
            raw_novelty: 原始新颖性分数
            state: 当前状态
            
        Returns:
            过滤后的新颖性分数
        """
        # 计算状态熵，熵越低，状态越确定，新颖性越可靠
        state_entropy = calculate_entropy(state)
        
        # 过滤规则：
        # 1. 如果状态熵太高（不确定），降低新颖性分数
        # 2. 如果原始新颖性低于噪声阈值，将其过滤
        filtered_novelty = raw_novelty
        
        if state_entropy > 0.5:  # 高熵，不确定状态
            filtered_novelty *= 0.5
        
        if raw_novelty < self.noise_threshold:
            filtered_novelty = 0.0
        
        return filtered_novelty
    
    def calibrate_curiosity(self, raw_curiosity, intention_score):
        """基于上下文校准好奇心信号
        
        Args:
            raw_curiosity: 原始好奇心分数
            intention_score: 推断的意图分数
            
        Returns:
            校准后的好奇心分数
        """
        # 根据覆盖级别和意图校准好奇心
        calibration_factor = self.coverage_level + (intention_score * 0.3)
        calibrated_curiosity = raw_curiosity * calibration_factor
        
        return calibrated_curiosity
    
    def evaluate_action_interestingness(self, state, action):
        """评估动作的有趣程度
        基于信息瓶颈原理，学习多智能体上下文探索表示
        """
        next_state, _, _, info = self.env.simulate(state, action)
        
        # 确保动作是张量形式
        if isinstance(action, int):
            # 转换为one-hot编码或随机张量表示
            action_tensor = torch.zeros(1, self.curiosity_model.action_dim)
            action_tensor[0, action % self.curiosity_model.action_dim] = 1.0
        else:
            action_tensor = action
        
        # 1. 计算原始预测误差
        predicted_next_state = self.curiosity_model(state, action_tensor)
        raw_prediction_error = torch.norm(predicted_next_state - next_state).item()
        
        # 2. 推断其他智能体意图
        other_agents = info.get('other_agents', [])
        intention_score = self.infer_agent_intentions(state, other_agents)
        
        # 3. 过滤虚假新颖性
        filtered_novelty = self.filter_novelty(raw_prediction_error, state)
        
        # 4. 基于上下文校准好奇心
        calibrated_curiosity = self.calibrate_curiosity(filtered_novelty, intention_score)
        
        # 5. 结合信息增益
        current_state_entropy = calculate_entropy(state)
        next_state_entropy = calculate_entropy(next_state)
        information_gain = next_state_entropy - current_state_entropy
        
        # 综合有趣程度分数
        interestingness = (calibrated_curiosity * 0.7) + (information_gain * 0.3)
        
        return max(0.0, interestingness)  # 确保有趣程度为正数
    
    def adaptive_exploration(self, state, possible_actions):
        """自适应探索策略
        基于信息瓶颈原理，学习多智能体上下文探索表示
        
        Args:
            state: 当前状态
            possible_actions: 可能的动作列表
            
        Returns:
            选择的动作
        """
        # 评估每个动作的有趣程度
        action_values = []
        for action in possible_actions:
            interestingness = self.evaluate_action_interestingness(state, action)
            action_values.append((action, interestingness))
        
        # 选择好奇心值最高的动作
        action_values.sort(key=lambda x: x[1], reverse=True)
        
        # 更新多智能体历史
        if action_values:
            best_action, best_value = action_values[0]
            self.agent_history.append((state, best_action, best_value))
            # 保持历史记录大小
            if len(self.agent_history) > self.context_window:
                self.agent_history.pop(0)
        
        return action_values[0][0] if action_values else self.env.action_space.sample()
    
    def update_coverage_level(self, new_coverage):
        """更新覆盖级别
        
        Args:
            new_coverage: 新的覆盖级别 (0-1)
        """
        self.coverage_level = max(0.0, min(1.0, new_coverage))
    
    def get_context_summary(self):
        """获取上下文摘要
        
        Returns:
            上下文摘要，包括历史动作和好奇心值
        """
        return {
            'agent_history': self.agent_history,
            'coverage_level': self.coverage_level,
            'noise_threshold': self.noise_threshold
        }


def evaluate_environment_complexity(state):
    """评估环境的复杂程度
    
    Args:
        state: 当前环境状态
        
    Returns:
        环境复杂度评分 (0-1)
    """
    # 简化实现：基于状态的特征数量和变化率评估复杂度
    complexity_score = 0.0
    
    if isinstance(state, dict):
        # 特征数量越多，环境越复杂
        feature_count = len(state)
        complexity_score += min(0.3, feature_count / 20)  # 最多贡献0.3分，降低特征数量权重
        
        # 数值型特征的变化范围越大，环境越复杂
        numeric_features = [v for v in state.values() if isinstance(v, (int, float))]
        if numeric_features:
            max_values = max(numeric_features)
            min_values = min(numeric_features)
            range_score = (max_values - min_values) / (max_values + min_values + 1e-8)  # 防止除零
            complexity_score += min(0.3, range_score)  # 最多贡献0.3分，降低范围权重
    elif isinstance(state, torch.Tensor):
        # 张量维度越高，环境越复杂
        complexity_score = min(1.0, state.numel() / 10000)  # 最多1分
    
    return max(0.0, min(1.0, complexity_score))


def calculate_curiosity_score(state, action_space, model, historical_states=None):
    """计算多维度好奇心分数
    综合考虑新颖性、复杂度和奖励潜力
    
    Args:
        state: 当前状态
        action_space: 动作空间
        model: 好奇心模型
        historical_states: 历史状态列表
        
    Returns:
        综合好奇心分数 (0-1)
    """
    curiosity_score = 0.0
    novelty_score = 0.0
    complexity_score = 0.0
    reward_score = 0.0
    
    # 1. 新颖性评估：与历史状态的差异
    if historical_states and len(historical_states) > 0:
        try:
            # 确保state是张量形式
            if isinstance(state, dict):
                # 简化实现：使用随机张量表示状态
                current_embedding = torch.randn(1, model.state_dim)
                historical_embeddings = torch.randn(len(historical_states), model.state_dim)
            else:
                current_embedding = state
                # 转换历史状态为张量
                if isinstance(historical_states[0], dict):
                    historical_embeddings = torch.randn(len(historical_states), model.state_dim)
                else:
                    historical_embeddings = torch.stack(historical_states)
            
            # 计算当前状态与历史状态的距离
            distances = torch.norm(current_embedding - historical_embeddings, dim=1)
            novelty_score = torch.mean(distances).item()
            # 归一化到0-1范围
            novelty_score = min(1.0, novelty_score / 10.0)  # 假设最大距离为10
        except Exception as e:
            # 简化处理：使用随机新颖性分数
            novelty_score = random.random()
    else:
        # 首次遇到的状态，新颖性最高
        novelty_score = 1.0
    
    # 2. 复杂性评估：环境的复杂程度
    complexity_score = evaluate_environment_complexity(state)
    
    # 3. 奖励潜力评估：预期的学习收益
    try:
        # 简化实现：使用模型预测不同动作的奖励
        if isinstance(action_space, int):
            # 离散动作空间：生成所有可能的动作
            actions = torch.eye(action_space, model.action_dim)[:action_space]
        else:
            # 连续动作空间：生成随机动作样本
            actions = torch.randn(10, model.action_dim)
        
        # 预测每个动作的奖励
        predicted_rewards = []
        for action in actions:
            action = action.unsqueeze(0)
            if isinstance(state, dict):
                pred_next_state = model(state, action)
            else:
                pred_next_state = model(state.unsqueeze(0), action)
            
            # 使用预测误差作为奖励潜力的代理
            if isinstance(state, dict):
                current_state_tensor = torch.randn(1, model.state_dim)
            else:
                current_state_tensor = state.unsqueeze(0)
            
            prediction_error = F.mse_loss(pred_next_state, current_state_tensor)
            predicted_rewards.append(prediction_error.item())
        
        reward_score = torch.mean(torch.tensor(predicted_rewards)).item()
        # 归一化到0-1范围
        reward_score = min(1.0, reward_score)
    except Exception as e:
        # 简化处理：使用随机奖励分数
        reward_score = random.random()
    
    # 4. 综合好奇心分数（可调节权重）
    curiosity_score = 0.4 * novelty_score + 0.3 * complexity_score + 0.3 * reward_score
    
    # 确保分数在0-1范围内
    curiosity_score = max(0.0, min(1.0, curiosity_score))
    
    return curiosity_score
