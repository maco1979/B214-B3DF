"""
区块链决策模块 - 基于强化学习的区块链积分分配和收益管理决策系统
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json

from .rl_decision_engine import DecisionType, DecisionState, DecisionAction

logger = logging.getLogger(__name__)


class ContributionType(Enum):
    """贡献类型枚举"""
    DATA_PROVIDING = "data_providing"      # 数据提供
    MODEL_TRAINING = "model_training"      # 模型训练
    SYSTEM_MAINTENANCE = "system_maintenance"  # 系统维护
    COMMUNITY_GOVERNANCE = "community_governance"  # 社区治理


class RevenueSource(Enum):
    """收入来源枚举"""
    USER_SUBSCRIPTION = "user_subscription"      # 用户订阅
    BUSINESS_SERVICE = "business_service"        # 企业服务
    DATA_SALES = "data_sales"                    # 数据销售
    HARDWARE_SALES = "hardware_sales"            # 硬件销售
    ADVERTISING = "advertising"                  # 广告收入


@dataclass
class UserContribution:
    """用户贡献数据类"""
    user_id: str
    contribution_type: ContributionType
    contribution_value: float
    data_quality: float
    timeliness: float
    frequency: float
    historical_consistency: float


@dataclass
class SystemRevenue:
    """系统收入数据类"""
    period: str
    total_revenue: float
    revenue_breakdown: Dict[RevenueSource, float]
    growth_rate: float
    predictability: float


@dataclass
class RewardDistribution:
    """奖励分配数据类"""
    user_id: str
    base_reward: float
    quality_bonus: float
    consistency_bonus: float
    special_bonus: float
    total_reward: float
    distribution_date: datetime


@dataclass
class BlockchainState:
    """区块链状态数据类"""
    user_contributions: List[UserContribution]
    system_revenue: SystemRevenue
    historical_distributions: List[RewardDistribution]
    community_governance_rules: Dict[str, Any]
    system_metrics: Dict[str, float]


class BlockchainDecisionModule:
    """区块链决策模块"""
    
    def __init__(self):
        # 奖励分配策略参数
        self.reward_strategy = self._initialize_reward_strategy()
        
        # 收益分配规则
        self.revenue_distribution_rules = {
            "platform_maintenance": 0.25,    # 平台维护
            "ai_development": 0.30,           # AI研发
            "user_rewards": 0.20,             # 用户奖励
            "marketing_operations": 0.15,     # 市场运营
            "reserve_fund": 0.10              # 储备基金
        }
        
        # 决策历史记录
        self.decision_history = []
        
        logger.info("区块链决策模块初始化完成")
    
    def _initialize_reward_strategy(self) -> Dict[str, Any]:
        """初始化奖励策略"""
        return {
            "base_reward_weights": {
                "data_providing": 0.4,
                "model_training": 0.3,
                "system_maintenance": 0.2,
                "community_governance": 0.1
            },
            "quality_bonus_curve": {
                "thresholds": [0.6, 0.8, 0.9],
                "bonuses": [0.1, 0.2, 0.3]
            },
            "consistency_multipliers": {
                "low": 1.0,
                "medium": 1.2,
                "high": 1.5
            },
            "special_bonus_criteria": {
                "early_adopter": 0.15,
                "high_frequency": 0.1,
                "innovation_contribution": 0.25
            }
        }
    
    def create_decision_state(self, blockchain_state: BlockchainState) -> DecisionState:
        """创建区块链决策状态向量"""
        
        state_features = []
        
        # 1. 用户贡献特征
        total_contributions = len(blockchain_state.user_contributions)
        avg_contribution_value = np.mean([uc.contribution_value for uc in blockchain_state.user_contributions]) if total_contributions > 0 else 0
        avg_data_quality = np.mean([uc.data_quality for uc in blockchain_state.user_contributions]) if total_contributions > 0 else 0
        
        state_features.extend([total_contributions, avg_contribution_value, avg_data_quality])
        
        # 2. 贡献类型分布
        contribution_types = {ct: 0 for ct in ContributionType}
        for uc in blockchain_state.user_contributions:
            contribution_types[uc.contribution_type] += uc.contribution_value
        
        total_contribution_value = sum(contribution_types.values())
        if total_contribution_value > 0:
            type_distribution = [contribution_types[ct] / total_contribution_value for ct in ContributionType]
        else:
            type_distribution = [0] * len(ContributionType)
        
        state_features.extend(type_distribution)
        
        # 3. 系统收入特征
        revenue = blockchain_state.system_revenue
        state_features.extend([
            revenue.total_revenue,
            revenue.growth_rate,
            revenue.predictability
        ])
        
        # 4. 收入来源分布
        revenue_sources = {rs: 0 for rs in RevenueSource}
        for source, amount in revenue.revenue_breakdown.items():
            revenue_sources[source] = amount
        
        total_revenue = sum(revenue_sources.values())
        if total_revenue > 0:
            source_distribution = [revenue_sources[rs] / total_revenue for rs in RevenueSource]
        else:
            source_distribution = [0] * len(RevenueSource)
        
        state_features.extend(source_distribution)
        
        # 5. 历史分配效果特征
        if blockchain_state.historical_distributions:
            recent_distributions = blockchain_state.historical_distributions[-10:]
            avg_reward = np.mean([d.total_reward for d in recent_distributions])
            reward_variance = np.var([d.total_reward for d in recent_distributions])
            user_satisfaction = self._estimate_user_satisfaction(recent_distributions)
        else:
            avg_reward, reward_variance, user_satisfaction = 0, 0, 0
        
        state_features.extend([avg_reward, reward_variance, user_satisfaction])
        
        # 6. 系统指标特征
        system_metrics = blockchain_state.system_metrics
        state_features.extend([
            system_metrics.get("active_users", 0),
            system_metrics.get("transaction_volume", 0),
            system_metrics.get("system_health", 1.0),
            system_metrics.get("community_engagement", 0.5)
        ])
        
        # 转换为numpy数组并归一化
        state_vector = np.array(state_features, dtype=np.float32)
        state_vector = (state_vector - np.min(state_vector)) / (np.max(state_vector) - np.min(state_vector) + 1e-8)
        
        return DecisionState(
            decision_type=DecisionType.BLOCKCHAIN,
            state_vector=state_vector,
            timestamp=datetime.now().timestamp(),
            context={
                "total_contributions": total_contributions,
                "total_revenue": revenue.total_revenue,
                "period": revenue.period
            }
        )
    
    def interpret_decision_action(self, action: DecisionAction, 
                                 current_state: BlockchainState) -> Dict[str, Any]:
        """解释决策动作为具体的区块链操作"""
        
        action_vector = action.action_vector
        
        # 解析动作向量为具体的分配策略
        decisions = {
            # 奖励分配策略调整
            "reward_strategy_adjustments": {
                "base_reward_multiplier": self._scale_value(action_vector[0], 0.8, 1.2),
                "quality_bonus_emphasis": self._scale_value(action_vector[1], 0.5, 2.0),
                "consistency_reward_factor": self._scale_value(action_vector[2], 1.0, 3.0),
                "special_bonus_threshold": self._scale_value(action_vector[3], 0.1, 0.5)
            },
            
            # 收益分配调整
            "revenue_distribution_adjustments": {
                "user_rewards_share": self._scale_value(action_vector[4], 0.15, 0.30),
                "ai_development_share": self._scale_value(action_vector[5], 0.25, 0.40),
                "platform_maintenance_share": self._scale_value(action_vector[6], 0.20, 0.30),
                "reserve_fund_share": self._scale_value(action_vector[7], 0.05, 0.15)
            },
            
            # 智能合约参数
            "smart_contract_parameters": {
                "minimum_contribution": self._scale_value(action_vector[8], 10, 100),
                "reward_distribution_frequency": int(self._scale_value(action_vector[9], 1, 30)),  # 天数
                "governance_participation_threshold": self._scale_value(action_vector[10], 0.01, 0.1),
                "penalty_severity": self._scale_value(action_vector[11], 0.1, 0.5)
            },
            
            # 激励机制设计
            "incentive_mechanisms": {
                "early_adopter_bonus": self._scale_value(action_vector[12], 0.05, 0.2),
                "referral_reward_rate": self._scale_value(action_vector[13], 0.1, 0.3),
                "loyalty_multiplier": self._scale_value(action_vector[14], 1.0, 2.0)
            }
        }
        
        # 生成具体的奖励分配方案
        reward_allocations = self._generate_reward_allocations(decisions, current_state)
        decisions["reward_allocations"] = reward_allocations
        
        # 添加决策推理
        reasoning = self._generate_reasoning(decisions, current_state, action.confidence)
        decisions["reasoning"] = reasoning
        decisions["confidence"] = action.confidence
        
        return decisions
    
    def _scale_value(self, normalized_value: float, min_val: float, max_val: float) -> float:
        """将归一化值缩放到实际范围"""
        return min_val + normalized_value * (max_val - min_val)
    
    def _generate_reward_allocations(self, decisions: Dict[str, Any], 
                                   current_state: BlockchainState) -> List[Dict[str, Any]]:
        """生成具体的奖励分配方案"""
        
        allocations = []
        strategy_adj = decisions["reward_strategy_adjustments"]
        
        for user_contrib in current_state.user_contributions:
            # 基础奖励
            base_weight = self.reward_strategy["base_reward_weights"][user_contrib.contribution_type.value]
            base_reward = user_contrib.contribution_value * base_weight * strategy_adj["base_reward_multiplier"]
            
            # 质量奖励
            quality_bonus = 0
            for threshold, bonus in zip(self.reward_strategy["quality_bonus_curve"]["thresholds"], 
                                       self.reward_strategy["quality_bonus_curve"]["bonuses"]):
                if user_contrib.data_quality >= threshold:
                    quality_bonus = bonus * strategy_adj["quality_bonus_emphasis"]
            
            # 一致性奖励
            consistency_level = "high" if user_contrib.historical_consistency > 0.8 else "medium" if user_contrib.historical_consistency > 0.6 else "low"
            consistency_multiplier = self.reward_strategy["consistency_multipliers"][consistency_level] * strategy_adj["consistency_reward_factor"]
            
            # 特殊奖励
            special_bonus = 0
            if user_contrib.frequency > 0.8:  # 高频贡献者
                special_bonus += self.reward_strategy["special_bonus_criteria"]["high_frequency"]
            
            # 总奖励计算
            total_reward = base_reward * (1 + quality_bonus) * consistency_multiplier + special_bonus
            
            allocations.append({
                "user_id": user_contrib.user_id,
                "contribution_type": user_contrib.contribution_type.value,
                "base_reward": base_reward,
                "quality_bonus": quality_bonus,
                "consistency_multiplier": consistency_multiplier,
                "special_bonus": special_bonus,
                "total_reward": total_reward
            })
        
        return allocations
    
    def _generate_reasoning(self, decisions: Dict[str, Any], 
                           current_state: BlockchainState, confidence: float) -> str:
        """生成决策推理说明"""
        
        reasoning_parts = []
        strategy_adj = decisions["reward_strategy_adjustments"]
        revenue_adj = decisions["revenue_distribution_adjustments"]
        
        # 奖励策略调整推理
        if strategy_adj["base_reward_multiplier"] > 1.1:
            reasoning_parts.append("提高基础奖励以激励更多用户参与")
        elif strategy_adj["base_reward_multiplier"] < 0.9:
            reasoning_parts.append("适当降低基础奖励以优化资源分配")
        
        if strategy_adj["quality_bonus_emphasis"] > 1.5:
            reasoning_parts.append("强调数据质量奖励，提升整体数据价值")
        
        # 收益分配推理
        if revenue_adj["user_rewards_share"] > 0.25:
            reasoning_parts.append("增加用户奖励份额，增强社区参与度")
        
        if revenue_adj["ai_development_share"] > 0.35:
            reasoning_parts.append("加大AI研发投入，提升系统长期竞争力")
        
        # 系统状态相关推理
        total_revenue = current_state.system_revenue.total_revenue
        if total_revenue > 1000000:  # 高收入时期
            reasoning_parts.append("高收入时期，适当增加储备基金")
        elif total_revenue < 100000:  # 低收入时期
            reasoning_parts.append("低收入时期，优先保障核心运营")
        
        # 置信度说明
        confidence_level = "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低"
        reasoning_parts.append(f"决策置信度: {confidence_level}({confidence:.2f})")
        
        return "; ".join(reasoning_parts)
    
    def calculate_reward(self, previous_state: BlockchainState, 
                        current_state: BlockchainState, 
                        decisions: Dict[str, Any]) -> float:
        """计算决策奖励"""
        
        reward = 0.0
        
        # 1. 用户参与度奖励
        previous_users = len(previous_state.user_contributions)
        current_users = len(current_state.user_contributions)
        user_growth = (current_users - previous_users) / max(previous_users, 1)
        user_reward = min(user_growth, 1.0) * 0.3
        reward += user_reward
        
        # 2. 贡献质量奖励
        if current_state.user_contributions:
            avg_quality = np.mean([uc.data_quality for uc in current_state.user_contributions])
            quality_reward = avg_quality * 0.2
            reward += quality_reward
        
        # 3. 收入增长奖励
        revenue_growth = current_state.system_revenue.growth_rate
        revenue_reward = max(0, revenue_growth) * 0.3
        reward += revenue_reward
        
        # 4. 分配公平性奖励
        fairness_reward = self._calculate_fairness_score(current_state, decisions) * 0.2
        reward += fairness_reward
        
        # 5. 惩罚项 - 系统健康度下降
        previous_health = previous_state.system_metrics.get("system_health", 1.0)
        current_health = current_state.system_metrics.get("system_health", 1.0)
        if current_health < previous_health:
            reward -= (previous_health - current_health) * 0.5
        
        return max(reward, 0)
    
    def _calculate_fairness_score(self, current_state: BlockchainState, 
                                 decisions: Dict[str, Any]) -> float:
        """计算分配公平性得分"""
        
        if not current_state.user_contributions:
            return 0.5  # 默认中等公平性
        
        # 计算奖励分配的基尼系数
        rewards = [alloc["total_reward"] for alloc in decisions.get("reward_allocations", [])]
        if not rewards:
            return 0.5
        
        # 简化版的基尼系数计算
        sorted_rewards = sorted(rewards)
        n = len(sorted_rewards)
        cumulative_rewards = np.cumsum(sorted_rewards)
        total_rewards = cumulative_rewards[-1]
        
        if total_rewards == 0:
            return 0.5
        
        # 计算基尼系数 (0表示完全平等，1表示完全不平等)
        gini_coefficient = 1 - 2 * np.trapz(cumulative_rewards / total_rewards, 
                                           np.linspace(0, 1, n)) / (n - 1)
        
        # 转换为公平性得分 (越高越好)
        fairness_score = 1 - gini_coefficient
        
        return max(0, min(1, fairness_score))
    
    def _estimate_user_satisfaction(self, distributions: List[RewardDistribution]) -> float:
        """估计用户满意度"""
        if not distributions:
            return 0.5
        
        # 基于奖励分配的公平性和可预测性估计满意度
        rewards = [d.total_reward for d in distributions]
        avg_reward = np.mean(rewards)
        reward_variance = np.var(rewards)
        
        # 奖励越高，方差越小，满意度越高
        satisfaction = min(avg_reward / 1000, 1.0)  # 假设1000为高奖励阈值
        predictability = 1 - min(reward_variance / (avg_reward ** 2 + 1e-8), 1.0)
        
        return (satisfaction + predictability) / 2
    
    def validate_decisions(self, decisions: Dict[str, Any], 
                          current_state: BlockchainState) -> Tuple[bool, Optional[str]]:
        """验证决策的合理性"""
        
        # 检查收益分配比例
        revenue_adj = decisions["revenue_distribution_adjustments"]
        total_share = sum(revenue_adj.values())
        
        if abs(total_share - 1.0) > 0.01:
            return False, f"收益分配比例总和必须为1.0，当前为{total_share:.3f}"
        
        # 检查用户奖励份额
        if revenue_adj["user_rewards_share"] < 0.1:
            return False, "用户奖励份额过低，可能影响参与积极性"
        
        # 检查智能合约参数
        contract_params = decisions["smart_contract_parameters"]
        if contract_params["minimum_contribution"] > 500:
            return False, "最低贡献门槛过高，可能阻碍新用户参与"
        
        return True, None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "total_distributions": len(self.decision_history),
            "average_reward_amount": np.mean([d.get("total_reward", 0) for d in self.decision_history]) if self.decision_history else 0,
            "user_satisfaction_estimate": self._estimate_recent_satisfaction()
        }
    
    def _estimate_recent_satisfaction(self) -> float:
        """估计近期用户满意度"""
        if len(self.decision_history) < 5:
            return 0.5
        
        recent_decisions = self.decision_history[-5:]
        avg_confidence = np.mean([d.get("confidence", 0) for d in recent_decisions])
        avg_reward = np.mean([d.get("reward", 0) for d in recent_decisions])
        
        return (avg_confidence + min(avg_reward, 1.0)) / 2


# 示例使用代码
if __name__ == "__main__":
    # 创建测试数据
    test_contributions = [
        UserContribution(
            user_id="user_001",
            contribution_type=ContributionType.DATA_PROVIDING,
            contribution_value=1500.0,
            data_quality=0.85,
            timeliness=0.9,
            frequency=0.8,
            historical_consistency=0.75
        ),
        UserContribution(
            user_id="user_002", 
            contribution_type=ContributionType.MODEL_TRAINING,
            contribution_value=800.0,
            data_quality=0.92,
            timeliness=0.85,
            frequency=0.6,
            historical_consistency=0.88
        )
    ]
    
    test_revenue = SystemRevenue(
        period="2024-Q1",
        total_revenue=500000.0,
        revenue_breakdown={
            RevenueSource.USER_SUBSCRIPTION: 200000,
            RevenueSource.BUSINESS_SERVICE: 150000,
            RevenueSource.DATA_SALES: 100000,
            RevenueSource.HARDWARE_SALES: 40000,
            RevenueSource.ADVERTISING: 10000
        },
        growth_rate=0.15,
        predictability=0.8
    )
    
    test_state = BlockchainState(
        user_contributions=test_contributions,
        system_revenue=test_revenue,
        historical_distributions=[],
        community_governance_rules={"voting_threshold": 0.6},
        system_metrics={
            "active_users": 1500,
            "transaction_volume": 2500000,
            "system_health": 0.95,
            "community_engagement": 0.72
        }
    )
    
    # 测试决策模块
    module = BlockchainDecisionModule()
    decision_state = module.create_decision_state(test_state)
    
    print(f"区块链决策状态向量维度: {len(decision_state.state_vector)}")
    print(f"决策状态上下文: {decision_state.context}")
    
    # 模拟决策动作
    test_action = DecisionAction(
        decision_type=DecisionType.BLOCKCHAIN,
        action_vector=np.random.rand(15),
        confidence=0.78,
        timestamp=datetime.now().timestamp()
    )
    
    decisions = module.interpret_decision_action(test_action, test_state)
    print(f"区块链决策结果: {json.dumps(decisions, indent=2, default=str)}")
    
    # 计算奖励
    reward = module.calculate_reward(test_state, test_state, decisions)
    print(f"决策奖励: {reward:.3f}")