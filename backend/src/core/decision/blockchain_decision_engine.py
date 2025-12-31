"""
区块链积分分配决策引擎 - 基于强化学习的自主积分分配系统
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time


class BlockchainAction(Enum):
    """区块链决策动作枚举"""
    ALLOCATE_POINTS = "allocate_points"
    APPROVE_TRANSACTION = "approve_transaction"
    SET_TRANSACTION_FEE = "set_transaction_fee"
    ADJUST_INCENTIVES = "adjust_incentives"
    RISK_CONTROL = "risk_control"
    NO_ACTION = "no_action"


class BlockchainObjective(Enum):
    """区块链决策目标枚举"""
    MAXIMIZE_ECOSYSTEM_GROWTH = "maximize_ecosystem_growth"
    OPTIMIZE_FAIRNESS = "optimize_fairness"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"


@dataclass
class BlockchainState:
    """区块链系统状态"""
    # 用户行为数据
    user_contribution: float  # 用户贡献度
    user_activity: float  # 用户活跃度
    user_reputation: float  # 用户信誉度
    user_balance: float  # 用户积分余额
    
    # 市场状态
    market_demand: float  # 市场需求
    market_supply: float  # 市场供应
    transaction_volume: float  # 交易量
    average_transaction_value: float  # 平均交易价值
    
    # 系统状态
    total_points_issued: float  # 总发行积分
    points_in_circulation: float  # 流通积分
    system_utilization: float  # 系统利用率
    risk_level: float  # 风险水平
    
    # 时间因素
    time_since_last_decision: float  # 距离上次决策时间


@dataclass
class BlockchainDecisionResult:
    """区块链决策结果"""
    action: BlockchainAction
    parameters: Dict[str, float]
    expected_reward: float
    confidence: float
    execution_time: float
    risk_assessment: Dict[str, float]


class BlockchainRLPolicy(nn.Module):
    """区块链强化学习策略网络"""
    hidden_dim: int = 256
    num_actions: int = 6
    
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


class BlockchainDecisionEngine:
    """区块链积分分配决策引擎"""
    
    def __init__(self):
        self.rl_policy = BlockchainRLPolicy()
        
        # 决策历史记录
        self.decision_history = []
        self.reward_history = []
        
        # 风险控制参数
        self.risk_thresholds = {
            "high_risk": 0.8,
            "medium_risk": 0.5,
            "low_risk": 0.2
        }
        
        # 初始化模型参数
        self._initialize_policy()
    
    def _initialize_policy(self):
        """初始化策略网络参数"""
        # 使用随机初始化
        dummy_state = jnp.ones(14)  # 14维状态特征
        self.policy_params = self.rl_policy.init(
            jax.random.PRNGKey(42), dummy_state
        )
    
    def make_decision(self, 
                     current_state: BlockchainState,
                     objective: BlockchainObjective) -> BlockchainDecisionResult:
        """
        基于当前状态和目标做出区块链决策
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
        action = list(BlockchainAction)[action_idx]
        
        # 生成动作参数
        parameters = self._generate_action_parameters(
            action, current_state, objective, risk_assessment
        )
        
        # 计算预期奖励
        expected_reward = self._calculate_expected_reward(
            current_state, action, parameters, objective, risk_assessment
        )
        
        return BlockchainDecisionResult(
            action=action,
            parameters=parameters,
            expected_reward=float(expected_reward),
            confidence=float(adjusted_probs[action_idx]),
            execution_time=0.05,  # 毫秒级响应
            risk_assessment=risk_assessment
        )
    
    def _assess_risk(self, state: BlockchainState) -> Dict[str, float]:
        """风险评估"""
        risk_scores = {}
        
        # 用户风险
        user_risk = 1.0 - min(state.user_reputation, 1.0)
        risk_scores["user_risk"] = user_risk
        
        # 市场风险
        market_imbalance = abs(state.market_demand - state.market_supply) / max(state.market_demand, state.market_supply, 1.0)
        risk_scores["market_risk"] = market_imbalance
        
        # 系统风险
        system_risk = max(0, state.system_utilization - 0.8) * 5.0  # 超过80%利用率风险增加
        risk_scores["system_risk"] = system_risk
        
        # 综合风险
        total_risk = (user_risk * 0.4 + market_risk * 0.3 + system_risk * 0.3)
        risk_scores["total_risk"] = total_risk
        
        # 风险等级
        if total_risk > self.risk_thresholds["high_risk"]:
            risk_scores["risk_level"] = 3.0  # 高风险
        elif total_risk > self.risk_thresholds["medium_risk"]:
            risk_scores["risk_level"] = 2.0  # 中风险
        else:
            risk_scores["risk_level"] = 1.0  # 低风险
        
        return risk_scores
    
    def _extract_state_features(self, 
                               state: BlockchainState,
                               objective: BlockchainObjective) -> jnp.ndarray:
        """提取状态特征向量"""
        features = [
            # 用户行为数据
            state.user_contribution,
            state.user_activity,
            state.user_reputation,
            state.user_balance / 1000.0,  # 归一化
            
            # 市场状态
            state.market_demand / 1000.0,
            state.market_supply / 1000.0,
            state.transaction_volume / 10000.0,
            state.average_transaction_value / 100.0,
            
            # 系统状态
            state.total_points_issued / 100000.0,
            state.points_in_circulation / 100000.0,
            state.system_utilization,
            state.risk_level,
            
            # 时间因素
            min(state.time_since_last_decision / 3600.0, 24.0),  # 最大24小时
            
            # 目标编码
            self._encode_objective(objective)
        ]
        
        return jnp.array(features, dtype=jnp.float32)
    
    def _encode_objective(self, objective: BlockchainObjective) -> float:
        """编码决策目标"""
        objective_encoding = {
            BlockchainObjective.MAXIMIZE_ECOSYSTEM_GROWTH: 0.0,
            BlockchainObjective.OPTIMIZE_FAIRNESS: 0.25,
            BlockchainObjective.MINIMIZE_RISK: 0.5,
            BlockchainObjective.MAXIMIZE_EFFICIENCY: 0.75
        }
        return objective_encoding.get(objective, 0.0)
    
    def _adjust_probs_by_risk(self, action_probs: jnp.ndarray, risk_assessment: Dict[str, float]) -> jnp.ndarray:
        """基于风险评估调整动作概率"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        # 高风险时倾向于保守动作
        if risk_level >= 3.0:
            # 增加风险控制和审批动作的概率
            risk_control_idx = list(BlockchainAction).index(BlockchainAction.RISK_CONTROL)
            approve_idx = list(BlockchainAction).index(BlockchainAction.APPROVE_TRANSACTION)
            
            adjusted_probs = jnp.array(action_probs)
            adjusted_probs = adjusted_probs.at[risk_control_idx].multiply(1.5)
            adjusted_probs = adjusted_probs.at[approve_idx].multiply(1.3)
            
            # 重新归一化
            return adjusted_probs / jnp.sum(adjusted_probs)
        
        return action_probs
    
    def _generate_action_parameters(self,
                                  action: BlockchainAction,
                                  state: BlockchainState,
                                  objective: BlockchainObjective,
                                  risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成动作具体参数"""
        if action == BlockchainAction.ALLOCATE_POINTS:
            return self._generate_points_allocation_parameters(state, objective, risk_assessment)
        elif action == BlockchainAction.APPROVE_TRANSACTION:
            return self._generate_transaction_approval_parameters(state, objective, risk_assessment)
        elif action == BlockchainAction.SET_TRANSACTION_FEE:
            return self._generate_fee_parameters(state, objective, risk_assessment)
        elif action == BlockchainAction.ADJUST_INCENTIVES:
            return self._generate_incentive_parameters(state, objective, risk_assessment)
        elif action == BlockchainAction.RISK_CONTROL:
            return self._generate_risk_control_parameters(state, objective, risk_assessment)
        else:
            return {}
    
    def _generate_points_allocation_parameters(self, 
                                             state: BlockchainState,
                                             objective: BlockchainObjective,
                                             risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成积分分配参数"""
        base_points = 100.0
        
        # 基于贡献度调整
        contribution_factor = state.user_contribution * 10.0
        
        # 基于活跃度调整
        activity_factor = state.user_activity * 5.0
        
        # 基于目标调整
        if objective == BlockchainObjective.MAXIMIZE_ECOSYSTEM_GROWTH:
            growth_bonus = 50.0
        elif objective == BlockchainObjective.OPTIMIZE_FAIRNESS:
            growth_bonus = 30.0
        else:
            growth_bonus = 20.0
        
        # 基于风险调整
        risk_factor = 1.0 - risk_assessment.get("total_risk", 0.0)
        
        total_points = base_points + contribution_factor + activity_factor + growth_bonus
        total_points *= risk_factor
        
        return {
            "points_to_allocate": max(10.0, total_points),
            "allocation_reason": "contribution_and_activity",
            "risk_adjustment_factor": risk_factor
        }
    
    def _generate_transaction_approval_parameters(self,
                                                 state: BlockchainState,
                                                 objective: BlockchainObjective,
                                                 risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成交易审批参数"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        # 高风险时增加审批严格度
        if risk_level >= 3.0:
            approval_threshold = 0.9
            review_required = 1.0
        elif risk_level >= 2.0:
            approval_threshold = 0.7
            review_required = 0.5
        else:
            approval_threshold = 0.5
            review_required = 0.0
        
        return {
            "approval_threshold": approval_threshold,
            "review_required": review_required,
            "auto_approve_limit": 100.0 * (1.0 - risk_level / 3.0)
        }
    
    def _generate_fee_parameters(self,
                                state: BlockchainState,
                                objective: BlockchainObjective,
                                risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成交易费用参数"""
        base_fee = 0.01  # 1%基础费用
        
        # 基于系统利用率调整
        utilization_factor = max(0, state.system_utilization - 0.5) * 0.02
        
        # 基于风险调整
        risk_factor = risk_assessment.get("total_risk", 0.0) * 0.03
        
        total_fee = base_fee + utilization_factor + risk_factor
        
        return {
            "transaction_fee_rate": min(0.1, total_fee),  # 最大10%
            "min_fee": 0.1,
            "max_fee": 10.0
        }
    
    def _generate_incentive_parameters(self,
                                      state: BlockchainState,
                                      objective: BlockchainObjective,
                                      risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成激励机制参数"""
        base_multiplier = 1.0
        
        if objective == BlockchainObjective.MAXIMIZE_ECOSYSTEM_GROWTH:
            base_multiplier = 1.5
        elif objective == BlockchainObjective.OPTIMIZE_FAIRNESS:
            base_multiplier = 1.2
        
        # 基于市场供需调整
        demand_supply_ratio = state.market_demand / max(state.market_supply, 1.0)
        market_factor = min(2.0, max(0.5, demand_supply_ratio))
        
        return {
            "point_multiplier": base_multiplier * market_factor,
            "bonus_period_days": 7.0,
            "referral_bonus": 50.0
        }
    
    def _generate_risk_control_parameters(self,
                                         state: BlockchainState,
                                         objective: BlockchainObjective,
                                         risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """生成风险控制参数"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        if risk_level >= 3.0:
            return {
                "transaction_limit": 100.0,
                "daily_limit": 1000.0,
                "manual_approval_required": 1.0,
                "risk_monitoring_intensity": 0.9
            }
        elif risk_level >= 2.0:
            return {
                "transaction_limit": 500.0,
                "daily_limit": 5000.0,
                "manual_approval_required": 0.5,
                "risk_monitoring_intensity": 0.7
            }
        else:
            return {
                "transaction_limit": 1000.0,
                "daily_limit": 10000.0,
                "manual_approval_required": 0.0,
                "risk_monitoring_intensity": 0.3
            }
    
    def _calculate_expected_reward(self,
                                  state: BlockchainState,
                                  action: BlockchainAction,
                                  parameters: Dict[str, float],
                                  objective: BlockchainObjective,
                                  risk_assessment: Dict[str, float]) -> float:
        """计算预期奖励"""
        base_reward = 0.0
        
        # 基于目标的奖励计算
        if objective == BlockchainObjective.MAXIMIZE_ECOSYSTEM_GROWTH:
            base_reward = state.user_activity * 20.0 + state.transaction_volume * 0.1
        elif objective == BlockchainObjective.OPTIMIZE_FAIRNESS:
            base_reward = state.user_reputation * 15.0 + (1.0 - abs(state.market_demand - state.market_supply)) * 10.0
        elif objective == BlockchainObjective.MINIMIZE_RISK:
            base_reward = (1.0 - risk_assessment.get("total_risk", 0.0)) * 25.0
        else:  # MAXIMIZE_EFFICIENCY
            base_reward = state.system_utilization * 20.0 + (1.0 - state.time_since_last_decision / 3600.0) * 5.0
        
        # 动作有效性奖励
        action_effectiveness = self._evaluate_action_effectiveness(action, state, risk_assessment)
        base_reward *= action_effectiveness
        
        # 参数合理性奖励
        parameter_quality = self._evaluate_parameter_quality(parameters, state)
        base_reward *= parameter_quality
        
        return float(base_reward)
    
    def _evaluate_action_effectiveness(self, 
                                      action: BlockchainAction, 
                                      state: BlockchainState,
                                      risk_assessment: Dict[str, float]) -> float:
        """评估动作有效性"""
        risk_level = risk_assessment.get("risk_level", 1.0)
        
        if action == BlockchainAction.RISK_CONTROL:
            # 高风险时风险控制更有效
            return max(0.3, risk_level / 3.0)
        elif action == BlockchainAction.ALLOCATE_POINTS:
            # 低风险时积分分配更有效
            return max(0.5, 1.0 - risk_level / 3.0)
        else:
            return 0.7  # 默认有效性
    
    def _evaluate_parameter_quality(self, parameters: Dict[str, float], state: BlockchainState) -> float:
        """评估参数质量"""
        quality_score = 1.0
        
        # 检查参数是否在合理范围内
        for param, value in parameters.items():
            if "fee" in param and value > 0.1:  # 费用过高
                quality_score *= 0.8
            elif "limit" in param and value < 10.0:  # 限制过低
                quality_score *= 0.7
        
        return quality_score
    
    def update_policy(self, state: BlockchainState, action: BlockchainAction, 
                     reward: float, next_state: BlockchainState):
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
            "risk_control_effectiveness": self._calculate_risk_control_effectiveness()
        }
    
    def _calculate_risk_control_effectiveness(self) -> float:
        """计算风险控制效果"""
        if len(self.decision_history) < 10:
            return 0.5
        
        # 分析最近的风险控制决策效果
        risk_control_decisions = [
            exp for exp in self.decision_history[-20:]
            if exp["action"] == BlockchainAction.RISK_CONTROL
        ]
        
        if not risk_control_decisions:
            return 0.5
        
        avg_reward = np.mean([exp["reward"] for exp in risk_control_decisions])
        return float(avg_reward / 10.0)  # 归一化