"""
决策服务核心引擎 - 基于强化学习的农业参数优化决策
"""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging

from ..models.decision_models import DecisionAction, DecisionObjective


logger = logging.getLogger(__name__)


@dataclass
class AgricultureState:
    """农业环境状态"""
    temperature: float
    humidity: float
    co2_level: float
    light_intensity: float
    spectrum_config: Dict[str, float]
    crop_type: str
    growth_day: int
    growth_rate: float
    health_score: float
    yield_potential: float
    energy_consumption: float
    resource_utilization: float


@dataclass
class DecisionResult:
    """决策结果"""
    action: DecisionAction
    parameters: Dict[str, float]
    expected_reward: float
    confidence: float
    execution_time: float


class AgricultureRLPolicy(nn.Module):
    """农业强化学习策略网络"""
    hidden_dim: int = 256
    num_actions: int = 5
    
    @nn.compact
    def __call__(self, state_features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """输入状态特征，输出动作概率和价值估计"""
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


class DecisionEngine:
    """决策引擎"""
    
    def __init__(self):
        self.rl_policy = AgricultureRLPolicy()
        self.decision_history = []
        self.reward_history = []
        self._initialize_policy()
        
        # 作物配置知识库
        self.crop_configs = self._load_crop_configs()
        
        logger.info("决策引擎初始化完成")
    
    def _initialize_policy(self):
        """初始化策略网络参数"""
        try:
            import jax.random
            dummy_state = jnp.ones(15)  # 15维状态特征
            self.policy_params = self.rl_policy.init(
                jax.random.PRNGKey(42), dummy_state
            )
            logger.info("策略网络参数初始化成功")
        except Exception as e:
            logger.error(f"策略网络初始化失败: {e}")
            # 使用备用初始化
            self.policy_params = None
    
    def make_decision(self, 
                     current_state: AgricultureState,
                     objective: DecisionObjective) -> DecisionResult:
        """基于当前状态和目标做出决策"""
        start_time = time.time()
        
        try:
            # 提取状态特征
            state_features = self._extract_state_features(current_state, objective)
            
            # 使用策略网络选择动作
            if self.policy_params is not None:
                action_probs, value_estimate = self.rl_policy.apply(
                    self.policy_params, state_features
                )
                action_idx = jnp.argmax(action_probs)
                action = list(DecisionAction)[action_idx]
                confidence = float(action_probs[action_idx])
            else:
                # 备用决策逻辑
                action, confidence = self._fallback_decision(current_state, objective)
            
            # 生成动作参数
            parameters = self._generate_action_parameters(
                action, current_state, objective
            )
            
            # 计算预期奖励
            expected_reward = self._calculate_expected_reward(
                current_state, action, parameters, objective
            )
            
            execution_time = time.time() - start_time
            
            logger.info(f"决策完成: {action.value}, 预期奖励: {expected_reward:.3f}, 置信度: {confidence:.3f}")
            
            return DecisionResult(
                action=action,
                parameters=parameters,
                expected_reward=expected_reward,
                confidence=confidence,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"决策失败: {e}")
            # 返回安全决策
            return DecisionResult(
                action=DecisionAction.NO_ACTION,
                parameters={},
                expected_reward=0.0,
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    def _extract_state_features(self, 
                               state: AgricultureState,
                               objective: DecisionObjective) -> jnp.ndarray:
        """提取状态特征向量"""
        features = [
            # 环境参数
            state.temperature / 50.0,  # 归一化
            state.humidity / 100.0,
            state.co2_level / 1000.0,
            state.light_intensity / 1000.0,
            
            # 光谱参数
            state.spectrum_config.get('uv_380nm', 0.05),
            state.spectrum_config.get('far_red_720nm', 0.1),
            state.spectrum_config.get('white_light', 0.7),
            state.spectrum_config.get('red_660nm', 0.15),
            
            # 作物状态
            state.growth_day / 100.0,
            state.growth_rate,
            state.health_score,
            state.yield_potential,
            
            # 系统状态
            state.energy_consumption / 10.0,
            state.resource_utilization,
            
            # 目标编码
            self._encode_objective(objective)
        ]
        
        return jnp.array(features, dtype=jnp.float32)
    
    def _encode_objective(self, objective: DecisionObjective) -> float:
        """编码决策目标"""
        objective_encoding = {
            DecisionObjective.MAXIMIZE_YIELD: 0.0,
            DecisionObjective.IMPROVE_QUALITY: 0.25,
            DecisionObjective.ENHANCE_RESISTANCE: 0.5,
            DecisionObjective.OPTIMIZE_EFFICIENCY: 0.75
        }
        return objective_encoding.get(objective, 0.0)
    
    def _fallback_decision(self, 
                          state: AgricultureState,
                          objective: DecisionObjective) -> tuple[DecisionAction, float]:
        """备用决策逻辑"""
        # 基于规则的简单决策
        if objective == DecisionObjective.MAXIMIZE_YIELD:
            return DecisionAction.ADJUST_SPECTRUM, 0.8
        elif objective == DecisionObjective.IMPROVE_QUALITY:
            return DecisionAction.ADJUST_NUTRIENTS, 0.7
        else:
            return DecisionAction.NO_ACTION, 0.5
    
    def _generate_action_parameters(self,
                                  action: DecisionAction,
                                  state: AgricultureState,
                                  objective: DecisionObjective) -> Dict[str, float]:
        """生成动作具体参数"""
        if action == DecisionAction.ADJUST_SPECTRUM:
            return self._generate_spectrum_parameters(state, objective)
        elif action == DecisionAction.ADJUST_TEMPERATURE:
            return self._generate_temperature_parameters(state, objective)
        elif action == DecisionAction.ADJUST_HUMIDITY:
            return self._generate_humidity_parameters(state, objective)
        elif action == DecisionAction.ADJUST_NUTRIENTS:
            return self._generate_nutrient_parameters(state, objective)
        else:
            return {}
    
    def _generate_spectrum_parameters(self, 
                                     state: AgricultureState,
                                     objective: DecisionObjective) -> Dict[str, float]:
        """生成光谱调整参数"""
        if objective == DecisionObjective.MAXIMIZE_YIELD:
            return {
                "red_660nm_adjustment": 0.1,
                "white_light_adjustment": -0.05,
                "uv_380nm_adjustment": 0.0,
                "far_red_720nm_adjustment": 0.0
            }
        elif objective == DecisionObjective.IMPROVE_QUALITY:
            return {
                "uv_380nm_adjustment": 0.15,
                "far_red_720nm_adjustment": 0.05,
                "red_660nm_adjustment": 0.0,
                "white_light_adjustment": -0.05
            }
        else:
            return {
                "red_660nm_adjustment": 0.02,
                "white_light_adjustment": -0.01,
                "uv_380nm_adjustment": 0.01,
                "far_red_720nm_adjustment": 0.01
            }
    
    def _generate_temperature_parameters(self,
                                        state: AgricultureState,
                                        objective: DecisionObjective) -> Dict[str, float]:
        """生成温度调整参数"""
        optimal_temp = self._get_optimal_temperature(state.crop_type, state.growth_day)
        current_temp = state.temperature
        
        temp_diff = optimal_temp - current_temp
        adjustment = np.clip(temp_diff / 5.0, -1.0, 1.0)
        
        return {"temperature_adjustment": adjustment}
    
    def _generate_humidity_parameters(self,
                                     state: AgricultureState,
                                     objective: DecisionObjective) -> Dict[str, float]:
        """生成湿度调整参数"""
        optimal_humidity = self._get_optimal_humidity(state.crop_type, state.growth_day)
        current_humidity = state.humidity
        
        humidity_diff = optimal_humidity - current_humidity
        adjustment = np.clip(humidity_diff / 20.0, -0.5, 0.5)
        
        return {"humidity_adjustment": adjustment}
    
    def _generate_nutrient_parameters(self,
                                     state: AgricultureState,
                                     objective: DecisionObjective) -> Dict[str, float]:
        """生成营养液调整参数"""
        growth_stage = self._get_growth_stage(state.crop_type, state.growth_day)
        
        if "苗期" in growth_stage:
            return {"npk_ratio_adjustment": 0.1, "micro_nutrients_adjustment": 0.05}
        elif "开花" in growth_stage:
            return {"npk_ratio_adjustment": -0.05, "phosphorus_boost": 0.15}
        elif "结果" in growth_stage:
            return {"npk_ratio_adjustment": 0.08, "potassium_boost": 0.1}
        else:
            return {"npk_ratio_adjustment": 0.02, "micro_nutrients_adjustment": 0.02}
    
    def _calculate_expected_reward(self,
                                  state: AgricultureState,
                                  action: DecisionAction,
                                  parameters: Dict[str, float],
                                  objective: DecisionObjective) -> float:
        """计算预期奖励"""
        base_reward = 0.0
        
        if objective == DecisionObjective.MAXIMIZE_YIELD:
            base_reward = state.yield_potential * 10.0
        elif objective == DecisionObjective.IMPROVE_QUALITY:
            base_reward = state.health_score * 8.0
        elif objective == DecisionObjective.ENHANCE_RESISTANCE:
            base_reward = (state.health_score + state.growth_rate) * 6.0
        else:  # OPTIMIZE_EFFICIENCY
            base_reward = (1.0 - state.energy_consumption) * 12.0
        
        # 动作有效性奖励
        action_effectiveness = self._evaluate_action_effectiveness(action, state)
        base_reward *= action_effectiveness
        
        # 参数合理性奖励
        parameter_quality = self._evaluate_parameter_quality(parameters, state)
        base_reward *= parameter_quality
        
        return float(base_reward)
    
    def _evaluate_action_effectiveness(self, action: DecisionAction, state: AgricultureState) -> float:
        """评估动作有效性"""
        if action == DecisionAction.ADJUST_SPECTRUM:
            deviation = abs(state.spectrum_config.get('white_red_ratio', 4.67) - 31.0)
            return max(0.5, 1.0 - deviation * 0.1)
        elif action == DecisionAction.ADJUST_TEMPERATURE:
            optimal_temp = self._get_optimal_temperature(state.crop_type, state.growth_day)
            temp_diff = abs(state.temperature - optimal_temp)
            return max(0.3, 1.0 - temp_diff / 10.0)
        else:
            return 0.7
    
    def _evaluate_parameter_quality(self, parameters: Dict[str, float], state: AgricultureState) -> float:
        """评估参数质量"""
        quality_score = 1.0
        
        for param, value in parameters.items():
            if "adjustment" in param:
                if abs(value) > 0.5:
                    quality_score *= 0.8
                elif abs(value) < 0.01:
                    quality_score *= 0.9
        
        return quality_score
    
    def _get_optimal_temperature(self, crop_type: str, growth_day: int) -> float:
        """获取最优温度"""
        crop_config = self.crop_configs.get(crop_type)
        if crop_config:
            return crop_config.get('optimal_temperature', 25.0)
        return 25.0
    
    def _get_optimal_humidity(self, crop_type: str, growth_day: int) -> float:
        """获取最优湿度"""
        crop_config = self.crop_configs.get(crop_type)
        if crop_config:
            return crop_config.get('optimal_humidity', 60.0)
        return 60.0
    
    def _get_growth_stage(self, crop_type: str, growth_day: int) -> str:
        """获取生长阶段"""
        crop_config = self.crop_configs.get(crop_type)
        if crop_config:
            stages = crop_config.get('growth_stages', [])
            for stage in stages:
                if stage['min_day'] <= growth_day <= stage['max_day']:
                    return stage['name']
        return "生长期"
    
    def _load_crop_configs(self) -> Dict[str, Dict]:
        """加载作物配置"""
        return {
            "番茄": {
                "optimal_temperature": 25.0,
                "optimal_humidity": 60.0,
                "growth_stages": [
                    {"name": "苗期", "min_day": 0, "max_day": 30},
                    {"name": "开花期", "min_day": 31, "max_day": 50},
                    {"name": "结果期", "min_day": 51, "max_day": 95}
                ]
            },
            "生菜": {
                "optimal_temperature": 22.0,
                "optimal_humidity": 70.0,
                "growth_stages": [
                    {"name": "苗期", "min_day": 0, "max_day": 15},
                    {"name": "生长期", "min_day": 16, "max_day": 40}
                ]
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取决策性能指标"""
        if not self.reward_history:
            return {"average_reward": 0.0, "decision_count": 0}
        
        return {
            "average_reward": float(np.mean(self.reward_history)),
            "decision_count": len(self.reward_history),
            "recent_success_rate": float(np.mean(self.reward_history[-10:]) > 0.5)
        }