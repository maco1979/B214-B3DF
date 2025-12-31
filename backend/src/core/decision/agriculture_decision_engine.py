"""
农业参数优化决策引擎 - 基于强化学习的自主决策系统
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..models.agriculture_model import AgricultureAIService, SpectrumConfig, CropConfig


class DecisionAction(Enum):
    """决策动作枚举"""
    ADJUST_SPECTRUM = "adjust_spectrum"
    ADJUST_TEMPERATURE = "adjust_temperature" 
    ADJUST_HUMIDITY = "adjust_humidity"
    ADJUST_NUTRIENTS = "adjust_nutrients"
    CONTROL_CAMERA = "control_camera"
    NO_ACTION = "no_action"


class DecisionObjective(Enum):
    """决策目标枚举"""
    MAXIMIZE_YIELD = "maximize_yield"
    IMPROVE_QUALITY = "improve_quality"
    ENHANCE_RESISTANCE = "enhance_resistance"
    OPTIMIZE_EFFICIENCY = "optimize_efficiency"


@dataclass
class AgricultureState:
    """农业环境状态"""
    # 环境参数
    temperature: float
    humidity: float
    co2_level: float
    light_intensity: float
    
    # 光谱参数
    spectrum_config: SpectrumConfig
    
    # 作物状态
    crop_type: str
    growth_day: int
    growth_rate: float
    health_score: float
    yield_potential: float
    
    # 系统状态
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
    num_actions: int = 6
    
    @nn.compact
    def __call__(self, state_features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        输入状态特征，输出动作概率和价值估计
        """
        # 特征编码层
        x = nn.Dense(self.hidden_dim)(state_features)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # 策略头
        policy_logits = nn.Dense(self.num_actions)(x)
        action_probs = nn.softmax(policy_logits)
        
        # 价值头
        value_estimate = nn.Dense(1)(x)
        
        return action_probs, value_estimate


from ..decision_engine import DecisionEngine

class AgricultureDecisionEngine(DecisionEngine):
    """农业参数优化决策引擎"""
    
    def __init__(self, agriculture_service: AgricultureAIService):
        super().__init__()
        self.agriculture_service = agriculture_service
        
        # 决策历史记录
        self.decision_history = []
        self.reward_history = []
        
        # 使用简化的规则引擎而不是神经网络（避免 Flax/Python 3.14 兼容性问题）
        self.use_simple_rules = True
    
    def _initialize_policy(self):
        """初始化策略（简化版本）"""
        # 使用简化规则，不需要神经网络
        pass
    
    async def make_decision(self, decision_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        实现DecisionEngine接口的异步决策方法
        
        Args:
            decision_request: 决策请求数据字典
            
        Returns:
            决策结果字典
        """
        # 将字典请求转换为AgricultureState
        spectrum_config = SpectrumConfig(**decision_request.get("spectrum_config", {}))
        current_state = AgricultureState(
            temperature=decision_request.get("temperature", 25.0),
            humidity=decision_request.get("humidity", 60.0),
            co2_level=decision_request.get("co2_level", 400.0),
            light_intensity=decision_request.get("light_intensity", 500.0),
            spectrum_config=spectrum_config,
            crop_type=decision_request.get("crop_type", "unknown"),
            growth_day=decision_request.get("growth_day", 1),
            growth_rate=decision_request.get("growth_rate", 1.0),
            health_score=decision_request.get("health_score", 1.0)
        )
        
        # 获取决策目标
        objective_str = decision_request.get("objective", "maximize_yield")
        try:
            objective = DecisionObjective(objective_str)
        except ValueError:
            objective = DecisionObjective.MAXIMIZE_YIELD
        
        # 调用同步决策方法
        decision_result = self._sync_make_decision(current_state, objective)
        
        # 转换结果为字典
        return {
            "action": decision_result.action.value,
            "parameters": decision_result.parameters,
            "expected_reward": decision_result.expected_reward,
            "confidence": decision_result.confidence,
            "execution_time": decision_result.execution_time
        }
    
    def _sync_make_decision(self, 
                     current_state: AgricultureState,
                     objective: DecisionObjective) -> DecisionResult:
        """
        基于当前状态和目标做出决策（同步方法，供内部使用）
        使用简化的规则引擎而不是神经网络
        """
        import random
        import time
        
        start_time = time.time()
        
        # 基于规则的决策逻辑
        action = self._select_action_by_rules(current_state, objective)
        
        # 生成动作参数
        parameters = self._generate_action_parameters(
            action, current_state, objective
        )
        
        # 计算预期奖励
        expected_reward = self._calculate_expected_reward(
            current_state, action, parameters, objective
        )
        
        # 计算置信度（基于状态健康度）
        confidence = min(0.95, current_state.health_score * 0.8 + 0.15 + random.uniform(0, 0.1))
        
        execution_time = time.time() - start_time
        
        return DecisionResult(
            action=action,
            parameters=parameters,
            expected_reward=float(expected_reward),
            confidence=float(confidence),
            execution_time=execution_time
        )
    
    def _select_action_by_rules(self, state: AgricultureState, objective: DecisionObjective) -> DecisionAction:
        """基于规则选择动作"""
        # 温度异常检测
        if state.temperature < 15 or state.temperature > 35:
            return DecisionAction.ADJUST_TEMPERATURE
        
        # 湿度异常检测
        if state.humidity < 40 or state.humidity > 85:
            return DecisionAction.ADJUST_HUMIDITY
        
        # 根据目标选择动作
        if objective == DecisionObjective.MAXIMIZE_YIELD:
            if state.growth_rate < 0.7:
                return DecisionAction.ADJUST_NUTRIENTS
            return DecisionAction.ADJUST_SPECTRUM
        
        elif objective == DecisionObjective.IMPROVE_QUALITY:
            return DecisionAction.ADJUST_SPECTRUM
        
        elif objective == DecisionObjective.ENHANCE_RESISTANCE:
            if state.health_score < 0.8:
                return DecisionAction.ADJUST_NUTRIENTS
            return DecisionAction.ADJUST_TEMPERATURE
        
        elif objective == DecisionObjective.OPTIMIZE_EFFICIENCY:
            if state.energy_consumption > 100:
                return DecisionAction.ADJUST_SPECTRUM
            return DecisionAction.NO_ACTION
        
        return DecisionAction.NO_ACTION
    
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
            state.spectrum_config.uv_380nm,
            state.spectrum_config.far_red_720nm,
            state.spectrum_config.white_light,
            state.spectrum_config.red_660nm,
            
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
        elif action == DecisionAction.CONTROL_CAMERA:
            return self._generate_camera_control_parameters(state, objective)
        else:
            return {}
    
    def _generate_spectrum_parameters(self, 
                                     state: AgricultureState,
                                     objective: DecisionObjective) -> Dict[str, float]:
        """生成光谱调整参数"""
        current_recipe = state.spectrum_config
        
        # 基于目标的光谱优化
        if objective == DecisionObjective.MAXIMIZE_YIELD:
            # 增加红光促进光合作用
            return {
                "red_660nm_adjustment": 0.1,
                "white_light_adjustment": -0.05,
                "uv_380nm_adjustment": 0.0,
                "far_red_720nm_adjustment": 0.0
            }
        elif objective == DecisionObjective.IMPROVE_QUALITY:
            # 增加紫外线促进次生代谢
            return {
                "uv_380nm_adjustment": 0.15,
                "far_red_720nm_adjustment": 0.05,
                "red_660nm_adjustment": 0.0,
                "white_light_adjustment": -0.05
            }
        else:
            # 默认微调
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
        
        # 计算温度调整量
        temp_diff = optimal_temp - current_temp
        adjustment = np.clip(temp_diff / 5.0, -1.0, 1.0)  # 限制调整幅度
        
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
        # 基于生长阶段和目标调整营养配比
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
        
        # 基于目标的奖励计算
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
        # 基于当前状态评估动作的适用性
        if action == DecisionAction.ADJUST_SPECTRUM:
            # 如果光谱参数偏离较大，调整更有效
            optimal_recipe = self._get_optimal_spectrum(state.crop_type, state.growth_day)
            current_recipe = state.spectrum_config
            
            deviation = abs(current_recipe.white_red_ratio - optimal_recipe.white_red_ratio)
            return max(0.5, 1.0 - deviation * 2.0)
        
        elif action == DecisionAction.ADJUST_TEMPERATURE:
            optimal_temp = self._get_optimal_temperature(state.crop_type, state.growth_day)
            temp_diff = abs(state.temperature - optimal_temp)
            return max(0.3, 1.0 - temp_diff / 10.0)
        
        elif action == DecisionAction.CONTROL_CAMERA:
            # 评估摄像头控制动作的有效性
            return self._evaluate_camera_control_effectiveness(state)
        
        else:
            return 0.7  # 默认有效性
    
    def _evaluate_parameter_quality(self, parameters: Dict[str, float], state: AgricultureState) -> float:
        """评估参数质量"""
        # 检查参数是否在合理范围内
        quality_score = 1.0
        
        for param, value in parameters.items():
            if "adjustment" in param:
                if abs(value) > 0.5:  # 调整幅度过大
                    quality_score *= 0.8
                elif abs(value) < 0.01:  # 调整幅度过小
                    quality_score *= 0.9
        
        return quality_score
    
    def _generate_camera_control_parameters(self, 
                                           state: AgricultureState,
                                           objective: DecisionObjective) -> Dict[str, float]:
        """生成摄像头控制参数"""
        # 基于作物状态和目标决定摄像头操作
        camera_params = {}
        
        # 如果作物健康状况不佳，启动视觉识别检测病虫害
        if state.health_score < 0.7:
            camera_params = {
                "camera_action": "start_recognition",
                "model_type": "haar",
                "domain": "camera_control"
            }
        
        # 如果是质量改进目标，定期进行视觉检查
        elif objective == DecisionObjective.IMPROVE_QUALITY:
            camera_params = {
                "camera_action": "start_tracking",
                "tracker_type": "CSRT",
                "domain": "camera_control"
            }
        
        # 如果是效率优化目标，确保摄像头按需使用
        elif objective == DecisionObjective.OPTIMIZE_EFFICIENCY:
            camera_params = {
                "camera_action": "list_cameras",
                "domain": "camera_control"
            }
        
        return camera_params
    
    def _evaluate_camera_control_effectiveness(self, state: AgricultureState) -> float:
        """评估摄像头控制动作的有效性"""
        # 根据当前状态评估摄像头控制的必要性
        if state.health_score < 0.7:
            # 健康状况不佳时，摄像头监测非常必要
            return 1.0
        elif state.growth_rate < 0.3:
            # 生长缓慢时，摄像头监测有一定必要性
            return 0.8
        else:
            # 正常生长状态下，摄像头监测必要性较低
            return 0.5
    
    def _get_optimal_temperature(self, crop_type: str, growth_day: int) -> float:
        """获取最优温度"""
        crop_config = self.agriculture_service.crop_configs.get(crop_type)
        if crop_config:
            current_stage = self.agriculture_service._get_current_stage(crop_config, growth_day)
            return sum(current_stage.optimal_temperature) / 2.0
        return 25.0  # 默认温度
    
    def _get_optimal_humidity(self, crop_type: str, growth_day: int) -> float:
        """获取最优湿度"""
        crop_config = self.agriculture_service.crop_configs.get(crop_type)
        if crop_config:
            current_stage = self.agriculture_service._get_current_stage(crop_config, growth_day)
            return sum(current_stage.optimal_humidity) / 2.0
        return 60.0  # 默认湿度
    
    def _get_growth_stage(self, crop_type: str, growth_day: int) -> str:
        """获取生长阶段"""
        crop_config = self.agriculture_service.crop_configs.get(crop_type)
        if crop_config:
            current_stage = self.agriculture_service._get_current_stage(crop_config, growth_day)
            return current_stage.stage_name
        return "生长期"
    
    def _get_optimal_spectrum(self, crop_type: str, growth_day: int) -> SpectrumConfig:
        """获取最优光谱配置"""
        # 使用农业服务生成基础配方
        environment = {"temperature": 25.0, "humidity": 60.0}
        try:
            result = self.agriculture_service.generate_light_recipe(
                crop_type, growth_day, "最大化产量", environment
            )
            return result["recipe"]
        except:
            return SpectrumConfig()  # 返回默认配置
    
    def update_policy(self, state: AgricultureState, action: DecisionAction, 
                     reward: float, next_state: AgricultureState):
        """更新策略网络（简化实现）"""
        # 这里应该实现完整的强化学习更新算法
        # 目前使用简单的经验回放
        
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
            "recent_success_rate": float(np.mean(self.reward_history[-10:]) > 0.5)
        }