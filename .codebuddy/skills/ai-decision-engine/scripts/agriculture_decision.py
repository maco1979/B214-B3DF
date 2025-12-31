"""
农业决策模块 - 基于强化学习的农业参数优化决策系统
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


class CropType(Enum):
    """作物类型枚举"""
    TOMATO = "tomato"
    LETTUCE = "lettuce"
    STRAWBERRY = "strawberry"
    CUCUMBER = "cucumber"
    PEPPER = "pepper"


class GrowthStage(Enum):
    """生长阶段枚举"""
    SEEDLING = "seedling"      # 幼苗期
    VEGETATIVE = "vegetative"  # 营养生长期
    FLOWERING = "flowering"    # 开花期
    FRUITING = "fruiting"      # 结果期
    MATURE = "mature"          # 成熟期


class SpectrumBand(Enum):
    """光谱波段枚举"""
    ULTRAVIOLET = "ultraviolet"  # 380nm 紫外线
    BLUE = "blue"               # 450nm 蓝光
    GREEN = "green"             # 520nm 绿光
    RED = "red"                 # 660nm 红光
    FAR_RED = "far_red"         # 720nm 远红外


@dataclass
class AgricultureEnvironment:
    """农业环境数据类"""
    temperature: float           # 温度 (°C)
    humidity: float              # 湿度 (%)
    co2_concentration: float     # CO2浓度 (ppm)
    light_intensity: float       # 光照强度 (lux)
    soil_moisture: float         # 土壤湿度 (%)
    nutrient_level: float        # 营养水平
    ph_level: float              # pH值


@dataclass
class SpectrumConfig:
    """光谱配置数据类"""
    ultraviolet_intensity: float  # 380nm 强度
    blue_intensity: float         # 450nm 强度
    green_intensity: float        # 520nm 强度
    red_intensity: float          # 660nm 强度
    far_red_intensity: float      # 720nm 强度
    white_red_ratio: float        # 白红配比 (31:1)
    photoperiod: int              # 光周期 (小时)


@dataclass
class CropConfig:
    """作物配置数据类"""
    crop_type: CropType
    growth_stage: GrowthStage
    planting_day: int
    target_yield: float
    quality_targets: Dict[str, float]  # 质量目标


@dataclass
class AgricultureState:
    """农业状态数据类"""
    crop_config: CropConfig
    environment: AgricultureEnvironment
    spectrum_config: SpectrumConfig
    historical_performance: List[Dict[str, float]]
    current_day: int


class AgricultureDecisionModule:
    """农业决策模块"""
    
    def __init__(self):
        # 作物生长模型参数
        self.crop_models = self._load_crop_models()
        
        # 优化目标权重
        self.objective_weights = {
            "yield_maximization": 0.4,
            "quality_optimization": 0.3,
            "resource_efficiency": 0.2,
            "growth_speed": 0.1
        }
        
        # 决策历史记录
        self.decision_history = []
        
        logger.info("农业决策模块初始化完成")
    
    def _load_crop_models(self) -> Dict[CropType, Dict[str, Any]]:
        """加载作物生长模型"""
        # 基于农业科学研究的作物模型参数
        return {
            CropType.TOMATO: {
                "optimal_temperature": {"min": 18.0, "max": 28.0},
                "optimal_humidity": {"min": 60.0, "max": 80.0},
                "spectrum_preferences": {
                    GrowthStage.SEEDLING: {"blue": 0.6, "red": 0.4},
                    GrowthStage.VEGETATIVE: {"blue": 0.4, "red": 0.6},
                    GrowthStage.FLOWERING: {"red": 0.7, "far_red": 0.3},
                    GrowthStage.FRUITING: {"red": 0.8, "far_red": 0.2}
                },
                "growth_duration": 90  # 生长期天数
            },
            CropType.LETTUCE: {
                "optimal_temperature": {"min": 15.0, "max": 25.0},
                "optimal_humidity": {"min": 70.0, "max": 85.0},
                "spectrum_preferences": {
                    GrowthStage.SEEDLING: {"blue": 0.7, "red": 0.3},
                    GrowthStage.VEGETATIVE: {"blue": 0.5, "red": 0.5},
                    GrowthStage.MATURE: {"blue": 0.3, "red": 0.7}
                },
                "growth_duration": 60
            },
            CropType.STRAWBERRY: {
                "optimal_temperature": {"min": 16.0, "max": 26.0},
                "optimal_humidity": {"min": 65.0, "max": 80.0},
                "spectrum_preferences": {
                    GrowthStage.VEGETATIVE: {"blue": 0.4, "red": 0.6},
                    GrowthStage.FLOWERING: {"red": 0.6, "far_red": 0.4},
                    GrowthStage.FRUITING: {"red": 0.8, "far_red": 0.2}
                },
                "growth_duration": 75
            }
        }
    
    def create_decision_state(self, agriculture_state: AgricultureState) -> DecisionState:
        """创建农业决策状态向量"""
        
        # 状态特征工程
        state_features = []
        
        # 1. 环境参数特征
        env = agriculture_state.environment
        state_features.extend([
            env.temperature,
            env.humidity,
            env.co2_concentration,
            env.light_intensity,
            env.soil_moisture,
            env.nutrient_level,
            env.ph_level
        ])
        
        # 2. 光谱配置特征
        spectrum = agriculture_state.spectrum_config
        state_features.extend([
            spectrum.ultraviolet_intensity,
            spectrum.blue_intensity,
            spectrum.green_intensity,
            spectrum.red_intensity,
            spectrum.far_red_intensity,
            spectrum.white_red_ratio,
            spectrum.photoperiod
        ])
        
        # 3. 作物状态特征
        crop = agriculture_state.crop_config
        crop_model = self.crop_models[crop.crop_type]
        
        # 生长阶段编码
        growth_stage_encoding = {stage: idx for idx, stage in enumerate(GrowthStage)}
        state_features.append(growth_stage_encoding[crop.growth_stage])
        
        # 生长进度 (0-1)
        growth_progress = agriculture_state.current_day / crop_model["growth_duration"]
        state_features.append(growth_progress)
        
        # 4. 历史性能特征
        if agriculture_state.historical_performance:
            recent_performance = agriculture_state.historical_performance[-5:]  # 最近5次
            avg_yield = np.mean([p.get("yield", 0) for p in recent_performance])
            avg_quality = np.mean([p.get("quality_score", 0) for p in recent_performance])
        else:
            avg_yield, avg_quality = 0, 0
        
        state_features.extend([avg_yield, avg_quality])
        
        # 5. 目标差异特征
        target_yield_diff = crop.target_yield - avg_yield
        state_features.append(target_yield_diff)
        
        # 转换为numpy数组并归一化
        state_vector = np.array(state_features, dtype=np.float32)
        
        # 简单的归一化（实际应用中应该更复杂）
        state_vector = (state_vector - np.min(state_vector)) / (np.max(state_vector) - np.min(state_vector) + 1e-8)
        
        return DecisionState(
            decision_type=DecisionType.AGRICULTURE,
            state_vector=state_vector,
            timestamp=datetime.now().timestamp(),
            context={
                "crop_type": crop.crop_type.value,
                "growth_stage": crop.growth_stage.value,
                "current_day": agriculture_state.current_day
            }
        )
    
    def interpret_decision_action(self, action: DecisionAction, 
                                 current_state: AgricultureState) -> Dict[str, Any]:
        """解释决策动作为具体的农业操作"""
        
        action_vector = action.action_vector
        
        # 解析动作向量为具体的调整参数
        decisions = {
            # 光谱调整决策
            "spectrum_adjustments": {
                "ultraviolet_intensity": self._scale_value(action_vector[0], 0, 100),
                "blue_intensity": self._scale_value(action_vector[1], 0, 200),
                "green_intensity": self._scale_value(action_vector[2], 0, 150),
                "red_intensity": self._scale_value(action_vector[3], 0, 300),
                "far_red_intensity": self._scale_value(action_vector[4], 0, 100),
                "white_red_ratio": self._scale_value(action_vector[5], 20, 40),
                "photoperiod": int(self._scale_value(action_vector[6], 8, 16))
            },
            
            # 环境控制决策
            "environment_control": {
                "temperature_target": self._scale_value(action_vector[7], 15, 30),
                "humidity_target": self._scale_value(action_vector[8], 50, 85),
                "co2_target": self._scale_value(action_vector[9], 400, 1200),
                "irrigation_amount": self._scale_value(action_vector[10], 0, 100),
                "nutrient_adjustment": self._scale_value(action_vector[11], -10, 10)
            },
            
            # 生长管理决策
            "growth_management": {
                "pruning_intensity": self._scale_value(action_vector[12], 0, 1),
                "harvest_timing": int(self._scale_value(action_vector[13], 0, 7)),  # 提前/延后天数
                "pest_control_level": self._scale_value(action_vector[14], 0, 1)
            }
        }
        
        # 添加决策推理
        reasoning = self._generate_reasoning(decisions, current_state, action.confidence)
        decisions["reasoning"] = reasoning
        decisions["confidence"] = action.confidence
        
        return decisions
    
    def _scale_value(self, normalized_value: float, min_val: float, max_val: float) -> float:
        """将归一化值缩放到实际范围"""
        return min_val + normalized_value * (max_val - min_val)
    
    def _generate_reasoning(self, decisions: Dict[str, Any], 
                           current_state: AgricultureState, confidence: float) -> str:
        """生成决策推理说明"""
        
        reasoning_parts = []
        crop_type = current_state.crop_config.crop_type
        growth_stage = current_state.crop_config.growth_stage
        
        # 光谱调整推理
        spectrum_adj = decisions["spectrum_adjustments"]
        if spectrum_adj["red_intensity"] > 200:
            reasoning_parts.append("增强红光促进光合作用和果实成熟")
        if spectrum_adj["blue_intensity"] > 100:
            reasoning_parts.append("增强蓝光促进植株健壮和营养生长")
        
        # 环境控制推理
        env_control = decisions["environment_control"]
        current_temp = current_state.environment.temperature
        if env_control["temperature_target"] > current_temp + 2:
            reasoning_parts.append("适当提高温度加速生长进程")
        elif env_control["temperature_target"] < current_temp - 2:
            reasoning_parts.append("降低温度以优化能量利用效率")
        
        # 生长阶段特定推理
        if growth_stage == GrowthStage.FLOWERING:
            reasoning_parts.append("开花期重点优化授粉环境和营养平衡")
        elif growth_stage == GrowthStage.FRUITING:
            reasoning_parts.append("结果期优先保障果实品质和糖分积累")
        
        # 置信度说明
        confidence_level = "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低"
        reasoning_parts.append(f"决策置信度: {confidence_level}({confidence:.2f})")
        
        return "; ".join(reasoning_parts)
    
    def calculate_reward(self, previous_state: AgricultureState, 
                        current_state: AgricultureState, 
                        decisions: Dict[str, Any]) -> float:
        """计算决策奖励"""
        
        reward = 0.0
        crop_type = current_state.crop_config.crop_type
        crop_model = self.crop_models[crop_type]
        
        # 1. 产量奖励
        if current_state.historical_performance:
            recent_yield = current_state.historical_performance[-1].get("yield", 0)
            target_yield = current_state.crop_config.target_yield
            yield_reward = min(recent_yield / target_yield, 1.0) * self.objective_weights["yield_maximization"]
            reward += yield_reward
        
        # 2. 质量奖励
        if current_state.historical_performance:
            quality_score = current_state.historical_performance[-1].get("quality_score", 0)
            quality_reward = quality_score * self.objective_weights["quality_optimization"]
            reward += quality_reward
        
        # 3. 资源效率奖励
        # 计算环境参数与最优范围的接近程度
        env = current_state.environment
        optimal_temp = crop_model["optimal_temperature"]
        temp_efficiency = 1 - abs(env.temperature - (optimal_temp["min"] + optimal_temp["max"]) / 2) / 10
        temp_efficiency = max(0, min(1, temp_efficiency))
        
        optimal_humidity = crop_model["optimal_humidity"]
        humidity_efficiency = 1 - abs(env.humidity - (optimal_humidity["min"] + optimal_humidity["max"]) / 2) / 20
        humidity_efficiency = max(0, min(1, humidity_efficiency))
        
        resource_reward = (temp_efficiency + humidity_efficiency) / 2 * self.objective_weights["resource_efficiency"]
        reward += resource_reward
        
        # 4. 生长速度奖励
        growth_progress = current_state.current_day / crop_model["growth_duration"]
        expected_progress = previous_state.current_day / crop_model["growth_duration"] + 0.01  # 每天预期进度
        growth_reward = max(0, growth_progress - expected_progress) * self.objective_weights["growth_speed"]
        reward += growth_reward
        
        # 5. 惩罚项 - 参数超出安全范围
        penalty = 0.0
        if env.temperature < optimal_temp["min"] or env.temperature > optimal_temp["max"]:
            penalty += 0.1
        if env.humidity < optimal_humidity["min"] or env.humidity > optimal_humidity["max"]:
            penalty += 0.1
        
        reward -= penalty
        
        return max(reward, 0)  # 确保奖励非负
    
    def validate_decisions(self, decisions: Dict[str, Any], 
                          current_state: AgricultureState) -> Tuple[bool, Optional[str]]:
        """验证决策的合理性"""
        
        # 检查光谱参数范围
        spectrum = decisions["spectrum_adjustments"]
        if spectrum["red_intensity"] > 350:
            return False, "红光强度超出安全范围"
        if spectrum["blue_intensity"] > 250:
            return False, "蓝光强度超出安全范围"
        
        # 检查环境参数
        env_control = decisions["environment_control"]
        if env_control["temperature_target"] < 10 or env_control["temperature_target"] > 35:
            return False, "目标温度超出作物耐受范围"
        if env_control["humidity_target"] < 40 or env_control["humidity_target"] > 95:
            return False, "目标湿度超出合理范围"
        
        return True, None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "total_decisions": len(self.decision_history),
            "average_confidence": np.mean([d.get("confidence", 0) for d in self.decision_history]) if self.decision_history else 0,
            "recent_success_rate": self._calculate_recent_success_rate()
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """计算近期决策成功率"""
        if len(self.decision_history) < 10:
            return 0.0
        
        recent_decisions = self.decision_history[-10:]
        successful = sum(1 for d in recent_decisions if d.get("reward", 0) > 0.5)
        return successful / len(recent_decisions)


# 示例使用代码
if __name__ == "__main__":
    # 创建测试数据
    test_env = AgricultureEnvironment(
        temperature=22.5,
        humidity=70.0,
        co2_concentration=800.0,
        light_intensity=15000.0,
        soil_moisture=65.0,
        nutrient_level=0.8,
        ph_level=6.5
    )
    
    test_spectrum = SpectrumConfig(
        ultraviolet_intensity=10.0,
        blue_intensity=120.0,
        green_intensity=80.0,
        red_intensity=180.0,
        far_red_intensity=30.0,
        white_red_ratio=31.0,
        photoperiod=12
    )
    
    test_crop = CropConfig(
        crop_type=CropType.TOMATO,
        growth_stage=GrowthStage.FRUITING,
        planting_day=60,
        target_yield=5.0,
        quality_targets={"sweetness": 8.0, "firmness": 7.0}
    )
    
    test_state = AgricultureState(
        crop_config=test_crop,
        environment=test_env,
        spectrum_config=test_spectrum,
        historical_performance=[{"yield": 4.2, "quality_score": 0.75}],
        current_day=65
    )
    
    # 测试决策模块
    module = AgricultureDecisionModule()
    decision_state = module.create_decision_state(test_state)
    
    print(f"决策状态向量维度: {len(decision_state.state_vector)}")
    print(f"决策状态上下文: {decision_state.context}")
    
    # 模拟决策动作
    test_action = DecisionAction(
        decision_type=DecisionType.AGRICULTURE,
        action_vector=np.random.rand(15),
        confidence=0.85,
        timestamp=datetime.now().timestamp()
    )
    
    decisions = module.interpret_decision_action(test_action, test_state)
    print(f"农业决策结果: {json.dumps(decisions, indent=2, default=str)}")
    
    # 计算奖励
    reward = module.calculate_reward(test_state, test_state, decisions)
    print(f"决策奖励: {reward:.3f}")