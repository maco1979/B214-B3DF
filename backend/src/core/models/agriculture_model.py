"""
农业AI核心模型 - 光谱分析、植物生长模型、光配方系统
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class SpectrumConfig:
    """光谱配置"""
    uv_380nm: float = 0.05  # 380nm紫外线强度
    far_red_720nm: float = 0.1  # 720nm远红外线强度
    white_light: float = 0.7  # 白光强度 (420-450nm, 18000K)
    red_660nm: float = 0.15  # 660nm红光强度
    
    @property
    def white_red_ratio(self) -> float:
        """白红配比"""
        return self.white_light / self.red_660nm if self.red_660nm > 0 else 0


@dataclass
class PlantGrowthStage:
    """植物生长阶段"""
    stage_name: str  # 阶段名称
    duration_days: int  # 持续时间(天)
    optimal_temperature: Tuple[float, float]  # 最佳温度范围
    optimal_humidity: Tuple[float, float]  # 最佳湿度范围
    light_hours: int  # 光照时长(小时)


@dataclass
class CropConfig:
    """作物配置"""
    crop_name: str  # 作物名称
    growth_stages: List[PlantGrowthStage]  # 生长阶段
    target_yield: str  # 目标产量类型
    quality_metrics: Dict[str, float]  # 质量指标


class SpectrumAnalyzer(nn.Module):
    """光谱分析器"""
    
    @nn.compact
    def __call__(self, spectrum_data: jnp.ndarray) -> Dict[str, float]:
        """分析光谱数据"""
        # 光谱特征提取
        uv_intensity = jnp.mean(spectrum_data[370:390])  # 380nm附近
        far_red_intensity = jnp.mean(spectrum_data[710:730])  # 720nm附近
        white_intensity = jnp.mean(spectrum_data[420:450])  # 白光区域
        red_intensity = jnp.mean(spectrum_data[650:670])  # 红光区域
        
        return {
            'uv_380nm': uv_intensity,
            'far_red_720nm': far_red_intensity,
            'white_light': white_intensity,
            'red_660nm': red_intensity,
            'white_red_ratio': white_intensity / red_intensity if red_intensity > 0 else 0
        }


class PlantGrowthModel(nn.Module):
    """植物生长模型"""
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, 
                 environmental_data: jnp.ndarray,
                 spectrum_data: jnp.ndarray,
                 growth_days: int) -> Dict[str, float]:
        """预测植物生长状态"""
        
        # 环境特征提取
        env_features = nn.Dense(self.hidden_dim)(environmental_data)
        env_features = nn.relu(env_features)
        
        # 光谱特征提取 - 先将一维光谱数据转换为适合Dense层的形状
        # 计算光谱的统计特征而不是直接使用800维数据
        spec_mean = jnp.mean(spectrum_data)
        spec_std = jnp.std(spectrum_data)
        spec_max = jnp.max(spectrum_data)
        spec_min = jnp.min(spectrum_data)
        
        # 使用统计特征作为输入
        spec_stats = jnp.array([spec_mean, spec_std, spec_max, spec_min])
        spec_features = nn.Dense(self.hidden_dim)(spec_stats)
        spec_features = nn.relu(spec_features)
        
        # 时间特征
        time_features = nn.Dense(self.hidden_dim)(
            jnp.array([growth_days / 100.0])  # 归一化
        )
        
        # 特征融合
        combined = jnp.concatenate([env_features, spec_features, time_features])
        combined = nn.Dense(self.hidden_dim)(combined)
        combined = nn.relu(combined)
        
        # 生长指标预测
        growth_rate = nn.Dense(1)(combined)
        health_score = nn.Dense(1)(combined)
        yield_potential = nn.Dense(1)(combined)
        
        return {
            'growth_rate': nn.sigmoid(growth_rate)[0],
            'health_score': nn.sigmoid(health_score)[0],
            'yield_potential': nn.sigmoid(yield_potential)[0]
        }


class LightRecipeGenerator(nn.Module):
    """光配方生成器"""
    
    @nn.compact
    def __call__(self, 
                 crop_config: CropConfig,
                 current_stage: PlantGrowthStage,
                 target_objective: str,
                 environmental_conditions: Dict[str, float]) -> SpectrumConfig:
        """生成最优光配方"""
        
        # 基于作物类型和生长阶段的基础配方
        base_recipe = self._get_base_recipe(crop_config, current_stage)
        
        # 根据目标优化配方
        optimized_recipe = self._optimize_for_objective(
            base_recipe, target_objective, environmental_conditions
        )
        
        # 验证31:1白红配比
        if optimized_recipe.white_red_ratio < 30:
            optimized_recipe.white_light = optimized_recipe.red_660nm * 31
        elif optimized_recipe.white_red_ratio > 32:
            optimized_recipe.red_660nm = optimized_recipe.white_light / 31
        
        return optimized_recipe
    
    def _get_base_recipe(self, crop_config: CropConfig, stage: PlantGrowthStage) -> SpectrumConfig:
        """获取基础光配方"""
        # 根据不同作物和生长阶段的基础配方
        if "苗期" in stage.stage_name:
            return SpectrumConfig(
                uv_380nm=0.02,  # 苗期减少紫外线
                far_red_720nm=0.08,
                white_light=0.75,
                red_660nm=0.15
            )
        elif "开花" in stage.stage_name:
            return SpectrumConfig(
                uv_380nm=0.08,  # 开花期增加紫外线促进次生代谢
                far_red_720nm=0.12,  # 调控开花时间
                white_light=0.65,
                red_660nm=0.15
            )
        else:  # 结果期
            return SpectrumConfig(
                uv_380nm=0.06,
                far_red_720nm=0.1,
                white_light=0.7,
                red_660nm=0.14
            )
    
    def _optimize_for_objective(self, 
                               recipe: SpectrumConfig,
                               objective: str,
                               env_conditions: Dict[str, float]) -> SpectrumConfig:
        """根据目标优化配方"""
        
        optimized = SpectrumConfig(
            uv_380nm=recipe.uv_380nm,
            far_red_720nm=recipe.far_red_720nm,
            white_light=recipe.white_light,
            red_660nm=recipe.red_660nm
        )
        
        if objective == "最大化产量":
            # 增加红光促进光合作用
            optimized.red_660nm *= 1.2
            optimized.white_light *= 0.9
        elif objective == "提升甜度":
            # 增加紫外线促进次生代谢物
            optimized.uv_380nm *= 1.5
            optimized.far_red_720nm *= 1.1
        elif objective == "提升抗性":
            # 平衡各波段，增强植物抗性
            optimized.uv_380nm *= 1.3
            optimized.far_red_720nm *= 0.9
        
        # 根据环境条件微调
        if env_conditions.get('temperature', 25) > 28:
            # 高温时减少光照强度
            optimized.white_light *= 0.9
            optimized.red_660nm *= 0.9
        
        # 归一化确保总和为1
        total = optimized.uv_380nm + optimized.far_red_720nm + optimized.white_light + optimized.red_660nm
        if total > 0:
            optimized.uv_380nm /= total
            optimized.far_red_720nm /= total
            optimized.white_light /= total
            optimized.red_660nm /= total
        
        return optimized


class AgricultureAIService:
    """农业AI服务"""
    
    def __init__(self):
        self.spectrum_analyzer = SpectrumAnalyzer()
        # 简化为直接返回模拟数据，避免Flax模型的复杂初始化
        self.growth_model = self._mock_growth_model
        self.recipe_generator = LightRecipeGenerator()
        
        # 预定义的作物配置
        self.crop_configs = self._load_crop_configs()
    
    def _mock_growth_model(self, environmental_data, spectrum_data, growth_days):
        """模拟生长模型，返回合理的预测结果"""
        import random
        return {
            'growth_rate': random.uniform(0.6, 0.95),
            'health_score': random.uniform(0.7, 0.98),
            'yield_potential': random.uniform(0.5, 0.9)
        }
    
    def _load_crop_configs(self) -> Dict[str, CropConfig]:
        """加载作物配置"""
        return {
            # 菜叶类
            "生菜": CropConfig(
                crop_name="生菜",
                growth_stages=[
                    PlantGrowthStage("苗期", 15, (18, 22), (70, 80), 14),
                    PlantGrowthStage("生长期", 25, (20, 24), (65, 75), 12)
                ],
                target_yield="提升品质",
                quality_metrics={"脆度": 0.9, "营养含量": 0.8}
            ),
            "白菜": CropConfig(
                crop_name="白菜",
                growth_stages=[
                    PlantGrowthStage("苗期", 20, (15, 20), (70, 80), 12),
                    PlantGrowthStage("莲座期", 30, (18, 23), (65, 75), 10),
                    PlantGrowthStage("结球期", 35, (20, 25), (60, 70), 8)
                ],
                target_yield="最大化产量",
                quality_metrics={"紧实度": 0.9, "口感": 0.8}
            ),
            "菠菜": CropConfig(
                crop_name="菠菜",
                growth_stages=[
                    PlantGrowthStage("苗期", 12, (10, 15), (65, 75), 12),
                    PlantGrowthStage("生长期", 28, (12, 18), (60, 70), 10)
                ],
                target_yield="提升品质",
                quality_metrics={"铁含量": 0.9, "鲜嫩度": 0.85}
            ),
            "芹菜": CropConfig(
                crop_name="芹菜",
                growth_stages=[
                    PlantGrowthStage("苗期", 25, (15, 20), (70, 80), 14),
                    PlantGrowthStage("生长前期", 35, (18, 23), (65, 75), 12),
                    PlantGrowthStage("生长后期", 40, (20, 25), (60, 70), 10)
                ],
                target_yield="最大化产量",
                quality_metrics={"纤维含量": 0.7, "风味": 0.8}
            ),
            "油菜": CropConfig(
                crop_name="油菜",
                growth_stages=[
                    PlantGrowthStage("苗期", 18, (12, 18), (65, 75), 12),
                    PlantGrowthStage("生长期", 27, (15, 22), (60, 70), 10)
                ],
                target_yield="提升品质",
                quality_metrics={"维生素C": 0.9, "水分含量": 0.85}
            ),
            
            # 水果类
            "番茄": CropConfig(
                crop_name="番茄",
                growth_stages=[
                    PlantGrowthStage("苗期", 30, (20, 25), (60, 70), 16),
                    PlantGrowthStage("开花期", 20, (22, 26), (50, 60), 14),
                    PlantGrowthStage("结果期", 45, (24, 28), (45, 55), 12)
                ],
                target_yield="最大化产量",
                quality_metrics={"甜度": 0.8, "维生素C": 0.7}
            ),
            "草莓": CropConfig(
                crop_name="草莓",
                growth_stages=[
                    PlantGrowthStage("苗期", 25, (15, 20), (60, 70), 14),
                    PlantGrowthStage("开花期", 15, (18, 22), (50, 60), 12),
                    PlantGrowthStage("结果期", 30, (20, 25), (45, 55), 10)
                ],
                target_yield="提升甜度",
                quality_metrics={"甜度": 0.9, "色泽": 0.85}
            ),
            "黄瓜": CropConfig(
                crop_name="黄瓜",
                growth_stages=[
                    PlantGrowthStage("苗期", 25, (18, 23), (65, 75), 14),
                    PlantGrowthStage("开花期", 15, (20, 25), (55, 65), 12),
                    PlantGrowthStage("结果期", 40, (22, 28), (50, 60), 10)
                ],
                target_yield="最大化产量",
                quality_metrics={"脆度": 0.85, "苦味": 0.9}
            ),
            "辣椒": CropConfig(
                crop_name="辣椒",
                growth_stages=[
                    PlantGrowthStage("苗期", 35, (20, 25), (60, 70), 14),
                    PlantGrowthStage("开花期", 20, (22, 27), (55, 65), 12),
                    PlantGrowthStage("结果期", 50, (24, 29), (50, 60), 10)
                ],
                target_yield="提升品质",
                quality_metrics={"辣度": 0.8, "维生素C": 0.9}
            ),
            
            # 花草类
            "玫瑰": CropConfig(
                crop_name="玫瑰",
                growth_stages=[
                    PlantGrowthStage("苗期", 40, (18, 23), (60, 70), 12),
                    PlantGrowthStage("营养生长期", 60, (20, 25), (55, 65), 14),
                    PlantGrowthStage("开花期", 30, (22, 27), (50, 60), 16)
                ],
                target_yield="提升品质",
                quality_metrics={"花期长度": 0.9, "花色鲜艳度": 0.85}
            ),
            "百合": CropConfig(
                crop_name="百合",
                growth_stages=[
                    PlantGrowthStage("苗期", 50, (15, 20), (65, 75), 12),
                    PlantGrowthStage("生长中期", 70, (18, 23), (60, 70), 14),
                    PlantGrowthStage("开花期", 25, (20, 25), (55, 65), 16)
                ],
                target_yield="提升品质",
                quality_metrics={"花茎长度": 0.8, "花期": 0.85}
            ),
            "郁金香": CropConfig(
                crop_name="郁金香",
                growth_stages=[
                    PlantGrowthStage("苗期", 35, (10, 15), (60, 70), 10),
                    PlantGrowthStage("生长中期", 45, (15, 20), (55, 65), 12),
                    PlantGrowthStage("开花期", 15, (18, 22), (50, 60), 14)
                ],
                target_yield="提升品质",
                quality_metrics={"花色": 0.9, "花型": 0.8}
            ),
            "康乃馨": CropConfig(
                crop_name="康乃馨",
                growth_stages=[
                    PlantGrowthStage("苗期", 30, (18, 23), (65, 75), 12),
                    PlantGrowthStage("生长中期", 50, (20, 25), (60, 70), 14),
                    PlantGrowthStage("开花期", 25, (22, 27), (55, 65), 16)
                ],
                target_yield="提升品质",
                quality_metrics={"花茎硬度": 0.85, "花期": 0.9}
            )
        }
    
    def generate_light_recipe(self, 
                             crop_type: str,
                             current_day: int,
                             target_objective: str,
                             environment: Dict[str, float]) -> Dict[str, any]:
        """生成光配方"""
        
        # 中英文作物名称映射
        crop_name_map = {
            "tomato": "番茄",
            "lettuce": "生菜"
        }
        
        # 检查是否为英文名称并转换
        if crop_type.lower() in crop_name_map:
            crop_type = crop_name_map[crop_type.lower()]
        
        if crop_type not in self.crop_configs:
            raise ValueError(f"未知作物类型: {crop_type}")
        
        crop_config = self.crop_configs[crop_type]
        
        # 确定当前生长阶段
        current_stage = self._get_current_stage(crop_config, current_day)
        
        # 生成光配方 - 由于是简单演示，直接使用规则生成而不使用Flax模型
        base_recipe = SpectrumConfig()
        
        # 根据生长阶段调整
        if "苗期" in current_stage.stage_name:
            base_recipe = SpectrumConfig(uv_380nm=0.02, far_red_720nm=0.08, white_light=0.75, red_660nm=0.15)
        elif "开花" in current_stage.stage_name:
            base_recipe = SpectrumConfig(uv_380nm=0.08, far_red_720nm=0.12, white_light=0.65, red_660nm=0.15)
        else:
            base_recipe = SpectrumConfig(uv_380nm=0.06, far_red_720nm=0.10, white_light=0.70, red_660nm=0.14)
        
        # 根据目标优化
        optimized = SpectrumConfig(
            uv_380nm=base_recipe.uv_380nm,
            far_red_720nm=base_recipe.far_red_720nm,
            white_light=base_recipe.white_light,
            red_660nm=base_recipe.red_660nm
        )
        
        if target_objective == "最大化产量":
            optimized.red_660nm *= 1.2
            optimized.white_light *= 0.9
        elif target_objective == "提升甜度":
            optimized.uv_380nm *= 1.5
            optimized.far_red_720nm *= 1.1
        elif target_objective == "提升抗性":
            optimized.uv_380nm *= 1.3
            optimized.far_red_720nm *= 0.9
        
        # 归一化
        total = optimized.uv_380nm + optimized.far_red_720nm + optimized.white_light + optimized.red_660nm
        if total > 0:
            optimized.uv_380nm /= total
            optimized.far_red_720nm /= total
            optimized.white_light /= total
            optimized.red_660nm /= total
        
        recipe = optimized
        
        return {
            "recipe": recipe,
            "current_stage": current_stage.stage_name,
            "light_hours": current_stage.light_hours,
            "recommendations": self._get_growth_recommendations(crop_config, current_stage)
        }
    
    def _get_current_stage(self, crop_config: CropConfig, current_day: int) -> PlantGrowthStage:
        """获取当前生长阶段"""
        accumulated_days = 0
        for stage in crop_config.growth_stages:
            accumulated_days += stage.duration_days
            if current_day <= accumulated_days:
                return stage
        # 如果超过所有阶段，返回最后一个阶段
        return crop_config.growth_stages[-1]
    
    def _get_growth_recommendations(self, 
                                  crop_config: CropConfig,
                                  stage: PlantGrowthStage) -> List[str]:
        """获取生长建议"""
        recommendations = []
        
        recommendations.append(f"保持温度在{stage.optimal_temperature[0]}-{stage.optimal_temperature[1]}°C")
        recommendations.append(f"保持湿度在{stage.optimal_humidity[0]}%-{stage.optimal_humidity[1]}%")
        recommendations.append(f"每日光照{stage.light_hours}小时")
        
        if "开花" in stage.stage_name:
            recommendations.append("适当增加磷钾肥促进开花")
        elif "结果" in stage.stage_name:
            recommendations.append("增加钙肥防止果实病害")
        
        return recommendations