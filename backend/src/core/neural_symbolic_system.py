"""Neural-Symbolic Hybrid System
Integration of Prolog reasoning engine with neural language modeling
Implementing NELLIE-inspired architecture for explainable reasoning
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import sys

logger = logging.getLogger(__name__)


# 尝试导入PySwip进行Prolog集成
try:
    from pyswip import Prolog, registerForeign
    PROLOG_AVAILABLE = True
    logger.info("✅ PySwip库导入成功，Prolog推理可用")
except ImportError as e:
    logger.warning(f"⚠️ PySwip库导入失败，将使用简化的符号推理: {str(e)}")
    PROLOG_AVAILABLE = False

# 导入统一知识表示
from .common_knowledge_base import common_knowledge_base, KnowledgeEntry

# 导入NELLIE服务
from .services.nellie_service import nellie_service


@dataclass
class ReasoningResult:
    """推理结果数据结构"""
    conclusion: str
    confidence: float
    reasoning_path: List[str]
    timestamp: datetime
    used_knowledge: List[str]  # 使用的知识ID列表
    is_symbolic: bool  # 是否为符号推理结果
    neural_support: Optional[float] = None  # 神经网络支持度
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class NeuralSymbolicRule:
    """神经符号规则数据结构"""
    rule_id: str
    antecedents: List[str]  # 前提条件
    consequent: str  # 结论
    confidence: float  # 规则置信度
    neural_weight: float  # 神经网络权重
    source: str  # 规则来源
    timestamp: datetime


class NeuralSymbolicSystem:
    """神经符号混合系统"""
    
    def __init__(self):
        self.prolog = None
        self.rules: List[NeuralSymbolicRule] = []
        self.rule_index: Dict[str, NeuralSymbolicRule] = {}
        self.knowledge_base = common_knowledge_base
        self.initialized = False
        self.use_external_nellie = True  # 是否使用外部NELLIE服务
        
        # 初始化Prolog引擎
        self._init_prolog()
        
        # 加载默认规则
        self._load_default_rules()
        
        logger.info(f"神经符号系统初始化完成，Prolog支持: {PROLOG_AVAILABLE}, 外部NELLIE服务可用: {nellie_service.is_enabled}")
    
    def _init_prolog(self):
        """初始化Prolog引擎"""
        if PROLOG_AVAILABLE:
            try:
                self.prolog = Prolog()
                # 初始化Prolog知识库
                self._init_prolog_knowledge()
                self.initialized = True
                logger.info("✅ Prolog引擎初始化成功")
            except Exception as e:
                logger.error(f"❌ Prolog引擎初始化失败: {str(e)}")
                self.prolog = None
                self.initialized = False
    
    def _init_prolog_knowledge(self):
        """初始化Prolog知识库，扩展农业领域规则和事实"""
        if not self.prolog:
            return
        
        # 初始Prolog规则和事实
        initial_facts = [
            # 基础分类
            "category(weather, natural_phenomenon)",
            "category(rain, weather)",
            "category(sunshine, weather)",
            "category(wind, weather)",
            "category(temperature, weather)",
            "category(humidity, weather)",
            "category(soil, natural_resource)",
            "category(water, natural_resource)",
            "category(fertilizer, agricultural_input)",
            "category(pesticide, agricultural_input)",
            "category(light, environmental_factor)",
            "category(ph, soil_property)",
            "category(nutrient, soil_property)",
            "category(moisture, soil_property)",
            
            # 作物分类
            "category(crop, agricultural_product)",
            "category(vegetable, crop)",
            "category(fruit, crop)",
            "category(grain, crop)",
            "category(flower, crop)",
            "category(leafy_vegetable, vegetable)",
            "category(root_vegetable, vegetable)",
            "category(fruit_vegetable, vegetable)",
            "category(legume, vegetable)",
            
            # 具体作物
            "is_a(lettuce, leafy_vegetable)",
            "is_a(spinach, leafy_vegetable)",
            "is_a(cabbage, leafy_vegetable)",
            "is_a(carrot, root_vegetable)",
            "is_a(potato, root_vegetable)",
            "is_a(tomato, fruit_vegetable)",
            "is_a(cucumber, fruit_vegetable)",
            "is_a(pepper, fruit_vegetable)",
            "is_a(bean, legume)",
            "is_a(pea, legume)",
            "is_a(rice, grain)",
            "is_a(wheat, grain)",
            "is_a(corn, grain)",
            "is_a(strawberry, fruit)",
            "is_a(apple, fruit)",
            "is_a(rose, flower)",
            "is_a(lily, flower)",
            
            # 生长条件
            "optimal_temperature(lettuce, 15, 20)",  # 生菜最适温度15-20度
            "optimal_temperature(tomato, 20, 28)",  # 番茄最适温度20-28度
            "optimal_temperature(cucumber, 25, 30)",  # 黄瓜最适温度25-30度
            "optimal_humidity(lettuce, 60, 70)",  # 生菜最适湿度60-70%
            "optimal_humidity(tomato, 45, 55)",  # 番茄最适湿度45-55%
            "optimal_humidity(cucumber, 70, 80)",  # 黄瓜最适湿度70-80%
            "optimal_ph(lettuce, 6.0, 7.0)",  # 生菜最适土壤pH6.0-7.0
            "optimal_ph(tomato, 6.0, 6.8)",  # 番茄最适土壤pH6.0-6.8
            "optimal_ph(cucumber, 6.0, 7.0)",  # 黄瓜最适土壤pH6.0-7.0
            "optimal_light(lettuce, 10, 12)",  # 生菜最适光照10-12小时
            "optimal_light(tomato, 12, 14)",  # 番茄最适光照12-14小时
            "optimal_light(cucumber, 8, 10)",  # 黄瓜最适光照8-10小时
            
            # 生长阶段
            "growth_stage(seed, initial)",
            "growth_stage(germination, early)",
            "growth_stage(seedling, early)",
            "growth_stage(vegetative_growth, middle)",
            "growth_stage(reproductive_growth, middle)",
            "growth_stage(flowering, late)",
            "growth_stage(fruiting, late)",
            "growth_stage(ripening, final)",
            "growth_stage(harvest, final)",
            
            # 农业操作
            "agricultural_operation(irrigation)",
            "agricultural_operation(fertilization)",
            "agricultural_operation(pesticide_application)",
            "agricultural_operation(pruning)",
            "agricultural_operation(harvesting)",
            "agricultural_operation(planting)",
            "agricultural_operation(weeding)",
            "agricultural_operation(tilling)",
            
            # 环境关系
            "relation(causes, rain, wet_soil)",
            "relation(causes, sunshine, warm_temperature)",
            "relation(causes, wind, evaporation)",
            "relation(causes, high_temperature, rapid_evaporation)",
            "relation(causes, low_temperature, slow_growth)",
            "relation(causes, high_humidity, fungal_growth)",
            "relation(causes, low_humidity, water_stress)",
            "relation(causes, sufficient_light, photosynthesis)",
            "relation(causes, insufficient_light, weak_growth)",
            
            # 农业操作效果
            "relation(causes, irrigation, increased_soil_moisture)",
            "relation(causes, fertilization, increased_soil_nutrients)",
            "relation(causes, pesticide_application, reduced_pests)",
            "relation(causes, pruning, better_air_circulation)",
            "relation(causes, pruning, increased_fruiting)",
            "relation(causes, weeding, reduced_competition)",
            "relation(causes, tilling, improved_soil_structure)",
            
            # 作物-操作关系
            "relation(requires, lettuce, irrigation)",
            "relation(requires, lettuce, fertilization)",
            "relation(requires, tomato, irrigation)",
            "relation(requires, tomato, fertilization)",
            "relation(requires, cucumber, irrigation)",
            "relation(requires, cucumber, fertilization)",
            "relation(requires, all_crops, water)",
            "relation(requires, all_crops, soil)",
            "relation(requires, all_crops, light)",
            
            # 病害关系
            "disease(downy_mildew, fungal)",
            "disease(powdery_mildew, fungal)",
            "disease(bacterial_wilt, bacterial)",
            "disease(aphid_infestation, pest)",
            "disease(whitefly_infestation, pest)",
            "disease(target_crop, downy_mildew, lettuce)",
            "disease(target_crop, powdery_mildew, cucumber)",
            "disease(target_crop, bacterial_wilt, tomato)",
            "disease(target_crop, aphid_infestation, many_crops)",
            "disease(target_crop, whitefly_infestation, many_crops)",
            "disease_cause(downy_mildew, high_humidity)",
            "disease_cause(powdery_mildew, high_humidity)",
            "disease_cause(bacterial_wilt, contaminated_soil)",
            "disease_cause(aphid_infestation, warm_temperature)",
            "disease_cause(whitefly_infestation, warm_temperature)",
            
            # 规则
            "rule(if_high_humidity_then_downy_mildew, [high_humidity, lettuce], downy_mildew_risk, 0.85)",
            "rule(if_high_humidity_then_powdery_mildew, [high_humidity, cucumber], powdery_mildew_risk, 0.88)",
            "rule(if_contaminated_soil_then_bacterial_wilt, [contaminated_soil, tomato], bacterial_wilt_risk, 0.90)",
            "rule(if_warm_temperature_then_aphids, [warm_temperature], aphid_risk, 0.82)",
            "rule(if_water_stress_then_weak_growth, [water_stress], weak_growth, 0.95)",
            "rule(if_weak_growth_then_low_yield, [weak_growth], low_yield, 0.90)",
            "rule(if_fungal_growth_then_disease, [fungal_growth], disease_risk, 0.92)",
            "rule(if_insufficient_light_then_leggy_growth, [insufficient_light], leggy_growth, 0.88)",
            "rule(if_leggy_growth_then_low_yield, [leggy_growth], low_yield, 0.85)",
            "rule(if_optimal_conditions_then_healthy_growth, [optimal_temperature, optimal_humidity, optimal_light, optimal_ph, sufficient_nutrients], healthy_growth, 0.98)",
            "rule(if_healthy_growth_then_high_yield, [healthy_growth], high_yield, 0.95)",
            "rule(if_irrigation_then_increased_moisture, [irrigation], increased_soil_moisture, 0.99)",
            "rule(if_increased_moisture_then_reduced_water_stress, [increased_soil_moisture], reduced_water_stress, 0.95)",
            "rule(if_fertilization_then_increased_nutrients, [fertilization], increased_soil_nutrients, 0.98)",
            "rule(if_pesticide_then_reduced_pests, [pesticide_application], reduced_pests, 0.90)",
            "rule(if_reduced_pests_then_increased_yield, [reduced_pests], increased_yield, 0.88)",
            "rule(if_cucumber_and_high_humidity_then_powdery_mildew, [is_a(crop, cucumber), high_humidity], powdery_mildew_risk, 0.85)",
            "rule(if_tomato_and_contaminated_soil_then_bacterial_wilt, [is_a(crop, tomato), contaminated_soil], bacterial_wilt_risk, 0.88)",
            "rule(if_lettuce_and_high_temperature_then_bolting, [is_a(crop, lettuce), high_temperature], bolting_risk, 0.90)",
            "rule(if_bolting_then_reduced_quality, [bolting_risk], reduced_quality, 0.95)",
            "rule(if_rain_and_ripe_fruit_then_cracking, [rain, ripe_fruit], fruit_cracking_risk, 0.92)",
            "rule(if_fruit_cracking_then_reduced_quality, [fruit_cracking_risk], reduced_quality, 0.98)",
            "rule(if_excessive_fertilization_then_burn, [excessive_fertilization], fertilizer_burn_risk, 0.85)",
            "rule(if_fertilizer_burn_then_plant_damage, [fertilizer_burn_risk], plant_damage, 0.95)",
            "rule(if_wind_and_tall_plants_then_lodging, [strong_wind, tall_plants], lodging_risk, 0.88)",
            "rule(if_lodging_then_reduced_yield, [lodging_risk], reduced_yield, 0.92)",
            "rule(if_sunscald_and_fruit_then_damage, [sunscald, fruit], fruit_damage, 0.90)",
            "rule(if_frost_and_tender_plants_then_damage, [frost, tender_plants], frost_damage, 0.98)",
            "rule(if_drought_and_long_duration_then_death, [drought, long_duration], plant_death, 0.95)",
            "rule(if_flood_and_long_duration_then_root_rot, [flood, long_duration], root_rot, 0.92)",
            "rule(if_root_rot_then_plant_death, [root_rot], plant_death, 0.98)",
            "rule(if_healthy_soil_then_healthy_plants, [healthy_soil], healthy_plants, 0.90)",
            "rule(if_healthy_plants_then_high_yield, [healthy_plants], high_yield, 0.95)",
            "rule(if_crop_rotation_then_reduced_pests, [crop_rotation], reduced_pests, 0.85)",
            "rule(if_intercropping_then_reduced_pests, [intercropping], reduced_pests, 0.82)",
            "rule(if_mulching_then_reduced_evaporation, [mulching], reduced_evaporation, 0.90)",
            "rule(if_reduced_evaporation_then_conserved_water, [reduced_evaporation], conserved_water, 0.95)",
            "rule(if_conserved_water_then_reduced_irrigation_needs, [conserved_water], reduced_irrigation_needs, 0.92)",
            "rule(if_cover_cropping_then_improved_soil_health, [cover_cropping], improved_soil_health, 0.88)",
            "rule(if_improved_soil_health_then_increased_nutrient_availability, [improved_soil_health], increased_nutrient_availability, 0.92)"
        ]
        
        # 添加事实到Prolog
        for fact in initial_facts:
            try:
                self.prolog.assertz(fact)
            except Exception as e:
                logger.error(f"添加Prolog事实失败: {fact}, 错误: {str(e)}")
    
    def _load_default_rules(self):
        """加载默认神经符号规则，扩展农业领域规则"""
        default_rules = [
            # 基础规则
            {
                "rule_id": "weather_umbrella_rule",
                "antecedents": ["rain"],
                "consequent": "need_umbrella",
                "confidence": 0.99,
                "neural_weight": 0.95,
                "source": "system"
            },
            {
                "rule_id": "temperature_rule",
                "antecedents": ["heating"],
                "consequent": "temperature_increase",
                "confidence": 0.98,
                "neural_weight": 0.90,
                "source": "system"
            },
            {
                "rule_id": "sleep_fatigue_rule",
                "antecedents": ["sleep_deprivation"],
                "consequent": "fatigue",
                "confidence": 0.95,
                "neural_weight": 0.85,
                "source": "system"
            },
            {
                "rule_id": "exercise_health_rule",
                "antecedents": ["exercise"],
                "consequent": "good_health",
                "confidence": 0.90,
                "neural_weight": 0.80,
                "source": "system"
            },
            
            # 农业环境规则
            {
                "rule_id": "high_humidity_fungal_rule",
                "antecedents": ["high_humidity"],
                "consequent": "fungal_growth",
                "confidence": 0.92,
                "neural_weight": 0.90,
                "source": "system"
            },
            {
                "rule_id": "low_humidity_stress_rule",
                "antecedents": ["low_humidity"],
                "consequent": "water_stress",
                "confidence": 0.95,
                "neural_weight": 0.92,
                "source": "system"
            },
            {
                "rule_id": "sufficient_light_photosynthesis_rule",
                "antecedents": ["sufficient_light"],
                "consequent": "photosynthesis",
                "confidence": 0.98,
                "neural_weight": 0.95,
                "source": "system"
            },
            {
                "rule_id": "insufficient_light_weak_rule",
                "antecedents": ["insufficient_light"],
                "consequent": "weak_growth",
                "confidence": 0.88,
                "neural_weight": 0.85,
                "source": "system"
            },
            
            # 农业操作规则
            {
                "rule_id": "irrigation_moisture_rule",
                "antecedents": ["irrigation"],
                "consequent": "increased_soil_moisture",
                "confidence": 0.99,
                "neural_weight": 0.97,
                "source": "system"
            },
            {
                "rule_id": "fertilization_nutrients_rule",
                "antecedents": ["fertilization"],
                "consequent": "increased_soil_nutrients",
                "confidence": 0.98,
                "neural_weight": 0.95,
                "source": "system"
            },
            {
                "rule_id": "pesticide_pests_rule",
                "antecedents": ["pesticide_application"],
                "consequent": "reduced_pests",
                "confidence": 0.90,
                "neural_weight": 0.85,
                "source": "system"
            },
            {
                "rule_id": "pruning_air_circulation_rule",
                "antecedents": ["pruning"],
                "consequent": "better_air_circulation",
                "confidence": 0.92,
                "neural_weight": 0.88,
                "source": "system"
            },
            
            # 作物特定规则
            {
                "rule_id": "lettuce_high_temp_bolting_rule",
                "antecedents": ["lettuce", "high_temperature"],
                "consequent": "bolting_risk",
                "confidence": 0.90,
                "neural_weight": 0.85,
                "source": "system"
            },
            {
                "rule_id": "tomato_contaminated_soil_wilt_rule",
                "antecedents": ["tomato", "contaminated_soil"],
                "consequent": "bacterial_wilt_risk",
                "confidence": 0.88,
                "neural_weight": 0.83,
                "source": "system"
            },
            {
                "rule_id": "cucumber_high_humidity_mildew_rule",
                "antecedents": ["cucumber", "high_humidity"],
                "consequent": "powdery_mildew_risk",
                "confidence": 0.85,
                "neural_weight": 0.80,
                "source": "system"
            },
            {
                "rule_id": "lettuce_irrigation_rule",
                "antecedents": ["lettuce", "water_stress"],
                "consequent": "need_irrigation",
                "confidence": 0.95,
                "neural_weight": 0.92,
                "source": "system"
            },
            {
                "rule_id": "tomato_fertilization_rule",
                "antecedents": ["tomato", "vegetative_growth"],
                "consequent": "need_fertilization",
                "confidence": 0.90,
                "neural_weight": 0.87,
                "source": "system"
            },
            {
                "rule_id": "cucumber_pruning_rule",
                "antecedents": ["cucumber", "vegetative_growth"],
                "consequent": "need_pruning",
                "confidence": 0.88,
                "neural_weight": 0.84,
                "source": "system"
            },
            
            # 作物生长规则
            {
                "rule_id": "optimal_conditions_healthy_rule",
                "antecedents": ["optimal_temperature", "optimal_humidity", "optimal_light", "optimal_ph", "sufficient_nutrients"],
                "consequent": "healthy_growth",
                "confidence": 0.98,
                "neural_weight": 0.95,
                "source": "system"
            },
            {
                "rule_id": "healthy_growth_high_yield_rule",
                "antecedents": ["healthy_growth"],
                "consequent": "high_yield",
                "confidence": 0.95,
                "neural_weight": 0.92,
                "source": "system"
            },
            {
                "rule_id": "water_stress_weak_growth_rule",
                "antecedents": ["water_stress"],
                "consequent": "weak_growth",
                "confidence": 0.95,
                "neural_weight": 0.90,
                "source": "system"
            },
            {
                "rule_id": "weak_growth_low_yield_rule",
                "antecedents": ["weak_growth"],
                "consequent": "low_yield",
                "confidence": 0.90,
                "neural_weight": 0.86,
                "source": "system"
            },
            {
                "rule_id": "fungal_growth_disease_rule",
                "antecedents": ["fungal_growth"],
                "consequent": "disease_risk",
                "confidence": 0.92,
                "neural_weight": 0.88,
                "source": "system"
            },
            {
                "rule_id": "disease_low_yield_rule",
                "antecedents": ["disease_risk"],
                "consequent": "low_yield",
                "confidence": 0.90,
                "neural_weight": 0.85,
                "source": "system"
            },
            
            # 土壤健康规则
            {
                "rule_id": "healthy_soil_healthy_plants_rule",
                "antecedents": ["healthy_soil"],
                "consequent": "healthy_plants",
                "confidence": 0.90,
                "neural_weight": 0.87,
                "source": "system"
            },
            {
                "rule_id": "improved_soil_structure_healthy_soil_rule",
                "antecedents": ["improved_soil_structure"],
                "consequent": "healthy_soil",
                "confidence": 0.88,
                "neural_weight": 0.84,
                "source": "system"
            },
            {
                "rule_id": "tilling_improved_structure_rule",
                "antecedents": ["tilling"],
                "consequent": "improved_soil_structure",
                "confidence": 0.85,
                "neural_weight": 0.80,
                "source": "system"
            },
            
            # 病害管理规则
            {
                "rule_id": "reduced_pests_increased_yield_rule",
                "antecedents": ["reduced_pests"],
                "consequent": "increased_yield",
                "confidence": 0.88,
                "neural_weight": 0.83,
                "source": "system"
            },
            {
                "rule_id": "crop_rotation_reduced_pests_rule",
                "antecedents": ["crop_rotation"],
                "consequent": "reduced_pests",
                "confidence": 0.85,
                "neural_weight": 0.80,
                "source": "system"
            },
            {
                "rule_id": "intercropping_reduced_pests_rule",
                "antecedents": ["intercropping"],
                "consequent": "reduced_pests",
                "confidence": 0.82,
                "neural_weight": 0.77,
                "source": "system"
            },
            
            # 水资源管理规则
            {
                "rule_id": "mulching_reduced_evaporation_rule",
                "antecedents": ["mulching"],
                "consequent": "reduced_evaporation",
                "confidence": 0.90,
                "neural_weight": 0.85,
                "source": "system"
            },
            {
                "rule_id": "reduced_evaporation_conserved_water_rule",
                "antecedents": ["reduced_evaporation"],
                "consequent": "conserved_water",
                "confidence": 0.95,
                "neural_weight": 0.90,
                "source": "system"
            },
            {
                "rule_id": "conserved_water_reduced_irrigation_rule",
                "antecedents": ["conserved_water"],
                "consequent": "reduced_irrigation_needs",
                "confidence": 0.92,
                "neural_weight": 0.87,
                "source": "system"
            }
        ]
        
        for rule_data in default_rules:
            rule = NeuralSymbolicRule(
                rule_id=rule_data["rule_id"],
                antecedents=rule_data["antecedents"],
                consequent=rule_data["consequent"],
                confidence=rule_data["confidence"],
                neural_weight=rule_data["neural_weight"],
                source=rule_data["source"],
                timestamp=datetime.now()
            )
            self.add_rule(rule)
    
    def add_rule(self, rule: NeuralSymbolicRule):
        """添加神经符号规则"""
        self.rules.append(rule)
        self.rule_index[rule.rule_id] = rule
        
        # 如果Prolog可用，将规则添加到Prolog知识库
        if self.prolog:
            try:
                # 将规则转换为Prolog格式
                antecedents_str = ", ".join(rule.antecedents)
                prolog_rule = f"rule({rule.rule_id}, [{antecedents_str}], {rule.consequent}, {rule.confidence})"
                self.prolog.assertz(prolog_rule)
            except Exception as e:
                logger.error(f"添加Prolog规则失败: {rule.rule_id}, 错误: {str(e)}")
    
    def add_prolog_fact(self, fact: str):
        """添加Prolog事实"""
        if self.prolog:
            try:
                self.prolog.assertz(fact)
                logger.debug(f"添加Prolog事实成功: {fact}")
                return True
            except Exception as e:
                logger.error(f"添加Prolog事实失败: {fact}, 错误: {str(e)}")
        return False
    
    async def symbolic_reasoning_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """异步符号推理，使用Prolog、简化的符号推理或外部NELLIE服务"""
        logger.debug(f"开始符号推理，查询: {query}")
        
        results = []
        
        # 优先使用外部NELLIE服务（如果可用且启用）
        if self.use_external_nellie and nellie_service.is_enabled:
            logger.debug(f"使用外部NELLIE服务进行推理")
            nellie_result = await nellie_service.perform_symbolic_reasoning(query, context)
            if nellie_result:
                # 将NELLIE结果转换为内部ReasoningResult格式
                result = ReasoningResult(
                    conclusion=nellie_result["conclusion"],
                    confidence=nellie_result["confidence"],
                    reasoning_path=nellie_result["reasoning_path"],
                    timestamp=datetime.now(),
                    used_knowledge=["nellie_api"],
                    is_symbolic=True,
                    neural_support=None,
                    attributes={
                        "symbolic_form": nellie_result["symbolic_form"],
                        "explanation": nellie_result["explanation"]
                    }
                )
                results.append(result)
                return results
        
        # 如果外部NELLIE服务不可用或未启用，使用内部推理
        if self.prolog and PROLOG_AVAILABLE:
            # 使用Prolog进行推理
            results.extend(self._prolog_reasoning(query, context))
        else:
            # 使用简化的符号推理
            results.extend(self._simple_symbolic_reasoning(query, context))
        
        return results
    
    def symbolic_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """符号推理（同步包装器），使用Prolog、简化的符号推理或外部NELLIE服务"""
        import asyncio
        
        # 使用asyncio运行异步方法
        return asyncio.run(self.symbolic_reasoning_async(query, context))
    
    def _prolog_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """使用Prolog进行推理"""
        results = []
        
        try:
            # 将自然语言查询转换为Prolog查询
            prolog_query = self._natural_to_prolog(query)
            logger.debug(f"转换后的Prolog查询: {prolog_query}")
            
            # 执行Prolog查询
            prolog_results = list(self.prolog.query(prolog_query))
            
            for result in prolog_results:
                # 解析Prolog结果
                conclusion = self._prolog_to_natural(result)
                
                reasoning_result = ReasoningResult(
                    conclusion=conclusion,
                    confidence=0.95,  # 默认置信度
                    reasoning_path=[f"Prolog查询: {prolog_query}", f"结果: {result}"],
                    timestamp=datetime.now(),
                    used_knowledge=["prolog_rule"],
                    is_symbolic=True
                )
                results.append(reasoning_result)
        except Exception as e:
            logger.error(f"Prolog推理失败: {str(e)}")
            # 回退到简化推理
            results.extend(self._simple_symbolic_reasoning(query, context))
        
        return results
    
    def _simple_symbolic_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """简化的符号推理，当Prolog不可用时使用"""
        results = []
        query_lower = query.lower()
        
        # 规则匹配推理
        matched_rules = []
        for rule in self.rules:
            # 检查规则的前提条件是否在查询中
            if all(antecedent.lower() in query_lower for antecedent in rule.antecedents):
                matched_rules.append(rule)
        
        for rule in matched_rules:
            # 生成推理结果
            conclusion = f"根据规则 {rule.rule_id}，{rule.consequent}"
            reasoning_result = ReasoningResult(
                conclusion=conclusion,
                confidence=rule.confidence,
                reasoning_path=[
                    f"前提: {', '.join(rule.antecedents)}",
                    f"规则: {rule.rule_id}",
                    f"结论: {rule.consequent}"
                ],
                timestamp=datetime.now(),
                used_knowledge=[rule.rule_id],
                is_symbolic=True
            )
            results.append(reasoning_result)
        
        # 如果没有匹配的规则，尝试从知识库中搜索相关知识
        if not results:
            # 搜索相关知识
            related_knowledge = self.knowledge_base.search_knowledge(query)
            if related_knowledge:
                for knowledge in related_knowledge:
                    conclusion = f"知识库中找到: {knowledge.content}"
                    reasoning_result = ReasoningResult(
                        conclusion=conclusion,
                        confidence=knowledge.confidence,
                        reasoning_path=[f"从知识库获取: {knowledge.id}"],
                        timestamp=datetime.now(),
                        used_knowledge=[knowledge.id],
                        is_symbolic=True
                    )
                    results.append(reasoning_result)
        
        return results
    
    def _natural_to_prolog(self, natural_query: str) -> str:
        """将自然语言查询转换为Prolog查询"""
        # 简单的查询转换示例
        query_lower = natural_query.lower()
        
        if "下雨" in query_lower or "rain" in query_lower:
            return "relation(causes, rain, X)"
        elif "加热" in query_lower or "heating" in query_lower:
            return "relation(causes, heating, X)"
        elif "睡眠" in query_lower or "sleep" in query_lower:
            return "relation(causes, sleep_deprivation, X)"
        elif "运动" in query_lower or "exercise" in query_lower:
            return "relation(causes, exercise, X)"
        elif "天气" in query_lower or "weather" in query_lower:
            return "category(X, weather)"
        elif "健康" in query_lower or "health" in query_lower:
            return "category(X, health)"
        elif "物理" in query_lower or "physics" in query_lower:
            return "category(X, physics)"
        elif "导致" in query_lower or "cause" in query_lower:
            return "relation(causes, X, Y)"
        elif "属于" in query_lower or "is a" in query_lower:
            return "category(X, Y)"
        elif "有" in query_lower or "has" in query_lower:
            return "relation(has_a, X, Y)"
        else:
            # 默认查询
            return f"category(X, Y)"
    
    def _prolog_to_natural(self, prolog_result: Dict[str, Any]) -> str:
        """将Prolog结果转换为自然语言"""
        # 更完善的结果转换
        result_str = ""
        for key, value in prolog_result.items():
            if key == "X" and "Y" in prolog_result:
                # 处理 category(X, Y) 结果
                return f"{value} 属于 {prolog_result['Y']} 类别"
            elif key == "Y" and "X" in prolog_result:
                # 处理 relation(causes, X, Y) 结果
                return f"{prolog_result['X']} 导致 {value}"
            else:
                result_str += f"{key} = {value}, "
        return result_str.rstrip(", ")
    
    def neural_symbolic_integration(self, neural_result: str, symbolic_context: List[str], neural_confidence: float = 0.85) -> ReasoningResult:
        """神经符号集成，结合神经网络结果和符号推理
        
        Args:
            neural_result: 神经网络结果
            symbolic_context: 符号推理上下文
            neural_confidence: 神经网络结果的置信度
            
        Returns:
            集成后的推理结果
        """
        logger.debug(f"开始神经符号集成，神经结果: {neural_result}, 符号上下文: {symbolic_context}, 神经置信度: {neural_confidence}")
        
        reasoning_path = [
            f"神经网络结果: {neural_result} (置信度: {neural_confidence:.2f})",
            f"符号上下文: {', '.join(symbolic_context)}"
        ]
        
        # 计算符号支持度
        symbolic_support, support_details = self._calculate_symbolic_support(neural_result, symbolic_context)
        reasoning_path.append(f"符号支持度计算: {support_details}")
        reasoning_path.append(f"符号支持度: {symbolic_support:.2f}")
        
        # 分析神经网络结果和符号上下文的关系
        relation_analysis = self._analyze_relation(neural_result, symbolic_context)
        reasoning_path.append(f"关系分析: {relation_analysis}")
        
        # 根据关系选择融合策略
        fusion_strategy = self._select_fusion_strategy(relation_analysis)
        reasoning_path.append(f"融合策略: {fusion_strategy}")
        
        # 计算综合置信度
        integrated_confidence = self._calculate_integrated_confidence(neural_confidence, symbolic_support, fusion_strategy)
        reasoning_path.append(f"综合置信度: {integrated_confidence:.2f}")
        
        # 生成集成结果
        result = ReasoningResult(
            conclusion=neural_result,
            confidence=integrated_confidence,
            reasoning_path=reasoning_path,
            timestamp=datetime.now(),
            used_knowledge=symbolic_context,
            is_symbolic=False,
            neural_support=neural_confidence,
            attributes={
                'symbolic_support': symbolic_support,
                'fusion_strategy': fusion_strategy,
                'relation_analysis': relation_analysis,
                'support_details': support_details
            }
        )
        
        return result
    
    def _calculate_symbolic_support(self, neural_result: str, symbolic_context: List[str]) -> Tuple[float, str]:
        """计算符号支持度，返回支持度和计算细节"""
        if not symbolic_context:
            return 0.0, "无符号上下文"
        
        support_score = 0.0
        matching_rules = []
        matching_knowledge = []
        
        neural_lower = neural_result.lower()
        
        # 1. 检查规则匹配
        for rule in self.rules:
            # 检查规则的前提条件是否在符号上下文中
            if all(antecedent in symbolic_context for antecedent in rule.antecedents):
                # 检查结论是否与神经网络结果相关
                if rule.consequent.lower() in neural_lower or neural_lower in rule.consequent.lower():
                    matching_rules.append((rule.rule_id, rule.confidence))
                    support_score += rule.confidence * rule.neural_weight
        
        # 2. 检查知识库匹配
        for context in symbolic_context:
            context_lower = context.lower()
            related_knowledge = self.knowledge_base.search_knowledge(context_lower)
            
            for knowledge in related_knowledge:
                if knowledge.content.lower() in neural_lower or neural_lower in knowledge.content.lower():
                    matching_knowledge.append((knowledge.id, knowledge.confidence))
                    support_score += knowledge.confidence
        
        # 3. 归一化支持度
        total_weight = len(symbolic_context)
        if matching_rules:
            total_weight += len(matching_rules)  # 规则匹配权重更高
        
        normalized_support = support_score / total_weight if total_weight > 0 else 0.0
        final_support = min(1.0, max(0.0, normalized_support))
        
        # 生成支持度计算细节
        details = f"规则匹配: {len(matching_rules)} 条, 知识库匹配: {len(matching_knowledge)} 条, 归一化后: {final_support:.2f}"
        if matching_rules:
            details += f"\n匹配规则: {', '.join([f'{rule_id} ({conf:.2f})' for rule_id, conf in matching_rules])}"
        if matching_knowledge:
            details += f"\n匹配知识: {', '.join([f'{k_id} ({conf:.2f})' for k_id, conf in matching_knowledge])}"
        
        return final_support, details
    
    def _analyze_relation(self, neural_result: str, symbolic_context: List[str]) -> str:
        """分析神经网络结果和符号上下文的关系"""
        neural_lower = neural_result.lower()
        context_lower = ' '.join(symbolic_context).lower()
        
        # 检查是否有直接匹配
        direct_match = any(term in neural_lower for term in symbolic_context)
        if direct_match:
            return "直接匹配 - 神经网络结果包含符号上下文中的术语"
        
        # 检查是否有间接关联
        related_knowledge = self.knowledge_base.search_knowledge(neural_lower)
        if related_knowledge:
            for knowledge in related_knowledge:
                if any(term in knowledge.content.lower() for term in symbolic_context):
                    return "间接关联 - 神经网络结果与符号上下文通过知识库关联"
        
        # 检查是否有规则支持
        for rule in self.rules:
            if all(antecedent in symbolic_context for antecedent in rule.antecedents):
                if rule.consequent.lower() in neural_lower or neural_lower in rule.consequent.lower():
                    return "规则支持 - 存在支持神经网络结果的符号规则"
        
        # 检查是否有矛盾
        for rule in self.rules:
            if all(antecedent in symbolic_context for antecedent in rule.antecedents):
                # 检查是否有相反的规则
                if "not_" + rule.consequent.lower() in neural_lower or "否定" + rule.consequent.lower() in neural_lower:
                    return "矛盾 - 神经网络结果与符号规则矛盾"
        
        return "弱关联 - 神经网络结果与符号上下文关联较弱"
    
    def _select_fusion_strategy(self, relation_analysis: str) -> str:
        """根据关系分析选择融合策略"""
        if "直接匹配" in relation_analysis or "规则支持" in relation_analysis:
            return "增强融合 - 符号推理增强神经网络结果"
        elif "间接关联" in relation_analysis:
            return "互补融合 - 神经网络结果与符号推理互补"
        elif "矛盾" in relation_analysis:
            return "谨慎融合 - 神经网络结果与符号推理存在矛盾，降低置信度"
        else:
            return "平衡融合 - 平衡神经网络结果和符号推理"
    
    def _calculate_integrated_confidence(self, neural_confidence: float, symbolic_support: float, fusion_strategy: str) -> float:
        """根据融合策略计算综合置信度"""
        if fusion_strategy == "增强融合":
            # 符号推理增强神经网络结果
            weight_neural = 0.6
            weight_symbolic = 0.4
        elif fusion_strategy == "互补融合":
            # 神经网络结果与符号推理互补
            weight_neural = 0.5
            weight_symbolic = 0.5
        elif fusion_strategy == "谨慎融合":
            # 神经网络结果与符号推理存在矛盾，降低置信度
            weight_neural = 0.4
            weight_symbolic = 0.6
            # 降低整体置信度
            confidence_penalty = 0.1
            neural_confidence = max(0.0, neural_confidence - confidence_penalty)
        else:  # 平衡融合
            weight_neural = 0.5
            weight_symbolic = 0.5
        
        # 计算加权平均
        integrated_confidence = (neural_confidence * weight_neural) + (symbolic_support * weight_symbolic)
        
        # 确保置信度在合理范围内
        return min(1.0, max(0.0, integrated_confidence))
    
    def explain_reasoning(self, result: ReasoningResult) -> str:
        """生成推理过程的自然语言解释"""
        explanation = f"结论: {result.conclusion}\n"
        explanation += f"置信度: {result.confidence:.2f}\n"
        explanation += "推理路径:\n"
        
        for i, step in enumerate(result.reasoning_path):
            explanation += f"  {i+1}. {step}\n"
        
        explanation += f"使用的知识: {', '.join(result.used_knowledge)}\n"
        explanation += f"推理类型: {'符号推理' if result.is_symbolic else '神经符号集成'}\n"
        explanation += f"推理时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return explanation
    
    def reason_about_knowledge(self, knowledge_id: str) -> List[ReasoningResult]:
        """对特定知识进行推理"""
        knowledge = self.knowledge_base.get_knowledge_by_id(knowledge_id)
        if not knowledge:
            logger.error(f"知识不存在: {knowledge_id}")
            return []
        
        # 获取知识的相关知识
        related_knowledge = self.knowledge_base.get_related_knowledge(knowledge_id)
        related_knowledge_ids = [k.id for k in related_knowledge]
        
        # 生成查询
        query = f"关于 {knowledge.content} 的推理"
        
        # 执行推理
        return self.symbolic_reasoning(query, {"knowledge_id": knowledge_id, "related_knowledge": related_knowledge_ids})
    
    def get_prolog_facts(self) -> List[str]:
        """获取所有Prolog事实"""
        if self.prolog and PROLOG_AVAILABLE:
            try:
                # 查询所有事实
                facts = []
                for fact in self.prolog.query("fact(X)"):
                    facts.append(str(fact))
                return facts
            except Exception as e:
                logger.error(f"获取Prolog事实失败: {str(e)}")
        return []
    
    def get_prolog_rules(self) -> List[str]:
        """获取所有Prolog规则"""
        if self.prolog and PROLOG_AVAILABLE:
            try:
                rules = []
                for rule in self.prolog.query("rule(X, Y, Z, W)"):
                    rules.append(str(rule))
                return rules
            except Exception as e:
                logger.error(f"获取Prolog规则失败: {str(e)}")
        return []


# 创建全局神经符号系统实例
neural_symbolic_system = NeuralSymbolicSystem()