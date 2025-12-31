"""
农业规则约束验证模块

负责迁移学习过程中农业领域规则的约束验证，包括：
- 农业生长周期规则验证
- 环境参数约束检查
- 作物特性匹配验证
- 农业最佳实践规则
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json


class RuleViolationType(Enum):
    """规则违反类型"""
    GROWTH_CYCLE = "growth_cycle"  # 生长周期违反
    ENVIRONMENTAL = "environmental"  # 环境参数违反
    CROP_SPECIFIC = "crop_specific"  # 作物特性违反
    BEST_PRACTICE = "best_practice"  # 最佳实践违反
    SAFETY = "safety"  # 安全性违反


class RuleSeverity(Enum):
    """规则违反严重程度"""
    LOW = "low"  # 低
    MEDIUM = "medium"  # 中
    HIGH = "high"  # 高
    CRITICAL = "critical"  # 严重


@dataclass
class RuleViolation:
    """规则违反记录"""
    violation_type: RuleViolationType
    severity: RuleSeverity
    rule_id: str
    description: str
    actual_value: Any
    expected_range: Tuple[Any, Any]
    recommendation: str


@dataclass
class RuleValidationResult:
    """规则验证结果"""
    validation_passed: bool
    overall_severity: RuleSeverity
    violations: List[RuleViolation]
    warning_count: int
    error_count: int
    recommendations: List[str]


class AgriculturalRuleValidator:
    """农业规则验证器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 加载农业规则库
        self.agricultural_rules = self._load_agricultural_rules()
        
        # 作物特性数据库
        self.crop_database = self._load_crop_database()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "strict_mode": True,  # 严格模式
            "allow_partial_violations": False,  # 是否允许部分违反
            "max_violations": 3,  # 最大允许违反数
            "severity_threshold": RuleSeverity.HIGH,  # 严重程度阈值
            "region_specific_rules": True  # 是否启用区域特定规则
        }
    
    def _load_agricultural_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载农业规则库"""
        # 内置农业规则库
        return {
            "growth_temperature": {
                "id": "GRW_TEMP_001",
                "type": RuleViolationType.ENVIRONMENTAL,
                "severity": RuleSeverity.HIGH,
                "description": "作物生长温度范围约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_growth_temperature
            },
            "humidity_range": {
                "id": "HUM_RNG_001", 
                "type": RuleViolationType.ENVIRONMENTAL,
                "severity": RuleSeverity.MEDIUM,
                "description": "湿度范围约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_humidity_range
            },
            "light_requirements": {
                "id": "LGT_REQ_001",
                "type": RuleViolationType.ENVIRONMENTAL,
                "severity": RuleSeverity.MEDIUM,
                "description": "光照需求约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_light_requirements
            },
            "growth_cycle_match": {
                "id": "GRW_CYC_001",
                "type": RuleViolationType.GROWTH_CYCLE,
                "severity": RuleSeverity.CRITICAL,
                "description": "生长周期匹配约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_growth_cycle_match
            },
            "nutrient_balance": {
                "id": "NUT_BAL_001",
                "type": RuleViolationType.BEST_PRACTICE,
                "severity": RuleSeverity.MEDIUM,
                "description": "营养平衡约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_nutrient_balance
            },
            "pesticide_safety": {
                "id": "PST_SAF_001",
                "type": RuleViolationType.SAFETY,
                "severity": RuleSeverity.CRITICAL,
                "description": "农药安全使用约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_pesticide_safety
            },
            "water_requirements": {
                "id": "WTR_REQ_001",
                "type": RuleViolationType.ENVIRONMENTAL,
                "severity": RuleSeverity.HIGH,
                "description": "水分需求约束",
                "applicable_crops": ["all"],
                "validation_function": self._validate_water_requirements
            }
        }
    
    def _load_crop_database(self) -> Dict[str, Dict[str, Any]]:
        """加载作物特性数据库"""
        return {
            "rice": {
                "name": "水稻",
                "growth_temperature": (20, 35),  # 适宜温度范围
                "optimal_temperature": (25, 30),  # 最佳温度范围
                "humidity_range": (60, 85),  # 湿度范围(%)
                "light_requirements": "moderate",  # 光照需求
                "growth_cycle_days": (90, 150),  # 生长周期(天)
                "water_requirements": "high",  # 水分需求
                "nutrient_needs": {"N": "high", "P": "medium", "K": "high"}
            },
            "wheat": {
                "name": "小麦",
                "growth_temperature": (10, 25),
                "optimal_temperature": (15, 20),
                "humidity_range": (40, 70),
                "light_requirements": "high",
                "growth_cycle_days": (120, 180),
                "water_requirements": "medium",
                "nutrient_needs": {"N": "high", "P": "high", "K": "medium"}
            },
            "corn": {
                "name": "玉米",
                "growth_temperature": (15, 35),
                "optimal_temperature": (20, 30),
                "humidity_range": (50, 80),
                "light_requirements": "high",
                "growth_cycle_days": (80, 120),
                "water_requirements": "medium",
                "nutrient_needs": {"N": "high", "P": "medium", "K": "high"}
            },
            "tomato": {
                "name": "番茄",
                "growth_temperature": (18, 30),
                "optimal_temperature": (20, 25),
                "humidity_range": (50, 70),
                "light_requirements": "high",
                "growth_cycle_days": (70, 100),
                "water_requirements": "medium",
                "nutrient_needs": {"N": "medium", "P": "high", "K": "high"}
            }
        }
    
    def validate_migration_rules(self,
                               source_context: Dict[str, Any],
                               target_context: Dict[str, Any],
                               crop_type: str) -> RuleValidationResult:
        """
        验证迁移学习的农业规则约束
        
        Args:
            source_context: 源领域上下文
            target_context: 目标领域上下文
            crop_type: 作物类型
            
        Returns:
            RuleValidationResult: 规则验证结果
        """
        try:
            violations = []
            
            # 1. 验证源领域规则
            source_violations = self._validate_context_rules(source_context, crop_type, "source")
            violations.extend(source_violations)
            
            # 2. 验证目标领域规则
            target_violations = self._validate_context_rules(target_context, crop_type, "target")
            violations.extend(target_violations)
            
            # 3. 验证跨领域规则一致性
            cross_domain_violations = self._validate_cross_domain_rules(
                source_context, target_context, crop_type)
            violations.extend(cross_domain_violations)
            
            # 4. 分析验证结果
            validation_passed = self._analyze_validation_result(violations)
            overall_severity = self._determine_overall_severity(violations)
            
            # 5. 统计违反情况
            warning_count, error_count = self._count_violations_by_severity(violations)
            
            # 6. 生成改进建议
            recommendations = self._generate_recommendations(violations, crop_type)
            
            return RuleValidationResult(
                validation_passed=validation_passed,
                overall_severity=overall_severity,
                violations=violations,
                warning_count=warning_count,
                error_count=error_count,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"农业规则验证失败: {e}")
            # 返回最严格的验证结果
            return RuleValidationResult(
                validation_passed=False,
                overall_severity=RuleSeverity.CRITICAL,
                violations=[RuleViolation(
                    violation_type=RuleViolationType.SAFETY,
                    severity=RuleSeverity.CRITICAL,
                    rule_id="VALIDATION_ERROR",
                    description="规则验证过程出现异常",
                    actual_value=None,
                    expected_range=(None, None),
                    recommendation="检查输入数据格式和完整性"
                )],
                warning_count=0,
                error_count=1,
                recommendations=["农业规则验证过程出现异常，建议检查系统配置"]
            )
    
    def _validate_context_rules(self, context: Dict[str, Any], crop_type: str, context_type: str) -> List[RuleViolation]:
        """验证特定上下文的农业规则"""
        violations = []
        
        # 获取作物特性
        crop_info = self.crop_database.get(crop_type, {})
        if not crop_info:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.CROP_SPECIFIC,
                severity=RuleSeverity.HIGH,
                rule_id="CROP_UNKNOWN",
                description=f"未知作物类型: {crop_type}",
                actual_value=crop_type,
                expected_range=list(self.crop_database.keys()),
                recommendation="请使用支持的作物类型或添加新的作物信息"
            ))
            return violations
        
        # 应用所有适用的规则
        for rule_id, rule_config in self.agricultural_rules.items():
            # 检查规则是否适用于当前作物
            applicable_crops = rule_config.get("applicable_crops", [])
            if "all" not in applicable_crops and crop_type not in applicable_crops:
                continue
            
            # 执行规则验证
            validation_func = rule_config["validation_function"]
            rule_violations = validation_func(context, crop_info, context_type)
            violations.extend(rule_violations)
        
        return violations
    
    def _validate_cross_domain_rules(self, 
                                   source_context: Dict[str, Any],
                                   target_context: Dict[str, Any],
                                   crop_type: str) -> List[RuleViolation]:
        """验证跨领域规则一致性"""
        violations = []
        
        # 检查环境参数变化是否合理
        env_changes = self._validate_environmental_changes(source_context, target_context, crop_type)
        violations.extend(env_changes)
        
        # 检查生长周期匹配度
        growth_cycle_match = self._validate_growth_cycle_transition(source_context, target_context, crop_type)
        violations.extend(growth_cycle_match)
        
        # 检查农业实践一致性
        practice_consistency = self._validate_agricultural_practices(source_context, target_context, crop_type)
        violations.extend(practice_consistency)
        
        return violations
    
    def _validate_growth_temperature(self, context: Dict[str, Any], crop_info: Dict[str, Any], 
                                   context_type: str) -> List[RuleViolation]:
        """验证生长温度规则"""
        violations = []
        
        temperature = context.get("temperature")
        if temperature is not None:
            min_temp, max_temp = crop_info.get("growth_temperature", (0, 50))
            
            if temperature < min_temp or temperature > max_temp:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.ENVIRONMENTAL,
                    severity=RuleSeverity.HIGH,
                    rule_id="GRW_TEMP_001",
                    description=f"{context_type}领域温度超出作物适宜范围",
                    actual_value=temperature,
                    expected_range=(min_temp, max_temp),
                    recommendation=f"调整温度至{min_temp}-{max_temp}℃范围内"
                ))
        
        return violations
    
    def _validate_humidity_range(self, context: Dict[str, Any], crop_info: Dict[str, Any],
                                context_type: str) -> List[RuleViolation]:
        """验证湿度范围规则"""
        violations = []
        
        humidity = context.get("humidity")
        if humidity is not None:
            min_humidity, max_humidity = crop_info.get("humidity_range", (30, 90))
            
            if humidity < min_humidity or humidity > max_humidity:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.ENVIRONMENTAL,
                    severity=RuleSeverity.MEDIUM,
                    rule_id="HUM_RNG_001",
                    description=f"{context_type}领域湿度超出适宜范围",
                    actual_value=humidity,
                    expected_range=(min_humidity, max_humidity),
                    recommendation=f"调整湿度至{min_humidity}-{max_humidity}%范围内"
                ))
        
        return violations
    
    def _validate_light_requirements(self, context: Dict[str, Any], crop_info: Dict[str, Any],
                                   context_type: str) -> List[RuleViolation]:
        """验证光照需求规则"""
        violations = []
        
        light_intensity = context.get("light_intensity")
        light_requirement = crop_info.get("light_requirements", "moderate")
        
        if light_intensity is not None:
            # 根据作物光照需求验证光照强度
            requirement_ranges = {
                "low": (1000, 3000),    # 低光照需求
                "moderate": (3000, 6000), # 中等光照需求
                "high": (6000, 10000)   # 高光照需求
            }
            
            min_light, max_light = requirement_ranges.get(light_requirement, (2000, 8000))
            
            if light_intensity < min_light or light_intensity > max_light:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.ENVIRONMENTAL,
                    severity=RuleSeverity.MEDIUM,
                    rule_id="LGT_REQ_001",
                    description=f"{context_type}领域光照强度不匹配作物需求",
                    actual_value=light_intensity,
                    expected_range=(min_light, max_light),
                    recommendation=f"调整光照强度至{min_light}-{max_light}lux范围内"
                ))
        
        return violations
    
    def _validate_growth_cycle_match(self, context: Dict[str, Any], crop_info: Dict[str, Any],
                                   context_type: str) -> List[RuleViolation]:
        """验证生长周期匹配规则"""
        violations = []
        
        growth_stage = context.get("growth_stage")
        if growth_stage:
            # 检查生长阶段是否合理
            valid_stages = ["seedling", "vegetative", "flowering", "fruiting", "harvest"]
            if growth_stage not in valid_stages:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.GROWTH_CYCLE,
                    severity=RuleSeverity.MEDIUM,
                    rule_id="GRW_CYC_001",
                    description=f"{context_type}领域生长阶段无效",
                    actual_value=growth_stage,
                    expected_range=valid_stages,
                    recommendation="使用有效的生长阶段标识"
                ))
        
        return violations
    
    def _validate_nutrient_balance(self, context: Dict[str, Any], crop_info: Dict[str, Any],
                                 context_type: str) -> List[RuleViolation]:
        """验证营养平衡规则"""
        violations = []
        
        nutrients = context.get("nutrients", {})
        if nutrients:
            crop_needs = crop_info.get("nutrient_needs", {})
            
            # 检查主要营养元素是否平衡
            for element, requirement in crop_needs.items():
                actual_level = nutrients.get(element, 0)
                
                # 根据需求级别设置阈值
                requirement_ranges = {
                    "low": (0.1, 0.3),
                    "medium": (0.3, 0.6),
                    "high": (0.6, 1.0)
                }
                
                min_level, max_level = requirement_ranges.get(requirement, (0.2, 0.8))
                
                if actual_level < min_level or actual_level > max_level:
                    violations.append(RuleViolation(
                        violation_type=RuleViolationType.BEST_PRACTICE,
                        severity=RuleSeverity.MEDIUM,
                        rule_id="NUT_BAL_001",
                        description=f"{context_type}领域{element}元素营养不平衡",
                        actual_value=actual_level,
                        expected_range=(min_level, max_level),
                        recommendation=f"调整{element}元素至适宜范围"
                    ))
        
        return violations
    
    def _validate_pesticide_safety(self, context: Dict[str, Any], crop_info: Dict[str, Any],
                                 context_type: str) -> List[RuleViolation]:
        """验证农药安全规则"""
        violations = []
        
        pesticides = context.get("pesticides", [])
        if pesticides:
            # 检查农药使用是否合规
            for pesticide in pesticides:
                if isinstance(pesticide, dict):
                    concentration = pesticide.get("concentration", 0)
                    application_date = pesticide.get("application_date")
                    
                    # 检查浓度是否安全
                    if concentration > 0.1:  # 假设安全阈值为0.1%
                        violations.append(RuleViolation(
                            violation_type=RuleViolationType.SAFETY,
                            severity=RuleSeverity.CRITICAL,
                            rule_id="PST_SAF_001",
                            description=f"{context_type}领域农药浓度超标",
                            actual_value=concentration,
                            expected_range=(0, 0.1),
                            recommendation="降低农药浓度至安全范围内"
                        ))
        
        return violations
    
    def _validate_water_requirements(self, context: Dict[str, Any], crop_info: Dict[str, Any],
                                   context_type: str) -> List[RuleViolation]:
        """验证水分需求规则"""
        violations = []
        
        water_supply = context.get("water_supply")
        water_requirement = crop_info.get("water_requirements", "medium")
        
        if water_supply is not None:
            # 根据作物水分需求验证供水量
            requirement_ranges = {
                "low": (100, 300),    # 低水分需求(mm/天)
                "medium": (300, 600), # 中等水分需求
                "high": (600, 1000)   # 高水分需求
            }
            
            min_water, max_water = requirement_ranges.get(water_requirement, (200, 800))
            
            if water_supply < min_water or water_supply > max_water:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.ENVIRONMENTAL,
                    severity=RuleSeverity.HIGH,
                    rule_id="WTR_REQ_001",
                    description=f"{context_type}领域水分供应不匹配作物需求",
                    actual_value=water_supply,
                    expected_range=(min_water, max_water),
                    recommendation=f"调整水分供应至{min_water}-{max_water}mm/天范围内"
                ))
        
        return violations
    
    # 跨领域验证方法
    def _validate_environmental_changes(self, source_context: Dict[str, Any],
                                       target_context: Dict[str, Any],
                                       crop_type: str) -> List[RuleViolation]:
        """验证环境参数变化合理性"""
        violations = []
        
        # 检查温度变化是否合理
        source_temp = source_context.get("temperature")
        target_temp = target_context.get("temperature")
        
        if source_temp is not None and target_temp is not None:
            temp_change = abs(target_temp - source_temp)
            if temp_change > 15:  # 温度变化超过15度可能不合理
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.ENVIRONMENTAL,
                    severity=RuleSeverity.MEDIUM,
                    rule_id="ENV_CHG_001",
                    description="源目标领域温度变化过大",
                    actual_value=temp_change,
                    expected_range=(0, 15),
                    recommendation="确保温度变化在合理范围内"
                ))
        
        return violations
    
    def _validate_growth_cycle_transition(self, source_context: Dict[str, Any],
                                        target_context: Dict[str, Any],
                                        crop_type: str) -> List[RuleViolation]:
        """验证生长周期过渡合理性"""
        violations = []
        
        source_stage = source_context.get("growth_stage")
        target_stage = target_context.get("growth_stage")
        
        if source_stage and target_stage:
            # 定义合理的生长阶段过渡
            valid_transitions = {
                "seedling": ["vegetative"],
                "vegetative": ["flowering"],
                "flowering": ["fruiting"],
                "fruiting": ["harvest"]
            }
            
            valid_targets = valid_transitions.get(source_stage, [])
            if target_stage not in valid_targets and target_stage != source_stage:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.GROWTH_CYCLE,
                    severity=RuleSeverity.HIGH,
                    rule_id="CYC_TRN_001",
                    description="生长阶段过渡不合理",
                    actual_value=f"{source_stage} -> {target_stage}",
                    expected_range=valid_targets,
                    recommendation="确保生长阶段过渡符合自然规律"
                ))
        
        return violations
    
    def _validate_agricultural_practices(self, source_context: Dict[str, Any],
                                       target_context: Dict[str, Any],
                                       crop_type: str) -> List[RuleViolation]:
        """验证农业实践一致性"""
        violations = []
        
        # 检查农业实践是否一致
        source_practices = source_context.get("agricultural_practices", [])
        target_practices = target_context.get("agricultural_practices", [])
        
        # 检查是否存在冲突的实践
        conflicting_practices = self._identify_conflicting_practices(
            source_practices, target_practices)
        
        for conflict in conflicting_practices:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.BEST_PRACTICE,
                severity=RuleSeverity.MEDIUM,
                rule_id="PRC_CNF_001",
                description="农业实践存在冲突",
                actual_value=conflict,
                expected_range="一致的农业实践",
                recommendation="协调源目标领域的农业实践"
            ))
        
        return violations
    
    def _identify_conflicting_practices(self, practices1: List[str], practices2: List[str]) -> List[str]:
        """识别冲突的农业实践"""
        conflicts = []
        
        # 定义冲突的实践对
        conflict_pairs = [
            ("organic_farming", "chemical_fertilizers"),
            ("drip_irrigation", "flood_irrigation"),
            ("conservation_tillage", "conventional_tillage")
        ]
        
        for practice1, practice2 in conflict_pairs:
            if (practice1 in practices1 and practice2 in practices2) or \
               (practice2 in practices1 and practice1 in practices2):
                conflicts.append(f"{practice1} vs {practice2}")
        
        return conflicts
    
    def _analyze_validation_result(self, violations: List[RuleViolation]) -> bool:
        """分析验证结果"""
        if not violations:
            return True
        
        # 检查是否有严重违反
        critical_violations = [v for v in violations if v.severity in [RuleSeverity.HIGH, RuleSeverity.CRITICAL]]
        
        if critical_violations:
            return False
        
        # 检查违反数量是否超过阈值
        if len(violations) > self.config["max_violations"]:
            return False
        
        return True
    
    def _determine_overall_severity(self, violations: List[RuleViolation]) -> RuleSeverity:
        """确定总体严重程度"""
        if not violations:
            return RuleSeverity.LOW
        
        severities = [v.severity for v in violations]
        
        if RuleSeverity.CRITICAL in severities:
            return RuleSeverity.CRITICAL
        elif RuleSeverity.HIGH in severities:
            return RuleSeverity.HIGH
        elif RuleSeverity.MEDIUM in severities:
            return RuleSeverity.MEDIUM
        else:
            return RuleSeverity.LOW
    
    def _count_violations_by_severity(self, violations: List[RuleViolation]) -> Tuple[int, int]:
        """按严重程度统计违反情况"""
        warnings = len([v for v in violations if v.severity in [RuleSeverity.LOW, RuleSeverity.MEDIUM]])
        errors = len([v for v in violations if v.severity in [RuleSeverity.HIGH, RuleSeverity.CRITICAL]])
        
        return warnings, errors
    
    def _generate_recommendations(self, violations: List[RuleViolation], crop_type: str) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not violations:
            recommendations.append("农业规则验证通过，可以安全进行迁移学习")
            return recommendations
        
        # 根据违反类型提供针对性建议
        violation_types = set(v.violation_type for v in violations)
        
        if RuleViolationType.ENVIRONMENTAL in violation_types:
            recommendations.append("建议调整环境参数至作物适宜范围")
        
        if RuleViolationType.GROWTH_CYCLE in violation_types:
            recommendations.append("建议确保生长周期匹配和过渡合理")
        
        if RuleViolationType.SAFETY in violation_types:
            recommendations.append("必须解决安全性问题后方可进行迁移")
        
        # 通用建议
        if len(violations) > 3:
            recommendations.append("建议分阶段进行迁移，先解决主要问题")
        
        return recommendations

    async def check_migration_constraints(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查迁移学习约束
        
        Args:
            validated_data: 验证后的数据
            
        Returns:
            约束检查结果
        """
        try:
            # 从validated_data中提取必要的信息
            source_context = validated_data.get("source_context", {})
            target_context = validated_data.get("target_context", {})
            crop_type = validated_data.get("crop_type", "rice")
            
            # 调用现有的规则验证方法
            result = self.validate_migration_rules(source_context, target_context, crop_type)
            
            # 转换为decision_integration.py期望的格式
            return {
                "satisfied": result.validation_passed,
                "severity": result.overall_severity.value,
                "violations": len(result.violations),
                "recommendations": result.recommendations
            }
        except Exception as e:
            self.logger.error(f"检查迁移学习约束失败: {e}")
            return {
                "satisfied": False,
                "severity": "critical",
                "violations": 1,
                "recommendations": ["规则验证过程出现异常，请检查输入数据"]
            }

    async def check_edge_constraints(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查边缘计算约束
        
        Args:
            validated_data: 验证后的数据
            
        Returns:
            约束检查结果
        """
        try:
            # 从validated_data中提取必要的信息
            edge_context = validated_data.get("edge_context", {})
            crop_type = validated_data.get("crop_type", "rice")
            
            # 对于边缘计算约束，我们可以复用现有的上下文验证逻辑
            edge_violations = self._validate_context_rules(edge_context, crop_type, "edge")
            
            # 分析结果
            satisfied = len(edge_violations) == 0
            overall_severity = self._determine_overall_severity(edge_violations)
            
            return {
                "satisfied": satisfied,
                "severity": overall_severity.value,
                "violations": len(edge_violations),
                "recommendations": [v.recommendation for v in edge_violations]
            }
        except Exception as e:
            self.logger.error(f"检查边缘计算约束失败: {e}")
            return {
                "satisfied": False,
                "severity": "critical",
                "violations": 1,
                "recommendations": ["边缘计算约束检查失败，请检查系统配置"]
            }

    async def check_decision_constraints(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查决策约束
        
        Args:
            validated_data: 验证后的数据
            
        Returns:
            约束检查结果
        """
        try:
            # 从validated_data中提取必要的信息
            source_context = validated_data.get("source_context", {})
            target_context = validated_data.get("target_context", {})
            crop_type = validated_data.get("crop_type", "rice")
            
            # 检查决策相关的约束，我们可以复用迁移学习规则验证
            result = self.validate_migration_rules(source_context, target_context, crop_type)
            
            # 额外检查决策特定的约束
            decision_violations = []
            
            # 检查隐私级别约束
            constraints = validated_data.get("constraints", {})
            privacy_level = constraints.get("privacy_level")
            data = validated_data.get("data", {})
            
            # 如果隐私级别要求为public但数据包含敏感信息，则标记为约束违反
            if privacy_level == "public":
                # 检查数据中是否包含敏感信息
                sensitive_fields = ["personal_info", "confidential_data", "private_analytics"]
                has_sensitive_data = any(field in data and data[field] for field in sensitive_fields)
                
                if has_sensitive_data:
                    decision_violations.append("隐私级别约束违反：公开级别的请求包含敏感信息")
            
            # 分析最终结果
            final_satisfied = result.validation_passed and len(decision_violations) == 0
            
            # 添加决策特定的建议
            recommendations = result.recommendations.copy()
            if decision_violations:
                recommendations.extend([f"决策约束违反：{violation}" for violation in decision_violations])
            
            return {
                "satisfied": final_satisfied,
                "severity": "high" if decision_violations else result.overall_severity.value,
                "violations": len(result.violations) + len(decision_violations),
                "recommendations": recommendations
            }
        except Exception as e:
            self.logger.error(f"检查决策约束失败: {e}")
            return {
                "satisfied": False,
                "severity": "critical",
                "violations": 1,
                "recommendations": ["决策约束检查失败，请检查系统配置"]
            }