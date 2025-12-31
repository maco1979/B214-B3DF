"""
迁移学习与AI决策系统集成模块

负责将迁移学习风险控制机制集成到AI自主决策系统中
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..migration_learning import MigrationRiskController, DataCredibilityValidator, AgriculturalRuleValidator
from ..core.decision_engine import DecisionEngine


@dataclass
class MigrationIntegrationConfig:
    """迁移学习集成配置"""
    enable_risk_control: bool = True
    enable_data_validation: bool = True
    enable_rule_constraints: bool = True
    risk_threshold: float = 0.7
    auto_fallback: bool = True
    logging_level: str = "INFO"


class MigrationLearningIntegration:
    """迁移学习集成管理器"""
    
    def __init__(self, 
                 decision_engine: DecisionEngine,
                 config: Optional[MigrationIntegrationConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.decision_engine = decision_engine
        self.config = config or MigrationIntegrationConfig()
        
        # 初始化迁移学习组件
        self.risk_controller = MigrationRiskController()
        self.data_validator = DataCredibilityValidator()
        self.rule_validator = AgriculturalRuleValidator()
        
        # 集成状态
        self.integration_status = {
            "risk_control_enabled": self.config.enable_risk_control,
            "data_validation_enabled": self.config.enable_data_validation,
            "rule_constraints_enabled": self.config.enable_rule_constraints,
            "last_risk_assessment": None,
            "integration_errors": []
        }
    
    async def integrate_migration_risk_control(self, 
                                              source_domain: Dict[str, Any],
                                              target_domain: Dict[str, Any],
                                              model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        集成迁移学习风险控制
        
        Args:
            source_domain: 源领域信息
            target_domain: 目标领域信息
            model_info: 模型信息
            
        Returns:
            Dict[str, Any]: 集成结果
        """
        try:
            integration_result = {
                "risk_assessment": None,
                "data_validation": None,
                "rule_validation": None,
                "integration_success": False,
                "recommendations": []
            }
            
            # 1. 风险评估
            if self.config.enable_risk_control:
                # 构建验证数据字典
                validated_data = {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "model_info": model_info
                }
                risk_assessment = await self.risk_controller.assess_migration_risk(validated_data)
                integration_result["risk_assessment"] = risk_assessment
                
                # 检查风险是否可接受
                if risk_assessment.get("risk_score", 0) > self.config.risk_threshold:
                    integration_result["recommendations"].append(
                        f"迁移风险较高({risk_assessment.get('risk_score', 0):.2f})，建议调整迁移策略")
            
            # 2. 数据可信度验证
            if self.config.enable_data_validation:
                data_credibility = self.data_validator.assess_data_credibility(
                    source_domain, target_domain)
                integration_result["data_validation"] = data_credibility
                
                if data_credibility.credibility_score < 0.8:
                    integration_result["recommendations"].append(
                        f"数据可信度较低({data_credibility.credibility_score:.2f})，建议验证数据质量")
            
            # 3. 农业规则约束验证
            if self.config.enable_rule_constraints:
                crop_type = target_domain.get("crop_type", "generic")
                rule_validation = self.rule_validator.validate_migration_rules(
                    source_domain, target_domain, crop_type)
                integration_result["rule_validation"] = rule_validation
                
                if not rule_validation.validation_passed:
                    integration_result["recommendations"].append(
                        "农业规则验证未通过，请检查领域约束")
            
            # 4. 综合决策
            integration_success = self._evaluate_integration_success(integration_result)
            integration_result["integration_success"] = integration_success
            
            # 5. 更新决策引擎
            if integration_success:
                await self._update_decision_engine(integration_result)
            elif self.config.auto_fallback:
                await self._apply_fallback_strategy(integration_result)
            
            # 更新集成状态
            self.integration_status["last_risk_assessment"] = integration_result
            
            self.logger.info(f"迁移学习风险控制集成完成: {integration_success}")
            return integration_result
            
        except Exception as e:
            self.logger.error(f"迁移学习集成失败: {e}")
            self.integration_status["integration_errors"].append(str(e))
            
            return {
                "risk_assessment": None,
                "data_validation": None,
                "rule_validation": None,
                "integration_success": False,
                "recommendations": [f"集成过程出现异常: {str(e)}"],
                "error": str(e)
            }
    
    def _evaluate_integration_success(self, integration_result: Dict[str, Any]) -> bool:
        """评估集成是否成功"""
        # 检查风险评估
        risk_assessment = integration_result.get("risk_assessment")
        if risk_assessment:
            # 处理字典类型的风险评估
            risk_score = risk_assessment.get("risk_score", 0)
            if risk_score > self.config.risk_threshold:
                return False
        
        # 检查数据验证
        data_validation = integration_result.get("data_validation")
        if data_validation:
            # 检查是对象还是字典
            if hasattr(data_validation, 'credibility_score'):
                credibility = data_validation.credibility_score
            else:
                credibility = data_validation.get("credibility_score", 1.0)
            
            if credibility < 0.7:
                return False
        
        # 检查规则验证
        rule_validation = integration_result.get("rule_validation")
        if rule_validation:
            # 检查是对象还是字典
            if hasattr(rule_validation, 'validation_passed'):
                validation_passed = rule_validation.validation_passed
            else:
                validation_passed = rule_validation.get("validation_passed", True)
            
            if not validation_passed:
                return False
        
        return True
    
    async def _update_decision_engine(self, integration_result: Dict[str, Any]):
        """更新决策引擎"""
        try:
            # 提取关键信息用于决策优化
            risk_info = integration_result.get("risk_assessment")
            data_info = integration_result.get("data_validation")
            
            # 处理风险信息（支持对象和字典类型）
            if risk_info:
                if hasattr(risk_info, 'risk_score'):
                    risk_score = risk_info.risk_score
                    recommended_strategy = risk_info.recommended_strategy.value if hasattr(risk_info.recommended_strategy, 'value') else "full_transfer"
                    risk_factors = risk_info.risk_factors
                else:
                    risk_score = risk_info.get("risk_score", 0.0)
                    recommended_strategy = risk_info.get("recommended_strategy", "full_transfer")
                    risk_factors = risk_info.get("risk_factors", [])
            else:
                risk_score = 0.0
                recommended_strategy = "full_transfer"
                risk_factors = []
            
            # 处理数据信息（支持对象和字典类型）
            if data_info:
                if hasattr(data_info, 'credibility_score'):
                    credibility_score = data_info.credibility_score
                else:
                    credibility_score = data_info.get("credibility_score", 1.0)
            else:
                credibility_score = 1.0
            
            # 构建决策参数
            decision_params = {
                "migration_risk_score": risk_score,
                "data_credibility_score": credibility_score,
                "recommended_strategy": recommended_strategy,
                "risk_factors": risk_factors
            }
            
            # 更新决策引擎的迁移学习参数
            # 这里需要根据实际的决策引擎接口进行调整
            if hasattr(self.decision_engine, 'update_migration_parameters'):
                await self.decision_engine.update_migration_parameters(decision_params)
            
            self.logger.info("决策引擎迁移学习参数已更新")
            
        except Exception as e:
            self.logger.error(f"更新决策引擎失败: {e}")
            raise
    
    async def integrate_migration_learning(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        集成迁移学习处理
        
        Args:
            validated_data: 验证后的数据
            
        Returns:
            Dict[str, Any]: 迁移学习集成结果
        """
        try:
            # 从验证数据中提取必要的信息
            source_domain = validated_data.get("source_domain", {})
            target_domain = validated_data.get("target_domain", {})
            model_info = validated_data.get("model_info", {})
            
            # 执行迁移学习风险控制集成
            integration_result = await self.integrate_migration_risk_control(
                source_domain, target_domain, model_info
            )
            
            return {
                "migration_learning_result": integration_result,
                "success": integration_result.get("integration_success", False),
                "timestamp": datetime.now(),
                "processing_mode": "integrated"
            }
        except Exception as e:
            self.logger.error(f"迁移学习集成处理失败: {e}")
            return {
                "migration_learning_result": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(),
                "processing_mode": "error"
            }
    
    async def _apply_fallback_strategy(self, integration_result: Dict[str, Any]):
        """
        应用降级策略
        """
        try:
            # 根据风险评估结果应用保守策略
            risk_assessment = integration_result.get("risk_assessment")
            
            if risk_assessment:
                fallback_strategy = self._select_fallback_strategy(risk_assessment)
                
                # 应用保守的决策参数
                conservative_params = {
                    "conservative_mode": True,
                    "risk_aversion_factor": 0.8,
                    "exploration_rate": 0.1,  # 降低探索率
                    "learning_rate": 0.001,   # 降低学习率
                    "fallback_strategy": fallback_strategy
                }
                
                # 更新决策引擎
                if hasattr(self.decision_engine, 'enable_conservative_mode'):
                    await self.decision_engine.enable_conservative_mode(conservative_params)
                
                self.logger.warning(f"应用降级策略: {fallback_strategy}")
            
        except Exception as e:
            self.logger.error(f"应用降级策略失败: {e}")
    
    def _select_fallback_strategy(self, risk_assessment) -> str:
        """选择降级策略"""
        # 支持对象和字典类型的风险评估
        if hasattr(risk_assessment, 'risk_level'):
            # 处理对象类型
            risk_level = risk_assessment.risk_level.value if hasattr(risk_assessment.risk_level, 'value') else str(risk_assessment.risk_level)
        else:
            # 处理字典类型
            risk_level = risk_assessment.get("risk_level", "low")
        
        if risk_level == "critical":
            return "reject_migration"
        elif risk_level == "high":
            return "conservative_fine_tuning"
        elif risk_level == "medium":
            return "domain_adaptation"
        else:
            return "monitored_transfer"
    
    def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        return {
            "config": {
                "enable_risk_control": self.config.enable_risk_control,
                "enable_data_validation": self.config.enable_data_validation,
                "enable_rule_constraints": self.config.enable_rule_constraints,
                "risk_threshold": self.config.risk_threshold
            },
            "status": self.integration_status,
            "component_versions": {
                "risk_controller": "1.0.0",
                "data_validator": "1.0.0", 
                "rule_validator": "1.0.0"
            }
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        return self.get_integration_status()
    
    async def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        更新集成配置
        
        Args:
            config_updates: 配置更新内容
        """
        try:
            # 更新配置参数
            for key, value in config_updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"配置参数 {key} 更新为 {value}")
                else:
                    self.logger.warning(f"未知的配置参数: {key}")
            
            # 更新集成状态
            self.integration_status["config_last_updated"] = datetime.now().isoformat()
            self.integration_status["updated_config_params"] = list(config_updates.keys())
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
            raise
    
    async def validate_migration_scenario(self,
                                        scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """验证迁移场景"""
        try:
            validation_result = {
                "scenario_valid": False,
                "constraint_violations": [],
                "optimization_opportunities": [],
                "risk_analysis": {}
            }
            
            # 分析场景约束
            constraints = scenario_config.get("constraints", {})
            for constraint_type, constraint_value in constraints.items():
                if not self._validate_constraint(constraint_type, constraint_value):
                    validation_result["constraint_violations"].append(constraint_type)
            
            # 识别优化机会
            optimization_opportunities = self._identify_optimization_opportunities(scenario_config)
            validation_result["optimization_opportunities"] = optimization_opportunities
            
            # 风险评估
            risk_analysis = await self._analyze_scenario_risk(scenario_config)
            validation_result["risk_analysis"] = risk_analysis
            
            # 综合验证结果
            validation_result["scenario_valid"] = (
                len(validation_result["constraint_violations"]) == 0 and
                risk_analysis.get("overall_risk", 1.0) < self.config.risk_threshold
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"迁移场景验证失败: {e}")
            return {
                "scenario_valid": False,
                "constraint_violations": ["验证过程异常"],
                "optimization_opportunities": [],
                "risk_analysis": {"error": str(e)},
                "error": str(e)
            }
    
    def _validate_constraint(self, constraint_type: str, constraint_value: Any) -> bool:
        """验证约束条件"""
        # 简化实现
        constraint_validators = {
            "model_size": lambda x: x < 500,  # 模型大小约束
            "latency": lambda x: x < 100,     # 延迟约束
            "accuracy": lambda x: x > 0.7,    # 准确率约束
            "resources": lambda x: x.get("memory", 0) < 4096  # 资源约束
        }
        
        validator = constraint_validators.get(constraint_type)
        return validator(constraint_value) if validator else True
    
    def _identify_optimization_opportunities(self, scenario_config: Dict[str, Any]) -> List[str]:
        """识别优化机会"""
        opportunities = []
        
        # 分析配置中的优化潜力
        model_complexity = scenario_config.get("model_complexity", "medium")
        if model_complexity == "high":
            opportunities.append("考虑模型压缩以降低资源需求")
        
        data_volume = scenario_config.get("data_volume", 0)
        if data_volume > 1000:
            opportunities.append("建议使用增量学习策略")
        
        return opportunities
    
    async def _analyze_scenario_risk(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """分析场景风险"""
        # 简化风险分析
        risk_factors = {}
        
        # 技术风险
        technical_risk = scenario_config.get("technical_complexity", 0.5)
        risk_factors["technical"] = technical_risk
        
        # 数据风险
        data_quality = scenario_config.get("data_quality", 0.8)
        risk_factors["data"] = 1.0 - data_quality
        
        # 资源风险
        resource_adequacy = scenario_config.get("resource_adequacy", 0.7)
        risk_factors["resource"] = 1.0 - resource_adequacy
        
        # 综合风险
        overall_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            "risk_factors": risk_factors,
            "overall_risk": overall_risk,
            "risk_level": "high" if overall_risk > 0.7 else "medium" if overall_risk > 0.4 else "low"
        }