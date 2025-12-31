"""
迁移学习风险控制模块

负责迁移学习过程中的风险评估和控制，包括：
- 迁移可行性评估
- 风险等级划分
- 迁移策略选择
- 风险缓解措施
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
from dataclasses import dataclass


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class MigrationStrategy(Enum):
    """迁移策略枚举"""
    FULL_TRANSFER = "full_transfer"  # 完全迁移
    FEATURE_EXTRACTION = "feature_extraction"  # 特征提取
    FINE_TUNING = "fine_tuning"  # 微调
    DOMAIN_ADAPTATION = "domain_adaptation"  # 领域自适应
    REJECT = "reject"  # 拒绝迁移


@dataclass
class MigrationRiskAssessment:
    """迁移风险评估结果"""
    risk_level: RiskLevel
    risk_score: float  # 0-1之间的风险评分
    risk_factors: List[str]  # 风险因素列表
    recommended_strategy: MigrationStrategy
    confidence_score: float  # 评估置信度
    mitigation_measures: List[str]  # 风险缓解措施


class MigrationRiskController:
    """迁移学习风险控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 风险阈值配置
        self.risk_thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6, 
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.9
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data_similarity_weight": 0.4,
            "model_compatibility_weight": 0.3,
            "domain_expertise_weight": 0.2,
            "resource_constraints_weight": 0.1,
            "min_confidence_threshold": 0.7
        }
    
    async def assess_migration_risk(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估迁移学习风险（适配决策集成接口）
        
        Args:
            validated_data: 验证后的数据
            
        Returns:
            Dict[str, Any]: 风险评估结果
        """
        try:
            # 从验证数据中提取所需信息
            source_domain = validated_data.get("source_domain", {})
            target_domain = validated_data.get("target_domain", {})
            model_info = validated_data.get("model_info", {})
            
            # 如果没有提取到足够信息，使用默认值
            if not source_domain:
                source_domain = {"data_quality": 0.8, "model_architecture": "default", "knowledge_depth": 0.7}
            if not target_domain:
                target_domain = {"data_quality": 0.8, "model_architecture": "default"}
            if not model_info:
                model_info = {"model_size_mb": 50, "edge_compatible": True, "memory_required_mb": 1000, "compute_required": 50}
            
            # 提取特征分布信息
            source_features = source_domain.get("feature_distribution", {})
            target_features = target_domain.get("feature_distribution", {})
            
            # 计算特征分布相似度（简化实现）
            similarity_score = self._calculate_distribution_similarity(
                source_features, target_features)
            data_similarity_risk = 1.0 - similarity_score
            
            # 计算模型兼容性风险
            architecture_compatibility = self._check_architecture_compatibility(
                source_domain, target_domain, model_info)
            dimension_compatibility = self._check_dimension_compatibility(
                source_domain, target_domain)
            model_compatibility_score = (architecture_compatibility + dimension_compatibility) / 2
            model_compatibility_risk = 1.0 - model_compatibility_score
            
            # 计算领域专家知识风险
            domain_gap = self._calculate_domain_gap(source_domain, target_domain)
            rule_constraints = self._check_agricultural_rules(source_domain, target_domain)
            domain_expertise_risk = (domain_gap + rule_constraints) / 2
            
            # 计算资源约束风险
            model_size_risk = self._check_model_size_risk(model_info)
            edge_compatibility_risk = self._check_edge_compatibility_risk(model_info)
            resource_demand_risk = self._check_resource_demand_risk(model_info)
            resource_risk = (model_size_risk + edge_compatibility_risk + resource_demand_risk) / 3
            
            # 综合风险评分
            weights = [
                self.config["data_similarity_weight"],
                self.config["model_compatibility_weight"], 
                self.config["domain_expertise_weight"],
                self.config["resource_constraints_weight"]
            ]
            risk_scores = [data_similarity_risk, model_compatibility_risk, domain_expertise_risk, resource_risk]
            total_risk_score = sum(score * weight for score, weight in zip(risk_scores, weights))
            total_risk_score = min(max(total_risk_score, 0.0), 1.0)
            
            # 确定风险等级
            if total_risk_score <= self.risk_thresholds[RiskLevel.LOW]:
                risk_level = "low"
            elif total_risk_score <= self.risk_thresholds[RiskLevel.MEDIUM]:
                risk_level = "medium"
            elif total_risk_score <= self.risk_thresholds[RiskLevel.HIGH]:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            # 识别风险因素
            risk_factors = []
            risk_names = ["数据相似度", "模型兼容性", "领域知识", "资源约束"]
            
            for i, score in enumerate(risk_scores):
                if score > 0.5:  # 高风险因素阈值
                    risk_factors.append(f"{risk_names[i]}风险较高")
            
            # 推荐迁移策略
            if total_risk_score < 0.3:
                recommended_strategy = "full_transfer"
            elif total_risk_score < 0.6:
                recommended_strategy = "fine_tuning"
            elif total_risk_score < 0.8:
                recommended_strategy = "domain_adaptation"
            else:
                recommended_strategy = "reject"
            
            # 计算置信度
            source_data_quality = source_domain.get("data_quality", 0.5)
            target_data_quality = target_domain.get("data_quality", 0.5)
            domain_knowledge_depth = source_domain.get("knowledge_depth", 0.5)
            confidence = (source_data_quality + target_data_quality + domain_knowledge_depth) / 3
            confidence = min(max(confidence, 0.0), 1.0)
            
            # 制定缓解措施
            mitigation_measures = []
            
            for factor in risk_factors:
                if "数据相似度" in factor:
                    mitigation_measures.append("增加目标领域数据采集")
                    mitigation_measures.append("进行数据增强和预处理")
                elif "模型兼容性" in factor:
                    mitigation_measures.append("调整模型架构适应目标领域")
                    mitigation_measures.append("使用适配器层进行特征映射")
                elif "领域知识" in factor:
                    mitigation_measures.append("引入领域专家知识")
                    mitigation_measures.append("增加农业规则约束验证")
                elif "资源约束" in factor:
                    mitigation_measures.append("进行模型轻量化处理")
                    mitigation_measures.append("优化边缘计算资源分配")
            
            if recommended_strategy == "fine_tuning":
                mitigation_measures.append("采用渐进式微调策略")
            elif recommended_strategy == "domain_adaptation":
                mitigation_measures.append("使用领域对抗训练")
            
            return {
                "risk_level": risk_level,
                "risk_score": total_risk_score,
                "risk_factors": risk_factors,
                "recommended_strategy": recommended_strategy,
                "confidence_score": confidence,
                "mitigation_measures": mitigation_measures
            }
            
        except Exception as e:
            self.logger.error(f"迁移风险评估失败: {e}")
            # 返回保守的风险评估结果
            return {
                "risk_level": "critical",
                "risk_score": 1.0,
                "risk_factors": ["评估过程出现异常"],
                "recommended_strategy": "reject",
                "confidence_score": 0.0,
                "mitigation_measures": ["检查输入数据完整性"]
            }
        try:
            # 1. 计算数据相似度风险
            data_similarity_risk = self._calculate_data_similarity_risk(
                source_domain, target_domain)
            
            # 2. 计算模型兼容性风险
            model_compatibility_risk = self._calculate_model_compatibility_risk(
                source_domain, target_domain, model_info)
            
            # 3. 计算领域专家知识风险
            domain_expertise_risk = self._calculate_domain_expertise_risk(
                source_domain, target_domain)
            
            # 4. 计算资源约束风险
            resource_constraints_risk = self._calculate_resource_constraints_risk(
                model_info)
            
            # 5. 综合风险评分
            total_risk_score = self._calculate_total_risk_score(
                data_similarity_risk, model_compatibility_risk,
                domain_expertise_risk, resource_constraints_risk)
            
            # 6. 确定风险等级
            risk_level = self._determine_risk_level(total_risk_score)
            
            # 7. 识别风险因素
            risk_factors = self._identify_risk_factors(
                data_similarity_risk, model_compatibility_risk,
                domain_expertise_risk, resource_constraints_risk)
            
            # 8. 推荐迁移策略
            recommended_strategy = self._recommend_migration_strategy(
                total_risk_score, risk_factors)
            
            # 9. 计算置信度
            confidence_score = self._calculate_confidence_score(
                source_domain, target_domain)
            
            # 10. 制定缓解措施
            mitigation_measures = self._generate_mitigation_measures(
                risk_factors, recommended_strategy)
            
            return MigrationRiskAssessment(
                risk_level=risk_level,
                risk_score=total_risk_score,
                risk_factors=risk_factors,
                recommended_strategy=recommended_strategy,
                confidence_score=confidence_score,
                mitigation_measures=mitigation_measures
            )
            
        except Exception as e:
            self.logger.error(f"迁移风险评估失败: {e}")
            # 返回保守的风险评估结果
            return MigrationRiskAssessment(
                risk_level=RiskLevel.CRITICAL,
                risk_score=1.0,
                risk_factors=["评估过程出现异常"],
                recommended_strategy=MigrationStrategy.REJECT,
                confidence_score=0.0,
                mitigation_measures=["检查输入数据完整性"]
            )
    
    def _calculate_data_similarity_risk(self, 
                                      source_domain: Dict[str, Any],
                                      target_domain: Dict[str, Any]) -> float:
        """计算数据相似度风险"""
        # 提取特征分布信息
        source_features = source_domain.get("feature_distribution", {})
        target_features = target_domain.get("feature_distribution", {})
        
        # 计算特征分布相似度（简化实现）
        similarity_score = self._calculate_distribution_similarity(
            source_features, target_features)
        
        # 相似度越低，风险越高
        return 1.0 - similarity_score
    
    def _calculate_model_compatibility_risk(self,
                                           source_domain: Dict[str, Any],
                                           target_domain: Dict[str, Any],
                                           model_info: Dict[str, Any]) -> float:
        """计算模型兼容性风险"""
        # 检查模型架构兼容性
        architecture_compatibility = self._check_architecture_compatibility(
            source_domain, target_domain, model_info)
        
        # 检查输入输出维度匹配
        dimension_compatibility = self._check_dimension_compatibility(
            source_domain, target_domain)
        
        # 综合兼容性评分
        compatibility_score = (architecture_compatibility + dimension_compatibility) / 2
        
        return 1.0 - compatibility_score
    
    def _calculate_domain_expertise_risk(self,
                                        source_domain: Dict[str, Any],
                                        target_domain: Dict[str, Any]) -> float:
        """计算领域专家知识风险"""
        # 检查领域知识差距
        domain_gap = self._calculate_domain_gap(source_domain, target_domain)
        
        # 检查农业规则约束
        rule_constraints = self._check_agricultural_rules(source_domain, target_domain)
        
        # 综合领域风险
        domain_risk = (domain_gap + rule_constraints) / 2
        
        return domain_risk
    
    def _calculate_resource_constraints_risk(self, model_info: Dict[str, Any]) -> float:
        """计算资源约束风险"""
        # 检查模型大小和计算需求
        model_size_risk = self._check_model_size_risk(model_info)
        
        # 检查边缘设备兼容性
        edge_compatibility_risk = self._check_edge_compatibility_risk(model_info)
        
        # 检查内存和存储需求
        resource_demand_risk = self._check_resource_demand_risk(model_info)
        
        # 综合资源风险
        resource_risk = (model_size_risk + edge_compatibility_risk + resource_demand_risk) / 3
        
        return resource_risk
    
    def _calculate_total_risk_score(self, *risk_scores: float) -> float:
        """计算综合风险评分"""
        weights = [
            self.config["data_similarity_weight"],
            self.config["model_compatibility_weight"], 
            self.config["domain_expertise_weight"],
            self.config["resource_constraints_weight"]
        ]
        
        # 加权平均
        total_score = sum(score * weight for score, weight in zip(risk_scores, weights))
        
        return min(max(total_score, 0.0), 1.0)  # 限制在0-1范围内
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """根据风险评分确定风险等级"""
        if risk_score <= self.risk_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif risk_score <= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif risk_score <= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _identify_risk_factors(self, *risk_scores: float) -> List[str]:
        """识别主要风险因素"""
        risk_factors = []
        risk_names = ["数据相似度", "模型兼容性", "领域知识", "资源约束"]
        
        for i, score in enumerate(risk_scores):
            if score > 0.5:  # 高风险因素阈值
                risk_factors.append(f"{risk_names[i]}风险较高")
        
        return risk_factors
    
    def _recommend_migration_strategy(self, 
                                    risk_score: float,
                                    risk_factors: List[str]) -> MigrationStrategy:
        """推荐迁移策略"""
        if risk_score < 0.3:
            return MigrationStrategy.FULL_TRANSFER
        elif risk_score < 0.6:
            return MigrationStrategy.FINE_TUNING
        elif risk_score < 0.8:
            return MigrationStrategy.DOMAIN_ADAPTATION
        else:
            return MigrationStrategy.REJECT
    
    def _calculate_confidence_score(self,
                                  source_domain: Dict[str, Any],
                                  target_domain: Dict[str, Any]) -> float:
        """计算评估置信度"""
        # 基于数据质量和完整性计算置信度
        source_data_quality = source_domain.get("data_quality", 0.5)
        target_data_quality = target_domain.get("data_quality", 0.5)
        
        # 基于领域知识深度
        domain_knowledge_depth = source_domain.get("knowledge_depth", 0.5)
        
        # 综合置信度
        confidence = (source_data_quality + target_data_quality + domain_knowledge_depth) / 3
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_mitigation_measures(self,
                                    risk_factors: List[str],
                                    strategy: MigrationStrategy) -> List[str]:
        """生成风险缓解措施"""
        measures = []
        
        # 根据风险因素制定措施
        for factor in risk_factors:
            if "数据相似度" in factor:
                measures.append("增加目标领域数据采集")
                measures.append("进行数据增强和预处理")
            elif "模型兼容性" in factor:
                measures.append("调整模型架构适应目标领域")
                measures.append("使用适配器层进行特征映射")
            elif "领域知识" in factor:
                measures.append("引入领域专家知识")
                measures.append("增加农业规则约束验证")
            elif "资源约束" in factor:
                measures.append("进行模型轻量化处理")
                measures.append("优化边缘计算资源分配")
        
        # 根据策略添加特定措施
        if strategy == MigrationStrategy.FINE_TUNING:
            measures.append("采用渐进式微调策略")
        elif strategy == MigrationStrategy.DOMAIN_ADAPTATION:
            measures.append("使用领域对抗训练")
        
        return measures
    
    # 辅助计算方法（简化实现）
    def _calculate_distribution_similarity(self, dist1: Dict, dist2: Dict) -> float:
        """计算分布相似度"""
        # 简化实现：实际应用中应使用更复杂的分布相似度度量
        if not dist1 or not dist2:
            return 0.5
        
        common_keys = set(dist1.keys()) & set(dist2.keys())
        if not common_keys:
            return 0.0
        
        # 计算平均相似度
        similarities = []
        for key in common_keys:
            val1 = dist1[key]
            val2 = dist2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1e-6)
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.5
    
    def _check_architecture_compatibility(self, 
                                         source_domain: Dict[str, Any],
                                         target_domain: Dict[str, Any],
                                         model_info: Dict[str, Any]) -> float:
        """检查模型架构兼容性"""
        # 简化实现
        source_arch = source_domain.get("model_architecture", "")
        target_arch = target_domain.get("model_architecture", "")
        
        if source_arch and target_arch:
            return 1.0 if source_arch == target_arch else 0.3
        
        return 0.5
    
    def _check_dimension_compatibility(self,
                                     source_domain: Dict[str, Any],
                                     target_domain: Dict[str, Any]) -> float:
        """检查输入输出维度匹配"""
        # 简化实现
        source_dims = source_domain.get("input_dimensions", [])
        target_dims = target_domain.get("input_dimensions", [])
        
        if source_dims and target_dims:
            if len(source_dims) == len(target_dims):
                return 0.8
            else:
                return 0.2
        
        return 0.5
    
    def _calculate_domain_gap(self,
                            source_domain: Dict[str, Any],
                            target_domain: Dict[str, Any]) -> float:
        """计算领域知识差距"""
        # 简化实现
        source_knowledge = source_domain.get("domain_knowledge", {})
        target_knowledge = target_domain.get("domain_knowledge", {})
        
        if not source_knowledge or not target_knowledge:
            return 0.5
        
        # 计算知识重叠度
        overlap = len(set(source_knowledge.keys()) & set(target_knowledge.keys()))
        total = len(set(source_knowledge.keys()) | set(target_knowledge.keys()))
        
        return 1.0 - (overlap / total) if total > 0 else 0.5
    
    def _check_agricultural_rules(self,
                                 source_domain: Dict[str, Any],
                                 target_domain: Dict[str, Any]) -> float:
        """检查农业规则约束"""
        # 简化实现
        source_rules = source_domain.get("agricultural_rules", [])
        target_rules = target_domain.get("agricultural_rules", [])
        
        if not source_rules or not target_rules:
            return 0.3
        
        # 检查规则一致性
        rule_similarity = len(set(source_rules) & set(target_rules)) / len(set(source_rules) | set(target_rules))
        
        return 1.0 - rule_similarity
    
    def _check_model_size_risk(self, model_info: Dict[str, Any]) -> float:
        """检查模型大小风险"""
        model_size = model_info.get("model_size_mb", 0)
        
        if model_size > 500:  # 超过500MB为高风险
            return 1.0
        elif model_size > 100:  # 100-500MB为中风险
            return 0.6
        else:  # 小于100MB为低风险
            return 0.2
    
    def _check_edge_compatibility_risk(self, model_info: Dict[str, Any]) -> float:
        """检查边缘设备兼容性风险"""
        edge_compatibility = model_info.get("edge_compatible", False)
        
        return 0.0 if edge_compatibility else 0.8
    
    def _check_resource_demand_risk(self, model_info: Dict[str, Any]) -> float:
        """检查资源需求风险"""
        memory_required = model_info.get("memory_required_mb", 0)
        compute_required = model_info.get("compute_required", 0)
        
        memory_risk = min(memory_required / 2000, 1.0)  # 假设2GB内存限制
        compute_risk = min(compute_required / 100, 1.0)  # 假设100TFLOPS计算限制
        
        return (memory_risk + compute_risk) / 2
    
    async def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        更新风险控制配置
        
        Args:
            config_updates: 配置更新内容
        """
        try:
            # 更新配置参数
            for key, value in config_updates.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.info(f"风险控制配置参数 {key} 更新为 {value}")
                else:
                    self.logger.warning(f"未知的风险控制配置参数: {key}")
        except Exception as e:
            self.logger.error(f"更新风险控制配置失败: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        获取风险控制系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        return {
            "status": "active",
            "config": self.config.copy(),
            "risk_thresholds": {level.value: threshold for level, threshold in self.risk_thresholds.items()}
        }

    async def assess_edge_computing_risk(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估边缘计算风险
        
        Args:
            validated_data: 验证后的数据
            
        Returns:
            Dict[str, Any]: 边缘计算风险评估结果
        """
        try:
            # 从验证数据中提取所需信息
            model_info = validated_data.get("model_info", {})
            edge_device_info = validated_data.get("edge_device_info", {})
            
            # 使用默认值处理缺失信息
            if not model_info:
                model_info = {"model_size_mb": 50, "edge_compatible": True, "memory_required_mb": 1000, "compute_required": 50}
            if not edge_device_info:
                edge_device_info = {"available_memory_mb": 2000, "compute_capacity": 80, "power_usage_w": 20}
            
            # 计算边缘设备兼容性风险
            edge_compatibility_risk = self._check_edge_compatibility_risk(model_info)
            
            # 计算资源需求风险
            memory_risk = min(model_info.get("memory_required_mb", 0) / edge_device_info.get("available_memory_mb", 1000), 1.0)
            compute_risk = min(model_info.get("compute_required", 0) / edge_device_info.get("compute_capacity", 50), 1.0)
            
            # 计算功耗风险
            power_risk = min(edge_device_info.get("power_usage_w", 0) / 50, 1.0)  # 假设50W为安全阈值
            
            # 综合风险评分
            total_risk_score = (edge_compatibility_risk + memory_risk + compute_risk + power_risk) / 4
            total_risk_score = min(max(total_risk_score, 0.0), 1.0)
            
            # 确定风险等级
            if total_risk_score <= self.risk_thresholds[RiskLevel.LOW]:
                risk_level = "low"
            elif total_risk_score <= self.risk_thresholds[RiskLevel.MEDIUM]:
                risk_level = "medium"
            elif total_risk_score <= self.risk_thresholds[RiskLevel.HIGH]:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            # 风险因素识别
            risk_factors = []
            if edge_compatibility_risk > 0.5:
                risk_factors.append("边缘设备兼容性风险较高")
            if memory_risk > 0.5:
                risk_factors.append("内存需求超过边缘设备容量")
            if compute_risk > 0.5:
                risk_factors.append("计算需求超过边缘设备能力")
            if power_risk > 0.5:
                risk_factors.append("功耗过高")
            
            # 缓解措施
            mitigation_measures = []
            if "边缘设备兼容性风险较高" in risk_factors:
                mitigation_measures.append("使用边缘优化的模型版本")
                mitigation_measures.append("增加边缘设备资源配置")
            if "内存需求超过边缘设备容量" in risk_factors:
                mitigation_measures.append("进行模型轻量化处理")
                mitigation_measures.append("优化内存使用")
            if "计算需求超过边缘设备能力" in risk_factors:
                mitigation_measures.append("降低模型精度以减少计算需求")
                mitigation_measures.append("使用更强大的边缘设备")
            if "功耗过高" in risk_factors:
                mitigation_measures.append("优化模型推理功耗")
                mitigation_measures.append("使用低功耗边缘设备")
            
            return {
                "risk_level": risk_level,
                "risk_score": total_risk_score,
                "risk_factors": risk_factors,
                "mitigation_measures": mitigation_measures,
                "edge_device_compatibility": "compatible" if edge_compatibility_risk < 0.5 else "incompatible",
                "resource_utilization": {
                    "memory": memory_risk * 100,
                    "compute": compute_risk * 100,
                    "power": power_risk * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"边缘计算风险评估失败: {e}")
            return {
                "risk_level": "critical",
                "risk_score": 1.0,
                "risk_factors": ["评估过程出现异常"],
                "mitigation_measures": ["检查输入数据完整性"],
                "edge_device_compatibility": "unknown",
                "resource_utilization": {
                    "memory": 0,
                    "compute": 0,
                    "power": 0
                }
            }

    async def assess_decision_risk(self, validated_data: Dict[str, Any], decision_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        评估决策风险
        
        Args:
            validated_data: 验证后的数据
            decision_context: 决策上下文信息
            
        Returns:
            Dict[str, Any]: 决策风险评估结果
        """
        try:
            # 处理None的decision_context
            if decision_context is None:
                decision_context = {}
            
            # 从验证数据和决策上下文中提取信息
            migration_risk = decision_context.get("migration_risk", {})
            edge_computing_risk = decision_context.get("edge_computing_risk", {})
            decision_confidence = decision_context.get("decision_confidence", 0.5)
            
            # 使用默认值处理缺失信息
            migration_risk_score = migration_risk.get("risk_score", 0.5)
            edge_risk_score = edge_computing_risk.get("risk_score", 0.5)
            
            # 计算综合决策风险
            total_risk_score = (migration_risk_score + edge_risk_score + (1 - decision_confidence)) / 3
            total_risk_score = min(max(total_risk_score, 0.0), 1.0)
            
            # 确定风险等级
            if total_risk_score <= self.risk_thresholds[RiskLevel.LOW]:
                risk_level = "low"
            elif total_risk_score <= self.risk_thresholds[RiskLevel.MEDIUM]:
                risk_level = "medium"
            elif total_risk_score <= self.risk_thresholds[RiskLevel.HIGH]:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            # 风险因素识别
            risk_factors = []
            if migration_risk_score > 0.5:
                risk_factors.append("迁移风险较高")
            if edge_risk_score > 0.5:
                risk_factors.append("边缘计算风险较高")
            if decision_confidence < 0.5:
                risk_factors.append("决策置信度低")
            
            # 缓解措施
            mitigation_measures = []
            if "迁移风险较高" in risk_factors:
                mitigation_measures.append("重新评估迁移策略")
                mitigation_measures.append("增加数据验证步骤")
            if "边缘计算风险较高" in risk_factors:
                mitigation_measures.append("优化边缘计算配置")
                mitigation_measures.append("使用云端备份处理")
            if "决策置信度低" in risk_factors:
                mitigation_measures.append("增加决策验证步骤")
                mitigation_measures.append("使用人工干预机制")
            
            return {
                "risk_level": risk_level,
                "risk_score": total_risk_score,
                "risk_factors": risk_factors,
                "mitigation_measures": mitigation_measures,
                "decision_confidence": decision_confidence,
                "risk_contributions": {
                    "migration_risk": migration_risk_score,
                    "edge_computing_risk": edge_risk_score,
                    "confidence_risk": 1 - decision_confidence
                }
            }
            
        except Exception as e:
            self.logger.error(f"决策风险评估失败: {e}")
            return {
                "risk_level": "critical",
                "risk_score": 1.0,
                "risk_factors": ["评估过程出现异常"],
                "mitigation_measures": ["检查输入数据完整性"],
                "decision_confidence": 0.0,
                "risk_contributions": {
                    "migration_risk": 0.0,
                    "edge_computing_risk": 0.0,
                    "confidence_risk": 1.0
                }
            }

    async def post_process_decision(self, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理决策结果
        
        Args:
            decision_result: 决策结果
            
        Returns:
            Dict[str, Any]: 后处理后的决策结果
        """
        try:
            # 检查决策结果是否包含必要信息
            if "decision" not in decision_result:
                decision_result["decision"] = "no_decision"
                decision_result["error"] = "决策结果不完整"
                
            # 添加后处理时间戳
            from datetime import datetime
            decision_result["post_processed_at"] = datetime.now().isoformat()
            
            # 根据风险等级添加建议
            risk_level = decision_result.get("risk_assessment", {}).get("risk_level", "low")
            if risk_level == "high" or risk_level == "critical":
                decision_result["additional_recommendation"] = "建议人工审核此决策"
                decision_result["monitoring_priority"] = "high"
            else:
                decision_result["additional_recommendation"] = "决策可以自动执行"
                decision_result["monitoring_priority"] = "medium"
            
            return decision_result
            
        except Exception as e:
            self.logger.error(f"决策结果后处理失败: {e}")
            # 添加错误信息但不影响原始决策结果
            if "errors" not in decision_result:
                decision_result["errors"] = []
            decision_result["errors"].append(f"后处理失败: {e}")
            return decision_result

    async def monitor_risk_changes(self, decision_id: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        监控风险变化
        
        Args:
            decision_id: 决策ID
            risk_data: 风险数据
            
        Returns:
            Dict[str, Any]: 风险监控结果
        """
        try:
            # 添加datetime导入
            from datetime import datetime
            # 简化实现：实际应用中应包含风险变化检测逻辑
            current_risk_level = risk_data.get("risk_level", "low")
            
            # 生成监控报告
            monitoring_result = {
                "decision_id": decision_id,
                "current_risk_level": current_risk_level,
                "monitoring_active": True,
                "last_monitored_at": datetime.now().isoformat(),
                "risk_change_detection": "stable",  # 可以是 "stable", "increasing", "decreasing"
                "alert_triggered": current_risk_level == "high" or current_risk_level == "critical"
            }
            
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"风险监控失败: {e}")
            return {
                "decision_id": decision_id,
                "current_risk_level": "unknown",
                "monitoring_active": False,
                "last_monitored_at": datetime.now().isoformat(),
                "risk_change_detection": "error",
                "alert_triggered": False,
                "error": str(e)
            }