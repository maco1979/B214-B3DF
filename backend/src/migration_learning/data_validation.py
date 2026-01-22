"""
数据可信度验证模块

负责迁移学习过程中数据质量的可信度验证，包括：
- 数据质量评估
- 异常检测和处理
- 数据分布验证
- 可信度评分计算
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 较差
    UNACCEPTABLE = "unacceptable"  # 不可接受


class AnomalyType(Enum):
    """异常类型"""
    MISSING_VALUES = "missing_values"  # 缺失值
    OUTLIERS = "outliers"  # 异常值
    INCONSISTENT_FORMAT = "inconsistent_format"  # 格式不一致
    DATA_DRIFT = "data_drift"  # 数据漂移
    CORRUPTED_DATA = "corrupted_data"  # 数据损坏


@dataclass
class DataQualityReport:
    """数据质量报告"""
    quality_level: DataQualityLevel
    overall_score: float  # 总体评分（0-1）
    anomaly_details: Dict[AnomalyType, float]  # 各类异常详情
    validation_passed: bool  # 是否通过验证
    recommendations: List[str]  # 改进建议


@dataclass
class DataCredibilityScore:
    """数据可信度评分"""
    credibility_score: float  # 可信度评分（0-1）
    confidence_level: float  # 置信度水平
    risk_factors: List[str]  # 风险因素
    validation_status: str  # 验证状态


class DataCredibilityValidator:
    """数据可信度验证器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 质量阈值配置
        self.quality_thresholds = {
            DataQualityLevel.EXCELLENT: 0.9,
            DataQualityLevel.GOOD: 0.8,
            DataQualityLevel.FAIR: 0.6,
            DataQualityLevel.POOR: 0.4,
            DataQualityLevel.UNACCEPTABLE: 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "missing_value_threshold": 0.05,  # 缺失值阈值
            "outlier_detection_method": "iqr",  # 异常值检测方法
            "drift_detection_threshold": 0.1,  # 漂移检测阈值
            "min_sample_size": 100,  # 最小样本量
            "max_missing_ratio": 0.1,  # 最大缺失比例
            "credibility_weights": {
                "completeness": 0.3,
                "consistency": 0.25,
                "accuracy": 0.25,
                "timeliness": 0.2
            }
        }
    
    def validate_data_quality(self, 
                            data: Dict[str, Any], 
                            metadata: Optional[Dict[str, Any]] = None) -> DataQualityReport:
        """
        验证数据质量
        
        Args:
            data: 待验证的数据
            metadata: 数据元数据
            
        Returns:
            DataQualityReport: 数据质量报告
        """
        try:
            # 1. 检查数据完整性
            completeness_score = self._check_completeness(data, metadata)
            
            # 2. 检查数据一致性
            consistency_score = self._check_consistency(data, metadata)
            
            # 3. 检查数据准确性
            accuracy_score = self._check_accuracy(data, metadata)
            
            # 4. 检查数据时效性
            timeliness_score = self._check_timeliness(data, metadata)
            
            # 5. 检测异常
            anomaly_details = self._detect_anomalies(data, metadata)
            
            # 6. 计算总体质量评分
            overall_score = self._calculate_overall_quality_score(
                completeness_score, consistency_score, accuracy_score, timeliness_score)
            
            # 7. 确定质量等级
            quality_level = self._determine_quality_level(overall_score)
            
            # 8. 判断是否通过验证
            validation_passed = self._check_validation_pass(overall_score, anomaly_details)
            
            # 9. 生成改进建议
            recommendations = self._generate_recommendations(
                overall_score, anomaly_details, quality_level)
            
            return DataQualityReport(
                quality_level=quality_level,
                overall_score=overall_score,
                anomaly_details=anomaly_details,
                validation_passed=validation_passed,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"数据质量验证失败: {e}")
            return DataQualityReport(
                quality_level=DataQualityLevel.UNACCEPTABLE,
                overall_score=0.0,
                anomaly_details={},
                validation_passed=False,
                recommendations=["数据验证过程出现异常，请检查数据格式和完整性"]
            )
    


    def assess_data_credibility(self,
                              source_data: Dict[str, Any],
                              target_data: Dict[str, Any],
                              domain_context: Optional[Dict[str, Any]] = None) -> DataCredibilityScore:
        """
        评估数据可信度
        
        Args:
            source_data: 源领域数据
            target_data: 目标领域数据
            domain_context: 领域上下文信息
            
        Returns:
            DataCredibilityScore: 数据可信度评分
        """
        try:
            # 1. 验证源数据质量
            source_quality = self.validate_data_quality(source_data, domain_context)
            
            # 2. 验证目标数据质量
            target_quality = self.validate_data_quality(target_data, domain_context)
            
            # 3. 检查数据分布相似性
            distribution_similarity = self._check_distribution_similarity(
                source_data, target_data)
            
            # 4. 检查领域适配性
            domain_adaptability = self._check_domain_adaptability(
                source_data, target_data, domain_context)
            
            # 5. 计算可信度评分
            credibility_score = self._calculate_credibility_score(
                source_quality.overall_score,
                target_quality.overall_score,
                distribution_similarity,
                domain_adaptability
            )
            
            # 6. 计算置信度水平
            confidence_level = self._calculate_confidence_level(
                source_quality, target_quality)
            
            # 7. 识别风险因素
            risk_factors = self._identify_credibility_risk_factors(
                source_quality, target_quality, distribution_similarity)
            
            # 8. 确定验证状态
            validation_status = self._determine_validation_status(
                credibility_score, risk_factors)
            
            return DataCredibilityScore(
                credibility_score=credibility_score,
                confidence_level=confidence_level,
                risk_factors=risk_factors,
                validation_status=validation_status
            )
            
        except Exception as e:
            self.logger.error(f"数据可信度评估失败: {e}")
            return DataCredibilityScore(
                credibility_score=0.0,
                confidence_level=0.0,
                risk_factors=["评估过程出现异常"],
                validation_status="failed"
            )
    
    def _check_completeness(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据完整性"""
        # 检查缺失值
        missing_ratio = self._calculate_missing_ratio(data)
        
        # 检查样本量充足性
        sample_size_adequacy = self._check_sample_size_adequacy(data)
        
        # 检查特征完整性
        feature_completeness = self._check_feature_completeness(data, metadata)
        
        # 综合完整性评分
        completeness_score = (1.0 - missing_ratio) * 0.5 + sample_size_adequacy * 0.3 + feature_completeness * 0.2
        
        return min(max(completeness_score, 0.0), 1.0)
    
    def _check_consistency(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据一致性"""
        # 检查格式一致性
        format_consistency = self._check_format_consistency(data)
        
        # 检查值域一致性
        value_range_consistency = self._check_value_range_consistency(data, metadata)
        
        # 检查时间序列一致性
        temporal_consistency = self._check_temporal_consistency(data)
        
        # 综合一致性评分
        consistency_score = (format_consistency + value_range_consistency + temporal_consistency) / 3
        
        return min(max(consistency_score, 0.0), 1.0)
    
    def _check_accuracy(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据准确性"""
        # 检查异常值
        outlier_ratio = self._calculate_outlier_ratio(data)
        
        # 检查数据合理性（基于领域知识）
        domain_reasonableness = self._check_domain_reasonableness(data, metadata)
        
        # 检查数据关联性
        data_correlation = self._check_data_correlation(data)
        
        # 综合准确性评分
        accuracy_score = (1.0 - outlier_ratio) * 0.4 + domain_reasonableness * 0.4 + data_correlation * 0.2
        
        return min(max(accuracy_score, 0.0), 1.0)
    
    def _check_timeliness(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据时效性"""
        # 检查数据新鲜度
        data_freshness = self._check_data_freshness(data, metadata)
        
        # 检查更新频率
        update_frequency = self._check_update_frequency(data, metadata)
        
        # 检查时效性需求匹配度
        timeliness_requirement_match = self._check_timeliness_requirement_match(data, metadata)
        
        # 综合时效性评分
        timeliness_score = (data_freshness + update_frequency + timeliness_requirement_match) / 3
        
        return min(max(timeliness_score, 0.0), 1.0)
    
    def _detect_anomalies(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[AnomalyType, float]:
        """检测数据异常"""
        anomalies = {}
        
        # 检测缺失值异常
        missing_ratio = self._calculate_missing_ratio(data)
        anomalies[AnomalyType.MISSING_VALUES] = missing_ratio
        
        # 检测异常值
        outlier_ratio = self._calculate_outlier_ratio(data)
        anomalies[AnomalyType.OUTLIERS] = outlier_ratio
        
        # 检测格式不一致异常
        format_inconsistency = 1.0 - self._check_format_consistency(data)
        anomalies[AnomalyType.INCONSISTENT_FORMAT] = format_inconsistency
        
        # 检测数据漂移（简化实现）
        data_drift = self._detect_data_drift(data, metadata)
        anomalies[AnomalyType.DATA_DRIFT] = data_drift
        
        # 检测数据损坏
        corruption_ratio = self._check_data_corruption(data)
        anomalies[AnomalyType.CORRUPTED_DATA] = corruption_ratio
        
        return anomalies
    
    def _calculate_overall_quality_score(self, *scores: float) -> float:
        """计算总体质量评分"""
        weights = [
            self.config["credibility_weights"]["completeness"],
            self.config["credibility_weights"]["consistency"],
            self.config["credibility_weights"]["accuracy"],
            self.config["credibility_weights"]["timeliness"]
        ]
        
        # 加权平均
        overall_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _determine_quality_level(self, overall_score: float) -> DataQualityLevel:
        """确定数据质量等级"""
        if overall_score >= self.quality_thresholds[DataQualityLevel.EXCELLENT]:
            return DataQualityLevel.EXCELLENT
        elif overall_score >= self.quality_thresholds[DataQualityLevel.GOOD]:
            return DataQualityLevel.GOOD
        elif overall_score >= self.quality_thresholds[DataQualityLevel.FAIR]:
            return DataQualityLevel.FAIR
        elif overall_score >= self.quality_thresholds[DataQualityLevel.POOR]:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNACCEPTABLE
    
    def _check_validation_pass(self, overall_score: float, anomalies: Dict[AnomalyType, float]) -> bool:
        """检查是否通过验证"""
        # 总体评分必须达到最低要求
        if overall_score < self.config.get("min_quality_threshold", 0.6):
            return False
        
        # 关键异常不能超过阈值
        critical_anomalies = [
            anomalies.get(AnomalyType.CORRUPTED_DATA, 0.0),
            anomalies.get(AnomalyType.MISSING_VALUES, 0.0)
        ]
        
        if any(anomaly > 0.2 for anomaly in critical_anomalies):
            return False
        
        return True
    
    def _generate_recommendations(self, 
                                overall_score: float,
                                anomalies: Dict[AnomalyType, float],
                                quality_level: DataQualityLevel) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 根据质量等级提供建议
        if quality_level in [DataQualityLevel.POOR, DataQualityLevel.UNACCEPTABLE]:
            recommendations.append("数据质量较差，建议重新采集或预处理数据")
        
        # 根据具体异常提供针对性建议
        for anomaly_type, ratio in anomalies.items():
            if ratio > 0.1:  # 异常比例超过10%时提供建议
                if anomaly_type == AnomalyType.MISSING_VALUES:
                    recommendations.append(f"缺失值比例较高({ratio:.1%})，建议进行缺失值处理")
                elif anomaly_type == AnomalyType.OUTLIERS:
                    recommendations.append(f"异常值比例较高({ratio:.1%})，建议进行异常值检测和处理")
                elif anomaly_type == AnomalyType.INCONSISTENT_FORMAT:
                    recommendations.append("数据格式不一致，建议统一数据格式标准")
                elif anomaly_type == AnomalyType.DATA_DRIFT:
                    recommendations.append("检测到数据漂移，建议更新模型或重新训练")
                elif anomaly_type == AnomalyType.CORRUPTED_DATA:
                    recommendations.append("检测到数据损坏，建议检查数据源和传输过程")
        
        # 通用建议
        if overall_score < 0.8:
            recommendations.append("建议增加数据质量监控和定期评估机制")
        
        return recommendations
    
    # 辅助计算方法（简化实现）
    def _calculate_missing_ratio(self, data: Dict[str, Any]) -> float:
        """计算缺失值比例"""
        # 简化实现
        if "values" in data and isinstance(data["values"], (list, np.ndarray)):
            values = np.array(data["values"])
            missing_count = np.sum(np.isnan(values))
            return missing_count / len(values)
        return 0.0
    
    def _check_sample_size_adequacy(self, data: Dict[str, Any]) -> float:
        """检查样本量充足性"""
        min_size = self.config.get("min_sample_size", 100)
        
        if "sample_size" in data:
            sample_size = data["sample_size"]
            if sample_size >= min_size:
                return 1.0
            else:
                return sample_size / min_size
        
        return 0.5
    
    def _check_feature_completeness(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查特征完整性"""
        # 简化实现
        if metadata and "required_features" in metadata:
            required_features = set(metadata["required_features"])
            actual_features = set(data.get("features", []))
            
            if required_features:
                completeness = len(required_features & actual_features) / len(required_features)
                return completeness
        
        return 0.8
    
    def _check_format_consistency(self, data: Dict[str, Any]) -> float:
        """检查格式一致性"""
        # 简化实现
        return 0.9  # 假设大部分情况下格式一致
    
    def _check_value_range_consistency(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查值域一致性"""
        # 简化实现
        return 0.85
    
    def _check_temporal_consistency(self, data: Dict[str, Any]) -> float:
        """检查时间序列一致性"""
        # 简化实现
        return 0.9
    
    def _check_domain_reasonableness(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据合理性（基于领域知识）"""
        # 简化实现
        return 0.9
    
    def _check_data_correlation(self, data: Dict[str, Any]) -> float:
        """检查数据关联性"""
        # 简化实现
        return 0.85
    
    def _detect_data_drift(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检测数据漂移"""
        # 简化实现
        return 0.05
    
    def _calculate_outlier_ratio(self, data: Dict[str, Any]) -> float:
        """计算异常值比例"""
        # 简化实现
        if "values" in data and isinstance(data["values"], (list, np.ndarray)):
            values = np.array(data["values"])
            if len(values) > 0:
                # 使用IQR方法检测异常值
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = np.sum((values < lower_bound) | (values > upper_bound))
                return outliers / len(values)
        
        return 0.0
    
    # 其他辅助方法...
    def _check_distribution_similarity(self, source_data: Dict[str, Any], target_data: Dict[str, Any]) -> float:
        """检查数据分布相似性"""
        # 简化实现
        return 0.8
    
    def _check_domain_adaptability(self, source_data: Dict[str, Any], target_data: Dict[str, Any], 
                                 domain_context: Optional[Dict[str, Any]]) -> float:
        """检查领域适配性"""
        # 简化实现
        return 0.7
    
    def _calculate_credibility_score(self, source_quality: float, target_quality: float,
                                   distribution_similarity: float, domain_adaptability: float) -> float:
        """计算可信度评分"""
        weights = [0.3, 0.3, 0.25, 0.15]  # 源质量、目标质量、分布相似性、领域适配性
        scores = [source_quality, target_quality, distribution_similarity, domain_adaptability]
        
        credibility_score = sum(score * weight for score, weight in zip(scores, weights))
        return min(max(credibility_score, 0.0), 1.0)
    
    def _calculate_confidence_level(self, source_quality: DataQualityReport, 
                                  target_quality: DataQualityReport) -> float:
        """计算置信度水平"""
        # 基于质量报告的详细程度和验证状态
        confidence = (source_quality.overall_score + target_quality.overall_score) / 2
        
        # 如果两者都通过验证，提高置信度
        if source_quality.validation_passed and target_quality.validation_passed:
            confidence *= 1.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _identify_credibility_risk_factors(self, source_quality: DataQualityReport,
                                        target_quality: DataQualityReport,
                                        distribution_similarity: float) -> List[str]:
        """识别可信度风险因素"""
        risk_factors = []
        
        if source_quality.overall_score < 0.7:
            risk_factors.append("源数据质量较低")
        
        if target_quality.overall_score < 0.7:
            risk_factors.append("目标数据质量较低")
        
        if distribution_similarity < 0.6:
            risk_factors.append("数据分布差异较大")
        
        if not source_quality.validation_passed:
            risk_factors.append("源数据验证未通过")
        
        if not target_quality.validation_passed:
            risk_factors.append("目标数据验证未通过")
        
        return risk_factors
    
    def _determine_validation_status(self, credibility_score: float, risk_factors: List[str]) -> str:
        """确定验证状态"""
        if credibility_score >= 0.8 and not risk_factors:
            return "excellent"
        elif credibility_score >= 0.7:
            return "good"
        elif credibility_score >= 0.6:
            return "fair"
        else:
            return "poor"
    
    async def validate_decision_data(self, decision_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证决策数据
        
        Args:
            decision_request: 决策请求数据
            
        Returns:
            Dict[str, Any]: 包含验证结果的字典
                - valid: bool - 是否通过验证
                - errors: List[str] - 错误信息列表
                - quality_report: DataQualityReport - 数据质量报告
        """
        try:
            # 确保请求包含必要字段 - task_type 是必须的
            errors = []
            
            if "task_type" not in decision_request:
                errors.append("缺少必要字段: task_type")
            elif not decision_request["task_type"]:
                errors.append("字段 task_type 的值不能为空")
            
            if errors:
                return {
                    "valid": False,
                    "errors": errors,
                    "quality_report": None
                }
            
            # 验证数据质量 - 如果有 data 字段使用它，否则使用整个请求作为数据
            data = decision_request.get("data", decision_request)
            quality_report = self.validate_data_quality(data)
            
            if quality_report.validation_passed:
                return {
                    "valid": True,
                    "errors": [],
                    "quality_report": quality_report
                }
            else:
                # 转换质量报告中的问题为错误信息
                error_messages = quality_report.recommendations
                if quality_report.overall_score < 0.4:
                    error_messages.insert(0, "数据质量严重不合格，无法用于决策")
                
                return {
                    "valid": False,
                    "errors": error_messages,
                    "quality_report": quality_report
                }
                
        except Exception as e:
            self.logger.error(f"决策数据验证失败: {e}")
            return {
                "valid": False,
                "errors": [f"数据验证过程异常: {str(e)}"],
                "quality_report": None
            }
    
    async def preprocess_data(self, decision_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理决策数据
        
        Args:
            decision_request: 决策请求数据
            
        Returns:
            Dict[str, Any]: 预处理后的数据
        """
        try:
            # 复制原始请求
            processed_data = decision_request.copy()
            
            # 这里可以添加数据预处理逻辑
            # 例如：数据标准化、特征工程等
            
            return processed_data
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            # 如果预处理失败，返回原始数据
            return decision_request
    
    def _check_data_freshness(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据新鲜度"""
        # 简化实现
        return 0.9
    
    def _check_update_frequency(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查数据更新频率"""
        # 简化实现
        return 0.8
    
    def _check_timeliness_requirement_match(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> float:
        """检查时效性需求匹配度"""
        # 简化实现
        return 0.85
    
    def _check_data_corruption(self, data: Dict[str, Any]) -> float:
        """检查数据损坏"""
        # 简化实现
        return 0.0