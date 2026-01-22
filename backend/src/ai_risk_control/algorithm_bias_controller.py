"""
算法偏见与系统性风险控制模块

负责识别和缓解AI决策中的算法偏见，防止历史数据偏差导致的不平等固化，
监控系统性风险，确保区块链经济系统的公平性和稳定性。
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
from scipy import stats
import pandas as pd


class BiasRiskType(Enum):
    """算法偏见风险类型"""
    DATA_BIAS = "data_bias"  # 数据偏见
    MODEL_BIAS = "model_bias"  # 模型偏见
    FAIRNESS_VIOLATION = "fairness_violation"  # 公平性违反
    SYSTEMIC_RISK = "systemic_risk"  # 系统性风险
    DECISION_CONVERGENCE = "decision_convergence"  # 决策趋同


class BiasSeverity(Enum):
    """偏见风险严重程度"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasRiskAlert:
    """算法偏见风险警报"""
    risk_type: BiasRiskType
    severity: BiasSeverity
    alert_id: str
    description: str
    bias_metrics: Dict[str, Any]
    confidence_score: float
    mitigation_action: str
    timestamp: datetime


@dataclass
class BiasRiskAssessment:
    """算法偏见风险评估结果"""
    overall_bias_level: BiasSeverity
    fairness_score: float  # 0-1之间的公平性评分
    active_alerts: List[BiasRiskAlert]
    systemic_risk_indicator: float
    decision_diversity_score: float
    compliance_status: bool
    recommendations: List[str]


class AlgorithmBiasController:
    """算法偏见与系统性风险控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 偏见监控状态
        self.bias_metrics_history = []
        self.alert_history = []
        self.fairness_baseline = {}
        
        # 偏见检测组件
        self.bias_detector = BiasDetector()
        
        # 公平性评估器
        self.fairness_evaluator = FairnessEvaluator()
        
        # 系统性风险分析器
        self.systemic_risk_analyzer = SystemicRiskAnalyzer()
        
        # 决策多样性监控器
        self.diversity_monitor = DecisionDiversityMonitor()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data_bias_threshold": 0.3,  # 数据偏见阈值
            "model_bias_threshold": 0.25,  # 模型偏见阈值
            "fairness_violation_threshold": 0.2,  # 公平性违反阈值
            "systemic_risk_threshold": 0.6,  # 系统性风险阈值
            "decision_convergence_threshold": 0.7,  # 决策趋同阈值
            "monitoring_window_days": 30,  # 监控时间窗口
            "baseline_calibration_period": 90,  # 基线校准周期
            "diversity_target": 0.8  # 决策多样性目标
        }
    
    async def assess_bias_risk(self,
                             training_data: Dict[str, Any],
                             model_decisions: List[Dict[str, Any]],
                             blockchain_context: Dict[str, Any],
                             historical_data: Dict[str, Any]) -> BiasRiskAssessment:
        """
        评估算法偏见与系统性风险
        
        Args:
            training_data: 训练数据信息
            model_decisions: 模型决策历史
            blockchain_context: 区块链上下文
            historical_data: 历史数据
            
        Returns:
            BiasRiskAssessment: 偏见风险评估结果
        """
        try:
            alerts = []
            
            # 1. 数据偏见风险评估
            data_bias_risk = await self._assess_data_bias_risk(training_data, historical_data)
            alerts.extend(data_bias_risk)
            
            # 2. 模型偏见风险评估
            model_bias_risk = await self._assess_model_bias_risk(model_decisions, blockchain_context)
            alerts.extend(model_bias_risk)
            
            # 3. 公平性违反风险评估
            fairness_violation_risk = await self._assess_fairness_violation_risk(
                model_decisions, blockchain_context)
            alerts.extend(fairness_violation_risk)
            
            # 4. 系统性风险评估
            systemic_risk = await self._assess_systemic_risk(model_decisions, blockchain_context)
            alerts.extend(systemic_risk)
            
            # 5. 决策趋同风险评估
            decision_convergence_risk = await self._assess_decision_convergence_risk(model_decisions)
            alerts.extend(decision_convergence_risk)
            
            # 6. 综合风险评估
            overall_bias_level = self._determine_overall_bias_level(alerts)
            fairness_score = self._calculate_fairness_score(alerts)
            
            # 7. 计算系统性风险指标
            systemic_risk_indicator = await self._calculate_systemic_risk_indicator(model_decisions)
            
            # 8. 计算决策多样性得分
            decision_diversity_score = await self._calculate_decision_diversity(model_decisions)
            
            # 9. 生成改进建议
            recommendations = self._generate_recommendations(alerts, overall_bias_level)
            
            # 10. 更新偏见指标
            self._update_bias_metrics(alerts, overall_bias_level)
            
            return BiasRiskAssessment(
                overall_bias_level=overall_bias_level,
                fairness_score=fairness_score,
                active_alerts=alerts,
                systemic_risk_indicator=systemic_risk_indicator,
                decision_diversity_score=decision_diversity_score,
                compliance_status=overall_bias_level in [BiasSeverity.LOW, BiasSeverity.MEDIUM],
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"算法偏见评估失败: {e}")
            # 返回最严格的风险评估结果
            return BiasRiskAssessment(
                overall_bias_level=BiasSeverity.CRITICAL,
                fairness_score=0.0,
                active_alerts=[BiasRiskAlert(
                    risk_type=BiasRiskType.DATA_BIAS,
                    severity=BiasSeverity.CRITICAL,
                    alert_id="ASSESSMENT_ERROR",
                    description="算法偏见评估过程出现异常",
                    bias_metrics={"error": str(e)},
                    confidence_score=1.0,
                    mitigation_action="立即停止AI决策并检查系统",
                    timestamp=datetime.utcnow()
                )],
                systemic_risk_indicator=1.0,
                decision_diversity_score=0.0,
                compliance_status=False,
                recommendations=["算法偏见评估失败，建议立即人工干预"]
            )
    
    async def _assess_data_bias_risk(self,
                                   training_data: Dict[str, Any],
                                   historical_data: Dict[str, Any]) -> List[BiasRiskAlert]:
        """评估数据偏见风险"""
        alerts = []
        
        # 分析训练数据中的偏见
        data_bias_score = await self.bias_detector.detect_data_bias(training_data, historical_data)
        
        if data_bias_score > self.config["data_bias_threshold"]:
            alerts.append(BiasRiskAlert(
                risk_type=BiasRiskType.DATA_BIAS,
                severity=BiasSeverity.HIGH,
                alert_id="DATA_BIAS_001",
                description="检测到训练数据存在显著偏见",
                bias_metrics={
                    "data_bias_score": data_bias_score,
                    "data_distribution": training_data.get("distribution_metrics", {})
                },
                confidence_score=data_bias_score,
                mitigation_action="重新采样或数据增强以平衡数据分布",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_model_bias_risk(self,
                                    model_decisions: List[Dict[str, Any]],
                                    blockchain_context: Dict[str, Any]) -> List[BiasRiskAlert]:
        """评估模型偏见风险"""
        alerts = []
        
        # 分析模型决策中的偏见
        model_bias_score = await self.bias_detector.detect_model_bias(model_decisions, blockchain_context)
        
        if model_bias_score > self.config["model_bias_threshold"]:
            alerts.append(BiasRiskAlert(
                risk_type=BiasRiskType.MODEL_BIAS,
                severity=BiasSeverity.MEDIUM,
                alert_id="MODEL_BIAS_001",
                description="检测到模型决策存在偏见",
                bias_metrics={
                    "model_bias_score": model_bias_score,
                    "decision_patterns": self._analyze_decision_patterns(model_decisions)
                },
                confidence_score=model_bias_score,
                mitigation_action="应用偏见缓解技术或重新训练模型",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_fairness_violation_risk(self,
                                            model_decisions: List[Dict[str, Any]],
                                            blockchain_context: Dict[str, Any]) -> List[BiasRiskAlert]:
        """评估公平性违反风险"""
        alerts = []
        
        # 评估决策公平性
        fairness_violation_score = await self.fairness_evaluator.evaluate_fairness(
            model_decisions, blockchain_context)
        
        if fairness_violation_score > self.config["fairness_violation_threshold"]:
            alerts.append(BiasRiskAlert(
                risk_type=BiasRiskType.FAIRNESS_VIOLATION,
                severity=BiasSeverity.CRITICAL,
                alert_id="FAIRNESS_VIOLATION_001",
                description="检测到AI决策违反公平性原则",
                bias_metrics={
                    "fairness_violation_score": fairness_violation_score,
                    "protected_groups": blockchain_context.get("protected_groups", [])
                },
                confidence_score=fairness_violation_score,
                mitigation_action="立即修正公平性算法并重新评估",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_systemic_risk(self,
                                  model_decisions: List[Dict[str, Any]],
                                  blockchain_context: Dict[str, Any]) -> List[BiasRiskAlert]:
        """评估系统性风险"""
        alerts = []
        
        # 分析系统性风险
        systemic_risk_score = await self.systemic_risk_analyzer.analyze_systemic_risk(
            model_decisions, blockchain_context)
        
        if systemic_risk_score > self.config["systemic_risk_threshold"]:
            alerts.append(BiasRiskAlert(
                risk_type=BiasRiskType.SYSTEMIC_RISK,
                severity=BiasSeverity.HIGH,
                alert_id="SYSTEMIC_RISK_001",
                description="检测到系统性风险积累",
                bias_metrics={
                    "systemic_risk_score": systemic_risk_score,
                    "risk_factors": self._identify_systemic_risk_factors(model_decisions)
                },
                confidence_score=systemic_risk_score,
                mitigation_action="分散决策风险并建立熔断机制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_decision_convergence_risk(self, model_decisions: List[Dict[str, Any]]) -> List[BiasRiskAlert]:
        """评估决策趋同风险"""
        alerts = []
        
        # 分析决策多样性
        convergence_score = await self.diversity_monitor.assess_convergence_risk(model_decisions)
        
        if convergence_score > self.config["decision_convergence_threshold"]:
            alerts.append(BiasRiskAlert(
                risk_type=BiasRiskType.DECISION_CONVERGENCE,
                severity=BiasSeverity.MEDIUM,
                alert_id="DECISION_CONVERGENCE_001",
                description="检测到决策趋同风险",
                bias_metrics={
                    "convergence_score": convergence_score,
                    "decision_diversity": await self._calculate_decision_diversity(model_decisions)
                },
                confidence_score=convergence_score,
                mitigation_action="引入决策多样性机制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _determine_overall_bias_level(self, alerts: List[BiasRiskAlert]) -> BiasSeverity:
        """确定总体偏见等级"""
        if not alerts:
            return BiasSeverity.LOW
        
        severities = [alert.severity for alert in alerts]
        
        if BiasSeverity.CRITICAL in severities:
            return BiasSeverity.CRITICAL
        elif BiasSeverity.HIGH in severities:
            return BiasSeverity.HIGH
        elif BiasSeverity.MEDIUM in severities:
            return BiasSeverity.MEDIUM
        else:
            return BiasSeverity.LOW
    
    def _calculate_fairness_score(self, alerts: List[BiasRiskAlert]) -> float:
        """计算公平性评分"""
        if not alerts:
            return 1.0
        
        severity_weights = {
            BiasSeverity.CRITICAL: 0.0,
            BiasSeverity.HIGH: 0.3,
            BiasSeverity.MEDIUM: 0.6,
            BiasSeverity.LOW: 0.9
        }
        
        total_score = sum(
            severity_weights[alert.severity] * (1.0 - alert.confidence_score)
            for alert in alerts
        )
        
        return total_score / len(alerts) if alerts else 1.0
    
    async def _calculate_systemic_risk_indicator(self, model_decisions: List[Dict[str, Any]]) -> float:
        """计算系统性风险指标"""
        if not model_decisions:
            return 0.0
        
        # 分析决策相关性
        decision_correlations = await self._analyze_decision_correlations(model_decisions)
        
        # 分析风险集中度
        risk_concentration = await self._analyze_risk_concentration(model_decisions)
        
        # 综合系统性风险指标
        systemic_risk = (decision_correlations + risk_concentration) / 2.0
        
        return systemic_risk
    
    async def _calculate_decision_diversity(self, model_decisions: List[Dict[str, Any]]) -> float:
        """计算决策多样性得分"""
        if len(model_decisions) < 2:
            return 1.0  # 单一决策视为完全多样
        
        # 分析决策策略的多样性
        decision_strategies = [decision.get("strategy", "default") for decision in model_decisions]
        unique_strategies = len(set(decision_strategies))
        
        diversity_score = unique_strategies / len(decision_strategies)
        
        return diversity_score
    
    def _generate_recommendations(self, 
                                alerts: List[BiasRiskAlert],
                                overall_bias_level: BiasSeverity) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_bias_level == BiasSeverity.LOW:
            recommendations.append("算法偏见控制良好，继续保持监控")
            return recommendations
        
        # 根据风险类型提供针对性建议
        risk_types = set(alert.risk_type for alert in alerts)
        
        if BiasRiskType.DATA_BIAS in risk_types:
            recommendations.append("建议重新采样或应用数据平衡技术")
        
        if BiasRiskType.MODEL_BIAS in risk_types:
            recommendations.append("建议应用偏见缓解算法或重新训练模型")
        
        if BiasRiskType.FAIRNESS_VIOLATION in risk_types:
            recommendations.append("必须立即修正公平性算法")
        
        if BiasRiskType.SYSTEMIC_RISK in risk_types:
            recommendations.append("建议分散决策风险并建立熔断机制")
        
        if BiasRiskType.DECISION_CONVERGENCE in risk_types:
            recommendations.append("建议引入决策多样性机制")
        
        # 紧急情况建议
        if overall_bias_level in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]:
            recommendations.insert(0, "⚖️ 高偏见风险：建议立即启动公平性修正机制")
        
        return recommendations
    
    def _update_bias_metrics(self, alerts: List[BiasRiskAlert], bias_level: BiasSeverity):
        """更新偏见指标"""
        current_time = datetime.utcnow()
        
        # 记录警报历史
        self.alert_history.extend(alerts)
        
        # 清理过时警报
        cutoff_time = current_time - timedelta(days=self.config["monitoring_window_days"])
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
        
        # 更新偏见指标历史
        self.bias_metrics_history.append({
            "timestamp": current_time,
            "bias_level": bias_level,
            "alert_count": len(alerts),
            "fairness_score": self._calculate_fairness_score(alerts)
        })
    
    # 辅助方法
    def _analyze_decision_patterns(self, model_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析决策模式"""
        if not model_decisions:
            return {}
        
        # 提取决策特征
        decision_features = []
        for decision in model_decisions:
            features = decision.get("features", {})
            decision_features.append(features)
        
        # 分析模式统计
        return {
            "total_decisions": len(model_decisions),
            "feature_variability": self._calculate_feature_variability(decision_features),
            "decision_consistency": self._calculate_decision_consistency(model_decisions)
        }
    
    def _identify_systemic_risk_factors(self, model_decisions: List[Dict[str, Any]]) -> List[str]:
        """识别系统性风险因素"""
        risk_factors = []
        
        # 分析决策相关性
        if len(model_decisions) > 10:
            # 假设大量相似决策可能引发系统性风险
            risk_factors.append("决策高度相关")
        
        # 分析风险集中度
        risk_scores = [decision.get("risk_score", 0) for decision in model_decisions]
        if max(risk_scores) > 0.8:
            risk_factors.append("高风险决策集中")
        
        return risk_factors
    
    async def _analyze_decision_correlations(self, model_decisions: List[Dict[str, Any]]) -> float:
        """分析决策相关性"""
        if len(model_decisions) < 2:
            return 0.0
        
        # 提取决策向量
        decision_vectors = []
        for decision in model_decisions:
            vector = self._decision_to_vector(decision)
            decision_vectors.append(vector)
        
        # 计算平均相关性
        correlations = []
        for i in range(len(decision_vectors)):
            for j in range(i+1, len(decision_vectors)):
                corr = np.corrcoef(decision_vectors[i], decision_vectors[j])[0, 1]
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    async def _analyze_risk_concentration(self, model_decisions: List[Dict[str, Any]]) -> float:
        """分析风险集中度"""
        if not model_decisions:
            return 0.0
        
        # 提取风险分数
        risk_scores = [decision.get("risk_score", 0) for decision in model_decisions]
        
        # 计算基尼系数作为风险集中度指标
        if risk_scores:
            sorted_scores = sorted(risk_scores)
            n = len(sorted_scores)
            cumulative_scores = np.cumsum(sorted_scores)
            total_score = cumulative_scores[-1]
            
            if total_score > 0:
                # 计算基尼系数
                gini = (n + 1 - 2 * np.sum(cumulative_scores) / total_score) / n
                return gini
        
        return 0.0
    
    def _decision_to_vector(self, decision: Dict[str, Any]) -> List[float]:
        """将决策转换为向量"""
        # 简化实现：提取关键决策特征
        vector = []
        
        # 添加风险分数
        vector.append(decision.get("risk_score", 0))
        
        # 添加置信度
        vector.append(decision.get("confidence", 0))
        
        # 添加时间特征（归一化）
        timestamp = decision.get("timestamp", datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # 将时间转换为数值特征
        hour = timestamp.hour / 24.0
        vector.append(hour)
        
        return vector
    
    def _calculate_feature_variability(self, decision_features: List[Dict[str, Any]]) -> float:
        """计算特征变异性"""
        if not decision_features:
            return 0.0
        
        # 分析特征值的变异性
        variabilities = []
        
        # 假设所有决策有相同的特征键
        feature_keys = decision_features[0].keys() if decision_features else []
        
        for key in feature_keys:
            values = [features.get(key, 0) for features in decision_features]
            if len(values) > 1:
                variability = np.std(values) / (np.mean(values) + 1e-6)
                variabilities.append(variability)
        
        return np.mean(variabilities) if variabilities else 0.0
    
    def _calculate_decision_consistency(self, model_decisions: List[Dict[str, Any]]) -> float:
        """计算决策一致性"""
        if len(model_decisions) < 2:
            return 1.0
        
        # 分析连续决策的一致性
        consistency_scores = []
        
        for i in range(len(model_decisions) - 1):
            decision1 = model_decisions[i]
            decision2 = model_decisions[i + 1]
            
            # 比较决策结果
            result1 = decision1.get("result", "")
            result2 = decision2.get("result", "")
            
            consistency = 1.0 if result1 == result2 else 0.0
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5


class BiasDetector:
    """偏见检测器"""
    
    async def detect_data_bias(self,
                             training_data: Dict[str, Any],
                             historical_data: Dict[str, Any]) -> float:
        """检测数据偏见"""
        bias_score = 0.0
        
        # 检查数据分布平衡性
        data_distribution = training_data.get("distribution", {})
        if data_distribution:
            # 计算不同群体的数据量差异
            group_sizes = list(data_distribution.values())
            if group_sizes:
                max_size = max(group_sizes)
                min_size = min(group_sizes)
                imbalance_ratio = (max_size - min_size) / max_size if max_size > 0 else 0.0
                bias_score += imbalance_ratio * 0.5
        
        # 检查历史数据偏见传递
        historical_bias = historical_data.get("historical_bias", 0.0)
        bias_score += historical_bias * 0.5
        
        return min(bias_score, 1.0)
    
    async def detect_model_bias(self,
                              model_decisions: List[Dict[str, Any]],
                              blockchain_context: Dict[str, Any]) -> float:
        """检测模型偏见"""
        if not model_decisions:
            return 0.0
        
        # 分析不同群体的决策差异
        protected_groups = blockchain_context.get("protected_groups", [])
        
        if not protected_groups:
            return 0.0
        
        # 计算群体间决策差异
        group_decisions = {}
        for decision in model_decisions:
            group = decision.get("user_group", "default")
            if group not in group_decisions:
                group_decisions[group] = []
            group_decisions[group].append(decision)
        
        # 计算群体间决策公平性
        fairness_scores = []
        for group1 in protected_groups:
            for group2 in protected_groups:
                if group1 != group2:
                    decisions1 = group_decisions.get(group1, [])
                    decisions2 = group_decisions.get(group2, [])
                    
                    if decisions1 and decisions2:
                        # 计算决策结果差异
                        results1 = [d.get("result", 0) for d in decisions1]
                        results2 = [d.get("result", 0) for d in decisions2]
                        
                        if results1 and results2:
                            mean1 = np.mean(results1)
                            mean2 = np.mean(results2)
                            disparity = abs(mean1 - mean2) / max(abs(mean1), abs(mean2), 1e-6)
                            fairness_scores.append(1.0 - disparity)
        
        bias_score = 1.0 - (np.mean(fairness_scores) if fairness_scores else 1.0)
        return bias_score


class FairnessEvaluator:
    """公平性评估器"""
    
    async def evaluate_fairness(self,
                              model_decisions: List[Dict[str, Any]],
                              blockchain_context: Dict[str, Any]) -> float:
        """评估公平性"""
        if not model_decisions:
            return 0.0
        
        # 应用多种公平性指标
        fairness_metrics = []
        
        # 1. 统计奇偶性
        statistical_parity = await self._calculate_statistical_parity(model_decisions, blockchain_context)
        fairness_metrics.append(statistical_parity)
        
        # 2. 平等机会
        equal_opportunity = await self._calculate_equal_opportunity(model_decisions, blockchain_context)
        fairness_metrics.append(equal_opportunity)
        
        # 3. 预测平等
        predictive_equality = await self._calculate_predictive_equality(model_decisions, blockchain_context)
        fairness_metrics.append(predictive_equality)
        
        # 综合公平性违反分数
        violation_score = 1.0 - np.mean(fairness_metrics) if fairness_metrics else 0.0
        
        return violation_score
    
    async def _calculate_statistical_parity(self,
                                          model_decisions: List[Dict[str, Any]],
                                          blockchain_context: Dict[str, Any]) -> float:
        """计算统计奇偶性"""
        # 简化实现
        protected_groups = blockchain_context.get("protected_groups", [])
        
        if not protected_groups:
            return 1.0
        
        group_acceptance_rates = {}
        for group in protected_groups:
            group_decisions = [d for d in model_decisions if d.get("user_group") == group]
            if group_decisions:
                accepted = [d for d in group_decisions if d.get("result") == "accepted"]
                acceptance_rate = len(accepted) / len(group_decisions)
                group_acceptance_rates[group] = acceptance_rate
        
        if len(group_acceptance_rates) > 1:
            rates = list(group_acceptance_rates.values())
            disparity = max(rates) - min(rates)
            return 1.0 - disparity
        
        return 1.0
    
    async def _calculate_equal_opportunity(self,
                                         model_decisions: List[Dict[str, Any]],
                                         blockchain_context: Dict[str, Any]) -> float:
        """计算平等机会"""
        # 简化实现
        return 0.8  # 假设基本公平
    
    async def _calculate_predictive_equality(self,
                                           model_decisions: List[Dict[str, Any]],
                                           blockchain_context: Dict[str, Any]) -> float:
        """计算预测平等"""
        # 简化实现
        return 0.8  # 假设基本公平


class SystemicRiskAnalyzer:
    """系统性风险分析器"""
    
    async def analyze_systemic_risk(self,
                                  model_decisions: List[Dict[str, Any]],
                                  blockchain_context: Dict[str, Any]) -> float:
        """分析系统性风险"""
        risk_factors = []
        
        # 1. 决策相关性风险
        correlation_risk = await self._analyze_decision_correlation(model_decisions)
        risk_factors.append(correlation_risk)
        
        # 2. 风险集中度
        concentration_risk = await self._analyze_risk_concentration(model_decisions)
        risk_factors.append(concentration_risk)
        
        # 3. 市场影响风险
        market_impact_risk = await self._analyze_market_impact(model_decisions, blockchain_context)
        risk_factors.append(market_impact_risk)
        
        return np.mean(risk_factors) if risk_factors else 0.0
    
    async def _analyze_decision_correlation(self, model_decisions: List[Dict[str, Any]]) -> float:
        """分析决策相关性"""
        # 简化实现
        if len(model_decisions) < 2:
            return 0.0
        
        # 假设决策高度相关时风险增加
        return min(len(model_decisions) / 100.0, 1.0)
    
    async def _analyze_risk_concentration(self, model_decisions: List[Dict[str, Any]]) -> float:
        """分析风险集中度"""
        if not model_decisions:
            return 0.0
        
        risk_scores = [d.get("risk_score", 0) for d in model_decisions]
        if risk_scores:
            # 计算风险集中度（方差）
            concentration = np.var(risk_scores)
            return min(concentration * 10, 1.0)
        
        return 0.0
    
    async def _analyze_market_impact(self,
                                   model_decisions: List[Dict[str, Any]],
                                   blockchain_context: Dict[str, Any]) -> float:
        """分析市场影响风险"""
        # 简化实现
        market_sensitivity = blockchain_context.get("market_sensitivity", 0.5)
        decision_volume = len(model_decisions)
        
        impact_risk = market_sensitivity * (decision_volume / 1000.0)
        return min(impact_risk, 1.0)


class DecisionDiversityMonitor:
    """决策多样性监控器"""
    
    async def assess_convergence_risk(self, model_decisions: List[Dict[str, Any]]) -> float:
        """评估决策趋同风险"""
        if len(model_decisions) < 2:
            return 0.0
        
        # 分析决策策略的趋同性
        strategies = [d.get("strategy", "default") for d in model_decisions]
        unique_strategies = len(set(strategies))
        
        # 趋同风险 = 1 - 多样性
        convergence_risk = 1.0 - (unique_strategies / len(strategies))
        
        return convergence_risk