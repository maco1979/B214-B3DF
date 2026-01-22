"""
技术失控与目标对齐风险控制模块

负责控制AI自主决策中的技术失控风险，确保AI决策与预设目标对齐，
防止算法黑箱特性导致的决策偏离，保障区块链经济系统稳定运行。
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib


class TechnicalRiskType(Enum):
    """技术风险类型"""
    TARGET_MISALIGNMENT = "target_misalignment"  # 目标偏离
    BLACKBOX_BEHAVIOR = "blackbox_behavior"  # 黑箱行为
    SAFETY_OVERRIDE = "safety_override"  # 安全机制绕过
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽
    DECISION_DRIFT = "decision_drift"  # 决策漂移


class RiskSeverity(Enum):
    """风险严重程度"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TechnicalRiskAlert:
    """技术风险警报"""
    risk_type: TechnicalRiskType
    severity: RiskSeverity
    alert_id: str
    description: str
    detected_behavior: Dict[str, Any]
    confidence_score: float
    mitigation_action: str
    timestamp: datetime


@dataclass
class TechnicalRiskAssessment:
    """技术风险评估结果"""
    overall_risk_level: RiskSeverity
    risk_score: float  # 0-1之间的风险评分
    active_alerts: List[TechnicalRiskAlert]
    compliance_status: bool
    emergency_stop_triggered: bool
    recommendations: List[str]


class TechnicalRiskController:
    """技术失控风险控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 风险监控状态
        self.risk_metrics = {}
        self.alert_history = []
        self.emergency_stop_count = 0
        self.last_assessment_time = datetime.utcnow()
        
        # 目标对齐验证器
        self.target_validator = TargetAlignmentValidator()
        
        # 行为异常检测器
        self.behavior_detector = BehaviorAnomalyDetector()
        
        # 紧急停止管理器
        self.emergency_manager = EmergencyStopManager()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "target_alignment_threshold": 0.8,  # 目标对齐阈值
            "blackbox_confidence_threshold": 0.3,  # 黑箱行为置信度阈值
            "safety_override_threshold": 3,  # 安全机制绕过次数阈值
            "resource_usage_threshold": 0.9,  # 资源使用率阈值
            "decision_drift_threshold": 0.15,  # 决策漂移阈值
            "emergency_stop_enabled": True,  # 是否启用紧急停止
            "monitoring_interval_seconds": 30,  # 监控间隔
            "max_alerts_per_hour": 10  # 每小时最大警报数
        }
    
    async def assess_technical_risk(self,
                                  ai_decision_data: Dict[str, Any],
                                  blockchain_context: Dict[str, Any],
                                  system_state: Dict[str, Any]) -> TechnicalRiskAssessment:
        """
        评估技术失控风险
        
        Args:
            ai_decision_data: AI决策数据
            blockchain_context: 区块链上下文
            system_state: 系统状态
            
        Returns:
            TechnicalRiskAssessment: 技术风险评估结果
        """
        try:
            alerts = []
            
            # 1. 目标对齐风险评估
            target_alignment_risk = await self._assess_target_alignment_risk(
                ai_decision_data, blockchain_context)
            alerts.extend(target_alignment_risk)
            
            # 2. 黑箱行为风险评估
            blackbox_risk = await self._assess_blackbox_behavior_risk(
                ai_decision_data, system_state)
            alerts.extend(blackbox_risk)
            
            # 3. 安全机制绕过风险评估
            safety_override_risk = await self._assess_safety_override_risk(
                ai_decision_data, blockchain_context)
            alerts.extend(safety_override_risk)
            
            # 4. 资源耗尽风险评估
            resource_exhaustion_risk = await self._assess_resource_exhaustion_risk(
                system_state)
            alerts.extend(resource_exhaustion_risk)
            
            # 5. 决策漂移风险评估
            decision_drift_risk = await self._assess_decision_drift_risk(
                ai_decision_data, blockchain_context)
            alerts.extend(decision_drift_risk)
            
            # 6. 综合风险评估
            overall_risk_level = self._determine_overall_risk_level(alerts)
            risk_score = self._calculate_risk_score(alerts)
            
            # 7. 检查是否触发紧急停止
            emergency_stop_triggered = self._check_emergency_stop_condition(alerts)
            
            # 8. 生成改进建议
            recommendations = self._generate_recommendations(alerts, overall_risk_level)
            
            # 9. 更新风险指标
            self._update_risk_metrics(alerts, overall_risk_level)
            
            return TechnicalRiskAssessment(
                overall_risk_level=overall_risk_level,
                risk_score=risk_score,
                active_alerts=alerts,
                compliance_status=overall_risk_level in [RiskSeverity.LOW, RiskSeverity.MEDIUM],
                emergency_stop_triggered=emergency_stop_triggered,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"技术风险评估失败: {e}")
            # 返回最严格的风险评估结果
            return TechnicalRiskAssessment(
                overall_risk_level=RiskSeverity.CRITICAL,
                risk_score=1.0,
                active_alerts=[TechnicalRiskAlert(
                    risk_type=TechnicalRiskType.TARGET_MISALIGNMENT,
                    severity=RiskSeverity.CRITICAL,
                    alert_id="ASSESSMENT_ERROR",
                    description="技术风险评估过程出现异常",
                    detected_behavior={"error": str(e)},
                    confidence_score=1.0,
                    mitigation_action="立即停止AI决策并检查系统",
                    timestamp=datetime.utcnow()
                )],
                compliance_status=False,
                emergency_stop_triggered=True,
                recommendations=["技术风险评估失败，建议立即人工干预"]
            )
    
    async def _assess_target_alignment_risk(self,
                                          ai_decision_data: Dict[str, Any],
                                          blockchain_context: Dict[str, Any]) -> List[TechnicalRiskAlert]:
        """评估目标对齐风险"""
        alerts = []
        
        # 检查决策目标是否与区块链经济目标对齐
        alignment_score = await self.target_validator.validate_target_alignment(
            ai_decision_data, blockchain_context)
        
        if alignment_score < self.config["target_alignment_threshold"]:
            alerts.append(TechnicalRiskAlert(
                risk_type=TechnicalRiskType.TARGET_MISALIGNMENT,
                severity=RiskSeverity.HIGH,
                alert_id="TARGET_MISALIGN_001",
                description="AI决策目标与区块链经济目标偏离",
                detected_behavior={
                    "alignment_score": alignment_score,
                    "threshold": self.config["target_alignment_threshold"]
                },
                confidence_score=1.0 - alignment_score,
                mitigation_action="调整AI决策目标或重新训练模型",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_blackbox_behavior_risk(self,
                                           ai_decision_data: Dict[str, Any],
                                           system_state: Dict[str, Any]) -> List[TechnicalRiskAlert]:
        """评估黑箱行为风险"""
        alerts = []
        
        # 检测AI决策的不可解释行为
        blackbox_confidence = await self.behavior_detector.detect_blackbox_behavior(
            ai_decision_data, system_state)
        
        if blackbox_confidence > self.config["blackbox_confidence_threshold"]:
            alerts.append(TechnicalRiskAlert(
                risk_type=TechnicalRiskType.BLACKBOX_BEHAVIOR,
                severity=RiskSeverity.MEDIUM,
                alert_id="BLACKBOX_BEHAVIOR_001",
                description="检测到AI黑箱决策行为",
                detected_behavior={
                    "blackbox_confidence": blackbox_confidence,
                    "decision_complexity": ai_decision_data.get("complexity", "high")
                },
                confidence_score=blackbox_confidence,
                mitigation_action="增加模型可解释性分析",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_safety_override_risk(self,
                                         ai_decision_data: Dict[str, Any],
                                         blockchain_context: Dict[str, Any]) -> List[TechnicalRiskAlert]:
        """评估安全机制绕过风险"""
        alerts = []
        
        # 检查AI是否尝试绕过安全机制
        safety_override_count = await self._check_safety_override_attempts(
            ai_decision_data, blockchain_context)
        
        if safety_override_count >= self.config["safety_override_threshold"]:
            alerts.append(TechnicalRiskAlert(
                risk_type=TechnicalRiskType.SAFETY_OVERRIDE,
                severity=RiskSeverity.CRITICAL,
                alert_id="SAFETY_OVERRIDE_001",
                description="检测到AI试图绕过安全机制",
                detected_behavior={
                    "override_attempts": safety_override_count,
                    "threshold": self.config["safety_override_threshold"]
                },
                confidence_score=min(safety_override_count / 10.0, 1.0),
                mitigation_action="立即启用紧急停止机制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_resource_exhaustion_risk(self, system_state: Dict[str, Any]) -> List[TechnicalRiskAlert]:
        """评估资源耗尽风险"""
        alerts = []
        
        # 检查系统资源使用情况
        resource_usage = system_state.get("resource_usage", {})
        cpu_usage = resource_usage.get("cpu", 0)
        memory_usage = resource_usage.get("memory", 0)
        
        if cpu_usage > self.config["resource_usage_threshold"] * 100:
            alerts.append(TechnicalRiskAlert(
                risk_type=TechnicalRiskType.RESOURCE_EXHAUSTION,
                severity=RiskSeverity.HIGH,
                alert_id="RESOURCE_CPU_001",
                description="CPU使用率过高，存在资源耗尽风险",
                detected_behavior={
                    "cpu_usage_percent": cpu_usage,
                    "threshold_percent": self.config["resource_usage_threshold"] * 100
                },
                confidence_score=cpu_usage / 100.0,
                mitigation_action="限制AI决策资源使用或扩容",
                timestamp=datetime.utcnow()
            ))
        
        if memory_usage > self.config["resource_usage_threshold"] * 100:
            alerts.append(TechnicalRiskAlert(
                risk_type=TechnicalRiskType.RESOURCE_EXHAUSTION,
                severity=RiskSeverity.HIGH,
                alert_id="RESOURCE_MEMORY_001",
                description="内存使用率过高，存在资源耗尽风险",
                detected_behavior={
                    "memory_usage_percent": memory_usage,
                    "threshold_percent": self.config["resource_usage_threshold"] * 100
                },
                confidence_score=memory_usage / 100.0,
                mitigation_action="优化内存使用或增加内存",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_decision_drift_risk(self,
                                        ai_decision_data: Dict[str, Any],
                                        blockchain_context: Dict[str, Any]) -> List[TechnicalRiskAlert]:
        """评估决策漂移风险"""
        alerts = []
        
        # 检测AI决策是否发生漂移
        drift_score = await self._calculate_decision_drift(ai_decision_data, blockchain_context)
        
        if drift_score > self.config["decision_drift_threshold"]:
            alerts.append(TechnicalRiskAlert(
                risk_type=TechnicalRiskType.DECISION_DRIFT,
                severity=RiskSeverity.MEDIUM,
                alert_id="DECISION_DRIFT_001",
                description="检测到AI决策漂移",
                detected_behavior={
                    "drift_score": drift_score,
                    "threshold": self.config["decision_drift_threshold"]
                },
                confidence_score=drift_score,
                mitigation_action="重新校准AI模型或调整决策策略",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _determine_overall_risk_level(self, alerts: List[TechnicalRiskAlert]) -> RiskSeverity:
        """确定总体风险等级"""
        if not alerts:
            return RiskSeverity.LOW
        
        severities = [alert.severity for alert in alerts]
        
        if RiskSeverity.CRITICAL in severities:
            return RiskSeverity.CRITICAL
        elif RiskSeverity.HIGH in severities:
            return RiskSeverity.HIGH
        elif RiskSeverity.MEDIUM in severities:
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW
    
    def _calculate_risk_score(self, alerts: List[TechnicalRiskAlert]) -> float:
        """计算综合风险评分"""
        if not alerts:
            return 0.0
        
        severity_weights = {
            RiskSeverity.CRITICAL: 1.0,
            RiskSeverity.HIGH: 0.7,
            RiskSeverity.MEDIUM: 0.4,
            RiskSeverity.LOW: 0.1
        }
        
        total_score = sum(
            severity_weights[alert.severity] * alert.confidence_score 
            for alert in alerts
        )
        
        return min(total_score / len(alerts), 1.0)
    
    def _check_emergency_stop_condition(self, alerts: List[TechnicalRiskAlert]) -> bool:
        """检查是否触发紧急停止条件"""
        if not self.config["emergency_stop_enabled"]:
            return False
        
        # 检查是否有严重风险警报
        critical_alerts = [
            alert for alert in alerts 
            if alert.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]
            and alert.confidence_score > 0.7
        ]
        
        return len(critical_alerts) > 0
    
    def _generate_recommendations(self, 
                                alerts: List[TechnicalRiskAlert],
                                overall_risk_level: RiskSeverity) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_risk_level == RiskSeverity.LOW:
            recommendations.append("技术风险控制良好，继续保持监控")
            return recommendations
        
        # 根据风险类型提供针对性建议
        risk_types = set(alert.risk_type for alert in alerts)
        
        if TechnicalRiskType.TARGET_MISALIGNMENT in risk_types:
            recommendations.append("建议加强AI目标对齐验证机制")
        
        if TechnicalRiskType.BLACKBOX_BEHAVIOR in risk_types:
            recommendations.append("建议增加模型可解释性分析和透明度")
        
        if TechnicalRiskType.SAFETY_OVERRIDE in risk_types:
            recommendations.append("必须立即修复安全机制漏洞")
        
        if TechnicalRiskType.RESOURCE_EXHAUSTION in risk_types:
            recommendations.append("建议优化资源分配策略")
        
        if TechnicalRiskType.DECISION_DRIFT in risk_types:
            recommendations.append("建议定期校准AI决策模型")
        
        # 紧急情况建议
        if overall_risk_level in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]:
            recommendations.insert(0, "⚠️ 高风险警报：建议立即启动应急响应机制")
        
        return recommendations
    
    def _update_risk_metrics(self, alerts: List[TechnicalRiskAlert], risk_level: RiskSeverity):
        """更新风险指标"""
        current_time = datetime.utcnow()
        
        # 记录警报历史
        self.alert_history.extend(alerts)
        
        # 清理过时警报（保留最近24小时）
        cutoff_time = current_time - timedelta(hours=24)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
        
        # 更新紧急停止计数
        if risk_level in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]:
            self.emergency_stop_count += 1
        
        self.last_assessment_time = current_time
    
    # 辅助方法（简化实现）
    async def _check_safety_override_attempts(self,
                                            ai_decision_data: Dict[str, Any],
                                            blockchain_context: Dict[str, Any]) -> int:
        """检查安全机制绕过尝试次数"""
        # 简化实现：实际应用中应分析决策日志
        decision_logs = ai_decision_data.get("decision_logs", [])
        override_attempts = 0
        
        for log in decision_logs:
            if log.get("safety_check_bypassed", False):
                override_attempts += 1
        
        return override_attempts
    
    async def _calculate_decision_drift(self,
                                      ai_decision_data: Dict[str, Any],
                                      blockchain_context: Dict[str, Any]) -> float:
        """计算决策漂移分数"""
        # 简化实现：实际应用中应使用更复杂的漂移检测算法
        decision_pattern = ai_decision_data.get("decision_pattern", {})
        historical_pattern = blockchain_context.get("historical_decisions", {})
        
        if not decision_pattern or not historical_pattern:
            return 0.0
        
        # 计算决策模式差异
        pattern_difference = 0.0
        common_keys = set(decision_pattern.keys()) & set(historical_pattern.keys())
        
        if not common_keys:
            return 0.5  # 无法比较，返回中等风险
        
        for key in common_keys:
            current_val = decision_pattern[key]
            historical_val = historical_pattern[key]
            
            if isinstance(current_val, (int, float)) and isinstance(historical_val, (int, float)):
                difference = abs(current_val - historical_val) / max(abs(historical_val), 1e-6)
                pattern_difference += difference
        
        return pattern_difference / len(common_keys) if common_keys else 0.0


class TargetAlignmentValidator:
    """目标对齐验证器"""
    
    async def validate_target_alignment(self,
                                      ai_decision_data: Dict[str, Any],
                                      blockchain_context: Dict[str, Any]) -> float:
        """验证AI决策目标与区块链经济目标对齐度"""
        # 简化实现：实际应用中应使用更复杂的对齐度计算
        ai_targets = ai_decision_data.get("decision_targets", [])
        blockchain_targets = blockchain_context.get("economic_targets", [])
        
        if not ai_targets or not blockchain_targets:
            return 0.5
        
        # 计算目标重叠度
        target_overlap = len(set(ai_targets) & set(blockchain_targets))
        total_targets = len(set(ai_targets) | set(blockchain_targets))
        
        return target_overlap / total_targets if total_targets > 0 else 0.0


class BehaviorAnomalyDetector:
    """行为异常检测器"""
    
    async def detect_blackbox_behavior(self,
                                     ai_decision_data: Dict[str, Any],
                                     system_state: Dict[str, Any]) -> float:
        """检测黑箱行为置信度"""
        # 简化实现：实际应用中应使用异常检测算法
        decision_explainability = ai_decision_data.get("explainability_score", 0.5)
        decision_confidence = ai_decision_data.get("confidence", 0.5)
        
        # 可解释性越低，黑箱行为置信度越高
        blackbox_confidence = (1.0 - decision_explainability) * decision_confidence
        
        return blackbox_confidence


class EmergencyStopManager:
    """紧急停止管理器"""
    
    async def trigger_emergency_stop(self, 
                                   risk_assessment: TechnicalRiskAssessment,
                                   blockchain_context: Dict[str, Any]) -> bool:
        """触发紧急停止机制"""
        try:
            # 执行紧急停止操作
            # 1. 暂停AI决策
            # 2. 记录停止原因
            # 3. 通知相关人员
            # 4. 启动应急响应
            
            self.logger.warning(
                f"紧急停止触发：风险等级{risk_assessment.overall_risk_level}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"紧急停止执行失败: {e}")
            return False