"""
数据安全与隐私保护风险控制模块

负责保护AI训练和决策过程中的数据安全，防止隐私泄露，
确保链上链下数据的机密性和完整性，防范数据窃取和滥用风险。
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class DataRiskType(Enum):
    """数据风险类型"""
    PRIVACY_LEAKAGE = "privacy_leakage"  # 隐私泄露
    DATA_BREACH = "data_breach"  # 数据泄露
    UNAUTHORIZED_ACCESS = "unauthorized_access"  # 未授权访问
    MODEL_PARAM_LEAKAGE = "model_param_leakage"  # 模型参数泄露
    FEDERATED_LEARNING_ATTACK = "federated_learning_attack"  # 联邦学习攻击


class DataRiskSeverity(Enum):
    """数据风险严重程度"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataSecurityAlert:
    """数据安全警报"""
    risk_type: DataRiskType
    severity: DataRiskSeverity
    alert_id: str
    description: str
    affected_data: Dict[str, Any]
    confidence_score: float
    protection_action: str
    timestamp: datetime


@dataclass
class DataSecurityAssessment:
    """数据安全评估结果"""
    overall_security_level: DataRiskSeverity
    security_score: float  # 0-1之间的安全评分
    active_alerts: List[DataSecurityAlert]
    encryption_status: bool
    privacy_protection_status: bool
    compliance_status: bool
    recommendations: List[str]


class DataSecurityController:
    """数据安全与隐私保护控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 安全监控状态
        self.security_metrics = {}
        self.alert_history = []
        self.encryption_keys = {}
        self.access_logs = []
        
        # 隐私保护组件
        self.privacy_protector = PrivacyProtector()
        
        # 加密管理器
        self.encryption_manager = EncryptionManager()
        
        # 访问控制管理器
        self.access_controller = AccessController()
        
        # 联邦学习安全器
        self.federated_security = FederatedLearningSecurity()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "privacy_leakage_threshold": 0.7,  # 隐私泄露风险阈值
            "data_breach_threshold": 0.8,  # 数据泄露风险阈值
            "unauthorized_access_threshold": 3,  # 未授权访问次数阈值
            "model_param_protection_threshold": 0.6,  # 模型参数保护阈值
            "federated_learning_security_threshold": 0.5,  # 联邦学习安全阈值
            "encryption_required": True,  # 是否要求加密
            "privacy_preservation_enabled": True,  # 是否启用隐私保护
            "access_control_enabled": True,  # 是否启用访问控制
            "audit_log_retention_days": 30  # 审计日志保留天数
        }
    
    async def assess_data_security_risk(self,
                                      training_data: Dict[str, Any],
                                      model_parameters: Dict[str, Any],
                                      blockchain_context: Dict[str, Any],
                                      access_logs: List[Dict[str, Any]]) -> DataSecurityAssessment:
        """
        评估数据安全与隐私保护风险
        
        Args:
            training_data: 训练数据信息
            model_parameters: 模型参数信息
            blockchain_context: 区块链上下文
            access_logs: 访问日志
            
        Returns:
            DataSecurityAssessment: 数据安全评估结果
        """
        try:
            alerts = []
            
            # 1. 隐私泄露风险评估
            privacy_leakage_risk = await self._assess_privacy_leakage_risk(
                training_data, blockchain_context)
            alerts.extend(privacy_leakage_risk)
            
            # 2. 数据泄露风险评估
            data_breach_risk = await self._assess_data_breach_risk(
                training_data, model_parameters, access_logs)
            alerts.extend(data_breach_risk)
            
            # 3. 未授权访问风险评估
            unauthorized_access_risk = await self._assess_unauthorized_access_risk(
                access_logs)
            alerts.extend(unauthorized_access_risk)
            
            # 4. 模型参数泄露风险评估
            model_param_leakage_risk = await self._assess_model_param_leakage_risk(
                model_parameters, blockchain_context)
            alerts.extend(model_param_leakage_risk)
            
            # 5. 联邦学习攻击风险评估
            federated_learning_risk = await self._assess_federated_learning_risk(
                training_data, model_parameters)
            alerts.extend(federated_learning_risk)
            
            # 6. 综合安全评估
            overall_security_level = self._determine_overall_security_level(alerts)
            security_score = self._calculate_security_score(alerts)
            
            # 7. 检查加密状态
            encryption_status = await self._check_encryption_status(training_data, model_parameters)
            
            # 8. 检查隐私保护状态
            privacy_protection_status = await self._check_privacy_protection_status(training_data)
            
            # 9. 生成改进建议
            recommendations = self._generate_recommendations(alerts, overall_security_level)
            
            # 10. 更新安全指标
            self._update_security_metrics(alerts, overall_security_level)
            
            return DataSecurityAssessment(
                overall_security_level=overall_security_level,
                security_score=security_score,
                active_alerts=alerts,
                encryption_status=encryption_status,
                privacy_protection_status=privacy_protection_status,
                compliance_status=overall_security_level in [DataRiskSeverity.LOW, DataRiskSeverity.MEDIUM],
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"数据安全评估失败: {e}")
            # 返回最严格的安全评估结果
            return DataSecurityAssessment(
                overall_security_level=DataRiskSeverity.CRITICAL,
                security_score=0.0,
                active_alerts=[DataSecurityAlert(
                    risk_type=DataRiskType.DATA_BREACH,
                    severity=DataRiskSeverity.CRITICAL,
                    alert_id="ASSESSMENT_ERROR",
                    description="数据安全评估过程出现异常",
                    affected_data={"error": str(e)},
                    confidence_score=1.0,
                    protection_action="立即停止数据处理并检查系统",
                    timestamp=datetime.utcnow()
                )],
                encryption_status=False,
                privacy_protection_status=False,
                compliance_status=False,
                recommendations=["数据安全评估失败，建议立即人工干预"]
            )
    
    async def _assess_privacy_leakage_risk(self,
                                         training_data: Dict[str, Any],
                                         blockchain_context: Dict[str, Any]) -> List[DataSecurityAlert]:
        """评估隐私泄露风险"""
        alerts = []
        
        # 检查训练数据中的隐私信息保护
        privacy_leakage_score = await self.privacy_protector.assess_privacy_risk(
            training_data, blockchain_context)
        
        if privacy_leakage_score > self.config["privacy_leakage_threshold"]:
            alerts.append(DataSecurityAlert(
                risk_type=DataRiskType.PRIVACY_LEAKAGE,
                severity=DataRiskSeverity.HIGH,
                alert_id="PRIVACY_LEAKAGE_001",
                description="检测到隐私信息泄露风险",
                affected_data={
                    "privacy_leakage_score": privacy_leakage_score,
                    "sensitive_fields": training_data.get("sensitive_fields", [])
                },
                confidence_score=privacy_leakage_score,
                protection_action="应用差分隐私或数据脱敏技术",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_data_breach_risk(self,
                                     training_data: Dict[str, Any],
                                     model_parameters: Dict[str, Any],
                                     access_logs: List[Dict[str, Any]]) -> List[DataSecurityAlert]:
        """评估数据泄露风险"""
        alerts = []
        
        # 检查数据安全防护措施
        data_breach_risk = await self._calculate_data_breach_risk(
            training_data, model_parameters, access_logs)
        
        if data_breach_risk > self.config["data_breach_threshold"]:
            alerts.append(DataSecurityAlert(
                risk_type=DataRiskType.DATA_BREACH,
                severity=DataRiskSeverity.CRITICAL,
                alert_id="DATA_BREACH_001",
                description="检测到数据泄露高风险",
                affected_data={
                    "data_breach_risk": data_breach_risk,
                    "data_size": training_data.get("size", 0),
                    "encryption_status": training_data.get("encrypted", False)
                },
                confidence_score=data_breach_risk,
                protection_action="加强数据加密和访问控制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_unauthorized_access_risk(self, access_logs: List[Dict[str, Any]]) -> List[DataSecurityAlert]:
        """评估未授权访问风险"""
        alerts = []
        
        # 分析访问日志，检测异常访问模式
        unauthorized_attempts = await self.access_controller.detect_unauthorized_access(access_logs)
        
        if unauthorized_attempts >= self.config["unauthorized_access_threshold"]:
            alerts.append(DataSecurityAlert(
                risk_type=DataRiskType.UNAUTHORIZED_ACCESS,
                severity=DataRiskSeverity.HIGH,
                alert_id="UNAUTHORIZED_ACCESS_001",
                description="检测到未授权访问尝试",
                affected_data={
                    "unauthorized_attempts": unauthorized_attempts,
                    "access_patterns": self._analyze_access_patterns(access_logs)
                },
                confidence_score=min(unauthorized_attempts / 10.0, 1.0),
                protection_action="强化身份认证和访问控制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_model_param_leakage_risk(self,
                                             model_parameters: Dict[str, Any],
                                             blockchain_context: Dict[str, Any]) -> List[DataSecurityAlert]:
        """评估模型参数泄露风险"""
        alerts = []
        
        # 检查模型参数保护措施
        model_protection_score = await self._assess_model_protection(model_parameters, blockchain_context)
        
        if model_protection_score < self.config["model_param_protection_threshold"]:
            alerts.append(DataSecurityAlert(
                risk_type=DataRiskType.MODEL_PARAM_LEAKAGE,
                severity=DataRiskSeverity.HIGH,
                alert_id="MODEL_PARAM_LEAKAGE_001",
                description="模型参数保护不足，存在泄露风险",
                affected_data={
                    "model_protection_score": model_protection_score,
                    "model_size": model_parameters.get("size", 0),
                    "protection_methods": model_parameters.get("protection_methods", [])
                },
                confidence_score=1.0 - model_protection_score,
                protection_action="应用模型加密或联邦学习技术",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_federated_learning_risk(self,
                                            training_data: Dict[str, Any],
                                            model_parameters: Dict[str, Any]) -> List[DataSecurityAlert]:
        """评估联邦学习攻击风险"""
        alerts = []
        
        # 检查联邦学习安全性
        federated_security_score = await self.federated_security.assess_security_risk(
            training_data, model_parameters)
        
        if federated_security_score < self.config["federated_learning_security_threshold"]:
            alerts.append(DataSecurityAlert(
                risk_type=DataRiskType.FEDERATED_LEARNING_ATTACK,
                severity=DataRiskSeverity.MEDIUM,
                alert_id="FEDERATED_LEARNING_ATTACK_001",
                description="联邦学习安全性不足，存在攻击风险",
                affected_data={
                    "federated_security_score": federated_security_score,
                    "participant_count": training_data.get("participants", 0),
                    "aggregation_method": model_parameters.get("aggregation_method", "fedavg")
                },
                confidence_score=1.0 - federated_security_score,
                protection_action="加强联邦学习安全机制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _determine_overall_security_level(self, alerts: List[DataSecurityAlert]) -> DataRiskSeverity:
        """确定总体安全等级"""
        if not alerts:
            return DataRiskSeverity.LOW
        
        severities = [alert.severity for alert in alerts]
        
        if DataRiskSeverity.CRITICAL in severities:
            return DataRiskSeverity.CRITICAL
        elif DataRiskSeverity.HIGH in severities:
            return DataRiskSeverity.HIGH
        elif DataRiskSeverity.MEDIUM in severities:
            return DataRiskSeverity.MEDIUM
        else:
            return DataRiskSeverity.LOW
    
    def _calculate_security_score(self, alerts: List[DataSecurityAlert]) -> float:
        """计算综合安全评分"""
        if not alerts:
            return 1.0
        
        severity_weights = {
            DataRiskSeverity.CRITICAL: 0.0,
            DataRiskSeverity.HIGH: 0.3,
            DataRiskSeverity.MEDIUM: 0.6,
            DataRiskSeverity.LOW: 0.9
        }
        
        total_score = sum(
            severity_weights[alert.severity] * (1.0 - alert.confidence_score)
            for alert in alerts
        )
        
        return total_score / len(alerts) if alerts else 1.0
    
    async def _check_encryption_status(self, training_data: Dict[str, Any], model_parameters: Dict[str, Any]) -> bool:
        """检查加密状态"""
        if not self.config["encryption_required"]:
            return True
        
        # 检查数据和模型参数是否加密
        data_encrypted = training_data.get("encrypted", False)
        model_encrypted = model_parameters.get("encrypted", False)
        
        return data_encrypted and model_encrypted
    
    async def _check_privacy_protection_status(self, training_data: Dict[str, Any]) -> bool:
        """检查隐私保护状态"""
        if not self.config["privacy_preservation_enabled"]:
            return True
        
        # 检查是否应用了隐私保护技术
        privacy_techniques = training_data.get("privacy_techniques", [])
        required_techniques = ["differential_privacy", "federated_learning", "homomorphic_encryption"]
        
        return any(tech in privacy_techniques for tech in required_techniques)
    
    def _generate_recommendations(self, 
                                alerts: List[DataSecurityAlert],
                                overall_security_level: DataRiskSeverity) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_security_level == DataRiskSeverity.LOW:
            recommendations.append("数据安全保护良好，继续保持监控")
            return recommendations
        
        # 根据风险类型提供针对性建议
        risk_types = set(alert.risk_type for alert in alerts)
        
        if DataRiskType.PRIVACY_LEAKAGE in risk_types:
            recommendations.append("建议应用差分隐私或数据脱敏技术")
        
        if DataRiskType.DATA_BREACH in risk_types:
            recommendations.append("建议加强数据加密和访问控制")
        
        if DataRiskType.UNAUTHORIZED_ACCESS in risk_types:
            recommendations.append("建议强化身份认证和权限管理")
        
        if DataRiskType.MODEL_PARAM_LEAKAGE in risk_types:
            recommendations.append("建议应用模型加密或联邦学习")
        
        if DataRiskType.FEDERATED_LEARNING_ATTACK in risk_types:
            recommendations.append("建议加强联邦学习安全机制")
        
        # 紧急情况建议
        if overall_security_level in [DataRiskSeverity.HIGH, DataRiskSeverity.CRITICAL]:
            recommendations.insert(0, "🔒 高安全风险：建议立即启动数据保护应急机制")
        
        return recommendations
    
    def _update_security_metrics(self, alerts: List[DataSecurityAlert], security_level: DataRiskSeverity):
        """更新安全指标"""
        current_time = datetime.utcnow()
        
        # 记录警报历史
        self.alert_history.extend(alerts)
        
        # 清理过时警报（保留配置的天数）
        cutoff_time = current_time - timedelta(days=self.config["audit_log_retention_days"])
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
        
        # 更新安全指标
        self.security_metrics["last_assessment"] = current_time
        self.security_metrics["current_security_level"] = security_level
        self.security_metrics["active_alerts_count"] = len(alerts)
    
    # 辅助方法（简化实现）
    async def _calculate_data_breach_risk(self,
                                        training_data: Dict[str, Any],
                                        model_parameters: Dict[str, Any],
                                        access_logs: List[Dict[str, Any]]) -> float:
        """计算数据泄露风险"""
        risk_factors = []
        
        # 数据加密状态
        data_encrypted = training_data.get("encrypted", False)
        risk_factors.append(0.0 if data_encrypted else 0.8)
        
        # 访问控制强度
        access_control_strength = training_data.get("access_control_strength", 0.5)
        risk_factors.append(1.0 - access_control_strength)
        
        # 异常访问模式
        abnormal_access = await self._detect_abnormal_access(access_logs)
        risk_factors.append(abnormal_access)
        
        return sum(risk_factors) / len(risk_factors)
    
    async def _assess_model_protection(self,
                                     model_parameters: Dict[str, Any],
                                     blockchain_context: Dict[str, Any]) -> float:
        """评估模型保护程度"""
        protection_score = 0.0
        
        # 模型加密状态
        if model_parameters.get("encrypted", False):
            protection_score += 0.3
        
        # 参数安全存储
        if model_parameters.get("secure_storage", False):
            protection_score += 0.3
        
        # 区块链溯源保护
        if blockchain_context.get("model_tracking_enabled", False):
            protection_score += 0.4
        
        return protection_score
    
    async def _detect_abnormal_access(self, access_logs: List[Dict[str, Any]]) -> float:
        """检测异常访问模式"""
        if not access_logs:
            return 0.0
        
        # 分析访问频率、时间模式等
        recent_logs = [log for log in access_logs 
                      if datetime.fromisoformat(log["timestamp"]) > datetime.utcnow() - timedelta(hours=24)]
        
        if len(recent_logs) > 100:  # 假设正常访问频率阈值
            return 0.7
        
        return 0.0
    
    def _analyze_access_patterns(self, access_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析访问模式"""
        if not access_logs:
            return {}
        
        recent_logs = [log for log in access_logs 
                      if datetime.fromisoformat(log["timestamp"]) > datetime.utcnow() - timedelta(hours=24)]
        
        return {
            "total_accesses": len(recent_logs),
            "unique_users": len(set(log["user_id"] for log in recent_logs if "user_id" in log)),
            "access_frequency": len(recent_logs) / 24.0  # 每小时访问次数
        }


class PrivacyProtector:
    """隐私保护器"""
    
    async def assess_privacy_risk(self,
                                training_data: Dict[str, Any],
                                blockchain_context: Dict[str, Any]) -> float:
        """评估隐私风险"""
        risk_score = 0.0
        
        # 检查敏感数据字段
        sensitive_fields = training_data.get("sensitive_fields", [])
        if sensitive_fields:
            risk_score += 0.4
        
        # 检查隐私保护技术应用
        privacy_techniques = training_data.get("privacy_techniques", [])
        if not privacy_techniques:
            risk_score += 0.6
        
        return risk_score


class EncryptionManager:
    """加密管理器"""
    
    def __init__(self):
        self.fernet = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """初始化加密系统"""
        # 简化实现：实际应用中应从安全存储获取密钥
        key = Fernet.generate_key()
        self.fernet = Fernet(key)
    
    async def encrypt_data(self, data: bytes) -> bytes:
        """加密数据"""
        if self.fernet:
            return self.fernet.encrypt(data)
        return data
    
    async def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """解密数据"""
        if self.fernet:
            return self.fernet.decrypt(encrypted_data)
        return encrypted_data


class AccessController:
    """访问控制器"""
    
    async def detect_unauthorized_access(self, access_logs: List[Dict[str, Any]]) -> int:
        """检测未授权访问尝试"""
        unauthorized_count = 0
        
        for log in access_logs:
            if log.get("access_granted", False) == False:
                unauthorized_count += 1
        
        return unauthorized_count


class FederatedLearningSecurity:
    """联邦学习安全器"""
    
    async def assess_security_risk(self,
                                 training_data: Dict[str, Any],
                                 model_parameters: Dict[str, Any]) -> float:
        """评估联邦学习安全风险"""
        security_score = 0.0
        
        # 检查安全聚合机制
        if model_parameters.get("secure_aggregation", False):
            security_score += 0.4
        
        # 检查差分隐私应用
        if training_data.get("differential_privacy_applied", False):
            security_score += 0.3
        
        # 检查参与方认证
        if training_data.get("participant_authentication", False):
            security_score += 0.3
        
        return security_score