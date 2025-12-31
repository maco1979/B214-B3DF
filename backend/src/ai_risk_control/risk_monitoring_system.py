"""
AI自主决策风险监控与预警系统

集成所有风险控制模块，实现全面的AI自主决策风险监控、预警和应急响应，
确保区块链经济模型中AI决策的安全、公平和稳定运行。
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np

from .technical_risk_controller import TechnicalRiskController, TechnicalRiskAssessment
from .data_security_controller import DataSecurityController, DataSecurityAssessment
from .algorithm_bias_controller import AlgorithmBiasController, BiasRiskAssessment
from .governance_conflict_controller import GovernanceConflictController, GovernanceConflictAssessment


class RiskCategory(Enum):
    """风险分类"""
    TECHNICAL = "technical"  # 技术风险
    DATA_SECURITY = "data_security"  # 数据安全风险
    ALGORITHM_BIAS = "algorithm_bias"  # 算法偏见风险
    GOVERNANCE_CONFLICT = "governance_conflict"  # 治理冲突风险


class SystemStatus(Enum):
    """系统状态"""
    NORMAL = "normal"  # 正常
    WARNING = "warning"  # 警告
    ALERT = "alert"  # 警报
    CRITICAL = "critical"  # 严重
    EMERGENCY = "emergency"  # 紧急


class AlertPriority(Enum):
    """警报优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """综合风险警报"""
    alert_id: str
    risk_category: RiskCategory
    priority: AlertPriority
    description: str
    risk_score: float
    affected_components: List[str]
    recommended_actions: List[str]
    timestamp: datetime
    expires_at: datetime


@dataclass
class SystemRiskReport:
    """系统风险报告"""
    report_id: str
    timestamp: datetime
    system_status: SystemStatus
    overall_risk_score: float
    risk_breakdown: Dict[RiskCategory, float]
    active_alerts: List[RiskAlert]
    emergency_actions_taken: List[str]
    recommendations: List[str]
    next_assessment_time: datetime


@dataclass
class EmergencyResponse:
    """应急响应"""
    response_id: str
    trigger_alert: RiskAlert
    actions_taken: List[str]
    response_status: str  # "executing", "completed", "failed"
    start_time: datetime
    completion_time: Optional[datetime]
    effectiveness_score: float


class AIRiskMonitoringSystem:
    """AI自主决策风险监控系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 初始化各风险控制器
        self.technical_controller = TechnicalRiskController()
        self.data_security_controller = DataSecurityController()
        self.algorithm_bias_controller = AlgorithmBiasController()
        self.governance_controller = GovernanceConflictController()
        
        # 系统状态
        self.system_status = SystemStatus.NORMAL
        self.active_alerts = []
        self.risk_history = []
        self.emergency_responses = []
        
        # 监控配置
        self.monitoring_interval = self.config["monitoring_interval_seconds"]
        self.alert_retention_days = self.config["alert_retention_days"]
        
        # 启动监控任务
        self.monitoring_task = None
        self.is_running = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitoring_interval_seconds": 60,  # 监控间隔
            "alert_retention_days": 30,  # 警报保留天数
            "risk_thresholds": {
                "normal": 0.3,
                "warning": 0.5,
                "alert": 0.7,
                "critical": 0.9
            },
            "emergency_response_enabled": True,
            "auto_mitigation_enabled": False,
            "report_generation_interval_hours": 24
        }
    
    async def start_monitoring(self):
        """启动风险监控"""
        if self.is_running:
            self.logger.warning("风险监控系统已在运行中")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("AI风险监控系统已启动")
    
    async def stop_monitoring(self):
        """停止风险监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("AI风险监控系统已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 执行综合风险评估
                risk_report = await self.perform_comprehensive_assessment()
                
                # 更新系统状态
                await self._update_system_status(risk_report)
                
                # 处理警报
                await self._process_alerts(risk_report)
                
                # 生成报告（如果达到间隔）
                await self._generate_periodic_report(risk_report)
                
                # 等待下一个监控周期
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环执行失败: {e}")
                await asyncio.sleep(10)  # 错误后短暂等待
    
    async def perform_comprehensive_assessment(self) -> SystemRiskReport:
        """执行综合风险评估"""
        try:
            # 收集系统数据
            system_data = await self._collect_system_data()
            
            # 并行执行各风险领域评估
            assessments = await asyncio.gather(
                self._assess_technical_risk(system_data),
                self._assess_data_security_risk(system_data),
                self._assess_algorithm_bias_risk(system_data),
                self._assess_governance_conflict_risk(system_data),
                return_exceptions=True
            )
            
            # 处理评估结果
            risk_breakdown = {}
            all_alerts = []
            
            for i, assessment in enumerate(assessments):
                category = list(RiskCategory)[i]
                
                if isinstance(assessment, Exception):
                    self.logger.error(f"{category.value}风险评估失败: {assessment}")
                    risk_breakdown[category] = 1.0  # 评估失败视为最高风险
                    all_alerts.append(self._create_assessment_error_alert(category, str(assessment)))
                else:
                    risk_score = self._extract_risk_score(assessment, category)
                    risk_breakdown[category] = risk_score
                    
                    # 转换警报格式
                    category_alerts = self._convert_to_system_alerts(assessment, category)
                    all_alerts.extend(category_alerts)
            
            # 计算综合风险评分
            overall_risk_score = self._calculate_overall_risk_score(risk_breakdown)
            
            # 确定系统状态
            system_status = self._determine_system_status(overall_risk_score)
            
            # 过滤有效警报
            active_alerts = self._filter_active_alerts(all_alerts)
            
            # 生成改进建议
            recommendations = self._generate_system_recommendations(risk_breakdown, active_alerts)
            
            # 记录应急响应
            emergency_actions = self._get_recent_emergency_actions()
            
            return SystemRiskReport(
                report_id=hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8],
                timestamp=datetime.utcnow(),
                system_status=system_status,
                overall_risk_score=overall_risk_score,
                risk_breakdown=risk_breakdown,
                active_alerts=active_alerts,
                emergency_actions_taken=emergency_actions,
                recommendations=recommendations,
                next_assessment_time=datetime.utcnow() + timedelta(seconds=self.monitoring_interval)
            )
            
        except Exception as e:
            self.logger.error(f"综合风险评估失败: {e}")
            return self._create_error_report(str(e))
    
    async def _collect_system_data(self) -> Dict[str, Any]:
        """收集系统数据"""
        # 简化实现：实际应用中应从各个系统组件收集数据
        return {
            "ai_decisions": await self._get_recent_ai_decisions(),
            "blockchain_data": await self._get_blockchain_state(),
            "system_metrics": await self._get_system_metrics(),
            "user_activity": await self._get_user_activity(),
            "security_logs": await self._get_security_logs()
        }
    
    async def _assess_technical_risk(self, system_data: Dict[str, Any]) -> TechnicalRiskAssessment:
        """评估技术风险"""
        ai_decisions = system_data.get("ai_decisions", [])
        blockchain_data = system_data.get("blockchain_data", {})
        system_metrics = system_data.get("system_metrics", {})
        
        return await self.technical_controller.assess_technical_risk(
            ai_decision_data={"decisions": ai_decisions},
            blockchain_context=blockchain_data,
            system_state=system_metrics
        )
    
    async def _assess_data_security_risk(self, system_data: Dict[str, Any]) -> DataSecurityAssessment:
        """评估数据安全风险"""
        training_data = system_data.get("training_data", {})
        model_params = system_data.get("model_parameters", {})
        blockchain_data = system_data.get("blockchain_data", {})
        access_logs = system_data.get("security_logs", [])
        
        return await self.data_security_controller.assess_data_security_risk(
            training_data=training_data,
            model_parameters=model_params,
            blockchain_context=blockchain_data,
            access_logs=access_logs
        )
    
    async def _assess_algorithm_bias_risk(self, system_data: Dict[str, Any]) -> BiasRiskAssessment:
        """评估算法偏见风险"""
        ai_decisions = system_data.get("ai_decisions", [])
        training_data = system_data.get("training_data", {})
        blockchain_data = system_data.get("blockchain_data", {})
        historical_data = system_data.get("historical_data", {})
        
        return await self.algorithm_bias_controller.assess_bias_risk(
            training_data=training_data,
            model_decisions=ai_decisions,
            blockchain_context=blockchain_data,
            historical_data=historical_data
        )
    
    async def _assess_governance_conflict_risk(self, system_data: Dict[str, Any]) -> GovernanceConflictAssessment:
        """评估治理冲突风险"""
        ai_decisions = system_data.get("ai_decisions", [])
        community_votes = system_data.get("community_votes", [])
        blockchain_governance = system_data.get("blockchain_data", {}).get("governance", {})
        system_state = system_data.get("system_metrics", {})
        
        return await self.governance_controller.assess_governance_conflict(
            ai_decisions=ai_decisions,
            community_votes=community_votes,
            blockchain_governance=blockchain_governance,
            system_state=system_state
        )
    
    def _extract_risk_score(self, assessment: Any, category: RiskCategory) -> float:
        """从评估结果中提取风险评分"""
        if category == RiskCategory.TECHNICAL and isinstance(assessment, TechnicalRiskAssessment):
            return assessment.risk_score
        elif category == RiskCategory.DATA_SECURITY and isinstance(assessment, DataSecurityAssessment):
            return 1.0 - assessment.security_score  # 安全评分转换为风险评分
        elif category == RiskCategory.ALGORITHM_BIAS and isinstance(assessment, BiasRiskAssessment):
            return 1.0 - assessment.fairness_score  # 公平性评分转换为风险评分
        elif category == RiskCategory.GOVERNANCE_CONFLICT and isinstance(assessment, GovernanceConflictAssessment):
            return 1.0 - assessment.collaboration_score  # 协作评分转换为风险评分
        else:
            return 1.0  # 未知类型返回最高风险
    
    def _convert_to_system_alerts(self, assessment: Any, category: RiskCategory) -> List[RiskAlert]:
        """将领域特定警报转换为系统警报"""
        alerts = []
        
        if category == RiskCategory.TECHNICAL and isinstance(assessment, TechnicalRiskAssessment):
            for alert in assessment.active_alerts:
                system_alert = RiskAlert(
                    alert_id=f"TECH_{alert.alert_id}",
                    risk_category=category,
                    priority=self._convert_severity_to_priority(alert.severity),
                    description=alert.description,
                    risk_score=alert.confidence_score,
                    affected_components=["AI决策引擎", "区块链系统"],
                    recommended_actions=alert.mitigation_action.split("，"),
                    timestamp=alert.timestamp,
                    expires_at=alert.timestamp + timedelta(hours=24)
                )
                alerts.append(system_alert)
        
        # 其他类别的转换逻辑类似...
        
        return alerts
    
    def _calculate_overall_risk_score(self, risk_breakdown: Dict[RiskCategory, float]) -> float:
        """计算综合风险评分"""
        if not risk_breakdown:
            return 0.0
        
        # 使用加权平均，技术风险权重最高
        weights = {
            RiskCategory.TECHNICAL: 0.4,
            RiskCategory.DATA_SECURITY: 0.3,
            RiskCategory.ALGORITHM_BIAS: 0.2,
            RiskCategory.GOVERNANCE_CONFLICT: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, score in risk_breakdown.items():
            weight = weights.get(category, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_system_status(self, overall_risk_score: float) -> SystemStatus:
        """根据风险评分确定系统状态"""
        thresholds = self.config["risk_thresholds"]
        
        if overall_risk_score >= thresholds["critical"]:
            return SystemStatus.EMERGENCY
        elif overall_risk_score >= thresholds["alert"]:
            return SystemStatus.CRITICAL
        elif overall_risk_score >= thresholds["warning"]:
            return SystemStatus.ALERT
        elif overall_risk_score >= thresholds["normal"]:
            return SystemStatus.WARNING
        else:
            return SystemStatus.NORMAL
    
    def _filter_active_alerts(self, all_alerts: List[RiskAlert]) -> List[RiskAlert]:
        """过滤有效警报"""
        current_time = datetime.utcnow()
        
        # 过滤未过期的警报
        active_alerts = [
            alert for alert in all_alerts 
            if alert.expires_at > current_time
        ]
        
        # 按优先级排序
        active_alerts.sort(key=lambda x: self._priority_to_value(x.priority), reverse=True)
        
        return active_alerts
    
    async def _update_system_status(self, risk_report: SystemRiskReport):
        """更新系统状态"""
        self.system_status = risk_report.system_status
        self.active_alerts = risk_report.active_alerts
        
        # 记录风险历史
        self.risk_history.append({
            "timestamp": risk_report.timestamp,
            "risk_score": risk_report.overall_risk_score,
            "status": risk_report.system_status.value
        })
        
        # 清理过时记录
        cutoff_time = datetime.utcnow() - timedelta(days=self.alert_retention_days)
        self.risk_history = [
            record for record in self.risk_history 
            if record["timestamp"] > cutoff_time
        ]
    
    async def _process_alerts(self, risk_report: SystemRiskReport):
        """处理警报"""
        # 检查是否需要应急响应
        critical_alerts = [
            alert for alert in risk_report.active_alerts 
            if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]
        ]
        
        if critical_alerts and self.config["emergency_response_enabled"]:
            await self._trigger_emergency_response(critical_alerts, risk_report)
        
        # 发送警报通知
        await self._send_alerts_notification(risk_report.active_alerts)
    
    async def _trigger_emergency_response(self, critical_alerts: List[RiskAlert], risk_report: SystemRiskReport):
        """触发应急响应"""
        response_id = f"EMERGENCY_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        emergency_response = EmergencyResponse(
            response_id=response_id,
            trigger_alert=critical_alerts[0],  # 以最高优先级警报为触发
            actions_taken=[],
            response_status="executing",
            start_time=datetime.utcnow(),
            completion_time=None,
            effectiveness_score=0.0
        )
        
        # 执行应急措施
        try:
            # 1. 暂停高风险AI决策
            if risk_report.system_status in [SystemStatus.CRITICAL, SystemStatus.EMERGENCY]:
                emergency_response.actions_taken.append("暂停高风险AI决策")
            
            # 2. 启用人工审核
            emergency_response.actions_taken.append("启用人工决策审核机制")
            
            # 3. 通知相关人员
            emergency_response.actions_taken.append("发送紧急通知给系统管理员")
            
            # 4. 记录应急响应
            emergency_response.actions_taken.append("记录应急响应过程")
            
            emergency_response.response_status = "completed"
            emergency_response.completion_time = datetime.utcnow()
            emergency_response.effectiveness_score = 0.8  # 假设效果良好
            
        except Exception as e:
            self.logger.error(f"应急响应执行失败: {e}")
            emergency_response.response_status = "failed"
            emergency_response.actions_taken.append(f"应急响应失败: {str(e)}")
        
        self.emergency_responses.append(emergency_response)
    
    async def _send_alerts_notification(self, alerts: List[RiskAlert]):
        """发送警报通知"""
        # 简化实现：实际应用中应集成通知系统
        if alerts:
            critical_count = len([a for a in alerts if a.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]])
            
            if critical_count > 0:
                self.logger.warning(f"检测到{critical_count}个高风险警报，请及时处理")
    
    async def _generate_periodic_report(self, risk_report: SystemRiskReport):
        """生成定期报告"""
        # 检查是否达到报告生成间隔
        last_report_time = getattr(self, '_last_report_time', None)
        report_interval = timedelta(hours=self.config["report_generation_interval_hours"])
        
        if last_report_time is None or datetime.utcnow() - last_report_time >= report_interval:
            await self._generate_detailed_report(risk_report)
            self._last_report_time = datetime.utcnow()
    
    async def _generate_detailed_report(self, risk_report: SystemRiskReport):
        """生成详细报告"""
        report_data = {
            "report_id": risk_report.report_id,
            "generated_at": risk_report.timestamp.isoformat(),
            "system_status": risk_report.system_status.value,
            "overall_risk_score": risk_report.overall_risk_score,
            "risk_breakdown": {k.value: v for k, v in risk_report.risk_breakdown.items()},
            "active_alerts_count": len(risk_report.active_alerts),
            "emergency_actions": risk_report.emergency_actions_taken,
            "recommendations": risk_report.recommendations
        }
        
        # 记录报告（实际应用中应保存到文件或数据库）
        self.logger.info(f"生成风险监控报告: {json.dumps(report_data, indent=2, ensure_ascii=False)}")
    
    def _generate_system_recommendations(self, 
                                       risk_breakdown: Dict[RiskCategory, float],
                                       active_alerts: List[RiskAlert]) -> List[str]:
        """生成系统改进建议"""
        recommendations = []
        
        # 根据风险分类提供建议
        for category, score in risk_breakdown.items():
            if score > 0.7:  # 高风险
                if category == RiskCategory.TECHNICAL:
                    recommendations.append("加强AI目标对齐验证和紧急停止机制")
                elif category == RiskCategory.DATA_SECURITY:
                    recommendations.append("强化数据加密和隐私保护措施")
                elif category == RiskCategory.ALGORITHM_BIAS:
                    recommendations.append("立即修正算法偏见和公平性问题")
                elif category == RiskCategory.GOVERNANCE_CONFLICT:
                    recommendations.append("优化人-AI协同决策机制")
        
        # 根据警报提供具体建议
        critical_alerts = [a for a in active_alerts if a.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]]
        if critical_alerts:
            recommendations.insert(0, "⚠️ 发现严重风险警报，建议立即处理")
        
        if not recommendations:
            recommendations.append("系统运行良好，继续保持监控")
        
        return recommendations
    
    def _get_recent_emergency_actions(self) -> List[str]:
        """获取最近的应急行动"""
        recent_responses = [
            r for r in self.emergency_responses 
            if r.start_time > datetime.utcnow() - timedelta(hours=24)
        ]
        
        actions = []
        for response in recent_responses:
            actions.extend(response.actions_taken)
        
        return actions[:5]  # 返回最近5个行动
    
    # 辅助方法
    def _convert_severity_to_priority(self, severity: Any) -> AlertPriority:
        """转换严重程度为警报优先级"""
        severity_mapping = {
            "critical": AlertPriority.CRITICAL,
            "high": AlertPriority.HIGH,
            "medium": AlertPriority.MEDIUM,
            "low": AlertPriority.LOW
        }
        
        return severity_mapping.get(severity.value if hasattr(severity, 'value') else severity, AlertPriority.MEDIUM)
    
    def _priority_to_value(self, priority: AlertPriority) -> int:
        """优先级转换为数值"""
        priority_values = {
            AlertPriority.CRITICAL: 4,
            AlertPriority.HIGH: 3,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 1
        }
        return priority_values.get(priority, 0)
    
    def _create_assessment_error_alert(self, category: RiskCategory, error: str) -> RiskAlert:
        """创建评估错误警报"""
        return RiskAlert(
            alert_id=f"ERROR_{category.value.upper()}",
            risk_category=category,
            priority=AlertPriority.HIGH,
            description=f"{category.value}风险评估失败",
            risk_score=1.0,
            affected_components=["风险监控系统"],
            recommended_actions=["检查评估模块运行状态", "查看详细错误日志"],
            timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
    
    def _create_error_report(self, error: str) -> SystemRiskReport:
        """创建错误报告"""
        return SystemRiskReport(
            report_id="ERROR",
            timestamp=datetime.utcnow(),
            system_status=SystemStatus.EMERGENCY,
            overall_risk_score=1.0,
            risk_breakdown={},
            active_alerts=[self._create_assessment_error_alert(RiskCategory.TECHNICAL, error)],
            emergency_actions_taken=["系统评估失败，启动紧急模式"],
            recommendations=["立即检查系统运行状态"],
            next_assessment_time=datetime.utcnow() + timedelta(minutes=5)
        )
    
    # 数据获取方法（简化实现）
    async def _get_recent_ai_decisions(self) -> List[Dict[str, Any]]:
        """获取最近的AI决策"""
        # 简化实现
        return [
            {
                "decision_id": f"decision_{i}",
                "timestamp": (datetime.utcnow() - timedelta(minutes=i*10)).isoformat(),
                "result": "approved" if i % 2 == 0 else "rejected",
                "confidence": 0.8 + i * 0.02,
                "risk_score": 0.1 + i * 0.05
            }
            for i in range(10)
        ]
    
    async def _get_blockchain_state(self) -> Dict[str, Any]:
        """获取区块链状态"""
        return {
            "block_height": 123456,
            "consensus_metrics": {
                "success_rate": 0.95,
                "stability_index": 0.88
            },
            "governance": {
                "proposal_count": 5,
                "voting_participation": 0.75
            }
        }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return {
            "cpu_usage": 0.65,
            "memory_usage": 0.72,
            "network_traffic": 1024,
            "active_connections": 150
        }
    
    async def _get_user_activity(self) -> Dict[str, Any]:
        """获取用户活动"""
        return {
            "active_users": 500,
            "transactions_per_hour": 1200,
            "average_session_duration": 1800
        }
    
    async def _get_security_logs(self) -> List[Dict[str, Any]]:
        """获取安全日志"""
        return [
            {
                "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "event_type": "access",
                "user_id": f"user_{i % 10}",
                "result": "success" if i % 3 != 0 else "failed"
            }
            for i in range(50)
        ]


# 系统启动和运行示例
async def main():
    """主函数示例"""
    # 创建风险监控系统
    risk_monitor = AIRiskMonitoringSystem()
    
    try:
        # 启动监控
        await risk_monitor.start_monitoring()
        
        # 运行一段时间（示例）
        await asyncio.sleep(300)  # 运行5分钟
        
        # 停止监控
        await risk_monitor.stop_monitoring()
        
    except KeyboardInterrupt:
        await risk_monitor.stop_monitoring()
    except Exception as e:
        logging.error(f"系统运行异常: {e}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行系统
    asyncio.run(main())