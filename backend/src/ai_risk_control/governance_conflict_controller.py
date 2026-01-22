"""
治理机制冲突解决模块

负责处理区块链去中心化治理与AI自主决策之间的权力冲突，
建立人-AI协同决策模式，防止治理失效和链上分叉风险。
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib


class GovernanceConflictType(Enum):
    """治理冲突类型"""
    POWER_STRUGGLE = "power_struggle"  # 权力争夺
    DECISION_CONFLICT = "decision_conflict"  # 决策冲突
    CONSENSUS_DISRUPTION = "consensus_disruption"  # 共识破坏
    GOVERNANCE_FAILURE = "governance_failure"  # 治理失效
    COMMUNITY_DIVISION = "community_division"  # 社区分裂


class ConflictSeverity(Enum):
    """冲突严重程度"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GovernanceConflictAlert:
    """治理冲突警报"""
    conflict_type: GovernanceConflictType
    severity: ConflictSeverity
    alert_id: str
    description: str
    conflict_details: Dict[str, Any]
    confidence_score: float
    resolution_action: str
    timestamp: datetime


@dataclass
class GovernanceConflictAssessment:
    """治理冲突评估结果"""
    overall_conflict_level: ConflictSeverity
    collaboration_score: float  # 0-1之间的协作评分
    active_alerts: List[GovernanceConflictAlert]
    consensus_stability: float
    community_engagement: float
    resolution_status: bool
    recommendations: List[str]


class GovernanceConflictController:
    """治理机制冲突解决控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 冲突监控状态
        self.conflict_history = []
        self.consensus_metrics = {}
        self.community_sentiment = {}
        
        # 冲突解决组件
        self.consensus_manager = ConsensusManager()
        self.community_mediator = CommunityMediator()
        self.human_ai_collaborator = HumanAICollaborator()
        
        # 治理审计器
        self.governance_auditor = GovernanceAuditor()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "power_struggle_threshold": 0.6,  # 权力争夺阈值
            "decision_conflict_threshold": 0.5,  # 决策冲突阈值
            "consensus_disruption_threshold": 0.4,  # 共识破坏阈值
            "governance_failure_threshold": 0.7,  # 治理失效阈值
            "community_division_threshold": 0.8,  # 社区分裂阈值
            "collaboration_target": 0.8,  # 协作目标
            "consensus_stability_target": 0.9,  # 共识稳定性目标
            "mediation_enabled": True,  # 是否启用调解
            "audit_frequency_hours": 24  # 审计频率
        }
    
    async def assess_governance_conflict(self,
                                       ai_decisions: List[Dict[str, Any]],
                                       community_votes: List[Dict[str, Any]],
                                       blockchain_governance: Dict[str, Any],
                                       system_state: Dict[str, Any]) -> GovernanceConflictAssessment:
        """
        评估治理机制冲突风险
        
        Args:
            ai_decisions: AI决策历史
            community_votes: 社区投票记录
            blockchain_governance: 区块链治理状态
            system_state: 系统状态
            
        Returns:
            GovernanceConflictAssessment: 治理冲突评估结果
        """
        try:
            alerts = []
            
            # 1. 权力争夺风险评估
            power_struggle_risk = await self._assess_power_struggle_risk(
                ai_decisions, community_votes, blockchain_governance)
            alerts.extend(power_struggle_risk)
            
            # 2. 决策冲突风险评估
            decision_conflict_risk = await self._assess_decision_conflict_risk(
                ai_decisions, community_votes)
            alerts.extend(decision_conflict_risk)
            
            # 3. 共识破坏风险评估
            consensus_disruption_risk = await self._assess_consensus_disruption_risk(
                blockchain_governance, system_state)
            alerts.extend(consensus_disruption_risk)
            
            # 4. 治理失效风险评估
            governance_failure_risk = await self._assess_governance_failure_risk(
                blockchain_governance, community_votes)
            alerts.extend(governance_failure_risk)
            
            # 5. 社区分裂风险评估
            community_division_risk = await self._assess_community_division_risk(
                community_votes, blockchain_governance)
            alerts.extend(community_division_risk)
            
            # 6. 综合冲突评估
            overall_conflict_level = self._determine_overall_conflict_level(alerts)
            collaboration_score = self._calculate_collaboration_score(alerts)
            
            # 7. 计算共识稳定性
            consensus_stability = await self._calculate_consensus_stability(blockchain_governance)
            
            # 8. 计算社区参与度
            community_engagement = await self._calculate_community_engagement(community_votes)
            
            # 9. 检查解决状态
            resolution_status = self._check_resolution_status(alerts, overall_conflict_level)
            
            # 10. 生成改进建议
            recommendations = self._generate_recommendations(alerts, overall_conflict_level)
            
            # 11. 更新冲突指标
            self._update_conflict_metrics(alerts, overall_conflict_level)
            
            return GovernanceConflictAssessment(
                overall_conflict_level=overall_conflict_level,
                collaboration_score=collaboration_score,
                active_alerts=alerts,
                consensus_stability=consensus_stability,
                community_engagement=community_engagement,
                resolution_status=resolution_status,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"治理冲突评估失败: {e}")
            # 返回最严格的冲突评估结果
            return GovernanceConflictAssessment(
                overall_conflict_level=ConflictSeverity.CRITICAL,
                collaboration_score=0.0,
                active_alerts=[GovernanceConflictAlert(
                    conflict_type=GovernanceConflictType.GOVERNANCE_FAILURE,
                    severity=ConflictSeverity.CRITICAL,
                    alert_id="ASSESSMENT_ERROR",
                    description="治理冲突评估过程出现异常",
                    conflict_details={"error": str(e)},
                    confidence_score=1.0,
                    resolution_action="立即启动应急治理机制",
                    timestamp=datetime.utcnow()
                )],
                consensus_stability=0.0,
                community_engagement=0.0,
                resolution_status=False,
                recommendations=["治理冲突评估失败，建议立即人工干预"]
            )
    
    async def _assess_power_struggle_risk(self,
                                        ai_decisions: List[Dict[str, Any]],
                                        community_votes: List[Dict[str, Any]],
                                        blockchain_governance: Dict[str, Any]) -> List[GovernanceConflictAlert]:
        """评估权力争夺风险"""
        alerts = []
        
        # 分析AI与社区决策权平衡
        power_balance_score = await self._analyze_power_balance(ai_decisions, community_votes)
        
        if power_balance_score < self.config["power_struggle_threshold"]:
            alerts.append(GovernanceConflictAlert(
                conflict_type=GovernanceConflictType.POWER_STRUGGLE,
                severity=ConflictSeverity.HIGH,
                alert_id="POWER_STRUGGLE_001",
                description="检测到AI与社区决策权失衡",
                conflict_details={
                    "power_balance_score": power_balance_score,
                    "ai_decision_ratio": len(ai_decisions) / max(len(community_votes), 1)
                },
                confidence_score=1.0 - power_balance_score,
                resolution_action="建立人-AI协同决策机制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_decision_conflict_risk(self,
                                           ai_decisions: List[Dict[str, Any]],
                                           community_votes: List[Dict[str, Any]]) -> List[GovernanceConflictAlert]:
        """评估决策冲突风险"""
        alerts = []
        
        # 分析AI决策与社区投票的冲突
        conflict_score = await self._analyze_decision_conflicts(ai_decisions, community_votes)
        
        if conflict_score > self.config["decision_conflict_threshold"]:
            alerts.append(GovernanceConflictAlert(
                conflict_type=GovernanceConflictType.DECISION_CONFLICT,
                severity=ConflictSeverity.MEDIUM,
                alert_id="DECISION_CONFLICT_001",
                description="检测到AI决策与社区意愿冲突",
                conflict_details={
                    "conflict_score": conflict_score,
                    "conflicting_decisions": self._identify_conflicting_decisions(ai_decisions, community_votes)
                },
                confidence_score=conflict_score,
                resolution_action="启动决策协调机制",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_consensus_disruption_risk(self,
                                              blockchain_governance: Dict[str, Any],
                                              system_state: Dict[str, Any]) -> List[GovernanceConflictAlert]:
        """评估共识破坏风险"""
        alerts = []
        
        # 分析共识机制稳定性
        disruption_risk = await self.consensus_manager.assess_consensus_risk(blockchain_governance, system_state)
        
        if disruption_risk > self.config["consensus_disruption_threshold"]:
            alerts.append(GovernanceConflictAlert(
                conflict_type=GovernanceConflictType.CONSENSUS_DISRUPTION,
                severity=ConflictSeverity.CRITICAL,
                alert_id="CONSENSUS_DISRUPTION_001",
                description="检测到共识机制破坏风险",
                conflict_details={
                    "disruption_risk": disruption_risk,
                    "consensus_metrics": blockchain_governance.get("consensus_metrics", {})
                },
                confidence_score=disruption_risk,
                resolution_action="加强共识机制安全防护",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_governance_failure_risk(self,
                                            blockchain_governance: Dict[str, Any],
                                            community_votes: List[Dict[str, Any]]) -> List[GovernanceConflictAlert]:
        """评估治理失效风险"""
        alerts = []
        
        # 分析治理机制有效性
        failure_risk = await self.governance_auditor.assess_governance_health(blockchain_governance, community_votes)
        
        if failure_risk > self.config["governance_failure_threshold"]:
            alerts.append(GovernanceConflictAlert(
                conflict_type=GovernanceConflictType.GOVERNANCE_FAILURE,
                severity=ConflictSeverity.CRITICAL,
                alert_id="GOVERNANCE_FAILURE_001",
                description="检测到治理机制失效风险",
                conflict_details={
                    "failure_risk": failure_risk,
                    "governance_indicators": blockchain_governance.get("governance_indicators", {})
                },
                confidence_score=failure_risk,
                resolution_action="启动治理机制改革",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _assess_community_division_risk(self,
                                            community_votes: List[Dict[str, Any]],
                                            blockchain_governance: Dict[str, Any]) -> List[GovernanceConflictAlert]:
        """评估社区分裂风险"""
        alerts = []
        
        # 分析社区意见分歧
        division_risk = await self.community_mediator.assess_community_division(community_votes, blockchain_governance)
        
        if division_risk > self.config["community_division_threshold"]:
            alerts.append(GovernanceConflictAlert(
                conflict_type=GovernanceConflictType.COMMUNITY_DIVISION,
                severity=ConflictSeverity.HIGH,
                alert_id="COMMUNITY_DIVISION_001",
                description="检测到社区意见严重分歧",
                conflict_details={
                    "division_risk": division_risk,
                    "vote_distribution": self._analyze_vote_distribution(community_votes)
                },
                confidence_score=division_risk,
                resolution_action="启动社区调解和共识建设",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _determine_overall_conflict_level(self, alerts: List[GovernanceConflictAlert]) -> ConflictSeverity:
        """确定总体冲突等级"""
        if not alerts:
            return ConflictSeverity.LOW
        
        severities = [alert.severity for alert in alerts]
        
        if ConflictSeverity.CRITICAL in severities:
            return ConflictSeverity.CRITICAL
        elif ConflictSeverity.HIGH in severities:
            return ConflictSeverity.HIGH
        elif ConflictSeverity.MEDIUM in severities:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW
    
    def _calculate_collaboration_score(self, alerts: List[GovernanceConflictAlert]) -> float:
        """计算协作评分"""
        if not alerts:
            return 1.0
        
        severity_weights = {
            ConflictSeverity.CRITICAL: 0.0,
            ConflictSeverity.HIGH: 0.3,
            ConflictSeverity.MEDIUM: 0.6,
            ConflictSeverity.LOW: 0.9
        }
        
        total_score = sum(
            severity_weights[alert.severity] * (1.0 - alert.confidence_score)
            for alert in alerts
        )
        
        return total_score / len(alerts) if alerts else 1.0
    
    async def _calculate_consensus_stability(self, blockchain_governance: Dict[str, Any]) -> float:
        """计算共识稳定性"""
        consensus_metrics = blockchain_governance.get("consensus_metrics", {})
        
        # 分析共识达成率和稳定性
        consensus_rate = consensus_metrics.get("success_rate", 0.9)
        stability_index = consensus_metrics.get("stability_index", 0.8)
        
        return (consensus_rate + stability_index) / 2.0
    
    async def _calculate_community_engagement(self, community_votes: List[Dict[str, Any]]) -> float:
        """计算社区参与度"""
        if not community_votes:
            return 0.0
        
        # 分析投票参与率和活跃度
        total_voters = len(set(vote.get("voter_id") for vote in community_votes))
        recent_votes = [vote for vote in community_votes 
                       if datetime.fromisoformat(vote["timestamp"]) > datetime.utcnow() - timedelta(days=7)]
        
        engagement_score = min(len(recent_votes) / 100.0, 1.0)  # 假设正常参与水平
        return engagement_score
    
    def _check_resolution_status(self, alerts: List[GovernanceConflictAlert], conflict_level: ConflictSeverity) -> bool:
        """检查解决状态"""
        if conflict_level == ConflictSeverity.LOW:
            return True
        
        # 检查是否有正在进行的冲突解决
        active_resolutions = [alert for alert in alerts if alert.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]]
        
        return len(active_resolutions) == 0
    
    def _generate_recommendations(self, 
                                alerts: List[GovernanceConflictAlert],
                                overall_conflict_level: ConflictSeverity) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_conflict_level == ConflictSeverity.LOW:
            recommendations.append("治理机制运行良好，继续保持监控")
            return recommendations
        
        # 根据冲突类型提供针对性建议
        conflict_types = set(alert.conflict_type for alert in alerts)
        
        if GovernanceConflictType.POWER_STRUGGLE in conflict_types:
            recommendations.append("建议建立人-AI协同决策机制")
        
        if GovernanceConflictType.DECISION_CONFLICT in conflict_types:
            recommendations.append("建议启动决策协调和冲突解决机制")
        
        if GovernanceConflictType.CONSENSUS_DISRUPTION in conflict_types:
            recommendations.append("必须加强共识机制安全防护")
        
        if GovernanceConflictType.GOVERNANCE_FAILURE in conflict_types:
            recommendations.append("建议启动治理机制改革")
        
        if GovernanceConflictType.COMMUNITY_DIVISION in conflict_types:
            recommendations.append("建议启动社区调解和共识建设")
        
        # 紧急情况建议
        if overall_conflict_level in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
            recommendations.insert(0, "⚖️ 高治理冲突风险：建议立即启动应急治理机制")
        
        return recommendations
    
    def _update_conflict_metrics(self, alerts: List[GovernanceConflictAlert], conflict_level: ConflictSeverity):
        """更新冲突指标"""
        current_time = datetime.utcnow()
        
        # 记录冲突历史
        self.conflict_history.extend(alerts)
        
        # 清理过时记录
        cutoff_time = current_time - timedelta(days=30)
        self.conflict_history = [
            alert for alert in self.conflict_history 
            if alert.timestamp > cutoff_time
        ]
        
        # 更新共识指标
        self.consensus_metrics["last_assessment"] = current_time
        self.consensus_metrics["current_conflict_level"] = conflict_level
    
    # 辅助方法
    async def _analyze_power_balance(self,
                                   ai_decisions: List[Dict[str, Any]],
                                   community_votes: List[Dict[str, Any]]) -> float:
        """分析权力平衡"""
        ai_decision_weight = len(ai_decisions)
        community_decision_weight = len(community_votes)
        
        if ai_decision_weight + community_decision_weight == 0:
            return 0.5  # 无决策时返回中性值
        
        # 计算AI决策权重比例
        ai_weight_ratio = ai_decision_weight / (ai_decision_weight + community_decision_weight)
        
        # 理想平衡点在0.3-0.7之间（AI建议，人类决策）
        if 0.3 <= ai_weight_ratio <= 0.7:
            return 1.0
        else:
            # 偏离理想平衡越远，分数越低
            deviation = min(abs(ai_weight_ratio - 0.3), abs(ai_weight_ratio - 0.7))
            return 1.0 - deviation * 2
    
    async def _analyze_decision_conflicts(self,
                                        ai_decisions: List[Dict[str, Any]],
                                        community_votes: List[Dict[str, Any]]) -> float:
        """分析决策冲突"""
        if not ai_decisions or not community_votes:
            return 0.0
        
        # 匹配时间相近的AI决策和社区投票
        conflict_count = 0
        total_comparisons = 0
        
        for ai_decision in ai_decisions[-10:]:  # 分析最近10个决策
            ai_timestamp = ai_decision.get("timestamp", datetime.utcnow())
            if isinstance(ai_timestamp, str):
                ai_timestamp = datetime.fromisoformat(ai_timestamp)
            
            # 查找时间相近的社区投票
            related_votes = [
                vote for vote in community_votes
                if abs((datetime.fromisoformat(vote["timestamp"]) - ai_timestamp).total_seconds()) < 3600  # 1小时内
            ]
            
            for vote in related_votes:
                total_comparisons += 1
                ai_result = ai_decision.get("result", "")
                vote_result = vote.get("vote_result", "")
                
                if ai_result != vote_result:
                    conflict_count += 1
        
        if total_comparisons > 0:
            return conflict_count / total_comparisons
        
        return 0.0
    
    def _identify_conflicting_decisions(self,
                                      ai_decisions: List[Dict[str, Any]],
                                      community_votes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别冲突决策"""
        conflicts = []
        
        for ai_decision in ai_decisions[-5:]:  # 分析最近5个决策
            ai_timestamp = ai_decision.get("timestamp", datetime.utcnow())
            if isinstance(ai_timestamp, str):
                ai_timestamp = datetime.fromisoformat(ai_timestamp)
            
            # 查找时间相近的冲突投票
            conflicting_votes = [
                vote for vote in community_votes
                if abs((datetime.fromisoformat(vote["timestamp"]) - ai_timestamp).total_seconds()) < 3600
                and vote.get("vote_result") != ai_decision.get("result")
            ]
            
            if conflicting_votes:
                conflicts.append({
                    "ai_decision": ai_decision,
                    "conflicting_votes": conflicting_votes
                })
        
        return conflicts
    
    def _analyze_vote_distribution(self, community_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析投票分布"""
        if not community_votes:
            return {}
        
        # 统计投票结果分布
        vote_results = {}
        for vote in community_votes:
            result = vote.get("vote_result", "abstain")
            vote_results[result] = vote_results.get(result, 0) + 1
        
        total_votes = len(community_votes)
        distribution = {result: count/total_votes for result, count in vote_results.items()}
        
        return {
            "total_votes": total_votes,
            "distribution": distribution,
            "majority_opinion": max(distribution.items(), key=lambda x: x[1])[0] if distribution else "unknown"
        }


class ConsensusManager:
    """共识管理器"""
    
    async def assess_consensus_risk(self,
                                  blockchain_governance: Dict[str, Any],
                                  system_state: Dict[str, Any]) -> float:
        """评估共识风险"""
        risk_factors = []
        
        # 1. 共识达成率风险
        consensus_rate = blockchain_governance.get("consensus_metrics", {}).get("success_rate", 0.9)
        risk_factors.append(1.0 - consensus_rate)
        
        # 2. 节点参与度风险
        participation_rate = blockchain_governance.get("node_participation", 0.8)
        risk_factors.append(1.0 - participation_rate)
        
        # 3. 算力集中度风险
        computing_power_concentration = system_state.get("computing_power_concentration", 0.3)
        risk_factors.append(computing_power_concentration)
        
        return np.mean(risk_factors) if risk_factors else 0.0


class CommunityMediator:
    """社区调解器"""
    
    async def assess_community_division(self,
                                      community_votes: List[Dict[str, Any]],
                                      blockchain_governance: Dict[str, Any]) -> float:
        """评估社区分裂风险"""
        if not community_votes:
            return 0.0
        
        # 分析投票意见分歧
        vote_results = [vote.get("vote_result") for vote in community_votes]
        
        if not vote_results:
            return 0.0
        
        # 计算投票结果的熵（衡量不确定性/分歧度）
        result_counts = {}
        for result in vote_results:
            result_counts[result] = result_counts.get(result, 0) + 1
        
        total_votes = len(vote_results)
        entropy = 0.0
        for count in result_counts.values():
            probability = count / total_votes
            entropy -= probability * np.log2(probability)
        
        # 归一化熵值到0-1范围
        max_entropy = np.log2(len(result_counts)) if result_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy


class HumanAICollaborator:
    """人-AI协作器"""
    
    async def facilitate_collaboration(self,
                                     ai_recommendations: List[Dict[str, Any]],
                                     human_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """促进人-AI协作"""
        collaboration_result = {
            "collaboration_score": 0.8,  # 假设基本协作良好
            "conflict_resolution": "automated",
            "decision_finalization": "human_override"
        }
        
        return collaboration_result


class GovernanceAuditor:
    """治理审计器"""
    
    async def assess_governance_health(self,
                                     blockchain_governance: Dict[str, Any],
                                     community_votes: List[Dict[str, Any]]) -> float:
        """评估治理健康度"""
        health_indicators = []
        
        # 1. 决策效率
        decision_efficiency = blockchain_governance.get("decision_efficiency", 0.7)
        health_indicators.append(decision_efficiency)
        
        # 2. 社区参与度
        community_engagement = len(community_votes) / 100.0  # 简化计算
        health_indicators.append(min(community_engagement, 1.0))
        
        # 3. 治理透明度
        transparency = blockchain_governance.get("transparency_index", 0.8)
        health_indicators.append(transparency)
        
        # 治理失效风险 = 1 - 健康度
        health_score = np.mean(health_indicators) if health_indicators else 0.5
        failure_risk = 1.0 - health_score
        
        return failure_risk