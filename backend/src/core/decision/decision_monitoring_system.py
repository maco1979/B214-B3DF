"""
AI决策效果监控和反馈系统 - 监控所有决策模块的性能和效果
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class DecisionModule(Enum):
    """决策模块枚举"""
    AGRICULTURE = "agriculture"
    BLOCKCHAIN = "blockchain"
    MODEL_TRAINING = "model_training"
    RESOURCE_ALLOCATION = "resource_allocation"


class DecisionPerformance(Enum):
    """决策性能等级"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"          # 良好
    FAIR = "fair"          # 一般
    POOR = "poor"          # 较差
    CRITICAL = "critical"  # 危急


@dataclass
class DecisionRecord:
    """决策记录"""
    decision_id: str
    module: DecisionModule
    action: str
    parameters: Dict[str, float]
    expected_reward: float
    actual_reward: Optional[float] = None
    confidence: float = 0.0
    execution_time: float = 0.0
    risk_assessment: Optional[Dict[str, float]] = None
    timestamp: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        result["module"] = self.module.value
        return result


@dataclass
class PerformanceMetrics:
    """性能指标"""
    module: DecisionModule
    total_decisions: int = 0
    successful_decisions: int = 0
    average_reward: float = 0.0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    recent_success_rate: float = 0.0
    risk_control_effectiveness: float = 0.0
    performance_level: DecisionPerformance = DecisionPerformance.FAIR
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result["module"] = self.module.value
        result["performance_level"] = self.performance_level.value
        return result


class DecisionMonitoringSystem:
    """AI决策效果监控和反馈系统"""
    
    def __init__(self):
        # 决策记录存储
        self.decision_records: List[DecisionRecord] = []
        
        # 性能指标
        self.performance_metrics: Dict[DecisionModule, PerformanceMetrics] = {}
        
        # 预警阈值
        self.alert_thresholds = {
            "success_rate_low": 0.6,      # 成功率低于60%预警
            "average_reward_low": 50.0,   # 平均奖励低于50预警
            "confidence_low": 0.5,        # 置信度低于0.5预警
            "execution_time_high": 0.5    # 执行时间超过0.5秒预警
        }
        
        # 反馈机制配置
        self.feedback_config = {
            "learning_rate": 0.1,         # 学习率
            "exploration_rate": 0.1,      # 探索率
            "forgetting_factor": 0.95,   # 遗忘因子
            "update_frequency": 60       # 更新频率（秒）
        }
        
        # 初始化性能指标
        for module in DecisionModule:
            self.performance_metrics[module] = PerformanceMetrics(module=module)
        
        self.last_update_time = datetime.now()
    
    def record_decision(self, 
                       decision_id: str,
                       module: DecisionModule,
                       action: str,
                       parameters: Dict[str, float],
                       expected_reward: float,
                       confidence: float,
                       execution_time: float,
                       risk_assessment: Optional[Dict[str, float]] = None) -> str:
        """
        记录决策
        
        Args:
            decision_id: 决策ID
            module: 决策模块
            action: 执行动作
            parameters: 动作参数
            expected_reward: 预期奖励
            confidence: 置信度
            execution_time: 执行时间
            risk_assessment: 风险评估
            
        Returns:
            记录ID
        """
        record = DecisionRecord(
            decision_id=decision_id,
            module=module,
            action=action,
            parameters=parameters,
            expected_reward=expected_reward,
            confidence=confidence,
            execution_time=execution_time,
            risk_assessment=risk_assessment,
            timestamp=datetime.now()
        )
        
        self.decision_records.append(record)
        
        # 保持记录数量，避免内存溢出
        if len(self.decision_records) > 10000:
            self.decision_records = self.decision_records[-5000:]
        
        logger.info(f"记录决策: {module.value} - {action} (ID: {decision_id})")
        return decision_id
    
    def update_decision_result(self, 
                             decision_id: str, 
                             actual_reward: float) -> bool:
        """
        更新决策结果
        
        Args:
            decision_id: 决策ID
            actual_reward: 实际获得的奖励
            
        Returns:
            是否成功更新
        """
        for record in self.decision_records:
            if record.decision_id == decision_id:
                record.actual_reward = actual_reward
                logger.info(f"更新决策结果: {decision_id}, 实际奖励: {actual_reward}")
                return True
        
        logger.warning(f"未找到决策记录: {decision_id}")
        return False
    
    def calculate_performance_metrics(self, module: DecisionModule) -> PerformanceMetrics:
        """
        计算指定模块的性能指标
        
        Args:
            module: 决策模块
            
        Returns:
            性能指标
        """
        # 获取该模块的决策记录
        module_records = [r for r in self.decision_records if r.module == module]
        
        if not module_records:
            return PerformanceMetrics(module=module)
        
        # 基础指标计算
        total_decisions = len(module_records)
        
        # 成功决策（实际奖励大于预期奖励的80%）
        successful_decisions = len([r for r in module_records 
                                  if r.actual_reward and r.actual_reward >= r.expected_reward * 0.8])
        
        # 平均指标
        avg_reward = np.mean([r.actual_reward or 0.0 for r in module_records])
        avg_confidence = np.mean([r.confidence for r in module_records])
        avg_execution_time = np.mean([r.execution_time for r in module_records])
        
        # 最近10次决策的成功率
        recent_records = module_records[-10:]
        recent_successful = len([r for r in recent_records 
                               if r.actual_reward and r.actual_reward >= r.expected_reward * 0.8])
        recent_success_rate = recent_successful / len(recent_records) if recent_records else 0.0
        
        # 风险控制效果
        risk_records = [r for r in module_records if r.risk_assessment]
        if risk_records:
            risk_effectiveness = np.mean([
                r.actual_reward / r.expected_reward if r.actual_reward and r.expected_reward > 0 else 0.0
                for r in risk_records
            ])
        else:
            risk_effectiveness = 0.0
        
        # 确定性能等级
        performance_level = self._determine_performance_level(
            success_rate=successful_decisions / total_decisions if total_decisions > 0 else 0.0,
            average_reward=avg_reward,
            average_confidence=avg_confidence,
            execution_time=avg_execution_time
        )
        
        metrics = PerformanceMetrics(
            module=module,
            total_decisions=total_decisions,
            successful_decisions=successful_decisions,
            average_reward=float(avg_reward),
            average_confidence=float(avg_confidence),
            average_execution_time=float(avg_execution_time),
            recent_success_rate=float(recent_success_rate),
            risk_control_effectiveness=float(risk_effectiveness),
            performance_level=performance_level
        )
        
        return metrics
    
    def _determine_performance_level(self, 
                                   success_rate: float, 
                                   average_reward: float,
                                   average_confidence: float,
                                   execution_time: float) -> DecisionPerformance:
        """确定性能等级"""
        score = 0.0
        
        # 成功率权重40%
        score += success_rate * 0.4
        
        # 平均奖励权重30%（归一化到0-1）
        reward_score = min(average_reward / 100.0, 1.0)
        score += reward_score * 0.3
        
        # 置信度权重20%
        score += average_confidence * 0.2
        
        # 执行时间权重10%（执行时间越短越好）
        time_score = max(0, 1.0 - min(execution_time / 1.0, 1.0))
        score += time_score * 0.1
        
        # 根据总分确定等级
        if score >= 0.9:
            return DecisionPerformance.EXCELLENT
        elif score >= 0.75:
            return DecisionPerformance.GOOD
        elif score >= 0.6:
            return DecisionPerformance.FAIR
        elif score >= 0.4:
            return DecisionPerformance.POOR
        else:
            return DecisionPerformance.CRITICAL
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        检查预警条件
        
        Returns:
            预警列表
        """
        alerts = []
        
        # 更新所有模块的性能指标
        for module in DecisionModule:
            metrics = self.calculate_performance_metrics(module)
            self.performance_metrics[module] = metrics
            
            # 检查预警条件
            if metrics.total_decisions > 10:  # 有足够数据才检查预警
                alert = self._check_module_alerts(module, metrics)
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def _check_module_alerts(self, module: DecisionModule, metrics: PerformanceMetrics) -> Optional[Dict[str, Any]]:
        """检查单个模块的预警条件"""
        alerts = []
        
        success_rate = metrics.successful_decisions / metrics.total_decisions
        
        if success_rate < self.alert_thresholds["success_rate_low"]:
            alerts.append({
                "type": "low_success_rate",
                "message": f"{module.value}模块决策成功率偏低: {success_rate:.2%}",
                "severity": "warning" if success_rate > 0.4 else "critical"
            })
        
        if metrics.average_reward < self.alert_thresholds["average_reward_low"]:
            alerts.append({
                "type": "low_average_reward",
                "message": f"{module.value}模块平均奖励偏低: {metrics.average_reward:.2f}",
                "severity": "warning"
            })
        
        if metrics.average_confidence < self.alert_thresholds["confidence_low"]:
            alerts.append({
                "type": "low_confidence",
                "message": f"{module.value}模块平均置信度偏低: {metrics.average_confidence:.2f}",
                "severity": "warning"
            })
        
        if metrics.average_execution_time > self.alert_thresholds["execution_time_high"]:
            alerts.append({
                "type": "high_execution_time",
                "message": f"{module.value}模块平均执行时间偏高: {metrics.average_execution_time:.3f}s",
                "severity": "warning"
            })
        
        if metrics.performance_level == DecisionPerformance.CRITICAL:
            alerts.append({
                "type": "critical_performance",
                "message": f"{module.value}模块性能处于危急状态",
                "severity": "critical"
            })
        
        if alerts:
            return {
                "module": module.value,
                "performance_level": metrics.performance_level.value,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def generate_feedback(self, module: DecisionModule) -> Dict[str, Any]:
        """
        为指定模块生成反馈建议
        
        Args:
            module: 决策模块
            
        Returns:
            反馈建议
        """
        metrics = self.performance_metrics[module]
        
        feedback = {
            "module": module.value,
            "performance_level": metrics.performance_level.value,
            "suggestions": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 根据性能等级生成建议
        if metrics.performance_level == DecisionPerformance.CRITICAL:
            feedback["suggestions"].extend([
                "立即停止使用当前决策策略",
                "重新初始化强化学习模型",
                "增加探索率以寻找更好的策略",
                "检查输入数据的质量和完整性"
            ])
            feedback["recommendations"].append("紧急人工干预")
        
        elif metrics.performance_level == DecisionPerformance.POOR:
            feedback["suggestions"].extend([
                "增加训练数据量",
                "调整奖励函数权重",
                "增加风险控制机制的严格度",
                "优化状态特征提取方法"
            ])
            feedback["recommendations"].append("需要优化决策策略")
        
        elif metrics.performance_level == DecisionPerformance.FAIR:
            feedback["suggestions"].extend([
                "微调超参数",
                "增加决策历史记录长度",
                "优化动作选择策略",
                "加强监控和反馈机制"
            ])
            feedback["recommendations"].append("持续优化和监控")
        
        elif metrics.performance_level == DecisionPerformance.GOOD:
            feedback["suggestions"].extend([
                "保持当前策略",
                "定期评估性能变化",
                "探索新的优化方向",
                "扩展决策范围"
            ])
            feedback["recommendations"].append("稳定运行，持续改进")
        
        else:  # EXCELLENT
            feedback["suggestions"].extend([
                "分享成功经验",
                "扩展到其他领域",
                "优化实时性能",
                "研究更先进的算法"
            ])
            feedback["recommendations"].append("优秀表现，可作为基准")
        
        # 具体指标改进建议
        if metrics.average_reward < 50:
            feedback["suggestions"].append("重新设计奖励函数以提高奖励值")
        
        if metrics.average_confidence < 0.6:
            feedback["suggestions"].append("改进状态特征表示以提高决策置信度")
        
        if metrics.average_execution_time > 0.3:
            feedback["suggestions"].append("优化算法实现以减少执行时间")
        
        return feedback
    
    def get_decision_analytics(self, 
                             module: Optional[DecisionModule] = None,
                             time_range_hours: int = 24) -> Dict[str, Any]:
        """
        获取决策分析数据
        
        Args:
            module: 指定模块（None表示所有模块）
            time_range_hours: 时间范围（小时）
            
        Returns:
            分析数据
        """
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        # 筛选指定时间范围内的记录
        filtered_records = [r for r in self.decision_records 
                          if r.timestamp >= cutoff_time]
        
        if module:
            filtered_records = [r for r in filtered_records if r.module == module]
        
        # 基本统计
        total_decisions = len(filtered_records)
        
        if total_decisions == 0:
            return {
                "total_decisions": 0,
                "time_range_hours": time_range_hours,
                "module": module.value if module else "all",
                "message": "指定时间段内无决策记录"
            }
        
        # 按模块分组统计
        module_stats = {}
        for record in filtered_records:
            module_name = record.module.value
            if module_name not in module_stats:
                module_stats[module_name] = {
                    "count": 0,
                    "total_reward": 0.0,
                    "successful_decisions": 0,
                    "actions": {}
                }
            
            stats = module_stats[module_name]
            stats["count"] += 1
            stats["total_reward"] += record.actual_reward or 0.0
            
            if record.actual_reward and record.actual_reward >= record.expected_reward * 0.8:
                stats["successful_decisions"] += 1
            
            # 动作统计
            action = record.action
            if action not in stats["actions"]:
                stats["actions"][action] = 0
            stats["actions"][action] += 1
        
        # 计算指标
        for module_name, stats in module_stats.items():
            stats["success_rate"] = stats["successful_decisions"] / stats["count"]
            stats["average_reward"] = stats["total_reward"] / stats["count"]
            
            # 动作分布
            total_actions = sum(stats["actions"].values())
            stats["action_distribution"] = {
                action: count / total_actions 
                for action, count in stats["actions"].items()
            }
        
        return {
            "total_decisions": total_decisions,
            "time_range_hours": time_range_hours,
            "module": module.value if module else "all",
            "module_statistics": module_stats,
            "performance_metrics": {
                m.value: self.calculate_performance_metrics(m).to_dict()
                for m in DecisionModule
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def export_performance_report(self) -> Dict[str, Any]:
        """
        导出性能报告
        
        Returns:
            性能报告
        """
        # 计算所有模块的性能指标
        performance_data = {}
        for module in DecisionModule:
            metrics = self.calculate_performance_metrics(module)
            performance_data[module.value] = metrics.to_dict()
        
        # 检查预警
        alerts = self.check_alerts()
        
        # 生成反馈建议
        feedback = {}
        for module in DecisionModule:
            feedback[module.value] = self.generate_feedback(module)
        
        return {
            "report_id": f"performance_report_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_decisions": len(self.decision_records),
                "modules_performance": {m.value: p["performance_level"] for m, p in performance_data.items()},
                "critical_alerts": len([a for a in alerts if any(alert["severity"] == "critical" for alert in a.get("alerts", []))])
            },
            "performance_data": performance_data,
            "alerts": alerts,
            "feedback": feedback,
            "analytics": self.get_decision_analytics(time_range_hours=24)
        }
    
    def auto_optimize(self) -> Dict[str, Any]:
        """
        自动优化决策系统
        
        Returns:
            优化结果
        """
        optimization_results = {}
        
        for module in DecisionModule:
            metrics = self.performance_metrics[module]
            
            # 根据性能等级自动调整参数
            if metrics.performance_level in [DecisionPerformance.POOR, DecisionPerformance.CRITICAL]:
                # 性能较差，需要激进优化
                optimization_results[module.value] = {
                    "action": "aggressive_optimization",
                    "suggestions": [
                        "重置策略网络参数",
                        "增加探索率",
                        "降低风险阈值",
                        "增加训练数据多样性"
                    ],
                    "parameters": {
                        "learning_rate": 0.2,
                        "exploration_rate": 0.3,
                        "risk_threshold_multiplier": 0.8
                    }
                }
            elif metrics.performance_level == DecisionPerformance.FAIR:
                # 性能一般，适度优化
                optimization_results[module.value] = {
                    "action": "moderate_optimization",
                    "suggestions": [
                        "微调学习率",
                        "优化状态特征",
                        "调整奖励权重"
                    ],
                    "parameters": {
                        "learning_rate": 0.15,
                        "exploration_rate": 0.15,
                        "risk_threshold_multiplier": 1.0
                    }
                }
            else:
                # 性能良好或优秀，保持稳定
                optimization_results[module.value] = {
                    "action": "maintain_stability",
                    "suggestions": [
                        "保持当前参数",
                        "持续监控性能",
                        "定期评估效果"
                    ],
                    "parameters": {
                        "learning_rate": 0.1,
                        "exploration_rate": 0.1,
                        "risk_threshold_multiplier": 1.0
                    }
                }
        
        return {
            "optimization_id": f"auto_optimize_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "results": optimization_results
        }


# 全局监控系统实例
monitoring_system = DecisionMonitoringSystem()