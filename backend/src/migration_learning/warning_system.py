"""
风险预警系统模块

负责迁移学习过程中的风险监控和预警，包括：
- 实时风险监控
- 多级预警机制
- 预警信息推送
- 预警历史记录
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import deque


class WarningLevel(Enum):
    """预警级别"""
    INFO = "info"  # 信息
    LOW = "low"  # 低风险
    MEDIUM = "medium"  # 中风险
    HIGH = "high"  # 高风险
    CRITICAL = "critical"  # 严重风险


class WarningType(Enum):
    """预警类型"""
    DATA_QUALITY = "data_quality"  # 数据质量
    MODEL_COMPATIBILITY = "model_compatibility"  # 模型兼容性
    DOMAIN_GAP = "domain_gap"  # 领域差距
    RESOURCE_CONSTRAINTS = "resource_constraints"  # 资源约束
    PERFORMANCE_DEGRADATION = "performance_degradation"  # 性能下降
    SAFETY_VIOLATION = "safety_violation"  # 安全违规


@dataclass
class WarningMessage:
    """预警消息"""
    warning_id: str
    warning_type: WarningType
    warning_level: WarningLevel
    title: str
    description: str
    timestamp: datetime
    source: str
    affected_components: List[str]
    risk_score: float
    mitigation_measures: List[str]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class WarningStatistics:
    """预警统计信息"""
    total_warnings: int
    active_warnings: int
    warning_distribution: Dict[WarningLevel, int]
    type_distribution: Dict[WarningType, int]
    recent_trend: str  # increasing, decreasing, stable
    average_response_time: timedelta


class RiskWarningSystem:
    """风险预警系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 预警存储
        self.warning_history: deque[WarningMessage] = deque(maxlen=1000)
        self.active_warnings: Dict[str, WarningMessage] = {}
        
        # 预警阈值配置
        self.warning_thresholds = self._setup_warning_thresholds()
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 预警订阅者
        self.subscribers: List[callable] = []
        
        # 统计信息
        self.statistics = self._initialize_statistics()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitoring_interval": 60,  # 监控间隔(秒)
            "warning_retention_days": 30,  # 预警保留天数
            "escalation_timeout": 3600,  # 预警升级超时(秒)
            "acknowledgement_required": True,  # 是否需要确认
            "auto_escalation": True,  # 是否自动升级
            "notification_channels": ["log", "api"],  # 通知渠道
            "trend_analysis_window": 24  # 趋势分析窗口(小时)
        }
    
    def _setup_warning_thresholds(self) -> Dict[WarningType, Dict[WarningLevel, float]]:
        """设置预警阈值"""
        return {
            WarningType.DATA_QUALITY: {
                WarningLevel.LOW: 0.3,
                WarningLevel.MEDIUM: 0.5,
                WarningLevel.HIGH: 0.7,
                WarningLevel.CRITICAL: 0.9
            },
            WarningType.MODEL_COMPATIBILITY: {
                WarningLevel.LOW: 0.25,
                WarningLevel.MEDIUM: 0.45,
                WarningLevel.HIGH: 0.65,
                WarningLevel.CRITICAL: 0.85
            },
            WarningType.DOMAIN_GAP: {
                WarningLevel.LOW: 0.35,
                WarningLevel.MEDIUM: 0.55,
                WarningLevel.HIGH: 0.75,
                WarningLevel.CRITICAL: 0.9
            },
            WarningType.RESOURCE_CONSTRAINTS: {
                WarningLevel.LOW: 0.4,
                WarningLevel.MEDIUM: 0.6,
                WarningLevel.HIGH: 0.8,
                WarningLevel.CRITICAL: 0.95
            },
            WarningType.PERFORMANCE_DEGRADATION: {
                WarningLevel.LOW: 0.2,
                WarningLevel.MEDIUM: 0.4,
                WarningLevel.HIGH: 0.6,
                WarningLevel.CRITICAL: 0.8
            },
            WarningType.SAFETY_VIOLATION: {
                WarningLevel.LOW: 0.1,
                WarningLevel.MEDIUM: 0.3,
                WarningLevel.HIGH: 0.5,
                WarningLevel.CRITICAL: 0.7
            }
        }
    
    def _initialize_statistics(self) -> WarningStatistics:
        """初始化统计信息"""
        return WarningStatistics(
            total_warnings=0,
            active_warnings=0,
            warning_distribution={level: 0 for level in WarningLevel},
            type_distribution={wtype: 0 for wtype in WarningType},
            recent_trend="stable",
            average_response_time=timedelta(0)
        )
    
    def start_monitoring(self) -> bool:
        """启动风险监控"""
        if self.monitoring_active:
            self.logger.warning("风险监控已经启动")
            return False
        
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("风险监控系统已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动风险监控失败: {e}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self) -> bool:
        """停止风险监控"""
        if not self.monitoring_active:
            self.logger.warning("风险监控未启动")
            return False
        
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            self.logger.info("风险监控系统已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止风险监控失败: {e}")
            return False
    
    def monitor_migration_risk(self, 
                            risk_assessment: Dict[str, Any],
                            context: Dict[str, Any]) -> List[WarningMessage]:
        """
        监控迁移学习风险
        
        Args:
            risk_assessment: 风险评估结果
            context: 监控上下文
            
        Returns:
            List[WarningMessage]: 生成的预警消息列表
        """
        warnings = []
        
        try:
            # 1. 监控数据质量风险
            data_quality_warnings = self._monitor_data_quality_risk(risk_assessment, context)
            warnings.extend(data_quality_warnings)
            
            # 2. 监控模型兼容性风险
            model_compatibility_warnings = self._monitor_model_compatibility_risk(risk_assessment, context)
            warnings.extend(model_compatibility_warnings)
            
            # 3. 监控领域差距风险
            domain_gap_warnings = self._monitor_domain_gap_risk(risk_assessment, context)
            warnings.extend(domain_gap_warnings)
            
            # 4. 监控资源约束风险
            resource_constraint_warnings = self._monitor_resource_constraint_risk(risk_assessment, context)
            warnings.extend(resource_constraint_warnings)
            
            # 5. 监控性能下降风险
            performance_warnings = self._monitor_performance_risk(risk_assessment, context)
            warnings.extend(performance_warnings)
            
            # 6. 监控安全违规风险
            safety_warnings = self._monitor_safety_risk(risk_assessment, context)
            warnings.extend(safety_warnings)
            
            # 7. 处理生成的预警
            for warning in warnings:
                self._process_warning(warning)
            
            return warnings
            
        except Exception as e:
            self.logger.error(f"迁移风险监控失败: {e}")
            # 生成系统错误预警
            error_warning = self._create_system_error_warning(e)
            self._process_warning(error_warning)
            return [error_warning]
    
    def _monitor_data_quality_risk(self, risk_assessment: Dict[str, Any], 
                                 context: Dict[str, Any]) -> List[WarningMessage]:
        """监控数据质量风险"""
        warnings = []
        
        data_quality_score = risk_assessment.get("data_quality_score", 1.0)
        risk_level = self._determine_warning_level(
            WarningType.DATA_QUALITY, 1.0 - data_quality_score)
        
        if risk_level != WarningLevel.INFO:
            warning = WarningMessage(
                warning_id=f"DQ_{int(time.time())}",
                warning_type=WarningType.DATA_QUALITY,
                warning_level=risk_level,
                title="数据质量风险预警",
                description=f"数据质量评分较低: {data_quality_score:.2f}",
                timestamp=datetime.now(),
                source="migration_learning",
                affected_components=["data_validation", "model_training"],
                risk_score=1.0 - data_quality_score,
                mitigation_measures=[
                    "检查数据源完整性",
                    "进行数据清洗和预处理",
                    "验证数据分布一致性"
                ]
            )
            warnings.append(warning)
        
        return warnings
    
    def _monitor_model_compatibility_risk(self, risk_assessment: Dict[str, Any],
                                         context: Dict[str, Any]) -> List[WarningMessage]:
        """监控模型兼容性风险"""
        warnings = []
        
        compatibility_score = risk_assessment.get("model_compatibility_score", 1.0)
        risk_level = self._determine_warning_level(
            WarningType.MODEL_COMPATIBILITY, 1.0 - compatibility_score)
        
        if risk_level != WarningLevel.INFO:
            warning = WarningMessage(
                warning_id=f"MC_{int(time.time())}",
                warning_type=WarningType.MODEL_COMPATIBILITY,
                warning_level=risk_level,
                title="模型兼容性风险预警",
                description=f"模型兼容性评分较低: {compatibility_score:.2f}",
                timestamp=datetime.now(),
                source="migration_learning",
                affected_components=["model_training", "inference"],
                risk_score=1.0 - compatibility_score,
                mitigation_measures=[
                    "检查模型架构匹配度",
                    "调整输入输出维度",
                    "使用适配器层进行特征映射"
                ]
            )
            warnings.append(warning)
        
        return warnings
    
    def _monitor_domain_gap_risk(self, risk_assessment: Dict[str, Any],
                                context: Dict[str, Any]) -> List[WarningMessage]:
        """监控领域差距风险"""
        warnings = []
        
        domain_gap_score = risk_assessment.get("domain_gap_score", 0.0)
        risk_level = self._determine_warning_level(
            WarningType.DOMAIN_GAP, domain_gap_score)
        
        if risk_level != WarningLevel.INFO:
            warning = WarningMessage(
                warning_id=f"DG_{int(time.time())}",
                warning_type=WarningType.DOMAIN_GAP,
                warning_level=risk_level,
                title="领域差距风险预警",
                description=f"领域差距评分较高: {domain_gap_score:.2f}",
                timestamp=datetime.now(),
                source="migration_learning",
                affected_components=["domain_adaptation", "feature_extraction"],
                risk_score=domain_gap_score,
                mitigation_measures=[
                    "引入领域专家知识",
                    "使用领域对抗训练",
                    "增加领域适配层"
                ]
            )
            warnings.append(warning)
        
        return warnings
    
    def _monitor_resource_constraint_risk(self, risk_assessment: Dict[str, Any],
                                        context: Dict[str, Any]) -> List[WarningMessage]:
        """监控资源约束风险"""
        warnings = []
        
        resource_risk_score = risk_assessment.get("resource_constraint_score", 0.0)
        risk_level = self._determine_warning_level(
            WarningType.RESOURCE_CONSTRAINTS, resource_risk_score)
        
        if risk_level != WarningLevel.INFO:
            warning = WarningMessage(
                warning_id=f"RC_{int(time.time())}",
                warning_type=WarningType.RESOURCE_CONSTRAINTS,
                warning_level=risk_level,
                title="资源约束风险预警",
                description=f"资源约束风险评分较高: {resource_risk_score:.2f}",
                timestamp=datetime.now(),
                source="migration_learning",
                affected_components=["edge_deployment", "model_serving"],
                risk_score=resource_risk_score,
                mitigation_measures=[
                    "进行模型轻量化",
                    "优化资源分配策略",
                    "使用模型压缩技术"
                ]
            )
            warnings.append(warning)
        
        return warnings
    
    def _monitor_performance_risk(self, risk_assessment: Dict[str, Any],
                                context: Dict[str, Any]) -> List[WarningMessage]:
        """监控性能下降风险"""
        warnings = []
        
        performance_degradation = risk_assessment.get("performance_degradation_score", 0.0)
        risk_level = self._determine_warning_level(
            WarningType.PERFORMANCE_DEGRADATION, performance_degradation)
        
        if risk_level != WarningLevel.INFO:
            warning = WarningMessage(
                warning_id=f"PD_{int(time.time())}",
                warning_type=WarningType.PERFORMANCE_DEGRADATION,
                warning_level=risk_level,
                title="性能下降风险预警",
                description=f"性能下降风险评分: {performance_degradation:.2f}",
                timestamp=datetime.now(),
                source="migration_learning",
                affected_components=["model_inference", "decision_making"],
                risk_score=performance_degradation,
                mitigation_measures=[
                    "优化模型架构",
                    "调整超参数",
                    "使用性能监控工具"
                ]
            )
            warnings.append(warning)
        
        return warnings
    
    def _monitor_safety_risk(self, risk_assessment: Dict[str, Any],
                           context: Dict[str, Any]) -> List[WarningMessage]:
        """监控安全违规风险"""
        warnings = []
        
        safety_violation_score = risk_assessment.get("safety_violation_score", 0.0)
        risk_level = self._determine_warning_level(
            WarningType.SAFETY_VIOLATION, safety_violation_score)
        
        if risk_level != WarningLevel.INFO:
            warning = WarningMessage(
                warning_id=f"SV_{int(time.time())}",
                warning_type=WarningType.SAFETY_VIOLATION,
                warning_level=risk_level,
                title="安全违规风险预警",
                description=f"安全违规风险评分: {safety_violation_score:.2f}",
                timestamp=datetime.now(),
                source="migration_learning",
                affected_components=["safety_monitoring", "rule_enforcement"],
                risk_score=safety_violation_score,
                mitigation_measures=[
                    "立即停止迁移过程",
                    "进行安全审查",
                    "修复安全漏洞"
                ]
            )
            warnings.append(warning)
        
        return warnings
    
    def _determine_warning_level(self, warning_type: WarningType, risk_score: float) -> WarningLevel:
        """根据风险评分确定预警级别"""
        thresholds = self.warning_thresholds.get(warning_type, {})
        
        if risk_score >= thresholds.get(WarningLevel.CRITICAL, 0.9):
            return WarningLevel.CRITICAL
        elif risk_score >= thresholds.get(WarningLevel.HIGH, 0.7):
            return WarningLevel.HIGH
        elif risk_score >= thresholds.get(WarningLevel.MEDIUM, 0.5):
            return WarningLevel.MEDIUM
        elif risk_score >= thresholds.get(WarningLevel.LOW, 0.3):
            return WarningLevel.LOW
        else:
            return WarningLevel.INFO
    
    def _process_warning(self, warning: WarningMessage):
        """处理预警消息"""
        # 添加到历史记录
        self.warning_history.append(warning)
        
        # 添加到活跃预警
        if warning.warning_level in [WarningLevel.MEDIUM, WarningLevel.HIGH, WarningLevel.CRITICAL]:
            self.active_warnings[warning.warning_id] = warning
        
        # 更新统计信息
        self._update_statistics(warning)
        
        # 发送通知
        self._notify_subscribers(warning)
        
        # 记录日志
        self._log_warning(warning)
    
    def _update_statistics(self, warning: WarningMessage):
        """更新统计信息"""
        self.statistics.total_warnings += 1
        
        # 更新级别分布
        current_count = self.statistics.warning_distribution.get(warning.warning_level, 0)
        self.statistics.warning_distribution[warning.warning_level] = current_count + 1
        
        # 更新类型分布
        current_type_count = self.statistics.type_distribution.get(warning.warning_type, 0)
        self.statistics.type_distribution[warning.warning_type] = current_type_count + 1
        
        # 更新活跃预警数量
        self.statistics.active_warnings = len(self.active_warnings)
        
        # 更新趋势分析
        self._update_trend_analysis()
    
    def _update_trend_analysis(self):
        """更新趋势分析"""
        # 分析最近24小时的预警趋势
        window_start = datetime.now() - timedelta(hours=self.config["trend_analysis_window"])
        recent_warnings = [w for w in self.warning_history 
                         if w.timestamp >= window_start]
        
        if len(recent_warnings) < 2:
            self.statistics.recent_trend = "stable"
            return
        
        # 按时间窗口分组统计
        time_windows = []
        current_time = window_start
        while current_time <= datetime.now():
            time_windows.append(current_time)
            current_time += timedelta(hours=1)
        
        warning_counts = []
        for i in range(len(time_windows) - 1):
            start_time = time_windows[i]
            end_time = time_windows[i + 1]
            count = len([w for w in recent_warnings 
                        if start_time <= w.timestamp < end_time])
            warning_counts.append(count)
        
        # 判断趋势
        if len(warning_counts) >= 2:
            recent_change = warning_counts[-1] - warning_counts[-2]
            if recent_change > 2:
                self.statistics.recent_trend = "increasing"
            elif recent_change < -2:
                self.statistics.recent_trend = "decreasing"
            else:
                self.statistics.recent_trend = "stable"
    
    def _notify_subscribers(self, warning: WarningMessage):
        """通知订阅者"""
        for subscriber in self.subscribers:
            try:
                subscriber(warning)
            except Exception as e:
                self.logger.error(f"通知订阅者失败: {e}")
    
    def _log_warning(self, warning: WarningMessage):
        """记录预警日志"""
        log_message = (f"[{warning.warning_level.value.upper()}] {warning.warning_type.value}: "
                      f"{warning.title} - {warning.description}")
    
    async def trigger_warning(self, warning_type: str, context: Dict[str, Any], details: Dict[str, Any]):
        """
        触发预警
        
        Args:
            warning_type: 预警类型
            context: 预警上下文
            details: 预警详情
        """
        try:
            # 将字符串转换为WarningType枚举
            warning_enum = WarningType(warning_type) if warning_type in WarningType.__members__ else WarningType.DATA_QUALITY
            
            # 从details中获取风险评分或使用默认值
            risk_score = details.get("risk_score", 0.7)
            
            # 确定预警级别
            risk_level = self._determine_warning_level(warning_enum, risk_score)
            
            # 创建预警消息
            warning = WarningMessage(
                warning_id=f"TRIGGER_{int(time.time())}",
                warning_type=warning_enum,
                warning_level=risk_level,
                title=f"手动触发预警: {warning_type}",
                description=f"预警详情: {details}",
                timestamp=datetime.now(),
                source="decision_integration",
                affected_components=["decision_engine"],
                risk_score=risk_score,
                mitigation_measures=["请检查系统状态和决策请求"]
            )
            
            # 处理预警
            self._process_warning(warning)
            return warning
            
        except Exception as e:
            self.logger.error(f"触发预警失败: {e}")
            # 生成系统错误预警
            error_warning = self._create_system_error_warning(e)
            self._process_warning(error_warning)
            return error_warning
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        获取预警系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        return {
            "status": "active" if self.monitoring_active else "inactive",
            "config": self.config.copy(),
            "active_warnings": len(self.active_warnings),
            "total_warnings": len(self.warning_history),
            "statistics": asdict(self.statistics)
        }
        
        if warning.warning_level == WarningLevel.CRITICAL:
            self.logger.critical(log_message)
        elif warning.warning_level == WarningLevel.HIGH:
            self.logger.error(log_message)
        elif warning.warning_level == WarningLevel.MEDIUM:
            self.logger.warning(log_message)
        elif warning.warning_level == WarningLevel.LOW:
            self.logger.info(log_message)
        else:
            self.logger.debug(log_message)
    
    def acknowledge_warning(self, warning_id: str, acknowledged_by: str) -> bool:
        """确认预警"""
        if warning_id not in self.active_warnings:
            self.logger.warning(f"预警ID不存在: {warning_id}")
            return False
        
        warning = self.active_warnings[warning_id]
        warning.acknowledged = True
        warning.acknowledged_by = acknowledged_by
        warning.acknowledged_at = datetime.now()
        
        # 从活跃预警中移除
        del self.active_warnings[warning_id]
        
        self.logger.info(f"预警 {warning_id} 已被 {acknowledged_by} 确认")
        return True
    
    def get_active_warnings(self) -> List[WarningMessage]:
        """获取活跃预警列表"""
        return list(self.active_warnings.values())
    
    def get_warning_history(self, days: int = 7) -> List[WarningMessage]:
        """获取预警历史记录"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [w for w in self.warning_history if w.timestamp >= cutoff_date]
    
    def get_statistics(self) -> WarningStatistics:
        """获取统计信息"""
        return self.statistics
    
    def subscribe(self, callback: callable):
        """订阅预警通知"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: callable):
        """取消订阅预警通知"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 检查预警升级
                self._check_warning_escalation()
                
                # 清理过期预警
                self._cleanup_expired_warnings()
                
                # 等待下一个监控周期
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(10)  # 异常后等待10秒再继续
    
    def _check_warning_escalation(self):
        """检查预警是否需要升级"""
        if not self.config["auto_escalation"]:
            return
        
        current_time = datetime.now()
        escalation_timeout = timedelta(seconds=self.config["escalation_timeout"])
        
        for warning_id, warning in list(self.active_warnings.items()):
            if warning.acknowledged:
                continue
            
            # 检查是否超过升级超时时间
            time_since_creation = current_time - warning.timestamp
            if time_since_creation > escalation_timeout:
                # 升级预警级别
                self._escalate_warning(warning)
    
    def _escalate_warning(self, warning: WarningMessage):
        """升级预警级别"""
        current_level = warning.warning_level
        
        if current_level == WarningLevel.LOW:
            new_level = WarningLevel.MEDIUM
        elif current_level == WarningLevel.MEDIUM:
            new_level = WarningLevel.HIGH
        elif current_level == WarningLevel.HIGH:
            new_level = WarningLevel.CRITICAL
        else:
            return  # 已经是最高级别
        
        # 更新预警级别
        warning.warning_level = new_level
        warning.description += f" (已自动升级至{new_level.value}级别)"
        
        self.logger.warning(f"预警 {warning.warning_id} 已升级至 {new_level.value} 级别")
    
    def _cleanup_expired_warnings(self):
        """清理过期预警"""
        retention_days = self.config["warning_retention_days"]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # 清理历史记录
        self.warning_history = deque(
            [w for w in self.warning_history if w.timestamp >= cutoff_date],
            maxlen=1000
        )
        
        # 清理活跃预警（只保留未确认的）
        expired_warnings = []
        for warning_id, warning in self.active_warnings.items():
            if warning.acknowledged and warning.acknowledged_at:
                if warning.acknowledged_at < cutoff_date:
                    expired_warnings.append(warning_id)
        
        for warning_id in expired_warnings:
            del self.active_warnings[warning_id]
    
    def _create_system_error_warning(self, error: Exception) -> WarningMessage:
        """创建系统错误预警"""
        return WarningMessage(
            warning_id=f"SYS_ERR_{int(time.time())}",
            warning_type=WarningType.SAFETY_VIOLATION,
            warning_level=WarningLevel.CRITICAL,
            title="风险监控系统异常",
            description=f"监控系统出现异常: {str(error)}",
            timestamp=datetime.now(),
            source="warning_system",
            affected_components=["risk_monitoring"],
            risk_score=1.0,
            mitigation_measures=[
                "检查系统日志",
                "重启监控服务",
                "联系系统管理员"
            ]
        )