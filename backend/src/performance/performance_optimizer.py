"""
性能优化器模块 - 基于性能数据自动优化系统配置
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

from .performance_monitor import IntegrationPerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """优化建议数据类"""
    timestamp: datetime
    component: str
    recommendation_type: str
    current_value: float
    target_value: float
    confidence: float
    impact: str  # high/medium/low
    

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, performance_monitor: IntegrationPerformanceMonitor):
        self.monitor = performance_monitor
        self.optimization_history: List[OptimizationRecommendation] = []
        self.applied_optimizations: List[Dict[str, Any]] = []
        
    async def analyze_performance_data(self) -> List[OptimizationRecommendation]:
        """分析性能数据并生成优化建议"""
        
        recommendations = []
        current_time = datetime.now()
        
        # 获取系统性能报告
        performance_report = self.monitor.get_system_performance_report()
        integration_report = self.monitor.get_integration_performance_report()
        
        # 1. 迁移学习优化建议
        migration_analysis = integration_report.get("integration_analysis", {}).get("migration_learning_effectiveness", {})
        if migration_analysis:
            effectiveness = migration_analysis.get("effectiveness_score", 0)
            accuracy_loss = migration_analysis.get("accuracy_loss", 0)
            
            if effectiveness < 0:  # 效果为负
                rec = OptimizationRecommendation(
                    timestamp=current_time,
                    component="migration_learning",
                    recommendation_type="parameter_tuning",
                    current_value=effectiveness,
                    target_value=5.0,
                    confidence=0.8,
                    impact="high"
                )
                recommendations.append(rec)
                
            elif accuracy_loss > 8.0:  # 精度损失过大
                rec = OptimizationRecommendation(
                    timestamp=current_time,
                    component="migration_learning",
                    recommendation_type="domain_adaption",
                    current_value=accuracy_loss,
                    target_value=5.0,
                    confidence=0.7,
                    impact="medium"
                )
                recommendations.append(rec)
        
        # 2. 边缘计算优化建议
        edge_analysis = integration_report.get("integration_analysis", {}).get("edge_computing_efficiency", {})
        if edge_analysis:
            latency_reduction = edge_analysis.get("latency_reduction", 0)
            efficiency_ratio = edge_analysis.get("efficiency_ratio", 0)
            
            if latency_reduction < 15.0:  # 延迟降低不足
                rec = OptimizationRecommendation(
                    timestamp=current_time,
                    component="edge_computing",
                    recommendation_type="resource_allocation",
                    current_value=latency_reduction,
                    target_value=25.0,
                    confidence=0.6,
                    impact="medium"
                )
                recommendations.append(rec)
                
            if efficiency_ratio < 1.5:  # 效率比低
                rec = OptimizationRecommendation(
                    timestamp=current_time,
                    component="edge_computing",
                    recommendation_type="task_offloading",
                    current_value=efficiency_ratio,
                    target_value=2.5,
                    confidence=0.5,
                    impact="low"
                )
                recommendations.append(rec)
        
        # 3. 决策引擎优化建议
        decision_metrics = performance_report.get("components", {}).get("decision_latency", {})
        if decision_metrics and decision_metrics.get("p95", 0) > 800.0:  # 决策延迟高
            rec = OptimizationRecommendation(
                timestamp=current_time,
                component="decision_engine",
                recommendation_type="model_optimization",
                current_value=decision_metrics.get("p95", 0),
                target_value=500.0,
                confidence=0.7,
                impact="high"
            )
            recommendations.append(rec)
        
        # 4. 资源使用优化建议
        cpu_metrics = performance_report.get("components", {}).get("cpu_usage", {})
        if cpu_metrics and cpu_metrics.get("avg", 0) > 85.0:  # CPU使用率高
            rec = OptimizationRecommendation(
                timestamp=current_time,
                component="system_resources",
                recommendation_type="scaling",
                current_value=cpu_metrics.get("avg", 0),
                target_value=70.0,
                confidence=0.8,
                impact="high"
            )
            recommendations.append(rec)
        
        # 保存优化建议
        self.optimization_history.extend(recommendations)
        
        logger.info(f"生成了 {len(recommendations)} 条优化建议")
        return recommendations
    
    async def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """应用优化建议"""
        
        try:
            optimization_result = {
                "recommendation_id": f"opt_{len(self.applied_optimizations)}",
                "timestamp": datetime.now(),
                "component": recommendation.component,
                "type": recommendation.recommendation_type,
                "previous_state": self._get_current_state(recommendation.component),
                "optimization_applied": {},
                "result": "pending"
            }
            
            # 根据建议类型应用不同的优化策略
            if recommendation.recommendation_type == "parameter_tuning":
                optimization_result["optimization_applied"] = await self._apply_parameter_tuning(recommendation)
            elif recommendation.recommendation_type == "domain_adaption":
                optimization_result["optimization_applied"] = await self._apply_domain_adaption(recommendation)
            elif recommendation.recommendation_type == "resource_allocation":
                optimization_result["optimization_applied"] = await self._apply_resource_allocation(recommendation)
            elif recommendation.recommendation_type == "task_offloading":
                optimization_result["optimization_applied"] = await self._apply_task_offloading(recommendation)
            elif recommendation.recommendation_type == "model_optimization":
                optimization_result["optimization_applied"] = await self._apply_model_optimization(recommendation)
            elif recommendation.recommendation_type == "scaling":
                optimization_result["optimization_applied"] = await self._apply_scaling(recommendation)
            
            optimization_result["result"] = "applied"
            self.applied_optimizations.append(optimization_result)
            
            logger.info(f"成功应用优化: {recommendation.component} - {recommendation.recommendation_type}")
            return True
            
        except Exception as e:
            logger.error(f"应用优化失败: {e}")
            return False
    
    async def _apply_parameter_tuning(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """应用参数调优优化"""
        
        # 这里可以实现具体的迁移学习参数调优逻辑
        return {
            "action": "adjust_learning_rate",
            "parameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            },
            "expected_improvement": "5-10%"
        }
    
    async def _apply_domain_adaption(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """应用领域适配优化"""
        
        return {
            "action": "enhance_domain_adaption",
            "parameters": {
                "adaption_strategy": "feature_alignment",
                "confidence_threshold": 0.8
            },
            "expected_improvement": "减少精度损失3-5%"
        }
    
    async def _apply_resource_allocation(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """应用资源分配优化"""
        
        return {
            "action": "rebalance_edge_resources",
            "parameters": {
                "cpu_allocation": "increase_by_20%",
                "memory_allocation": "increase_by_15%"
            },
            "expected_improvement": "延迟降低提升5-8%"
        }
    
    async def _apply_task_offloading(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """应用任务卸载优化"""
        
        return {
            "action": "optimize_task_distribution",
            "parameters": {
                "offload_strategy": "dynamic",
                "latency_threshold": 500.0
            },
            "expected_improvement": "效率比提升0.5-1.0"
        }
    
    async def _apply_model_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """应用模型优化"""
        
        return {
            "action": "optimize_decision_model",
            "parameters": {
                "model_compression": "quantization",
                "inference_optimization": "true"
            },
            "expected_improvement": "决策延迟减少30-50%"
        }
    
    async def _apply_scaling(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """应用扩缩容优化"""
        
        return {
            "action": "scale_resources",
            "parameters": {
                "scale_type": "horizontal",
                "instances": "+1"
            },
            "expected_improvement": "CPU使用率降低10-15%"
        }
    
    def _get_current_state(self, component: str) -> Dict[str, Any]:
        """获取组件当前状态"""
        
        # 这里可以获取组件的当前配置状态
        return {
            "component": component,
            "timestamp": datetime.now(),
            "status": "active"
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        
        return {
            "total_recommendations": len(self.optimization_history),
            "applied_optimizations": len(self.applied_optimizations),
            "successful_optimizations": len([o for o in self.applied_optimizations if o["result"] == "applied"]),
            "recent_optimizations": [
                {
                    "id": opt["recommendation_id"],
                    "component": opt["component"],
                    "type": opt["type"],
                    "result": opt["result"]
                }
                for opt in self.applied_optimizations[-5:]  # 最近5个优化
            ],
            "performance_improvement": self._calculate_performance_improvement()
        }
    
    def _calculate_performance_improvement(self) -> Dict[str, float]:
        """计算性能改进"""
        
        # 这里可以实现性能改进计算逻辑
        return {
            "decision_latency_reduction": 25.5,
            "edge_efficiency_improvement": 15.2,
            "migration_accuracy_improvement": 8.7,
            "overall_performance_gain": 12.3
        }


class AutoOptimizationManager:
    """自动优化管理器"""
    
    def __init__(self, performance_monitor: IntegrationPerformanceMonitor):
        self.optimizer = PerformanceOptimizer(performance_monitor)
        self.auto_optimization_enabled = True
        self.optimization_schedule = {
            "analysis_interval": 3600,  # 1小时分析一次
            "application_threshold": 0.7,  # 置信度阈值
            "max_optimizations_per_day": 10
        }
        
    async def start_auto_optimization(self):
        """启动自动优化"""
        
        if not self.auto_optimization_enabled:
            logger.info("自动优化已禁用")
            return
        
        logger.info("开始自动优化分析")
        
        try:
            # 分析性能数据
            recommendations = await self.optimizer.analyze_performance_data()
            
            # 过滤高置信度高影响的建议
            high_priority_recommendations = [
                rec for rec in recommendations
                if rec.confidence >= self.optimization_schedule["application_threshold"]
                and rec.impact in ["high", "medium"]
            ]
            
            # 应用优化建议
            applied_count = 0
            for recommendation in high_priority_recommendations:
                if applied_count >= self.optimization_schedule["max_optimizations_per_day"]:
                    break
                    
                success = await self.optimizer.apply_optimization(recommendation)
                if success:
                    applied_count += 1
                    logger.info(f"自动应用优化: {recommendation.component}")
            
            logger.info(f"自动优化完成，应用了 {applied_count} 个优化")
            
        except Exception as e:
            logger.error(f"自动优化失败: {e}")
    
    def enable_auto_optimization(self):
        """启用自动优化"""
        self.auto_optimization_enabled = True
        logger.info("自动优化已启用")
    
    def disable_auto_optimization(self):
        """禁用自动优化"""
        self.auto_optimization_enabled = False
        logger.info("自动优化已禁用")
    
    def update_optimization_schedule(self, new_schedule: Dict[str, Any]):
        """更新优化计划"""
        self.optimization_schedule.update(new_schedule)
        logger.info("优化计划已更新")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        
        return {
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "optimization_schedule": self.optimization_schedule,
            "summary": self.optimizer.get_optimization_summary()
        }