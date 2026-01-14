"""Meta-Cognitive Controller Module
Implementing self-awareness, reflection mechanisms, and meta-level decision making
Building on ACT-R/Soar cognitive architecture with autonomous goal management
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import time
from enum import Enum
from collections import deque

from .cognitive_architecture import CognitiveArchitecture, CognitiveModuleType, CognitiveModule
from .neural_symbolic_system import neural_symbolic_system
from .common_knowledge_base import common_knowledge_base
from .environment_perception import environment_perception_system

logger = logging.getLogger(__name__)


class SelfAwarenessLevel(int, Enum):
    """自我意识水平枚举"""
    NONE = 0
    BASIC = 1  # 基础自我感知
    AWARE = 2  # 自我意识
    REFLECTIVE = 3  # 反思性自我意识
    META_AWARE = 4  # 元认知意识


class ReflectionType(str, Enum):
    """反思类型枚举"""
    DECISION_QUALITY = "decision_quality"
    LEARNING_EFFICIENCY = "learning_efficiency"
    GOAL_PROGRESS = "goal_progress"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_METRICS = "performance_metrics"
    ETHICAL_COMPLIANCE = "ethical_compliance"


@dataclass
class SelfAssessmentResult:
    """自我评估结果数据结构"""
    timestamp: datetime
    overall_score: float  # 整体评分，0-1
    performance_metrics: Dict[str, float]  # 各项性能指标
    strengths: List[str]  # 系统优势
    weaknesses: List[str]  # 系统弱点
    recommended_actions: List[str]  # 推荐改进措施
    self_awareness_level: SelfAwarenessLevel


@dataclass
class ReflectionRecord:
    """反思记录数据结构"""
    reflection_id: str
    reflection_type: ReflectionType
    content: Dict[str, Any]
    timestamp: datetime
    insights: List[str]  # 反思洞察
    actions_taken: List[str]  # 采取的行动
    effectiveness: float = 0.0  # 行动有效性评分，0-1


@dataclass
class MetaGoal:
    """元目标数据结构"""
    goal_id: str
    description: str
    priority: int
    status: str  # active, completed, suspended, abandoned
    creation_time: datetime
    completion_time: Optional[datetime] = None
    resources_required: Optional[Dict[str, Any]] = None
    expected_impact: float = 0.5  # 预期影响，0-1


class MetaCognitiveModule(CognitiveModule):
    """元认知模块，负责自我意识和反思机制"""
    
    def __init__(self, cognitive_architecture: CognitiveArchitecture, name: str = "meta_cognitive_module"):
        super().__init__(CognitiveModuleType.METACOGNITIVE_MODULE, name)
        self.cognitive_architecture = cognitive_architecture
        self.self_awareness_level = SelfAwarenessLevel.AWARE
        self.reflection_history = deque(maxlen=100)  # 保存最近100条反思记录
        self.self_assessment_history = deque(maxlen=50)  # 保存最近50次自我评估
        self.meta_goals: Dict[str, MetaGoal] = {}  # 元目标字典
        self.active_reflection: Optional[ReflectionType] = None
        self.last_self_assessment = None
        
        # 性能监控参数
        self.performance_window = 10  # 性能评估窗口大小
        self.decision_history = deque(maxlen=100)
        self.learning_history = deque(maxlen=50)
        
    async def initialize(self):
        """初始化元认知模块"""
        # 创建初始元目标
        initial_goals = [
            MetaGoal(
                goal_id="meta_goal_maintain_awareness",
                description="持续监控系统状态和性能",
                priority=1,
                status="active",
                creation_time=datetime.now(),
                expected_impact=0.8
            ),
            MetaGoal(
                goal_id="meta_goal_improve_learning",
                description="优化学习效率和知识获取",
                priority=2,
                status="active",
                creation_time=datetime.now(),
                expected_impact=0.7
            ),
            MetaGoal(
                goal_id="meta_goal_ethical_compliance",
                description="确保系统决策符合伦理规范",
                priority=1,
                status="active",
                creation_time=datetime.now(),
                expected_impact=0.9
            )
        ]
        
        for goal in initial_goals:
            self.meta_goals[goal.goal_id] = goal
        
        logger.info(f"元认知模块初始化完成，初始自我意识水平: {self.self_awareness_level.name}")
        await super().initialize()
    
    def update_self_awareness_level(self, new_level: SelfAwarenessLevel):
        """更新自我意识水平"""
        old_level = self.self_awareness_level
        if new_level != old_level:
            self.self_awareness_level = new_level
            logger.info(f"自我意识水平变化: {old_level.name} -> {new_level.name}")
    
    def monitor_system_state(self) -> Dict[str, Any]:
        """监控系统状态"""
        # 获取认知架构状态
        arch_status = self.cognitive_architecture.get_status()
        
        # 获取环境感知状态
        env_state = environment_perception_system.get_environment_context()
        
        # 构建系统状态监控结果
        system_state = {
            "timestamp": datetime.now(),
            "cognitive_architecture": arch_status,
            "environment": env_state,
            "self_awareness_level": self.self_awareness_level.name,
            "active_meta_goals": [g for g in self.meta_goals.values() if g.status == "active"],
            "reflection_count": len(self.reflection_history),
            "assessment_count": len(self.self_assessment_history)
        }
        
        return system_state
    
    def perform_self_assessment(self) -> SelfAssessmentResult:
        """执行自我评估"""
        start_time = time.time()
        
        # 获取当前系统状态
        system_state = self.monitor_system_state()
        
        # 评估各项性能指标
        performance_metrics = {
            "decision_quality": self._assess_decision_quality(),
            "learning_efficiency": self._assess_learning_efficiency(),
            "goal_progress": self._assess_goal_progress(),
            "resource_usage": self._assess_resource_usage(),
            "ethical_compliance": self._assess_ethical_compliance()
        }
        
        # 计算整体评分（加权平均）
        weights = {
            "decision_quality": 0.25,
            "learning_efficiency": 0.20,
            "goal_progress": 0.25,
            "resource_usage": 0.15,
            "ethical_compliance": 0.15
        }
        
        overall_score = sum(performance_metrics[key] * weights[key] for key in weights)
        
        # 识别优势和弱点
        strengths = [key for key, value in performance_metrics.items() if value > 0.7]
        weaknesses = [key for key, value in performance_metrics.items() if value < 0.5]
        
        # 生成推荐行动
        recommended_actions = self._generate_recommendations(performance_metrics)
        
        # 创建评估结果
        assessment = SelfAssessmentResult(
            timestamp=datetime.now(),
            overall_score=overall_score,
            performance_metrics=performance_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            recommended_actions=recommended_actions,
            self_awareness_level=self.self_awareness_level
        )
        
        # 保存评估结果
        self.self_assessment_history.append(assessment)
        self.last_self_assessment = assessment
        
        # 根据评估结果调整自我意识水平
        if overall_score > 0.8 and len(self.self_assessment_history) > 5:
            self.update_self_awareness_level(SelfAwarenessLevel.META_AWARE)
        elif overall_score < 0.4:
            self.update_self_awareness_level(SelfAwarenessLevel.BASIC)
        
        logger.info(f"自我评估完成，整体评分: {overall_score:.2f}, 自我意识水平: {self.self_awareness_level.name}")
        return assessment
    
    def _assess_decision_quality(self) -> float:
        """评估决策质量"""
        # 简单的决策质量评估，基于最近决策的成功率
        if not self.decision_history:
            return 0.7  # 默认值
        
        # 假设决策历史中包含成功/失败标记
        successful_decisions = sum(1 for d in self.decision_history if d.get("success", False))
        quality = successful_decisions / len(self.decision_history)
        return min(1.0, quality)
    
    def _assess_learning_efficiency(self) -> float:
        """评估学习效率"""
        # 基于学习历史评估
        if not self.learning_history:
            return 0.6  # 默认值
        
        # 计算平均学习效率得分
        avg_efficiency = sum(h.get("efficiency", 0.0) for h in self.learning_history) / len(self.learning_history)
        return min(1.0, avg_efficiency)
    
    def _assess_goal_progress(self) -> float:
        """评估目标进度"""
        # 检查活跃元目标的进度
        active_goals = [g for g in self.meta_goals.values() if g.status == "active"]
        if not active_goals:
            return 0.8  # 没有活跃目标，视为完成
        
        # 简单的进度评估，实际应用中应基于具体目标进度
        return 0.7  # 默认值，实际应根据目标类型实现
    
    def _assess_resource_usage(self) -> float:
        """评估资源使用效率"""
        # 简单的资源使用评估
        # 实际应用中应监控CPU、内存、API调用等资源
        return 0.8  # 默认值，实际应实现具体监控
    
    def _assess_ethical_compliance(self) -> float:
        """评估伦理合规性"""
        # 简单的伦理合规性检查
        # 实际应用中应集成伦理规则引擎
        return 0.9  # 默认值，实际应实现伦理检查
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """基于性能指标生成改进建议"""
        recommendations = []
        
        if metrics["decision_quality"] < 0.6:
            recommendations.append("改进决策算法，增加决策验证步骤")
        
        if metrics["learning_efficiency"] < 0.6:
            recommendations.append("优化学习算法，增加知识检索效率")
        
        if metrics["goal_progress"] < 0.6:
            recommendations.append("重新评估目标优先级，调整资源分配")
        
        if metrics["resource_usage"] < 0.6:
            recommendations.append("优化资源使用，减少不必要的计算")
        
        if metrics["ethical_compliance"] < 0.8:
            recommendations.append("加强伦理规则检查，增加伦理审核步骤")
        
        return recommendations
    
    def reflect_on_decision(self, decision_id: str, decision_context: Dict[str, Any]):
        """反思特定决策"""
        # 使用神经符号系统进行决策反思
        reflection_content = {
            "decision_id": decision_id,
            "context": decision_context,
            "timestamp": datetime.now()
        }
        
        # 生成反思查询
        query = f"反思决策 {decision_id} 的质量和改进空间"
        reasoning_results = neural_symbolic_system.symbolic_reasoning(query, decision_context)
        
        # 生成反思洞察
        insights = []
        for result in reasoning_results:
            insights.append(result.conclusion)
        
        # 创建反思记录
        reflection = {
            "reflection_id": f"ref_{decision_id}_{int(time.time())}",
            "reflection_type": ReflectionType.DECISION_QUALITY,
            "content": reflection_content,
            "timestamp": datetime.now(),
            "insights": insights,
            "actions_taken": []
        }
        
        # 保存反思记录
        self.reflection_history.append(reflection)
        
        # 根据反思结果改进系统
        self._implement_reflection_insights(insights)
        
        logger.debug(f"完成决策反思: {decision_id}, 生成 {len(insights)} 条洞察")
    
    def _implement_reflection_insights(self, insights: List[str]):
        """实现反思洞察，改进系统"""
        for insight in insights:
            # 简单的洞察实现，实际应用中应根据具体洞察类型采取行动
            if "决策质量" in insight and "改进" in insight:
                # 改进决策机制
                logger.info(f"根据反思洞察改进决策机制: {insight}")
            elif "学习效率" in insight and "优化" in insight:
                # 优化学习算法
                logger.info(f"根据反思洞察优化学习算法: {insight}")
            elif "伦理" in insight and "合规" in insight:
                # 加强伦理检查
                logger.info(f"根据反思洞察加强伦理检查: {insight}")
    
    def manage_goals(self):
        """管理元目标，包括创建、更新和优先级调整"""
        # 基于自我评估结果调整目标优先级
        if self.last_self_assessment:
            # 检查性能指标，调整目标优先级
            if self.last_self_assessment.performance_metrics["learning_efficiency"] < 0.5:
                # 提高学习相关目标的优先级
                for goal in self.meta_goals.values():
                    if "learning" in goal.description.lower():
                        goal.priority = 1
            
            if self.last_self_assessment.performance_metrics["ethical_compliance"] < 0.7:
                # 提高伦理相关目标的优先级
                for goal in self.meta_goals.values():
                    if "ethical" in goal.description.lower():
                        goal.priority = 0
    
    def make_meta_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """做出元级决策，关于系统的行为和目标"""
        # 收集元决策所需的信息
        self_assessment = self.perform_self_assessment()
        system_state = self.monitor_system_state()
        
        # 使用神经符号系统进行元级推理
        meta_query = "基于当前系统状态和自我评估，做出最佳元级决策"
        reasoning_results = neural_symbolic_system.symbolic_reasoning(meta_query, {
            "self_assessment": self_assessment.__dict__,
            "system_state": system_state
        })
        
        # 生成元决策
        meta_decision = {
            "decision_id": f"meta_{int(time.time())}",
            "timestamp": datetime.now(),
            "self_assessment": self_assessment,
            "system_state": system_state,
            "reasoning_results": reasoning_results,
            "selected_action": reasoning_results[0].conclusion if reasoning_results else "维持当前状态",
            "confidence": reasoning_results[0].confidence if reasoning_results else 0.7
        }
        
        # 保存决策到历史
        self.decision_history.append({
            "decision_id": meta_decision["decision_id"],
            "type": "meta",
            "action": meta_decision["selected_action"],
            "confidence": meta_decision["confidence"],
            "timestamp": meta_decision["timestamp"],
            "success": True  # 元决策默认视为成功
        })
        
        logger.info(f"做出元决策: {meta_decision['selected_action']}, 置信度: {meta_decision['confidence']:.2f}")
        
        # 执行元决策
        self._execute_meta_decision(meta_decision)
        
        return meta_decision
    
    def _execute_meta_decision(self, meta_decision: Dict[str, Any]):
        """执行元决策"""
        action = meta_decision["selected_action"]
        
        if "调整目标" in action or "优先级" in action:
            # 调整目标优先级
            self.manage_goals()
        elif "改进决策" in action:
            # 改进决策机制
            logger.info(f"执行元决策: 改进决策机制")
        elif "优化学习" in action:
            # 优化学习算法
            logger.info(f"执行元决策: 优化学习算法")
        elif "加强伦理" in action:
            # 加强伦理检查
            logger.info(f"执行元决策: 加强伦理检查")
        elif "提高自我意识" in action:
            # 提高自我意识水平
            new_level = min(SelfAwarenessLevel.META_AWARE, self.self_awareness_level + 1)
            self.update_self_awareness_level(new_level)
        
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """处理输入数据，执行元认知功能"""
        # 监控系统状态
        system_state = self.monitor_system_state()
        
        # 执行自我评估
        self_assessment = self.perform_self_assessment()
        
        # 管理目标
        self.manage_goals()
        
        # 做出元决策
        meta_decision = self.make_meta_decision(input_data)
        
        return {
            "system_state": system_state,
            "self_assessment": self_assessment,
            "meta_decision": meta_decision,
            "self_awareness_level": self.self_awareness_level.name,
            "timestamp": datetime.now()
        }
    
    def get_self_assessment_summary(self) -> Dict[str, Any]:
        """获取自我评估摘要"""
        if not self.self_assessment_history:
            return {
                "message": "尚未进行自我评估",
                "self_awareness_level": self.self_awareness_level.name
            }
        
        latest_assessment = self.self_assessment_history[-1]
        
        return {
            "latest_assessment": {
                "overall_score": latest_assessment.overall_score,
                "timestamp": latest_assessment.timestamp,
                "performance_metrics": latest_assessment.performance_metrics,
                "strengths": latest_assessment.strengths,
                "weaknesses": latest_assessment.weaknesses
            },
            "self_awareness_level": self.self_awareness_level.name,
            "assessment_count": len(self.self_assessment_history),
            "reflection_count": len(self.reflection_history),
            "active_meta_goals": [g.goal_id for g in self.meta_goals.values() if g.status == "active"]
        }
    
    def get_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取反思历史"""
        return list(self.reflection_history)[-limit:]


class MetaCognitiveSystem:
    """元认知系统主类，整合所有元认知功能"""
    
    def __init__(self, cognitive_architecture: CognitiveArchitecture):
        self.cognitive_architecture = cognitive_architecture
        self.meta_module = MetaCognitiveModule(cognitive_architecture)
        self.initialized = False
        self.last_update = datetime.now()
        
        logger.info("元认知系统初始化完成")
    
    async def initialize(self):
        """初始化元认知系统"""
        await self.meta_module.initialize()
        self.initialized = True
        self.last_update = datetime.now()
        logger.info("元认知系统初始化完成")
    
    def update(self):
        """更新元认知系统状态"""
        # 监控系统状态
        system_state = self.meta_module.monitor_system_state()
        
        # 执行自我评估
        self.meta_module.perform_self_assessment()
        
        # 管理目标
        self.meta_module.manage_goals()
        
        self.last_update = datetime.now()
    
    def process_input(self, input_data: Any) -> Dict[str, Any]:
        """处理输入数据，执行元认知处理"""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，使用create_task
            task = loop.create_task(self.meta_module.process(input_data))
            return {"status": "processing", "task_id": id(task)}
        else:
            # 如果事件循环没有运行，直接执行
            return loop.run_until_complete(self.meta_module.process(input_data))
    
    def get_status(self) -> Dict[str, Any]:
        """获取元认知系统状态"""
        return {
            "initialized": self.initialized,
            "last_update": self.last_update,
            "meta_module": self.meta_module.get_status(),
            "self_awareness_level": self.meta_module.self_awareness_level.name,
            "assessment_summary": self.meta_module.get_self_assessment_summary()
        }
    
    def reflect_on_decision(self, decision_id: str, decision_context: Dict[str, Any]):
        """反思特定决策"""
        self.meta_module.reflect_on_decision(decision_id, decision_context)
    
    def get_self_awareness_level(self) -> SelfAwarenessLevel:
        """获取当前自我意识水平"""
        return self.meta_module.self_awareness_level


# 创建全局元认知系统实例
from .cognitive_architecture import cognitive_architecture
meta_cognitive_system = MetaCognitiveSystem(cognitive_architecture)