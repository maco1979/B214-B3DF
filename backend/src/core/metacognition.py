# Metacognition Monitoring System
# Implements hierarchical metacognitive architecture for AGI systems
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class AbilityEvaluator:
    """能力评估器 - 评估当前任务的难度和自身能力"""
    
    def __init__(self):
        self.task_difficulty_history = []
        self.ability_scores = {
            'decision_making': 0.8,  # 决策能力初始分数
            'learning_speed': 0.7,   # 学习速度初始分数
            'problem_solving': 0.75,  # 问题解决能力初始分数
            'pattern_recognition': 0.85  # 模式识别能力初始分数
        }
    
    def evaluate_task_difficulty(self, task: Dict[str, Any]) -> float:
        """评估任务难度
        
        Args:
            task: 任务描述字典
            
        Returns:
            任务难度评分 (0-1)
        """
        difficulty_score = 0.5  # 默认难度
        
        # 基于任务复杂度评估
        if 'complexity' in task:
            difficulty_score = task['complexity']
        elif 'parameters' in task:
            # 参数数量越多，任务越复杂
            param_count = len(task['parameters'])
            difficulty_score = min(1.0, param_count / 20)  # 最多1.0
        
        # 记录任务难度
        self.task_difficulty_history.append({
            'task': task,
            'difficulty': difficulty_score,
            'timestamp': datetime.now()
        })
        
        return difficulty_score
    
    def evaluate_ability(self, ability_type: str, task_difficulty: float, success: bool) -> float:
        """评估特定能力
        
        Args:
            ability_type: 能力类型
            task_difficulty: 任务难度
            success: 是否成功完成任务
            
        Returns:
            更新后的能力评分
        """
        if ability_type not in self.ability_scores:
            return 0.5  # 默认分数
        
        current_score = self.ability_scores[ability_type]
        
        # 根据任务难度和成功情况调整能力评分
        if success:
            # 完成困难任务提升分数更多
            improvement = task_difficulty * 0.05
            new_score = min(1.0, current_score + improvement)
        else:
            # 失败降低分数，困难任务失败降低更少
            decrease = (1 - task_difficulty) * 0.05
            new_score = max(0.0, current_score - decrease)
        
        self.ability_scores[ability_type] = new_score
        return new_score
    
    def get_ability_scores(self) -> Dict[str, float]:
        """获取当前能力评分"""
        return self.ability_scores.copy()


class ConfidenceCalculator:
    """信心计算器 - 计算对答案正确性的置信度"""
    
    def __init__(self):
        self.confidence_history = []
    
    def calculate_confidence(self, decision: Dict[str, Any], evidence: List[Dict[str, Any]]) -> float:
        """计算决策置信度
        
        Args:
            decision: 决策描述
            evidence: 支持决策的证据列表
            
        Returns:
            置信度评分 (0-1)
        """
        # 基础置信度
        confidence = 0.7
        
        # 基于证据数量调整置信度
        if evidence:
            confidence += min(0.3, len(evidence) / 10)  # 最多增加0.3
        
        # 基于推理链长度调整置信度
        if 'reasoning' in decision:
            reasoning_length = len(decision['reasoning'])
            # 推理链越长，置信度越高（在一定范围内）
            confidence += min(0.2, reasoning_length / 500)  # 最多增加0.2
        
        # 基于参数一致性调整置信度
        if 'parameters' in decision:
            params = decision['parameters']
            # 参数值越接近历史平均值，置信度越高
            # 简化实现：随机调整
            confidence += (np.random.rand() - 0.5) * 0.1
        
        # 确保置信度在0-1范围内
        confidence = max(0.0, min(1.0, confidence))
        
        # 记录置信度
        self.confidence_history.append({
            'decision': decision,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        return confidence
    
    def get_confidence_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的置信度历史"""
        return self.confidence_history[-limit:]


class ReasoningTraceLogger:
    """推理轨迹记录器 - 记录推理过程的每一步"""
    
    def __init__(self):
        self.reasoning_traces = []
    
    def log_reasoning_step(self, step: Dict[str, Any]):
        """记录推理步骤
        
        Args:
            step: 推理步骤描述
        """
        reasoning_step = {
            **step,
            'timestamp': datetime.now()
        }
        self.reasoning_traces.append(reasoning_step)
    
    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的推理轨迹
        
        Args:
            limit: 返回的最大轨迹数量
            
        Returns:
            最近的推理轨迹列表
        """
        return self.reasoning_traces[-limit:]
    
    def clear_traces(self):
        """清除推理轨迹"""
        self.reasoning_traces = []


class AnomalyDetector:
    """异常检测器 - 检测推理中的逻辑矛盾或错误"""
    
    def __init__(self):
        self.anomalies = []
    
    def detect_anomalies(self, reasoning_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测推理过程中的异常
        
        Args:
            reasoning_steps: 推理步骤列表
            
        Returns:
            检测到的异常列表
        """
        detected_anomalies = []
        
        if not reasoning_steps:
            return detected_anomalies
        
        # 检查前后步骤的矛盾
        for i in range(1, len(reasoning_steps)):
            current_step = reasoning_steps[i]
            previous_step = reasoning_steps[i-1]
            
            # 简单的矛盾检测：检查结论是否相反
            if ('conclusion' in current_step and 'conclusion' in previous_step):
                current_conclusion = current_step['conclusion'].lower()
                previous_conclusion = previous_step['conclusion'].lower()
                
                # 只有同一属性的相反描述才被视为矛盾
                # 检查明显的矛盾关键词对，且必须涉及同一属性
                
                # 1. 检查温度相关矛盾
                if ('温度' in current_conclusion and '温度' in previous_conclusion):
                    if ('高于' in current_conclusion and '低于' in previous_conclusion) or \
                       ('低于' in current_conclusion and '高于' in previous_conclusion) or \
                       ('高温' in current_conclusion and '低温' in previous_conclusion) or \
                       ('低温' in current_conclusion and '高温' in previous_conclusion):
                        anomaly = {
                            'type': 'contradiction',
                            'step_index': i,
                            'message': f"步骤 {i} 与步骤 {i-1} 存在温度矛盾",
                            'current_conclusion': current_conclusion,
                            'previous_conclusion': previous_conclusion
                        }
                        detected_anomalies.append(anomaly)
                
                # 2. 检查湿度相关矛盾
                elif ('湿度' in current_conclusion and '湿度' in previous_conclusion):
                    if ('高于' in current_conclusion and '低于' in previous_conclusion) or \
                       ('低于' in current_conclusion and '高于' in previous_conclusion) or \
                       ('高湿度' in current_conclusion and '低湿度' in previous_conclusion) or \
                       ('低湿度' in current_conclusion and '高湿度' in previous_conclusion):
                        anomaly = {
                            'type': 'contradiction',
                            'step_index': i,
                            'message': f"步骤 {i} 与步骤 {i-1} 存在湿度矛盾",
                            'current_conclusion': current_conclusion,
                            'previous_conclusion': previous_conclusion
                        }
                        detected_anomalies.append(anomaly)
                
                # 3. 检查通用操作矛盾
                elif ('开启' in current_conclusion and '关闭' in previous_conclusion) or \
                     ('关闭' in current_conclusion and '开启' in previous_conclusion) or \
                     ('增加' in current_conclusion and '减少' in previous_conclusion) or \
                     ('减少' in current_conclusion and '增加' in previous_conclusion) or \
                     ('上升' in current_conclusion and '下降' in previous_conclusion) or \
                     ('下降' in current_conclusion and '上升' in previous_conclusion) or \
                     ('是' in current_conclusion and '否' in previous_conclusion) or \
                     ('否' in current_conclusion and '是' in previous_conclusion):
                    # 只有当这些词描述同一属性时才视为矛盾
                    # 简化实现：检查是否包含相同的名词
                    import re
                    # 提取名词
                    def extract_nouns(text):
                        # 简化实现：提取中文名词
                        return [word for word in text if re.match(r'[\u4e00-\u9fa5]{2,}', word)]
                    
                    current_nouns = extract_nouns(current_conclusion)
                    previous_nouns = extract_nouns(previous_conclusion)
                    
                    # 如果有共同名词，则视为矛盾
                    if any(noun in previous_nouns for noun in current_nouns):
                        anomaly = {
                            'type': 'contradiction',
                            'step_index': i,
                            'message': f"步骤 {i} 与步骤 {i-1} 存在矛盾",
                            'current_conclusion': current_conclusion,
                            'previous_conclusion': previous_conclusion
                        }
                        detected_anomalies.append(anomaly)
        
        # 记录检测到的异常
        for anomaly in detected_anomalies:
            self.anomalies.append({
                **anomaly,
                'timestamp': datetime.now()
            })
        
        return detected_anomalies
    
    def get_recent_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近检测到的异常"""
        return self.anomalies[-limit:]


class LearningStrategySelector:
    """学习策略选择器 - 根据任务类型选择合适的学习方法"""
    
    def __init__(self):
        self.strategy_history = []
        self.available_strategies = [
            'reinforcement_learning',
            'supervised_learning',
            'unsupervised_learning',
            'transfer_learning',
            'meta_learning'
        ]
    
    def select_strategy(self, task: Dict[str, Any]) -> str:
        """选择合适的学习策略
        
        Args:
            task: 任务描述
            
        Returns:
            选择的学习策略名称
        """
        # 基于任务类型选择策略
        if 'task_type' in task:
            task_type = task['task_type'].lower()
            
            if task_type in ['classification', 'regression']:
                strategy = 'supervised_learning'
            elif task_type in ['clustering', 'dimensionality_reduction']:
                strategy = 'unsupervised_learning'
            elif task_type in ['control', 'game_playing']:
                strategy = 'reinforcement_learning'
            elif task_type in ['cross_domain', 'knowledge_transfer']:
                strategy = 'transfer_learning'
            elif task_type in ['few_shot', 'rapid_adaptation']:
                strategy = 'meta_learning'
            else:
                # 默认使用强化学习
                strategy = 'reinforcement_learning'
        else:
            # 基于任务复杂度选择
            complexity = task.get('complexity', 0.5)
            if complexity > 0.7:
                strategy = 'meta_learning'  # 复杂任务使用元学习
            elif complexity > 0.5:
                strategy = 'transfer_learning'  # 中等复杂度任务使用迁移学习
            else:
                strategy = 'reinforcement_learning'  # 简单任务使用强化学习
        
        # 记录策略选择
        self.strategy_history.append({
            'task': task,
            'strategy': strategy,
            'timestamp': datetime.now()
        })
        
        return strategy
    
    def evaluate_strategy_effectiveness(self, strategy: str, success: bool, performance: float) -> Dict[str, Any]:
        """评估学习策略的有效性
        
        Args:
            strategy: 学习策略名称
            success: 是否成功
            performance: 性能指标
            
        Returns:
            策略评估结果
        """
        effectiveness_score = performance if success else 0.0
        
        evaluation = {
            'strategy': strategy,
            'success': success,
            'performance': performance,
            'effectiveness_score': effectiveness_score,
            'timestamp': datetime.now()
        }
        
        return evaluation


class ResourceAllocator:
    """资源分配器 - 动态调整计算资源和时间分配"""
    
    def __init__(self):
        self.resource_usage_history = []
        self.current_allocation = {
            'cpu': 0.5,  # CPU使用率比例
            'memory': 0.5,  # 内存使用率比例
            'time_budget': 60.0  # 时间预算（秒）
        }
    
    def allocate_resources(self, task_difficulty: float, urgency: float = 0.5) -> Dict[str, Any]:
        """分配资源
        
        Args:
            task_difficulty: 任务难度
            urgency: 任务紧急程度
            
        Returns:
            资源分配方案
        """
        # 基于任务难度和紧急程度分配资源
        cpu_allocation = min(1.0, 0.3 + (task_difficulty * 0.5) + (urgency * 0.2))
        memory_allocation = min(1.0, 0.3 + (task_difficulty * 0.5))
        time_budget = 30.0 + (task_difficulty * 60.0) + (urgency * 30.0)
        
        allocation = {
            'cpu': cpu_allocation,
            'memory': memory_allocation,
            'time_budget': time_budget
        }
        
        # 更新当前分配
        self.current_allocation = allocation
        
        # 记录资源分配
        self.resource_usage_history.append({
            'allocation': allocation,
            'task_difficulty': task_difficulty,
            'urgency': urgency,
            'timestamp': datetime.now()
        })
        
        return allocation
    
    def get_current_allocation(self) -> Dict[str, Any]:
        """获取当前资源分配"""
        return self.current_allocation.copy()
    
    def update_resource_usage(self, actual_usage: Dict[str, Any]):
        """更新资源使用情况
        
        Args:
            actual_usage: 实际资源使用情况
        """
        self.resource_usage_history.append({
            'actual_usage': actual_usage,
            'timestamp': datetime.now()
        })


class MetacognitionSystem:
    """元认知监控系统 - 分层元认知架构"""
    
    def __init__(self):
        # 初始化子模块
        self.ability_evaluator = AbilityEvaluator()
        self.confidence_calculator = ConfidenceCalculator()
        self.reasoning_logger = ReasoningTraceLogger()
        self.anomaly_detector = AnomalyDetector()
        self.strategy_selector = LearningStrategySelector()
        self.resource_allocator = ResourceAllocator()
        
        logger.info("元认知监控系统初始化成功")
    
    def self_assessment(self, task: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """自我评估
        
        Args:
            task: 任务描述
            decision: 决策结果
            
        Returns:
            自我评估结果
        """
        # 评估任务难度
        task_difficulty = self.ability_evaluator.evaluate_task_difficulty(task)
        
        # 计算决策置信度
        confidence = self.confidence_calculator.calculate_confidence(decision, [])
        
        # 获取能力评分
        ability_scores = self.ability_evaluator.get_ability_scores()
        
        return {
            'task_difficulty': task_difficulty,
            'confidence': confidence,
            'ability_scores': ability_scores
        }
    
    def monitor_process(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """过程监控
        
        Args:
            reasoning_steps: 推理步骤列表
            
        Returns:
            过程监控结果
        """
        # 记录推理步骤
        for step in reasoning_steps:
            self.reasoning_logger.log_reasoning_step(step)
        
        # 检测异常
        anomalies = self.anomaly_detector.detect_anomalies(reasoning_steps)
        
        return {
            'reasoning_steps_count': len(reasoning_steps),
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies
        }
    
    def adjust_strategy(self, task: Dict[str, Any], assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        """策略调整
        
        Args:
            task: 任务描述
            assessment_result: 自我评估结果
            
        Returns:
            策略调整建议
        """
        # 选择学习策略
        strategy = self.strategy_selector.select_strategy(task)
        
        # 分配资源
        resource_allocation = self.resource_allocator.allocate_resources(
            assessment_result['task_difficulty']
        )
        
        return {
            'selected_strategy': strategy,
            'resource_allocation': resource_allocation
        }
    
    def learn_from_experience(self, task: Dict[str, Any], success: bool, performance: float):
        """从经验中学习
        
        Args:
            task: 任务描述
            success: 是否成功
            performance: 性能指标
        """
        # 更新能力评分
        task_difficulty = self.ability_evaluator.evaluate_task_difficulty(task)
        for ability_type in self.ability_evaluator.get_ability_scores():
            self.ability_evaluator.evaluate_ability(ability_type, task_difficulty, success)
        
        # 评估策略有效性
        strategy = self.strategy_selector.select_strategy(task)
        self.strategy_selector.evaluate_strategy_effectiveness(strategy, success, performance)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取元认知系统状态
        
        Returns:
            系统状态信息
        """
        return {
            'ability_scores': self.ability_evaluator.get_ability_scores(),
            'current_confidence': self.confidence_calculator.calculate_confidence({}, []),
            'recent_anomalies': len(self.anomaly_detector.get_recent_anomalies(limit=5)),
            'current_allocation': self.resource_allocator.get_current_allocation(),
            'reasoning_steps_count': len(self.reasoning_logger.get_recent_traces(limit=10))
        }
