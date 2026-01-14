"""
好奇心机制评估指标模块
实现技术文档中描述的探索效率、学习效果和适应性评估指标
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class CuriosityMetric:
    """好奇心机制指标数据类"""
    timestamp: datetime
    metric_type: str
    value: float
    tags: Dict[str, Any]


class CuriosityEvaluator:
    """好奇心机制评估器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self.explored_states = set()  # 已探索的独特状态集合
        self.total_states = 0  # 环境中总状态数（需要外部设置）
        self.exploration_steps = 0  # 总探索步数
        self.new_states_discovered = 0  # 发现的新状态数
        self.cumulative_information_gain = 0.0  # 累计信息增益
        self.exploration_start_time = time.time()  # 探索开始时间
        self.task_success_count = 0  # 成功完成的任务数
        self.total_tasks = 0  # 总任务数
        self.environment_changes = []  # 环境变化记录
        self.strategy_switches = []  # 策略切换记录
        self.exploration_time = 0.0  # 探索时间
        self.exploitation_time = 0.0  # 利用时间
        
    async def set_total_states(self, total_states: int):
        """设置环境中总状态数"""
        self.total_states = total_states
        logger.info(f"设置环境总状态数: {total_states}")
    
    async def record_explored_state(self, state: Any, is_new: bool = False):
        """记录探索的状态"""
        # 将状态转换为可哈希类型以便存储
        state_hash = hash(str(state))
        
        if is_new and state_hash not in self.explored_states:
            self.explored_states.add(state_hash)
            self.new_states_discovered += 1
        
        self.exploration_steps += 1
    
    async def record_information_gain(self, gain: float):
        """记录信息增益"""
        self.cumulative_information_gain += gain
    
    async def record_task_completion(self, success: bool):
        """记录任务完成情况"""
        self.total_tasks += 1
        if success:
            self.task_success_count += 1
    
    async def record_environment_change(self, timestamp: float = None):
        """记录环境变化"""
        if timestamp is None:
            timestamp = time.time()
        self.environment_changes.append(timestamp)
    
    async def record_strategy_switch(self, success: bool):
        """记录策略切换情况"""
        self.strategy_switches.append(success)
    
    async def record_exploration_time(self, duration: float):
        """记录探索时间"""
        self.exploration_time += duration
    
    async def record_exploitation_time(self, duration: float):
        """记录利用时间"""
        self.exploitation_time += duration
    
    async def record_metric(self, metric_type: str, value: float, tags: Dict[str, Any] = None):
        """记录好奇心机制指标"""
        
        metric = CuriosityMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            tags=tags or {}
        )
        
        self.metrics_history[metric_type].append(metric)
        logger.debug(f"记录好奇心指标: {metric_type} = {value}")
    
    def calculate_exploration_coverage(self) -> float:
        """计算探索覆盖率
        探索覆盖率 = 已探索的独特状态数 / 环境中总状态数
        """
        if self.total_states == 0:
            return 0.0
        
        coverage = len(self.explored_states) / self.total_states
        return round(coverage, 4)
    
    def calculate_path_efficiency(self, shortest_path: float, actual_path: float) -> float:
        """计算探索路径效率
        探索路径效率 = 最短路径长度 / 实际探索路径长度
        """
        if actual_path == 0:
            return 0.0
        
        efficiency = shortest_path / actual_path
        return round(efficiency, 4)
    
    def calculate_novelty_discovery_rate(self) -> float:
        """计算新颖性发现率
        新颖性发现率 = 发现的新状态数 / 总探索步数
        """
        if self.exploration_steps == 0:
            return 0.0
        
        rate = self.new_states_discovered / self.exploration_steps
        return round(rate, 6)
    
    def calculate_information_gain_rate(self) -> float:
        """计算信息增益率
        信息增益率 = 累计信息增益 / 总探索时间
        """
        total_time = time.time() - self.exploration_start_time
        if total_time == 0:
            return 0.0
        
        rate = self.cumulative_information_gain / total_time
        return round(rate, 4)
    
    def calculate_task_completion_rate(self) -> float:
        """计算任务完成率
        任务完成率 = 成功完成的任务数 / 总任务数
        """
        if self.total_tasks == 0:
            return 0.0
        
        rate = self.task_success_count / self.total_tasks
        return round(rate, 4)
    
    def calculate_environment_adaptation_time(self) -> Optional[float]:
        """计算环境适应时间
        环境适应时间 = 从环境变化到系统稳定的时间
        """
        if len(self.environment_changes) < 2:
            return None
        
        # 简单实现：计算相邻环境变化之间的时间差
        # 实际实现应根据系统稳定的定义来计算
        adaptation_times = []
        for i in range(1, len(self.environment_changes)):
            adaptation_time = self.environment_changes[i] - self.environment_changes[i-1]
            adaptation_times.append(adaptation_time)
        
        if not adaptation_times:
            return None
        
        return round(statistics.mean(adaptation_times), 4)
    
    def calculate_strategy_switch_success_rate(self) -> float:
        """计算策略切换成功率
        策略切换成功率 = 成功切换策略的次数 / 总切换次数
        """
        if not self.strategy_switches:
            return 0.0
        
        success_count = sum(1 for success in self.strategy_switches if success)
        rate = success_count / len(self.strategy_switches)
        return round(rate, 4)
    
    def calculate_exploration_exploitation_balance(self) -> float:
        """计算探索-利用平衡
        平衡度 = |探索时间比例 - 利用时间比例|
        平衡度越接近0，平衡越好
        """
        total_time = self.exploration_time + self.exploitation_time
        if total_time == 0:
            return 1.0  # 初始状态，完全不平衡
        
        exploration_ratio = self.exploration_time / total_time
        exploitation_ratio = self.exploitation_time / total_time
        balance = abs(exploration_ratio - exploitation_ratio)
        return round(balance, 4)
    
    async def calculate_all_metrics(self) -> Dict[str, float]:
        """计算所有好奇心机制指标"""
        # 计算探索效率指标
        exploration_coverage = self.calculate_exploration_coverage()
        novelty_discovery_rate = self.calculate_novelty_discovery_rate()
        information_gain_rate = self.calculate_information_gain_rate()
        
        # 计算学习效果指标
        task_completion_rate = self.calculate_task_completion_rate()
        
        # 计算适应性评估指标
        environment_adaptation_time = self.calculate_environment_adaptation_time() or 0.0
        strategy_switch_success_rate = self.calculate_strategy_switch_success_rate()
        exploration_exploitation_balance = self.calculate_exploration_exploitation_balance()
        
        # 记录所有指标
        await self.record_metric("exploration_coverage", exploration_coverage)
        await self.record_metric("novelty_discovery_rate", novelty_discovery_rate)
        await self.record_metric("information_gain_rate", information_gain_rate)
        await self.record_metric("task_completion_rate", task_completion_rate)
        await self.record_metric("environment_adaptation_time", environment_adaptation_time)
        await self.record_metric("strategy_switch_success_rate", strategy_switch_success_rate)
        await self.record_metric("exploration_exploitation_balance", exploration_exploitation_balance)
        
        return {
            "exploration_coverage": exploration_coverage,
            "novelty_discovery_rate": novelty_discovery_rate,
            "information_gain_rate": information_gain_rate,
            "task_completion_rate": task_completion_rate,
            "environment_adaptation_time": environment_adaptation_time,
            "strategy_switch_success_rate": strategy_switch_success_rate,
            "exploration_exploitation_balance": exploration_exploitation_balance
        }
    
    def get_metrics_summary(self, metric_type: str, time_range: str = "1h") -> Dict[str, Any]:
        """获取指标摘要"""
        
        if metric_type not in self.metrics_history:
            return {"error": f"未知指标类型: {metric_type}"}
        
        # 计算时间范围
        end_time = datetime.now()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(days=1)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)  # 默认1小时
        
        # 过滤时间范围内的指标
        metrics_in_range = [
            m for m in self.metrics_history[metric_type]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not metrics_in_range:
            return {"count": 0, "message": "指定时间范围内无数据"}
        
        values = [m.value for m in metrics_in_range]
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "count": len(metrics_in_range),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "latest_value": values[-1] if values else 0,
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势（上升/下降/稳定）"""
        
        if len(values) < 2:
            return "stable"
        
        # 简单趋势计算
        first_half_avg = statistics.mean(values[:len(values)//2])
        second_half_avg = statistics.mean(values[len(values)//2:])
        
        if second_half_avg > first_half_avg * 1.1:  # 上升10%
            return "rising"
        elif second_half_avg < first_half_avg * 0.9:  # 下降10%
            return "falling"
        else:
            return "stable"
    
    async def get_evaluation_report(self, time_range: str = "24h") -> Dict[str, Any]:
        """获取好奇心机制评估报告"""
        # 先计算所有指标
        metrics = await self.calculate_all_metrics()
        
        # 获取各指标的时间范围摘要
        exploration_coverage_summary = self.get_metrics_summary("exploration_coverage", time_range)
        novelty_discovery_summary = self.get_metrics_summary("novelty_discovery_rate", time_range)
        information_gain_summary = self.get_metrics_summary("information_gain_rate", time_range)
        task_completion_summary = self.get_metrics_summary("task_completion_rate", time_range)
        
        # 生成评估报告
        report = {
            "timestamp": datetime.now(),
            "metrics": metrics,
            "exploration_efficiency": {
                "exploration_coverage": exploration_coverage_summary,
                "novelty_discovery_rate": novelty_discovery_summary,
                "information_gain_rate": information_gain_summary
            },
            "learning_effect": {
                "task_completion_rate": task_completion_summary,
                "total_tasks": self.total_tasks,
                "successful_tasks": self.task_success_count
            },
            "adaptability": {
                "environment_adaptation_time": metrics["environment_adaptation_time"],
                "strategy_switch_success_rate": metrics["strategy_switch_success_rate"],
                "exploration_exploitation_balance": metrics["exploration_exploitation_balance"]
            },
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于探索覆盖率的建议
        if metrics["exploration_coverage"] < 0.3:
            recommendations.append("探索覆盖率较低，建议增加探索策略的多样性")
        elif metrics["exploration_coverage"] < 0.7:
            recommendations.append("探索覆盖率中等，可考虑调整探索与利用的平衡")
        else:
            recommendations.append("探索覆盖率较高，建议增加利用时间以巩固学习成果")
        
        # 基于新颖性发现率的建议
        if metrics["novelty_discovery_rate"] < 0.01:
            recommendations.append("新颖性发现率较低，建议使用更激进的探索策略")
        
        # 基于探索-利用平衡的建议
        if metrics["exploration_exploitation_balance"] > 0.5:
            recommendations.append("探索与利用严重不平衡，建议调整策略平衡两者比例")
        
        # 基于任务完成率的建议
        if metrics["task_completion_rate"] < 0.5:
            recommendations.append("任务完成率较低，建议优化学习算法或调整探索策略")
        
        return recommendations
    
    def clear_old_metrics(self, older_than_days: int = 7):
        """清理旧指标数据"""
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        cleared_count = 0
        
        for metric_type, metrics in self.metrics_history.items():
            original_count = len(metrics)
            # 保留最近的数据
            self.metrics_history[metric_type] = deque(
                [m for m in metrics if m.timestamp >= cutoff_time],
                maxlen=self.max_history_size
            )
            cleared_count += (original_count - len(self.metrics_history[metric_type]))
        
        logger.info(f"清理了 {cleared_count} 条旧好奇心指标数据")
        return cleared_count
