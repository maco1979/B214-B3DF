import pytest
import asyncio
import sys
import os

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from evaluation.curiosity_metrics import CuriosityEvaluator


@pytest.mark.asyncio
async def test_curiosity_evaluator_initialization():
    """测试好奇心评估器初始化"""
    evaluator = CuriosityEvaluator()
    
    assert evaluator.max_history_size == 1000
    assert len(evaluator.explored_states) == 0
    assert evaluator.total_states == 0
    assert evaluator.exploration_steps == 0
    assert evaluator.new_states_discovered == 0
    assert evaluator.cumulative_information_gain == 0.0
    assert evaluator.task_success_count == 0
    assert evaluator.total_tasks == 0
    assert len(evaluator.environment_changes) == 0
    assert len(evaluator.strategy_switches) == 0
    assert evaluator.exploration_time == 0.0
    assert evaluator.exploitation_time == 0.0


@pytest.mark.asyncio
async def test_set_total_states():
    """测试设置总状态数"""
    evaluator = CuriosityEvaluator()
    total_states = 1000
    
    await evaluator.set_total_states(total_states)
    
    assert evaluator.total_states == total_states


@pytest.mark.asyncio
async def test_record_explored_state():
    """测试记录探索状态"""
    evaluator = CuriosityEvaluator()
    
    # 记录新状态
    await evaluator.record_explored_state("state1", is_new=True)
    assert len(evaluator.explored_states) == 1
    assert evaluator.new_states_discovered == 1
    assert evaluator.exploration_steps == 1
    
    # 记录已探索过的状态
    await evaluator.record_explored_state("state1", is_new=True)
    assert len(evaluator.explored_states) == 1
    assert evaluator.new_states_discovered == 1  # 不增加新状态计数
    assert evaluator.exploration_steps == 2  # 增加探索步数
    
    # 记录多个不同状态
    await evaluator.record_explored_state("state2", is_new=True)
    await evaluator.record_explored_state("state3", is_new=True)
    assert len(evaluator.explored_states) == 3
    assert evaluator.new_states_discovered == 3
    assert evaluator.exploration_steps == 4


@pytest.mark.asyncio
async def test_record_information_gain():
    """测试记录信息增益"""
    evaluator = CuriosityEvaluator()
    
    await evaluator.record_information_gain(1.5)
    await evaluator.record_information_gain(2.5)
    await evaluator.record_information_gain(3.0)
    
    assert evaluator.cumulative_information_gain == 7.0


@pytest.mark.asyncio
async def test_record_task_completion():
    """测试记录任务完成情况"""
    evaluator = CuriosityEvaluator()
    
    # 记录成功任务
    await evaluator.record_task_completion(success=True)
    assert evaluator.task_success_count == 1
    assert evaluator.total_tasks == 1
    
    # 记录失败任务
    await evaluator.record_task_completion(success=False)
    assert evaluator.task_success_count == 1
    assert evaluator.total_tasks == 2
    
    # 记录更多成功任务
    for _ in range(3):
        await evaluator.record_task_completion(success=True)
    
    assert evaluator.task_success_count == 4
    assert evaluator.total_tasks == 5


@pytest.mark.asyncio
async def test_record_environment_change():
    """测试记录环境变化"""
    evaluator = CuriosityEvaluator()
    
    await evaluator.record_environment_change(100.0)
    await evaluator.record_environment_change(200.0)
    await evaluator.record_environment_change(300.0)
    
    assert len(evaluator.environment_changes) == 3
    assert evaluator.environment_changes == [100.0, 200.0, 300.0]


@pytest.mark.asyncio
async def test_record_strategy_switch():
    """测试记录策略切换"""
    evaluator = CuriosityEvaluator()
    
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=False)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=True)
    
    assert len(evaluator.strategy_switches) == 4
    assert evaluator.strategy_switches == [True, False, True, True]


@pytest.mark.asyncio
async def test_record_exploration_exploitation_time():
    """测试记录探索和利用时间"""
    evaluator = CuriosityEvaluator()
    
    await evaluator.record_exploration_time(10.0)
    await evaluator.record_exploration_time(15.0)
    await evaluator.record_exploitation_time(5.0)
    await evaluator.record_exploitation_time(8.0)
    
    assert evaluator.exploration_time == 25.0
    assert evaluator.exploitation_time == 13.0


@pytest.mark.asyncio
async def test_calculate_exploration_coverage():
    """测试计算探索覆盖率"""
    evaluator = CuriosityEvaluator()
    
    # 总状态数为0时，覆盖率为0
    coverage = evaluator.calculate_exploration_coverage()
    assert coverage == 0.0
    
    # 设置总状态数并记录探索状态
    await evaluator.set_total_states(100)
    
    for i in range(30):
        await evaluator.record_explored_state(f"state{i}", is_new=True)
    
    coverage = evaluator.calculate_exploration_coverage()
    assert coverage == 0.30  # 30/100


@pytest.mark.asyncio
async def test_calculate_path_efficiency():
    """测试计算探索路径效率"""
    evaluator = CuriosityEvaluator()
    
    # 实际路径长度为0时，效率为0
    efficiency = evaluator.calculate_path_efficiency(5, 0)
    assert efficiency == 0.0
    
    # 正常情况
    efficiency = evaluator.calculate_path_efficiency(5, 10)
    assert efficiency == 0.5
    
    efficiency = evaluator.calculate_path_efficiency(8, 10)
    assert efficiency == 0.8
    
    efficiency = evaluator.calculate_path_efficiency(10, 10)
    assert efficiency == 1.0


@pytest.mark.asyncio
async def test_calculate_novelty_discovery_rate():
    """测试计算新颖性发现率"""
    evaluator = CuriosityEvaluator()
    
    # 探索步数为0时，发现率为0
    rate = evaluator.calculate_novelty_discovery_rate()
    assert rate == 0.0
    
    # 记录一些状态
    for i in range(50):
        # 每5步记录一个新状态
        is_new = (i % 5 == 0)
        await evaluator.record_explored_state(f"state{i//5}", is_new=is_new)
    
    rate = evaluator.calculate_novelty_discovery_rate()
    assert rate == 0.2  # 10个新状态 / 50步


@pytest.mark.asyncio
async def test_calculate_information_gain_rate():
    """测试计算信息增益率"""
    evaluator = CuriosityEvaluator()
    
    # 总探索时间为0时，增益率为0
    rate = evaluator.calculate_information_gain_rate()
    assert rate == 0.0
    
    # 记录信息增益并等待一小段时间
    await evaluator.record_information_gain(10.0)
    await asyncio.sleep(0.1)  # 等待0.1秒
    
    rate = evaluator.calculate_information_gain_rate()
    assert rate > 0


@pytest.mark.asyncio
async def test_calculate_task_completion_rate():
    """测试计算任务完成率"""
    evaluator = CuriosityEvaluator()
    
    # 总任务数为0时，完成率为0
    rate = evaluator.calculate_task_completion_rate()
    assert rate == 0.0
    
    # 记录任务完成情况
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=False)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=False)
    
    rate = evaluator.calculate_task_completion_rate()
    assert rate == 0.6  # 3个成功 / 5个总任务


@pytest.mark.asyncio
async def test_calculate_environment_adaptation_time():
    """测试计算环境适应时间"""
    evaluator = CuriosityEvaluator()
    
    # 环境变化少于2次时，返回None
    time = evaluator.calculate_environment_adaptation_time()
    assert time is None
    
    # 记录环境变化
    await evaluator.record_environment_change(100.0)
    await evaluator.record_environment_change(200.0)
    await evaluator.record_environment_change(350.0)
    
    time = evaluator.calculate_environment_adaptation_time()
    assert time == 125.0  # 平均适应时间: (100 + 150) / 2 = 125


@pytest.mark.asyncio
async def test_calculate_strategy_switch_success_rate():
    """测试计算策略切换成功率"""
    evaluator = CuriosityEvaluator()
    
    # 策略切换次数为0时，成功率为0
    rate = evaluator.calculate_strategy_switch_success_rate()
    assert rate == 0.0
    
    # 记录策略切换
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=False)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=False)
    
    rate = evaluator.calculate_strategy_switch_success_rate()
    assert rate == 0.6  # 3次成功 / 5次总切换


@pytest.mark.asyncio
async def test_calculate_exploration_exploitation_balance():
    """测试计算探索-利用平衡"""
    evaluator = CuriosityEvaluator()
    
    # 总时间为0时，平衡度为1
    balance = evaluator.calculate_exploration_exploitation_balance()
    assert balance == 1.0
    
    # 探索时间远大于利用时间
    await evaluator.record_exploration_time(90.0)
    await evaluator.record_exploitation_time(10.0)
    balance = evaluator.calculate_exploration_exploitation_balance()
    assert balance == 0.8  # |0.9 - 0.1| = 0.8
    
    # 探索和利用时间相等
    evaluator = CuriosityEvaluator()  # 重置
    await evaluator.record_exploration_time(50.0)
    await evaluator.record_exploitation_time(50.0)
    balance = evaluator.calculate_exploration_exploitation_balance()
    assert balance == 0.0  # |0.5 - 0.5| = 0.0


@pytest.mark.asyncio
async def test_calculate_all_metrics():
    """测试计算所有指标"""
    evaluator = CuriosityEvaluator()
    
    # 设置总状态数
    await evaluator.set_total_states(100)
    
    # 记录一些数据
    for i in range(20):
        is_new = (i % 4 == 0)
        await evaluator.record_explored_state(f"state{i//4}", is_new=is_new)
    
    await evaluator.record_information_gain(20.0)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=False)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_environment_change(100.0)
    await evaluator.record_environment_change(200.0)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=False)
    await evaluator.record_exploration_time(60.0)
    await evaluator.record_exploitation_time(40.0)
    
    # 计算所有指标
    metrics = await evaluator.calculate_all_metrics()
    
    # 验证指标存在
    assert "exploration_coverage" in metrics
    assert "novelty_discovery_rate" in metrics
    assert "information_gain_rate" in metrics
    assert "task_completion_rate" in metrics
    assert "environment_adaptation_time" in metrics
    assert "strategy_switch_success_rate" in metrics
    assert "exploration_exploitation_balance" in metrics
    
    # 验证部分指标值
    assert metrics["exploration_coverage"] == 0.05  # 5个状态 / 100总状态
    assert metrics["task_completion_rate"] == 0.6667  # 2个成功 / 3个总任务
    assert metrics["strategy_switch_success_rate"] == 0.6667  # 2次成功 / 3次总切换
    assert metrics["exploration_exploitation_balance"] == 0.2  # |0.6 - 0.4| = 0.2


@pytest.mark.asyncio
async def test_get_evaluation_report():
    """测试获取评估报告"""
    evaluator = CuriosityEvaluator()
    
    # 设置总状态数
    await evaluator.set_total_states(100)
    
    # 记录一些数据
    for i in range(15):
        is_new = (i % 5 == 0)
        await evaluator.record_explored_state(f"state{i//5}", is_new=is_new)
    
    await evaluator.record_information_gain(15.0)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=True)
    await evaluator.record_task_completion(success=False)
    await evaluator.record_environment_change(100.0)
    await evaluator.record_environment_change(150.0)
    await evaluator.record_environment_change(250.0)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_strategy_switch(success=False)
    await evaluator.record_strategy_switch(success=True)
    await evaluator.record_exploration_time(70.0)
    await evaluator.record_exploitation_time(30.0)
    
    # 获取评估报告
    report = await evaluator.get_evaluation_report()
    
    # 验证报告结构
    assert "timestamp" in report
    assert "metrics" in report
    assert "exploration_efficiency" in report
    assert "learning_effect" in report
    assert "adaptability" in report
    assert "recommendations" in report
    
    # 验证探索效率部分
    assert "exploration_coverage" in report["exploration_efficiency"]
    assert "novelty_discovery_rate" in report["exploration_efficiency"]
    assert "information_gain_rate" in report["exploration_efficiency"]
    
    # 验证学习效果部分
    assert "task_completion_rate" in report["learning_effect"]
    assert "total_tasks" in report["learning_effect"]
    assert "successful_tasks" in report["learning_effect"]
    
    # 验证适应性部分
    assert "environment_adaptation_time" in report["adaptability"]
    assert "strategy_switch_success_rate" in report["adaptability"]
    assert "exploration_exploitation_balance" in report["adaptability"]


@pytest.mark.asyncio
async def test_clear_old_metrics():
    """测试清理旧指标"""
    evaluator = CuriosityEvaluator()
    
    # 记录一些指标
    await evaluator.record_metric("test_metric", 1.0)
    await evaluator.record_metric("test_metric", 2.0)
    await evaluator.record_metric("test_metric", 3.0)
    
    # 验证记录了3个指标
    assert len(evaluator.metrics_history["test_metric"]) == 3
    
    # 清理旧指标（所有指标都在7天内，所以不会清理）
    cleared_count = evaluator.clear_old_metrics(older_than_days=7)
    assert cleared_count == 0
    assert len(evaluator.metrics_history["test_metric"]) == 3
    
    # 清理旧指标（所有指标都超过0天，所以会清理所有）
    cleared_count = evaluator.clear_old_metrics(older_than_days=0)
    assert cleared_count == 3
    assert len(evaluator.metrics_history["test_metric"]) == 0
