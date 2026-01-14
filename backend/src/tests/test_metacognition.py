# Test cases for metacognition system
import pytest
from src.core.metacognition import (
    MetacognitionSystem,
    AbilityEvaluator,
    ConfidenceCalculator,
    ReasoningTraceLogger,
    AnomalyDetector,
    LearningStrategySelector,
    ResourceAllocator
)


class TestAbilityEvaluator:
    """测试能力评估器"""
    
    def test_initialization(self):
        """测试初始化"""
        evaluator = AbilityEvaluator()
        assert isinstance(evaluator, AbilityEvaluator)
        assert hasattr(evaluator, 'task_difficulty_history')
        assert hasattr(evaluator, 'ability_scores')
    
    def test_evaluate_task_difficulty(self):
        """测试评估任务难度"""
        evaluator = AbilityEvaluator()
        task = {
            'parameters': {
                'param1': 1.0,
                'param2': 2.0,
                'param3': 3.0
            }
        }
        
        difficulty = evaluator.evaluate_task_difficulty(task)
        assert isinstance(difficulty, float)
        assert 0.0 <= difficulty <= 1.0
    
    def test_evaluate_ability(self):
        """测试评估特定能力"""
        evaluator = AbilityEvaluator()
        ability_type = 'decision_making'
        task_difficulty = 0.7
        success = True
        
        # 保存初始分数
        initial_score = evaluator.ability_scores[ability_type]

        new_score = evaluator.evaluate_ability(ability_type, task_difficulty, success)
        assert isinstance(new_score, float)
        assert 0.0 <= new_score <= 1.0
        assert new_score > initial_score  # 成功完成任务提升分数


class TestConfidenceCalculator:
    """测试信心计算器"""
    
    def test_initialization(self):
        """测试初始化"""
        calculator = ConfidenceCalculator()
        assert isinstance(calculator, ConfidenceCalculator)
        assert hasattr(calculator, 'confidence_history')
    
    def test_calculate_confidence(self):
        """测试计算决策置信度"""
        calculator = ConfidenceCalculator()
        decision = {
            'reasoning': "温度高于30度且湿度低于40%，需要增加灌溉",
            'parameters': {
                'duration': 30,
                'amount': 50
            }
        }
        
        confidence = calculator.calculate_confidence(decision, [])
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_confidence_history(self):
        """测试置信度历史记录"""
        calculator = ConfidenceCalculator()
        
        # 添加多个决策的置信度
        for i in range(5):
            decision = {
                'reasoning': f"决策 {i}",
                'parameters': {
                    'param1': i * 0.5
                }
            }
            calculator.calculate_confidence(decision, [])
        
        history = calculator.get_confidence_history()
        assert len(history) == 5


class TestReasoningTraceLogger:
    """测试推理轨迹记录器"""
    
    def test_initialization(self):
        """测试初始化"""
        logger = ReasoningTraceLogger()
        assert isinstance(logger, ReasoningTraceLogger)
        assert hasattr(logger, 'reasoning_traces')
    
    def test_log_reasoning_step(self):
        """测试记录推理步骤"""
        logger = ReasoningTraceLogger()
        step = {'step': 'decision_evaluation', 'conclusion': "决策质量: 0.8"}
        
        logger.log_reasoning_step(step)
        assert len(logger.reasoning_traces) == 1
        assert 'timestamp' in logger.reasoning_traces[0]
    
    def test_get_recent_traces(self):
        """测试获取最近的推理轨迹"""
        logger = ReasoningTraceLogger()
        
        # 记录多个推理步骤
        for i in range(15):
            step = {'step': f'step_{i}', 'conclusion': f"结论 {i}"}
            logger.log_reasoning_step(step)
        
        # 获取最近10个轨迹
        recent_traces = logger.get_recent_traces(limit=10)
        assert len(recent_traces) == 10
    
    def test_clear_traces(self):
        """测试清除推理轨迹"""
        logger = ReasoningTraceLogger()
        step = {'step': 'test_step', 'conclusion': "测试结论"}
        logger.log_reasoning_step(step)
        
        logger.clear_traces()
        assert len(logger.reasoning_traces) == 0


class TestAnomalyDetector:
    """测试异常检测器"""
    
    def test_initialization(self):
        """测试初始化"""
        detector = AnomalyDetector()
        assert isinstance(detector, AnomalyDetector)
        assert hasattr(detector, 'anomalies')
    
    def test_detect_anomalies(self):
        """测试检测异常"""
        detector = AnomalyDetector()
        
        # 无矛盾的推理步骤
        consistent_steps = [
            {'conclusion': "温度高于30度"},
            {'conclusion': "湿度低于40%"},
            {'conclusion': "需要增加灌溉"}
        ]
        anomalies = detector.detect_anomalies(consistent_steps)
        assert len(anomalies) == 0
        
        # 有矛盾的推理步骤
        inconsistent_steps = [
            {'conclusion': "温度高于30度"},
            {'conclusion': "温度低于20度"},
            {'conclusion': "需要增加灌溉"}
        ]
        anomalies = detector.detect_anomalies(inconsistent_steps)
        assert len(anomalies) == 1
        assert 'contradiction' in anomalies[0]['type']


class TestLearningStrategySelector:
    """测试学习策略选择器"""
    
    def test_initialization(self):
        """测试初始化"""
        selector = LearningStrategySelector()
        assert isinstance(selector, LearningStrategySelector)
        assert hasattr(selector, 'strategy_history')
    
    def test_select_strategy(self):
        """测试选择学习策略"""
        selector = LearningStrategySelector()
        
        # 分类任务
        classification_task = {
            'task_type': 'classification',
            'complexity': 0.5
        }
        strategy = selector.select_strategy(classification_task)
        assert strategy == 'supervised_learning'
        
        # 控制任务
        control_task = {
            'task_type': 'control',
            'complexity': 0.5
        }
        strategy = selector.select_strategy(control_task)
        assert strategy == 'reinforcement_learning'
    
    def test_evaluate_strategy_effectiveness(self):
        """测试评估学习策略的有效性"""
        selector = LearningStrategySelector()
        strategy = 'reinforcement_learning'
        success = True
        performance = 0.8
        
        evaluation = selector.evaluate_strategy_effectiveness(strategy, success, performance)
        assert isinstance(evaluation, dict)
        assert evaluation['strategy'] == strategy
        assert evaluation['success'] == success
        assert evaluation['performance'] == performance


class TestResourceAllocator:
    """测试资源分配器"""
    
    def test_initialization(self):
        """测试初始化"""
        allocator = ResourceAllocator()
        assert isinstance(allocator, ResourceAllocator)
        assert hasattr(allocator, 'resource_usage_history')
        assert hasattr(allocator, 'current_allocation')
    
    def test_allocate_resources(self):
        """测试分配资源"""
        allocator = ResourceAllocator()
        task_difficulty = 0.7
        urgency = 0.8
        
        allocation = allocator.allocate_resources(task_difficulty, urgency)
        assert isinstance(allocation, dict)
        assert 'cpu' in allocation
        assert 'memory' in allocation
        assert 'time_budget' in allocation
        assert 0.0 <= allocation['cpu'] <= 1.0
        assert 0.0 <= allocation['memory'] <= 1.0
        assert allocation['time_budget'] > 0.0
    
    def test_get_current_allocation(self):
        """测试获取当前资源分配"""
        allocator = ResourceAllocator()
        original_allocation = allocator.get_current_allocation()
        
        # 更新资源分配
        allocator.allocate_resources(0.5, 0.5)
        new_allocation = allocator.get_current_allocation()
        
        assert original_allocation != new_allocation


class TestMetacognitionSystem:
    """测试元认知监控系统"""
    
    def test_initialization(self):
        """测试初始化"""
        metacognition = MetacognitionSystem()
        assert isinstance(metacognition, MetacognitionSystem)
        assert hasattr(metacognition, 'ability_evaluator')
        assert hasattr(metacognition, 'confidence_calculator')
        assert hasattr(metacognition, 'reasoning_logger')
        assert hasattr(metacognition, 'anomaly_detector')
        assert hasattr(metacognition, 'strategy_selector')
        assert hasattr(metacognition, 'resource_allocator')
    
    def test_self_assessment(self):
        """测试自我评估"""
        metacognition = MetacognitionSystem()
        task = {
            'type': 'test_task',
            'complexity': 0.6,
            'parameters': {
                'param1': 1.0
            }
        }
        decision = {
            'action': 'test_action',
            'reasoning': "测试推理",
            'confidence': 0.8
        }
        
        assessment = metacognition.self_assessment(task, decision)
        assert isinstance(assessment, dict)
        assert 'task_difficulty' in assessment
        assert 'confidence' in assessment
        assert 'ability_scores' in assessment
    
    def test_monitor_process(self):
        """测试过程监控"""
        metacognition = MetacognitionSystem()
        reasoning_steps = [
            {'step': 'step1', 'conclusion': "温度高于30度"},
            {'step': 'step2', 'conclusion': "湿度低于40%"},
            {'step': 'step3', 'conclusion': "需要增加灌溉"}
        ]
        
        monitoring_result = metacognition.monitor_process(reasoning_steps)
        assert isinstance(monitoring_result, dict)
        assert 'reasoning_steps_count' in monitoring_result
        assert 'anomalies_detected' in monitoring_result
        assert monitoring_result['reasoning_steps_count'] == 3
        assert monitoring_result['anomalies_detected'] == 0
    
    def test_adjust_strategy(self):
        """测试策略调整"""
        metacognition = MetacognitionSystem()
        task = {
            'type': 'test_task',
            'complexity': 0.7
        }
        assessment_result = {
            'task_difficulty': 0.7,
            'confidence': 0.8,
            'ability_scores': {
                'decision_making': 0.7,
                'learning_speed': 0.6
            }
        }
        
        strategy_adjustment = metacognition.adjust_strategy(task, assessment_result)
        assert isinstance(strategy_adjustment, dict)
        assert 'selected_strategy' in strategy_adjustment
        assert 'resource_allocation' in strategy_adjustment
    
    def test_learn_from_experience(self):
        """测试从经验中学习"""
        metacognition = MetacognitionSystem()
        task = {
            'type': 'test_task',
            'complexity': 0.5
        }
        success = True
        performance = 0.8
        
        metacognition.learn_from_experience(task, success, performance)
        # 验证方法执行成功，无异常
        assert True
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        metacognition = MetacognitionSystem()
        status = metacognition.get_system_status()
        assert isinstance(status, dict)
        assert 'ability_scores' in status
        assert 'current_confidence' in status
        assert 'recent_anomalies' in status
        assert 'current_allocation' in status
        assert 'reasoning_steps_count' in status
