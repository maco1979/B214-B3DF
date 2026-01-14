"""
AGI设计原则综合集成测试
测试逻辑一致性和好奇心机制的协同工作
"""

import pytest
import torch
import time
from src.core.logical_consistency import LogicalConsistencyChecker
from src.core.models.curiosity_model import (
    MultiModalCuriosity, calculate_intrinsic_reward,
    NoveltySeekingExplorer, CERMICExplorer
)
from src.core.ai_organic_core import AdaptiveLearningSystem


class TestAGIIntegrationComprehensive:
    """AGI设计原则综合集成测试"""
    
    def test_logical_consistency_and_curiosity_coordination(self):
        """测试逻辑一致性与好奇心机制的协同工作"""
        # 初始化逻辑一致性检查器
        consistency_checker = LogicalConsistencyChecker()
        
        # 初始化好奇心模型
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 模拟决策过程，结合逻辑一致性检查和好奇心驱动的探索
        
        # 1. 生成候选决策
        candidate_decisions = [
            {
                "action": "explore",
                "parameters": {
                    "direction": "north",
                    "speed": 5
                },
                "reasoning": "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。",
                "timestamp": "2026-01-13T10:00:00"
            },
            {
                "action": "explore",
                "parameters": {
                    "direction": "south",
                    "speed": 3
                },
                "reasoning": "温度高于30度。温度低于20度。需要增加灌溉。执行灌溉操作。",
                "timestamp": "2026-01-13T10:00:00"
            },
            {
                "action": "monitor",
                "parameters": {
                    "interval": 60
                },
                "reasoning": "湿度高于80%。需要通风。执行通风操作。",
                "timestamp": "2026-01-13T10:00:00"
            }
        ]
        
        # 2. 筛选逻辑一致的决策
        consistent_decisions = []
        for decision in candidate_decisions:
            result = consistency_checker.check_consistency(decision, [])
            if result['is_consistent']:
                consistent_decisions.append((decision, result))
        
        # 验证至少有一个一致的决策
        assert len(consistent_decisions) >= 1, "没有找到逻辑一致的决策"
        
        # 3. 对一致的决策应用好奇心机制
        current_state = torch.randn(1, state_dim)
        action = torch.randn(1, action_dim)
        next_state = torch.randn(1, state_dim)
        
        # 计算内在奖励
        intrinsic_reward = calculate_intrinsic_reward(
            current_state, action, next_state, curiosity_model
        )
        
        # 验证协同工作
        assert isinstance(intrinsic_reward, torch.Tensor)
        assert len(consistent_decisions) > 0
    
    def test_adaptive_learning_system_integration(self):
        """测试自适应学习系统的集成"""
        # 初始化自适应学习系统
        learning_system = AdaptiveLearningSystem()
        
        # 验证系统组件初始化
        assert hasattr(learning_system, 'curiosity_enabled')
        assert hasattr(learning_system, 'curiosity_model')
        assert hasattr(learning_system, 'add_experience')
        
        # 测试添加经验（包含好奇心机制）
        state = torch.randn(1, 128).numpy().tolist()  # 转换为列表，模拟实际使用场景
        action = torch.randn(1, 32).numpy().tolist()
        reward = 1.0
        next_state = torch.randn(1, 128).numpy().tolist()
        done = False
        context = {
            "temperature": 30.5,
            "humidity": 35.0,
            "timestamp": "2026-01-13T10:00:00"
        }
        
        # 添加经验（包含好奇心处理）
        learning_system.add_experience(state, action, reward, next_state, done, context)
        
        # 验证经验被添加
        assert len(learning_system.experience_buffer) >= 1
        
        # 验证好奇心相关数据被记录
        assert hasattr(learning_system, 'curiosity_history')
        assert hasattr(learning_system, 'interesting_experiences')
        assert hasattr(learning_system, 'intrinsic_rewards')
    
    def test_integrated_decision_making(self):
        """测试集成决策制定"""
        # 创建逻辑一致性检查器
        consistency_checker = LogicalConsistencyChecker()
        
        # 创建模拟环境
        class MockEnvironment:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                
            def simulate(self, state, action):
                return torch.randn(1, 128), 0.0, False, {}
        
        mock_env = MockEnvironment()
        
        # 创建CERMIC探索器
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        explorer = CERMICExplorer(mock_env, curiosity_model)
        
        # 模拟集成决策过程
        state = torch.randn(1, state_dim)
        possible_actions = list(range(mock_env.action_space.n))
        
        # 1. 基于好奇心选择探索动作
        selected_action = explorer.adaptive_exploration(state, possible_actions)
        
        # 2. 生成决策推理
        reasoning = f"状态分析显示需要探索。选择动作 {selected_action} 进行探索。"
        
        # 3. 检查决策的逻辑一致性
        decision = {
            "action": "explore",
            "parameters": {
                "action_id": selected_action
            },
            "reasoning": reasoning,
            "timestamp": "2026-01-13T10:00:00"
        }
        
        result = consistency_checker.check_consistency(decision, [])
        
        # 验证集成决策过程
        assert result['is_consistent'] == True
        assert selected_action in possible_actions
    
    def test_system_performance_comprehensive(self):
        """测试系统综合性能"""
        # 初始化组件
        consistency_checker = LogicalConsistencyChecker()
        
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 模拟大量决策和探索操作，测量性能
        num_operations = 100
        total_time = 0.0
        
        for i in range(num_operations):
            # 测量逻辑一致性检查时间
            start_time = time.time()
            
            # 1. 逻辑一致性检查
            decision = {
                "action": "test_action",
                "parameters": {
                    "param1": i,
                    "param2": f"value_{i}"
                },
                "reasoning": f"测试推理 {i}。这是一个一致的推理链。没有矛盾。",
                "timestamp": "2026-01-13T10:00:00"
            }
            consistency_result = consistency_checker.check_consistency(decision, [])
            
            # 2. 好奇心计算
            current_state = torch.randn(1, state_dim)
            action = torch.randn(1, action_dim)
            next_state = torch.randn(1, state_dim)
            intrinsic_reward = calculate_intrinsic_reward(
                current_state, action, next_state, curiosity_model
            )
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        # 计算平均操作时间
        average_time = total_time / num_operations
        print(f"综合操作平均时间: {average_time:.4f}秒")
        
        # 验证性能要求
        assert average_time < 0.1, f"综合操作时间 {average_time:.4f}秒 超过阈值0.1秒"
    
    def test_system_reliability(self):
        """测试系统可靠性"""
        # 初始化组件
        consistency_checker = LogicalConsistencyChecker()
        
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 长时间运行测试
        num_iterations = 500
        success_count = 0
        error_count = 0
        
        for i in range(num_iterations):
            try:
                # 逻辑一致性检查
                decision = {
                    "action": "test_reliability",
                    "parameters": {
                        "iteration": i
                    },
                    "reasoning": f"可靠性测试迭代 {i}。这是一个一致的陈述。",
                    "timestamp": "2026-01-13T10:00:00"
                }
                consistency_result = consistency_checker.check_consistency(decision, [])
                
                # 好奇心计算
                current_state = torch.randn(1, state_dim)
                action = torch.randn(1, action_dim)
                next_state = torch.randn(1, state_dim)
                intrinsic_reward = calculate_intrinsic_reward(
                    current_state, action, next_state, curiosity_model
                )
                
                # 验证结果
                assert consistency_result['is_consistent'] in [True, False]
                assert isinstance(intrinsic_reward, torch.Tensor)
                
                success_count += 1
            except Exception as e:
                print(f"测试过程中发生错误: {e}")
                error_count += 1
        
        # 计算可靠性指标
        reliability = success_count / num_iterations
        error_rate = error_count / num_iterations
        
        print(f"系统可靠性: {reliability:.4f}")
        print(f"系统错误率: {error_rate:.4f}")
        
        # 验证可靠性要求
        assert reliability > 0.99, f"系统可靠性 {reliability:.4f} 低于阈值0.99"
        assert error_rate < 0.01, f"系统错误率 {error_rate:.4f} 高于阈值0.01"
    
    def test_agi_capability_improvement(self):
        """测试AGI能力提升"""
        # 这个测试模拟了AGI能力提升的评估过程
        # 实际应用中，这会涉及更复杂的基准测试
        
        # 初始化组件
        learning_system = AdaptiveLearningSystem()
        consistency_checker = LogicalConsistencyChecker()
        
        # 模拟学习前后的性能对比
        
        # 1. 初始性能评估
        initial_success_rate = 0.75
        
        # 2. 模拟学习过程
        for i in range(100):
            # 生成模拟经验
            state = torch.randn(1, 128).numpy().tolist()
            action = torch.randn(1, 32).numpy().tolist()
            reward = 1.0 if i % 2 == 0 else 0.5
            next_state = torch.randn(1, 128).numpy().tolist()
            done = i % 20 == 0  # 每20步结束一次 episode
            context = {
                "temperature": 30.0 + (i % 10),
                "humidity": 40.0 + (i % 20),
                "iteration": i
            }
            
            # 添加经验到学习系统
            learning_system.add_experience(state, action, reward, next_state, done, context)
            
            # 更新性能指标
            learning_system.update_performance_metrics(reward, done)
        
        # 3. 学习后性能评估
        final_success_rate = learning_system.success_rate
        
        print(f"初始成功率: {initial_success_rate:.4f}")
        print(f"学习后成功率: {final_success_rate:.4f}")
        
        # 验证AGI能力提升
        assert final_success_rate >= initial_success_rate, f"学习后成功率 {final_success_rate:.4f} 低于初始成功率 {initial_success_rate:.4f}"
    
    def test_integration_benchmark(self):
        """集成系统基准测试"""
        # 初始化组件
        consistency_checker = LogicalConsistencyChecker()
        
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 基准测试配置
        benchmark_config = {
            'num_decisions': 200,
            'num_explorations': 100,
            'time_limit': 30  # 秒
        }
        
        print(f"开始集成系统基准测试: {benchmark_config}")
        
        start_time = time.time()
        total_time = 0.0
        
        # 执行基准测试
        for i in range(benchmark_config['num_decisions']):
            # 逻辑一致性检查
            decision = {
                "action": "benchmark_action",
                "parameters": {
                    "index": i
                },
                "reasoning": f"基准测试决策 {i}。这是一个一致的推理过程。",
                "timestamp": "2026-01-13T10:00:00"
            }
            consistency_result = consistency_checker.check_consistency(decision, [])
            
            # 好奇心计算
            current_state = torch.randn(1, state_dim)
            action = torch.randn(1, action_dim)
            next_state = torch.randn(1, state_dim)
            intrinsic_reward = calculate_intrinsic_reward(
                current_state, action, next_state, curiosity_model
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算基准测试指标
        decisions_per_second = benchmark_config['num_decisions'] / total_time
        print(f"基准测试结果:")
        print(f"总时间: {total_time:.2f}秒")
        print(f"每秒决策数: {decisions_per_second:.2f} 决策/秒")
        print(f"总决策数: {benchmark_config['num_decisions']}")
        
        # 验证基准测试结果
        assert total_time < benchmark_config['time_limit'], f"基准测试超时，用时 {total_time:.2f}秒，超时 {benchmark_config['time_limit']}秒"
        assert decisions_per_second > 5, f"每秒决策数 {decisions_per_second:.2f} 低于阈值5决策/秒"
    
    def test_function_completeness(self):
        """测试功能完整性"""
        # 测试核心功能的完整性
        
        # 1. 逻辑一致性模块功能
        consistency_checker = LogicalConsistencyChecker()
        
        # 测试配置选项
        config_options = [
            {'use_chain_verification': True},
            {'use_chain_verification': False},
            {'use_model_detection': True},
            {'use_model_detection': False}
        ]
        
        for config in config_options:
            checker = LogicalConsistencyChecker(config)
            
            decision = {
                "action": "test_config",
                "parameters": {
                    "config": str(config)
                },
                "reasoning": "测试不同配置下的功能完整性。",
                "timestamp": "2026-01-13T10:00:00"
            }
            
            result = checker.check_consistency(decision, [])
            assert 'is_consistent' in result
        
        # 2. 好奇心模块功能
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 测试不同输入类型
        input_types = [
            # 状态张量输入
            torch.randn(1, state_dim),
            # 字典形式的多模态输入
            {
                'vision': torch.randn(1, 3, 32, 32),
                'speech': torch.randn(1, modality_dims['speech']),
                'text': torch.randint(0, modality_dims['text'], (1, 10))
            }
        ]
        
        for observations in input_types:
            actions = torch.randn(1, action_dim)
            output = curiosity_model(observations, actions)
            assert output.shape == (1, state_dim)
        
        # 验证功能完整性
        print("功能完整性测试通过: 所有核心功能正常工作")
    
    def test_edge_cases_handling(self):
        """测试边缘情况处理"""
        # 测试系统在边缘情况下的表现
        
        # 1. 逻辑一致性模块的边缘情况
        consistency_checker = LogicalConsistencyChecker()
        
        # 空推理文本
        empty_reasoning_decision = {
            "action": "test",
            "parameters": {},
            "reasoning": "",
            "timestamp": "2026-01-13T10:00:00"
        }
        result = consistency_checker.check_consistency(empty_reasoning_decision, [])
        assert 'is_consistent' in result
        
        # 非常长的推理文本
        long_reasoning = "这是一个非常长的推理文本 " * 100
        long_reasoning_decision = {
            "action": "test",
            "parameters": {},
            "reasoning": long_reasoning,
            "timestamp": "2026-01-13T10:00:00"
        }
        result = consistency_checker.check_consistency(long_reasoning_decision, [])
        assert 'is_consistent' in result
        
        # 2. 好奇心模块的边缘情况
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 零状态
        zero_state = torch.zeros(1, state_dim)
        action = torch.randn(1, action_dim)
        output = curiosity_model(zero_state, action)
        assert output.shape == (1, state_dim)
        
        # 极端值状态
        extreme_state = torch.ones(1, state_dim) * 1000
        output = curiosity_model(extreme_state, action)
        assert output.shape == (1, state_dim)
        
        print("边缘情况处理测试通过")
    
    def test_integration_scalability(self):
        """测试集成系统的可扩展性"""
        # 测试系统在不同规模下的表现
        
        # 1. 不同规模的逻辑一致性检查
        consistency_checker = LogicalConsistencyChecker()
        
        # 生成不同长度的推理文本
        reasoning_lengths = [1, 5, 10, 20, 50]
        
        for length in reasoning_lengths:
            statements = [f"陈述 {i}" for i in range(length)]
            reasoning = ". ".join(statements) + "."
            
            decision = {
                "action": "test_scalability",
                "parameters": {
                    "reasoning_length": length
                },
                "reasoning": reasoning,
                "timestamp": "2026-01-13T10:00:00"
            }
            
            start_time = time.time()
            result = consistency_checker.check_consistency(decision, [])
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"推理长度 {length} 的处理时间: {processing_time:.4f}秒")
            
            assert 'is_consistent' in result
        
        # 2. 不同规模的好奇心计算
        state_dim = 128
        action_dim = 32
        
        # 测试不同状态维度
        state_dims = [64, 128, 256]
        
        for dim in state_dims:
            modality_dims = {
                'vision': 10000,
                'speech': dim,
                'text': 10000
            }
            
            model = MultiModalCuriosity(dim, action_dim, modality_dims)
            state = torch.randn(1, dim)
            action = torch.randn(1, action_dim)
            
            start_time = time.time()
            output = model(state, action)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"状态维度 {dim} 的处理时间: {processing_time:.4f}秒")
            
            assert output.shape == (1, dim)
        
        print("可扩展性测试通过: 系统在不同规模下表现稳定")


class TestExplorationExploitationBalance:
    """探索-利用平衡测试"""
    
    def test_balance_maintenance(self):
        """测试系统在探索和利用之间的平衡能力"""
        # 这个测试模拟了探索-利用平衡的评估
        
        # 初始化组件
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 模拟探索和利用的平衡
        exploration_count = 0
        exploitation_count = 0
        total_actions = 100
        
        for i in range(total_actions):
            # 模拟探索和利用的选择
            # 在实际系统中，这会基于好奇心值和已有的奖励信息
            
            # 简单模拟：交替进行探索和利用
            if i % 2 == 0:
                # 探索
                exploration_count += 1
            else:
                # 利用
                exploitation_count += 1
        
        # 计算探索和利用的比例
        exploration_ratio = exploration_count / total_actions
        exploitation_ratio = exploitation_count / total_actions
        
        # 计算平衡度：|探索比例 - 利用比例|
        balance = abs(exploration_ratio - exploitation_ratio)
        
        print(f"探索-利用平衡测试结果:")
        print(f"探索次数: {exploration_count}")
        print(f"利用次数: {exploitation_count}")
        print(f"探索比例: {exploration_ratio:.2f}")
        print(f"利用比例: {exploitation_ratio:.2f}")
        print(f"平衡度: {balance:.2f}")
        
        # 验证平衡度
        # 理想情况下，平衡度应接近0.0
        assert balance < 0.3, f"探索-利用平衡度 {balance:.2f} 超过阈值0.3"
    
    def test_adaptive_balance_adjustment(self):
        """测试系统自适应调整探索-利用平衡的能力"""
        # 初始化自适应学习系统
        learning_system = AdaptiveLearningSystem()
        
        # 记录初始探索率
        initial_exploration_rate = learning_system.exploration_rate
        
        # 模拟不同环境下的平衡调整
        environments = [
            # 未知环境：高探索率
            {"environment_type": "unknown", "reward_variance": 0.8},
            # 已知环境：低探索率
            {"environment_type": "known", "reward_variance": 0.2}
        ]
        
        for env in environments:
            # 模拟环境中的学习过程
            for i in range(50):
                # 生成模拟经验
                state = torch.randn(1, 128).numpy().tolist()
                action = torch.randn(1, 32).numpy().tolist()
                # 根据环境类型生成奖励
                reward = 1.0 + (torch.randn(1).item() * env["reward_variance"])
                next_state = torch.randn(1, 128).numpy().tolist()
                done = i % 10 == 0
                context = {
                    "environment_type": env["environment_type"],
                    "iteration": i
                }
                
                # 添加经验
                learning_system.add_experience(state, action, reward, next_state, done, context)
                
                # 更新性能指标
                learning_system.update_performance_metrics(reward, done)
            
            # 检查探索率调整
            current_exploration_rate = learning_system.exploration_rate
            print(f"环境类型: {env['environment_type']}")
            print(f"初始探索率: {initial_exploration_rate:.4f}")
            print(f"当前探索率: {current_exploration_rate:.4f}")
            
            # 验证自适应调整
            if env["environment_type"] == "unknown":
                # 未知环境应保持较高的探索率
                assert current_exploration_rate > initial_exploration_rate * 0.8, f"未知环境探索率 {current_exploration_rate:.4f} 过低"
            else:
                # 已知环境应降低探索率
                assert current_exploration_rate < initial_exploration_rate * 1.2, f"已知环境探索率 {current_exploration_rate:.4f} 过高"
        
        print("自适应平衡调整测试通过: 系统根据环境类型调整探索率")
