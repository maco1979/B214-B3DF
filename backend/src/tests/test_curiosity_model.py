# Test cases for curiosity model
import pytest
import torch
from src.core.models.curiosity_model import (
    MultiModalCuriosity,
    calculate_intrinsic_reward,
    calculate_entropy,
    NoveltySeekingExplorer,
    CERMICExplorer,
    calculate_curiosity_score,
    evaluate_environment_complexity
)


class TestMultiModalCuriosity:
    """测试多模态好奇心模型"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        assert isinstance(model, MultiModalCuriosity)
        assert hasattr(model, 'prediction_net')
        assert hasattr(model, 'state_dim')
        assert hasattr(model, 'action_dim')
    
    def test_model_forward(self):
        """测试模型前向传播"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 创建测试输入
        observations = {
            'vision': torch.randn(1, 3, 32, 32),  # 3x32x32图像
            'speech': torch.randn(1, modality_dims['speech']),
            'text': torch.randint(0, modality_dims['text'], (1, 10))  # 10个token
        }
        actions = torch.randn(1, action_dim)
        
        # 执行前向传播
        output = model(observations, actions)
        
        # 验证输出形状
        assert output.shape == (1, state_dim)
    
    def test_model_forward_with_partial_modalities(self):
        """测试模型使用部分模态输入的情况"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 创建只有部分模态的输入
        observations = {
            'speech': torch.randn(1, modality_dims['speech']),
            'text': torch.randint(0, modality_dims['text'], (1, 10))
        }
        actions = torch.randn(1, action_dim)
        
        # 执行前向传播
        output = model(observations, actions)
        
        # 验证输出形状
        assert output.shape == (1, state_dim)


class TestIntrinsicReward:
    """测试内在奖励计算"""
    
    def test_calculate_entropy(self):
        """测试熵计算"""
        # 创建测试状态
        states = torch.randn(100, 128)
        
        # 计算熵
        entropy = calculate_entropy(states)
        
        # 验证熵值为正数
        assert entropy > 0
    
    def test_calculate_intrinsic_reward(self):
        """测试内在奖励计算"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 创建测试数据
        current_state = torch.randn(1, state_dim)
        action = torch.randn(1, action_dim)
        next_state = torch.randn(1, state_dim)
        
        # 计算内在奖励
        reward = calculate_intrinsic_reward(current_state, action, next_state, model)
        
        # 验证奖励值为张量
        assert isinstance(reward, torch.Tensor)
        # 验证奖励值不是NaN
        assert not torch.isnan(reward)


class TestNoveltySeekingExplorer:
    """测试新颖性寻求探索器"""
    
    def test_initialization(self):
        """测试初始化"""
        # 创建模拟环境
        class MockEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                
            def simulate(self, state, action):
                return torch.randn(1, 128), 0.0, False, {}
        
        env = MockEnv()
        explorer = NoveltySeekingExplorer(env)
        
        assert isinstance(explorer, NoveltySeekingExplorer)
        assert hasattr(explorer, 'history')
        assert hasattr(explorer, 'novelty_seeking_exploration')
    
    def test_add_to_history(self):
        """测试添加到历史记录"""
        # 创建模拟环境
        class MockEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                
            def simulate(self, state, action):
                return torch.randn(1, 128), 0.0, False, {}
        
        env = MockEnv()
        explorer = NoveltySeekingExplorer(env)
        
        # 添加历史记录
        explorer.add_to_history(
            torch.randn(1, 128),
            0,
            torch.randn(1, 128)
        )
        
        assert len(explorer.history) == 1
        
        # 添加更多历史记录，测试历史长度限制
        for i in range(100):
            explorer.add_to_history(
                torch.randn(1, 128),
                i % 5,
                torch.randn(1, 128)
            )
        
        assert len(explorer.history) == 50  # 默认历史长度为50


class TestCERMICExplorer:
    """测试CERMIC探索器"""
    
    def test_initialization(self):
        """测试初始化"""
        # 创建模拟环境
        class MockEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                
            def simulate(self, state, action):
                return torch.randn(1, 128), 0.0, False, {}
        
        env = MockEnv()
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        explorer = CERMICExplorer(env, curiosity_model)
        
        assert isinstance(explorer, CERMICExplorer)
        assert hasattr(explorer, 'curiosity_model')
        assert hasattr(explorer, 'evaluate_action_interestingness')
        assert hasattr(explorer, 'adaptive_exploration')
    
    def test_evaluate_action_interestingness(self):
        """测试评估动作有趣程度"""
        # 创建模拟环境
        class MockEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                
            def simulate(self, state, action):
                return torch.randn(1, 128), 0.0, False, {}
        
        env = MockEnv()
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        explorer = CERMICExplorer(env, curiosity_model)
        
        # 创建测试状态和动作
        state = torch.randn(1, state_dim)
        action = 0
        
        # 评估动作有趣程度
        interestingness = explorer.evaluate_action_interestingness(state, action)
        
        # 验证结果是数值类型
        assert isinstance(interestingness, float)
        assert not torch.isnan(torch.tensor(interestingness))
    
    def test_adaptive_exploration(self):
        """测试自适应探索"""
        # 创建模拟环境
        class MockEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                
            def simulate(self, state, action):
                return torch.randn(1, 128), 0.0, False, {}
        
        env = MockEnv()
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        explorer = CERMICExplorer(env, curiosity_model)
        
        # 创建测试状态
        state = torch.randn(1, state_dim)
        possible_actions = [0, 1, 2, 3, 4]
        
        # 执行自适应探索
        selected_action = explorer.adaptive_exploration(state, possible_actions)
        
        # 验证结果是可能的动作之一
        assert selected_action in possible_actions


class TestEnvironmentComplexity:
    """测试环境复杂度评估"""
    
    def test_evaluate_simple_dict(self):
        """测试评估简单字典状态的复杂度"""
        simple_state = {
            'temperature': 25.0,
            'humidity': 60.0
        }
        
        complexity = evaluate_environment_complexity(simple_state)
        assert isinstance(complexity, float)
        assert 0.0 <= complexity <= 1.0
        assert complexity < 0.5  # 简单状态复杂度较低
    
    def test_evaluate_complex_dict(self):
        """测试评估复杂字典状态的复杂度"""
        complex_state = {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.25,
            'wind_speed': 5.0,
            'wind_direction': 180.0,
            'solar_radiation': 800.0,
            'soil_moisture': 45.0,
            'leaf_wetness': 0.2,
            'co2_level': 400.0,
            'voc_level': 0.5
        }
        
        complexity = evaluate_environment_complexity(complex_state)
        assert isinstance(complexity, float)
        assert 0.0 <= complexity <= 1.0
        assert complexity > 0.5  # 复杂状态复杂度较高
    
    def test_evaluate_tensor(self):
        """测试评估张量状态的复杂度"""
        simple_tensor = torch.randn(1, 128)
        complex_tensor = torch.randn(10, 10, 128)  # 更高维度的张量
        
        simple_complexity = evaluate_environment_complexity(simple_tensor)
        complex_complexity = evaluate_environment_complexity(complex_tensor)
        
        assert isinstance(simple_complexity, float)
        assert isinstance(complex_complexity, float)
        assert simple_complexity < complex_complexity  # 更高维度的张量复杂度更高


class TestCuriosityScoring:
    """测试多维度好奇心分数计算"""
    
    def test_calculate_curiosity_score(self):
        """测试计算好奇心分数"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        state = torch.randn(state_dim)
        action_space = action_dim
        
        # 无历史状态
        curiosity_score = calculate_curiosity_score(state, action_space, model)
        assert isinstance(curiosity_score, float)
        assert 0.0 <= curiosity_score <= 1.0
        
        # 有历史状态
        historical_states = [torch.randn(state_dim) for _ in range(5)]
        curiosity_score_with_history = calculate_curiosity_score(
            state, action_space, model, historical_states
        )
        assert isinstance(curiosity_score_with_history, float)
        assert 0.0 <= curiosity_score_with_history <= 1.0
    
    def test_calculate_curiosity_score_dict(self):
        """测试使用字典状态计算好奇心分数"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        state = {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.25
        }
        action_space = action_dim
        
        curiosity_score = calculate_curiosity_score(state, action_space, model)
        assert isinstance(curiosity_score, float)
        assert 0.0 <= curiosity_score <= 1.0
    
    def test_calculate_curiosity_score_with_high_novelty(self):
        """测试高新颖性状态的好奇心分数"""
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 创建差异较大的历史状态和当前状态
        historical_states = [torch.zeros(state_dim) for _ in range(5)]
        current_state = torch.ones(state_dim) * 10.0  # 与历史状态差异很大
        
        curiosity_score = calculate_curiosity_score(
            current_state, action_dim, model, historical_states
        )
        assert isinstance(curiosity_score, float)
        assert 0.0 <= curiosity_score <= 1.0
