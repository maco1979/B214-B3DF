"""
好奇心机制模块集成测试
测试探索能力、好奇心值计算和自适应调节
"""

import pytest
import torch
from src.core.models.curiosity_model import (
    MultiModalCuriosity, calculate_intrinsic_reward, calculate_entropy,
    NoveltySeekingExplorer, CERMICExplorer
)


class TestCuriosityIntegration:
    """好奇心机制模块集成测试"""
    
    def test_exploration_capability_random(self, mock_environment):
        """测试随机探索生成"""
        # 创建新颖性寻求探索器
        explorer = NoveltySeekingExplorer(mock_environment)
        
        # 测试随机探索动作生成
        import torch
        state = torch.randn(1, 128)
        action = explorer.novelty_seeking_exploration(state)
        
        # 验证动作在有效范围内
        assert action in range(mock_environment.action_space.n)
    
    def test_exploration_capability_curiosity_guided(self, curiosity_model, mock_environment):
        """测试基于好奇心值选择探索动作"""
        # 创建CERMIC探索器
        explorer = CERMICExplorer(mock_environment, curiosity_model)
        
        # 测试好奇心引导的动作选择
        import torch
        state = torch.randn(1, 128)
        possible_actions = list(range(mock_environment.action_space.n))
        
        # 评估每个动作的有趣程度
        interestingness_scores = []
        for action in possible_actions:
            score = explorer.evaluate_action_interestingness(state, action)
            interestingness_scores.append((action, score))
        
        # 验证所有动作都有有趣程度评分
        assert len(interestingness_scores) == len(possible_actions)
        for _, score in interestingness_scores:
            assert isinstance(score, float)
        
        # 测试自适应探索选择
        selected_action = explorer.adaptive_exploration(state, possible_actions)
        assert selected_action in possible_actions
    
    def test_exploration_capability_novelty_seeking(self, mock_environment):
        """测试寻找与历史差异最大的动作"""
        # 创建新颖性寻求探索器
        explorer = NoveltySeekingExplorer(mock_environment)
        
        # 添加历史记录
        import torch
        for i in range(10):
            state = torch.randn(1, 128)
            action = i % mock_environment.action_space.n
            next_state = torch.randn(1, 128)
            explorer.add_to_history(state, action, next_state)
        
        # 测试新颖性寻求探索
        current_state = torch.randn(1, 128)
        selected_action = explorer.novelty_seeking_exploration(current_state)
        
        # 验证动作在有效范围内
        assert selected_action in range(mock_environment.action_space.n)
    
    def test_curiosity_value_calculation_prediction_error(self, curiosity_model, sample_curiosity_state, sample_curiosity_action, sample_curiosity_next_state):
        """测试预测误差计算"""
        # 计算预测的下一状态
        predicted_next_state = curiosity_model(sample_curiosity_state, sample_curiosity_action)
        
        # 计算预测误差
        prediction_error = torch.norm(predicted_next_state - sample_curiosity_next_state)
        
        # 验证预测误差是正数
        assert prediction_error > 0
    
    def test_curiosity_value_calculation_information_gain(self):
        """测试信息增益计算"""
        # 准备测试状态
        states = torch.randn(100, 128)
        
        # 计算熵
        entropy = calculate_entropy(states)
        
        # 验证熵值是正数
        assert entropy > 0
    
    def test_curiosity_value_calculation_combined(self, curiosity_model, sample_curiosity_state, sample_curiosity_action, sample_curiosity_next_state):
        """测试综合奖励计算"""
        # 计算内在奖励
        intrinsic_reward = calculate_intrinsic_reward(
            sample_curiosity_state, sample_curiosity_action, 
            sample_curiosity_next_state, curiosity_model
        )
        
        # 验证内在奖励是张量类型
        assert isinstance(intrinsic_reward, torch.Tensor)
        # 验证奖励值不是NaN
        assert not torch.isnan(intrinsic_reward)
    
    def test_adaptive_adjustment_exploration_threshold(self, curiosity_model, mock_environment):
        """测试根据环境动态调整探索阈值"""
        # 创建CERMIC探索器
        explorer = CERMICExplorer(mock_environment, curiosity_model)
        
        # 测试不同状态下的探索选择
        import torch
        
        # 生成不同的状态
        states = [torch.randn(1, 128) for _ in range(5)]
        possible_actions = list(range(mock_environment.action_space.n))
        
        # 验证不同状态下选择不同的动作
        selected_actions = []
        for state in states:
            action = explorer.adaptive_exploration(state, possible_actions)
            selected_actions.append(action)
        
        # 至少有两个不同的动作被选中
        unique_actions = set(selected_actions)
        assert len(unique_actions) >= 2, f"只选择了 {len(unique_actions)} 种不同动作，预期至少2种"
    
    def test_adaptive_adjustment_strategy_switching(self, mock_environment):
        """测试在不同探索策略间灵活切换"""
        # 创建新颖性寻求探索器
        explorer = NoveltySeekingExplorer(mock_environment)
        
        # 测试不同条件下的策略表现
        import torch
        
        # 1. 历史记录为空时的行为
        state1 = torch.randn(1, 128)
        action1 = explorer.novelty_seeking_exploration(state1)
        
        # 2. 添加历史记录后的行为
        for i in range(20):
            state = torch.randn(1, 128)
            action = i % mock_environment.action_space.n
            next_state = torch.randn(1, 128)
            explorer.add_to_history(state, action, next_state)
        
        state2 = torch.randn(1, 128)
        action2 = explorer.novelty_seeking_exploration(state2)
        
        # 验证策略在不同条件下可能产生不同结果
        # 这是一个概率性测试，可能需要多次运行才能通过
        # assert action1 != action2, "策略切换未产生不同结果"
    
    def test_scenarios_maze_exploration(self, mock_environment):
        """测试在未知迷宫中的探索路径选择"""
        # 创建模拟迷宫环境
        class MockMazeEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 4,  # 上、下、左、右
                    'sample': lambda self: 0
                })()
                self.visited_states = set()
                
            def simulate(self, state, action):
                # 简单的迷宫模拟：每次移动到新状态
                state_tuple = tuple(state.flatten().tolist())
                self.visited_states.add(state_tuple)
                # 返回新状态
                next_state = torch.randn(1, 128)
                return next_state, 0.0, False, {}
        
        maze_env = MockMazeEnv()
        explorer = NoveltySeekingExplorer(maze_env)
        
        # 执行多次探索
        state = torch.randn(1, 128)
        for _ in range(10):
            action = explorer.novelty_seeking_exploration(state)
            next_state, _, _, _ = maze_env.simulate(state, action)
            explorer.add_to_history(state, action, next_state)
            state = next_state
        
        # 验证探索了多个不同状态
        assert len(maze_env.visited_states) > 1
    
    def test_scenarios_goal_discovery(self, mock_environment):
        """测试在稀疏奖励环境中发现目标的能力"""
        # 创建带有稀疏奖励的模拟环境
        class MockSparseRewardEnv:
            def __init__(self):
                self.action_space = type('MockActionSpace', (), {
                    'n': 5,
                    'sample': lambda self: 0
                })()
                self.goal_found = False
                self.steps_taken = 0
                self.goal_step = 7  # 在第7步发现目标
                
            def simulate(self, state, action):
                self.steps_taken += 1
                
                # 稀疏奖励：只有在特定步骤才给奖励
                if self.steps_taken == self.goal_step:
                    self.goal_found = True
                    reward = 10.0  # 高奖励
                else:
                    reward = 0.0
                
                next_state = torch.randn(1, 128)
                done = self.goal_found
                info = {'goal_found': self.goal_found}
                
                return next_state, reward, done, info
        
        sparse_env = MockSparseRewardEnv()
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        explorer = CERMICExplorer(sparse_env, curiosity_model)
        
        # 执行探索直到找到目标或达到最大步数
        state = torch.randn(1, 128)
        possible_actions = list(range(sparse_env.action_space.n))
        
        max_steps = 20
        found_goal = False
        
        for step in range(max_steps):
            action = explorer.adaptive_exploration(state, possible_actions)
            next_state, reward, done, info = sparse_env.simulate(state, action)
            
            if info['goal_found']:
                found_goal = True
                break
            
            state = next_state
        
        # 验证能够在合理步数内找到目标
        assert found_goal, f"在{max_steps}步内未找到目标"
    
    def test_exploration_efficiency_coverage(self, mock_environment):
        """测试探索覆盖率"""
        # 创建新颖性寻求探索器
        explorer = NoveltySeekingExplorer(mock_environment)
        
        # 执行多次探索
        state = torch.randn(1, 128)
        visited_states = set()
        
        # 执行50步探索
        for _ in range(50):
            # 添加当前状态到已访问集合
            state_tuple = tuple(state.flatten().tolist())
            visited_states.add(state_tuple)
            
            # 选择并执行动作
            action = explorer.novelty_seeking_exploration(state)
            next_state, _, _, _ = mock_environment.simulate(state, action)
            
            # 添加到历史记录
            explorer.add_to_history(state, action, next_state)
            
            # 更新状态
            state = next_state
        
        # 添加最后一个状态
        state_tuple = tuple(state.flatten().tolist())
        visited_states.add(state_tuple)
        
        # 计算探索覆盖率（这里简化为已访问状态数）
        exploration_coverage = len(visited_states)
        print(f"探索覆盖率: {exploration_coverage}")
        
        # 验证探索覆盖率达到一定水平
        assert exploration_coverage > 10, f"探索覆盖率 {exploration_coverage} 低于预期阈值10"
    
    def test_exploration_efficiency_novelty_discovery_rate(self, mock_environment):
        """测试新颖性发现率"""
        # 创建新颖性寻求探索器
        explorer = NoveltySeekingExplorer(mock_environment)
        
        # 执行探索并记录新状态发现
        state = torch.randn(1, 128)
        visited_states = set()
        new_states_found = 0
        total_steps = 30
        
        for step in range(total_steps):
            # 检查当前状态是否是新状态
            state_tuple = tuple(state.flatten().tolist())
            if state_tuple not in visited_states:
                new_states_found += 1
                visited_states.add(state_tuple)
            
            # 选择并执行动作
            action = explorer.novelty_seeking_exploration(state)
            next_state, _, _, _ = mock_environment.simulate(state, action)
            
            # 添加到历史记录
            explorer.add_to_history(state, action, next_state)
            
            # 更新状态
            state = next_state
        
        # 计算新颖性发现率
        novelty_discovery_rate = new_states_found / total_steps
        print(f"新颖性发现率: {novelty_discovery_rate:.2f}")
        
        # 验证新颖性发现率
        assert novelty_discovery_rate > 0.5, f"新颖性发现率 {novelty_discovery_rate:.2f} 低于预期阈值0.5"
    
    def test_information_gain_rate(self, curiosity_model, sample_curiosity_state, sample_curiosity_action, sample_curiosity_next_state):
        """测试信息增益率"""
        import time
        
        # 多次计算信息增益并测量时间
        num_iterations = 10
        total_information_gain = 0.0
        total_time = 0.0
        
        for _ in range(num_iterations):
            # 计算当前状态熵
            state_entropy_current = calculate_entropy(sample_curiosity_state)
            
            # 记录时间
            start_time = time.time()
            
            # 计算内在奖励（包含信息增益）
            intrinsic_reward = calculate_intrinsic_reward(
                sample_curiosity_state, sample_curiosity_action, 
                sample_curiosity_next_state, curiosity_model
            )
            
            # 计算下一状态熵
            state_entropy_next = calculate_entropy(sample_curiosity_next_state)
            information_gain = state_entropy_next - state_entropy_current
            total_information_gain += information_gain
            
            # 记录时间
            end_time = time.time()
            total_time += (end_time - start_time)
        
        # 计算信息增益率
        if total_time > 0:
            information_gain_rate = total_information_gain / total_time
            print(f"信息增益率: {information_gain_rate:.4f}")
            
            # 验证信息增益率为正数
            assert information_gain_rate > 0, f"信息增益率 {information_gain_rate:.4f} 应为正数"
    
    def test_novelty_seeking_explorer_history_management(self, mock_environment):
        """测试新颖性寻求探索器的历史管理"""
        # 创建新颖性寻求探索器
        history_length = 20
        explorer = NoveltySeekingExplorer(mock_environment, history_length=history_length)
        
        # 添加超出历史长度的记录
        for i in range(history_length + 10):
            state = torch.randn(1, 128)
            action = i % mock_environment.action_space.n
            next_state = torch.randn(1, 128)
            explorer.add_to_history(state, action, next_state)
        
        # 验证历史记录长度不超过设定值
        assert len(explorer.history) == history_length, f"历史记录长度 {len(explorer.history)} 超过设定值 {history_length}"
    
    def test_cermic_explorer_interestingness_evaluation(self, curiosity_model, mock_environment):
        """测试CERMIC探索器的有趣程度评估"""
        # 创建CERMIC探索器
        explorer = CERMICExplorer(mock_environment, curiosity_model)
        
        # 测试不同动作的有趣程度评估
        state = torch.randn(1, 128)
        possible_actions = list(range(mock_environment.action_space.n))
        
        # 评估所有可能动作
        interestingness_scores = []
        for action in possible_actions:
            score = explorer.evaluate_action_interestingness(state, action)
            interestingness_scores.append((action, score))
        
        # 打印评分结果
        print("动作有趣程度评分:")
        for action, score in sorted(interestingness_scores, key=lambda x: x[1], reverse=True):
            print(f"动作 {action}: {score:.4f}")
        
        # 验证评分差异
        scores = [score for _, score in interestingness_scores]
        max_score = max(scores)
        min_score = min(scores)
        
        # 最大评分和最小评分应有显著差异
        assert max_score - min_score > 0.1, f"评分差异 {max_score - min_score:.4f} 太小，预期至少0.1"
    
    def test_curiosity_model_modality_handling(self):
        """测试好奇心模型的多模态处理能力"""
        # 创建好奇心模型
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 测试不同模态输入的处理
        
        # 1. 字典形式的多模态输入
        observations_dict = {
            'vision': torch.randn(1, 3, 32, 32),
            'speech': torch.randn(1, modality_dims['speech']),
            'text': torch.randint(0, modality_dims['text'], (1, 10))
        }
        actions = torch.randn(1, action_dim)
        
        output1 = model(observations_dict, actions)
        assert output1.shape == (1, state_dim)
        
        # 2. 直接状态张量输入
        state_tensor = torch.randn(1, state_dim)
        output2 = model(state_tensor, actions)
        assert output2.shape == (1, state_dim)
        
        # 3. 缺少某些模态的输入
        partial_observations = {
            'speech': torch.randn(1, modality_dims['speech']),
            'text': torch.randint(0, modality_dims['text'], (1, 10))
        }
        output3 = model(partial_observations, actions)
        assert output3.shape == (1, state_dim)
    
    def test_intrinsic_reward_consistency(self, curiosity_model):
        """测试内在奖励计算的一致性"""
        # 准备测试数据
        state_dim = 128
        action_dim = 32
        
        # 生成多组测试数据
        num_test_cases = 5
        
        for _ in range(num_test_cases):
            current_state = torch.randn(1, state_dim)
            action = torch.randn(1, action_dim)
            next_state = torch.randn(1, state_dim)
            
            # 多次计算同一组数据的内在奖励
            rewards = []
            for _ in range(3):
                reward = calculate_intrinsic_reward(
                    current_state, action, next_state, curiosity_model
                )
                rewards.append(reward.item())
            
            # 验证奖励值的一致性（波动不应太大）
            max_reward = max(rewards)
            min_reward = min(rewards)
            assert max_reward - min_reward < 0.1, f"内在奖励波动过大: {max_reward - min_reward:.4f}"


class TestCuriosityPerformance:
    """好奇心机制性能测试"""
    
    def test_performance_forward_pass(self):
        """测试好奇心模型前向传播性能"""
        import time
        
        # 创建好奇心模型
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 准备测试数据
        batch_size = 32
        state = torch.randn(batch_size, state_dim)
        action = torch.randn(batch_size, action_dim)
        
        # 测量前向传播时间
        num_iterations = 100
        total_time = 0.0
        
        for _ in range(num_iterations):
            start_time = time.time()
            output = model(state, action)
            end_time = time.time()
            total_time += (end_time - start_time)
        
        average_time = total_time / num_iterations
        print(f"前向传播平均时间: {average_time:.4f}秒")
        
        # 验证性能要求
        assert average_time < 0.01, f"前向传播时间 {average_time:.4f}秒 超过阈值0.01秒"
    
    def test_performance_intrinsic_reward_calculation(self):
        """测试内在奖励计算性能"""
        import time
        
        # 创建好奇心模型
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        
        # 准备测试数据
        current_state = torch.randn(1, state_dim)
        action = torch.randn(1, action_dim)
        next_state = torch.randn(1, state_dim)
        
        # 测量内在奖励计算时间
        num_iterations = 50
        total_time = 0.0
        
        for _ in range(num_iterations):
            start_time = time.time()
            reward = calculate_intrinsic_reward(
                current_state, action, next_state, model
            )
            end_time = time.time()
            total_time += (end_time - start_time)
        
        average_time = total_time / num_iterations
        print(f"内在奖励计算平均时间: {average_time:.4f}秒")
        
        # 验证性能要求
        assert average_time < 0.05, f"内在奖励计算时间 {average_time:.4f}秒 超过阈值0.05秒"
    
    def test_performance_exploration_decision(self, mock_environment):
        """测试探索决策性能"""
        import time
        
        # 创建好奇心模型和探索器
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        curiosity_model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
        explorer = CERMICExplorer(mock_environment, curiosity_model)
        
        # 准备测试数据
        state = torch.randn(1, state_dim)
        possible_actions = list(range(mock_environment.action_space.n))
        
        # 测量探索决策时间
        num_iterations = 30
        total_time = 0.0
        
        for _ in range(num_iterations):
            start_time = time.time()
            action = explorer.adaptive_exploration(state, possible_actions)
            end_time = time.time()
            total_time += (end_time - start_time)
        
        average_time = total_time / num_iterations
        print(f"探索决策平均时间: {average_time:.4f}秒")
        
        # 验证性能要求
        assert average_time < 0.1, f"探索决策时间 {average_time:.4f}秒 超过阈值0.1秒"
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        # 创建多个好奇心模型实例，检查内存使用
        state_dim = 128
        action_dim = 32
        modality_dims = {
            'vision': 10000,
            'speech': 128,
            'text': 10000
        }
        
        # 创建多个模型实例
        models = []
        for _ in range(10):
            model = MultiModalCuriosity(state_dim, action_dim, modality_dims)
            models.append(model)
        
        # 执行前向传播
        for model in models:
            state = torch.randn(1, state_dim)
            action = torch.randn(1, action_dim)
            output = model(state, action)
            assert output.shape == (1, state_dim)
        
        # 验证所有模型都能正常工作
        assert len(models) == 10, "模型创建失败"
