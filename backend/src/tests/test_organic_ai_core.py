#!/usr/bin/env python3
"""
有机体AI核心测试脚本
测试学习能力、决策能力和协同学习能力
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将项目根目录添加到Python路径
import sys
import os
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(src_dir))
sys.path.append(project_root)

from src.core.ai_organic_core import OrganicAICore, get_organic_ai_core


class OrganicAITester:
    """有机体AI核心测试器"""
    
    def __init__(self):
        self.ai_core: OrganicAICore = None
        self.test_results: Dict[str, Any] = {
            'learning_tests': [],
            'decision_tests': [],
            'collaboration_tests': []
        }
    
    async def initialize(self):
        """初始化AI核心"""
        self.ai_core = await get_organic_ai_core()
        logger.info(f"AI核心初始化完成，当前状态: {self.ai_core.state.value}")
    
    async def test_learning_ability(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        测试AI核心的学习能力
        
        Args:
            num_iterations: 测试迭代次数
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始学习能力测试，迭代次数: {num_iterations}")
        
        # 记录测试开始时间
        start_time = time.time()
        
        # 生成模拟环境数据
        environment_states = self._generate_environment_states(num_iterations)
        
        # 初始性能指标
        initial_success_rate = self.ai_core.learning_system.success_rate
        initial_average_reward = self.ai_core.learning_system.average_reward
        
        # 测试学习过程
        learning_results = []
        
        for i in range(num_iterations):
            # 选择一个环境状态
            state = environment_states[i % len(environment_states)]
            
            # 做出决策
            decision = await self.ai_core.make_decision(state)
            
            # 生成模拟奖励（基于决策合理性）
            reward = self._generate_reward(decision, state)
            
            # 从经验中学习
            await self.ai_core.learn_from_experience(
                state=state,
                action=decision.action,
                reward=reward,
                next_state=self._generate_next_state(state, decision),
                done=False
            )
            
            # 记录学习结果
            learning_results.append({
                'iteration': i,
                'decision_action': decision.action,
                'confidence': decision.confidence,
                'reward': reward,
                'success_rate': self.ai_core.learning_system.success_rate,
                'average_reward': self.ai_core.learning_system.average_reward
            })
            
            # 每10次迭代记录一次日志
            if (i + 1) % 10 == 0:
                logger.info(f"学习测试迭代 {i+1}/{num_iterations} 完成")
        
        # 测试结束时间
        end_time = time.time()
        
        # 最终性能指标
        final_success_rate = self.ai_core.learning_system.success_rate
        final_average_reward = self.ai_core.learning_system.average_reward
        
        # 计算性能提升
        success_rate_improvement = final_success_rate - initial_success_rate
        reward_improvement = final_average_reward - initial_average_reward
        
        # 记录测试结果
        test_result = {
            'test_name': 'learning_ability',
            'num_iterations': num_iterations,
            'duration_seconds': end_time - start_time,
            'initial_success_rate': initial_success_rate,
            'final_success_rate': final_success_rate,
            'success_rate_improvement': success_rate_improvement,
            'initial_average_reward': initial_average_reward,
            'final_average_reward': final_average_reward,
            'reward_improvement': reward_improvement,
            'learning_results': learning_results
        }
        
        self.test_results['learning_tests'].append(test_result)
        
        logger.info(f"学习能力测试完成: 成功率提升 {success_rate_improvement:.4f}, 奖励提升 {reward_improvement:.4f}")
        
        return test_result
    
    async def test_decision_ability(self, num_test_cases: int = 50) -> Dict[str, Any]:
        """
        测试AI核心的决策能力
        
        Args:
            num_test_cases: 测试用例数量
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始决策能力测试，测试用例数量: {num_test_cases}")
        
        # 记录测试开始时间
        start_time = time.time()
        
        # 生成多样化的测试状态
        test_states = self._generate_diverse_test_states(num_test_cases)
        
        # 测试决策过程
        decision_results = []
        decision_times = []
        confidence_scores = []
        risk_scores = []
        
        for i, state in enumerate(test_states):
            # 记录决策开始时间
            decision_start = time.time()
            
            # 做出决策
            decision = await self.ai_core.make_decision(state)
            
            # 记录决策结束时间
            decision_end = time.time()
            
            # 计算决策时间
            decision_time = decision_end - decision_start
            
            # 记录决策结果
            decision_results.append({
                'test_case': i,
                'state': state,
                'decision': decision,
                'decision_time': decision_time
            })
            
            # 收集统计数据
            decision_times.append(decision_time)
            confidence_scores.append(decision.confidence)
            risk_scores.append(decision.risk_assessment['total_risk'])
            
            # 每10个测试用例记录一次日志
            if (i + 1) % 10 == 0:
                logger.info(f"决策测试用例 {i+1}/{num_test_cases} 完成")
        
        # 测试结束时间
        end_time = time.time()
        
        # 计算统计指标
        avg_decision_time = np.mean(decision_times)
        std_decision_time = np.std(decision_times)
        avg_confidence = np.mean(confidence_scores)
        avg_risk = np.mean(risk_scores)
        
        # 评估决策多样性
        action_types = [result['decision'].action for result in decision_results]
        unique_actions = set(action_types)
        action_diversity = len(unique_actions) / len(action_types)
        
        # 记录测试结果
        test_result = {
            'test_name': 'decision_ability',
            'num_test_cases': num_test_cases,
            'duration_seconds': end_time - start_time,
            'avg_decision_time_ms': avg_decision_time * 1000,
            'std_decision_time_ms': std_decision_time * 1000,
            'avg_confidence': avg_confidence,
            'avg_risk_score': avg_risk,
            'action_diversity': action_diversity,
            'unique_actions': list(unique_actions),
            'decision_results': decision_results
        }
        
        self.test_results['decision_tests'].append(test_result)
        
        logger.info(f"决策能力测试完成: 平均决策时间 {avg_decision_time*1000:.2f}ms, 平均置信度 {avg_confidence:.4f}, 决策多样性 {action_diversity:.4f}")
        
        return test_result
    
    async def test_collaboration_ability(self, num_ai_instances: int = 3, num_iterations: int = 50) -> Dict[str, Any]:
        """
        测试AI核心的协同学习能力
        
        Args:
            num_ai_instances: AI实例数量
            num_iterations: 测试迭代次数
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始协同学习能力测试，AI实例数量: {num_ai_instances}, 迭代次数: {num_iterations}")
        
        # 记录测试开始时间
        start_time = time.time()
        
        # 创建多个AI实例
        ai_instances = [await get_organic_ai_core() for _ in range(num_ai_instances)]
        
        # 生成共享环境状态
        environment_states = self._generate_environment_states(num_iterations)
        
        # 初始性能指标
        initial_performance = []
        for ai in ai_instances:
            initial_performance.append({
                'success_rate': ai.learning_system.success_rate,
                'average_reward': ai.learning_system.average_reward
            })
        
        # 协同学习测试
        collaboration_results = []
        
        for i in range(num_iterations):
            # 选择一个共享环境状态
            shared_state = environment_states[i % len(environment_states)]
            
            # 每个AI实例独立决策
            decisions = []
            for j, ai in enumerate(ai_instances):
                decision = await ai.make_decision(shared_state)
                decisions.append(decision)
            
            # 生成共享奖励
            shared_reward = self._generate_shared_reward(decisions, shared_state)
            
            # 每个AI实例从经验中学习
            for j, (ai, decision) in enumerate(zip(ai_instances, decisions)):
                await ai.learn_from_experience(
                    state=shared_state,
                    action=decision.action,
                    reward=shared_reward,
                    next_state=self._generate_next_state(shared_state, decision),
                    done=False
                )
            
            # 实现简单的知识共享（同步性能指标）
            await self._share_knowledge(ai_instances)
            
            # 记录协作学习结果
            instance_results = []
            for j, ai in enumerate(ai_instances):
                instance_results.append({
                    'ai_instance': j,
                    'success_rate': ai.learning_system.success_rate,
                    'average_reward': ai.learning_system.average_reward
                })
            
            collaboration_results.append({
                'iteration': i,
                'shared_state': shared_state,
                'instance_results': instance_results
            })
            
            # 每10次迭代记录一次日志
            if (i + 1) % 10 == 0:
                logger.info(f"协同学习测试迭代 {i+1}/{num_iterations} 完成")
        
        # 测试结束时间
        end_time = time.time()
        
        # 最终性能指标
        final_performance = []
        for ai in ai_instances:
            final_performance.append({
                'success_rate': ai.learning_system.success_rate,
                'average_reward': ai.learning_system.average_reward
            })
        
        # 计算性能提升
        performance_improvement = []
        for initial, final in zip(initial_performance, final_performance):
            performance_improvement.append({
                'success_rate_improvement': final['success_rate'] - initial['success_rate'],
                'reward_improvement': final['average_reward'] - initial['average_reward']
            })
        
        # 记录测试结果
        test_result = {
            'test_name': 'collaboration_ability',
            'num_ai_instances': num_ai_instances,
            'num_iterations': num_iterations,
            'duration_seconds': end_time - start_time,
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'performance_improvement': performance_improvement,
            'collaboration_results': collaboration_results
        }
        
        self.test_results['collaboration_tests'].append(test_result)
        
        logger.info(f"协同学习能力测试完成")
        
        return test_result
    
    async def run_all_tests(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        运行所有测试
        
        Args:
            config: 测试配置
            
        Returns:
            所有测试结果
        """
        logger.info("开始运行所有测试")
        
        # 默认配置
        default_config = {
            'learning_iterations': 100,
            'decision_test_cases': 50,
            'collaboration_instances': 3,
            'collaboration_iterations': 50
        }
        
        test_config = {**default_config, **(config or {})}
        
        # 运行学习能力测试
        learning_result = await self.test_learning_ability(test_config['learning_iterations'])
        
        # 运行决策能力测试
        decision_result = await self.test_decision_ability(test_config['decision_test_cases'])
        
        # 运行协同学习能力测试
        collaboration_result = await self.test_collaboration_ability(
            num_ai_instances=test_config['collaboration_instances'],
            num_iterations=test_config['collaboration_iterations']
        )
        
        # 汇总测试结果
        summary = {
            'total_tests': 3,
            'completed_at': datetime.now().isoformat(),
            'test_config': test_config,
            'results': self.test_results
        }
        
        logger.info("所有测试运行完成")
        
        # 打印测试汇总
        self._print_test_summary(summary)
        
        return summary
    
    def _generate_environment_states(self, count: int) -> List[Dict[str, Any]]:
        """生成模拟环境状态"""
        states = []
        for _ in range(count):
            state = {
                'temperature': np.random.uniform(10, 50),  # 温度10-50度
                'humidity': np.random.uniform(0, 100),  # 湿度0-100%
                'co2_level': np.random.uniform(300, 2000),  # CO2浓度300-2000 ppm
                'light_intensity': np.random.uniform(0, 2000),  # 光照强度0-2000 lux
                'energy_consumption': np.random.uniform(0, 1000),  # 能耗0-1000
                'resource_utilization': np.random.uniform(0, 100),  # 资源利用率0-100%
                'health_score': np.random.uniform(0, 100),  # 健康分数0-100
                'yield_potential': np.random.uniform(0, 100)  # 产量潜力0-100
            }
            states.append(state)
        return states
    
    def _generate_diverse_test_states(self, count: int) -> List[Dict[str, Any]]:
        """生成多样化的测试状态"""
        states = []
        
        # 生成各种极端状态
        extreme_conditions = [
            # 高温高湿
            {'temperature': 45, 'humidity': 90, 'co2_level': 1500, 'light_intensity': 1800},
            # 低温低湿
            {'temperature': 15, 'humidity': 20, 'co2_level': 400, 'light_intensity': 200},
            # 高能耗状态
            {'energy_consumption': 900, 'resource_utilization': 95, 'health_score': 60},
            # 健康状态差
            {'health_score': 30, 'yield_potential': 40, 'temperature': 35},
            # 理想状态
            {'temperature': 25, 'humidity': 65, 'co2_level': 600, 'light_intensity': 1000, 'health_score': 90}
        ]
        
        # 生成多样化状态
        for i in range(count):
            # 交替使用极端状态和随机状态
            if i % 2 == 0:
                # 使用极端状态
                base_state = extreme_conditions[i % len(extreme_conditions)].copy()
            else:
                # 使用随机状态
                base_state = self._generate_environment_states(1)[0]
            
            # 添加一些随机噪声
            for key, value in base_state.items():
                if isinstance(value, (int, float)):
                    noise = value * 0.1  # 10%的噪声
                    base_state[key] += np.random.uniform(-noise, noise)
            
            states.append(base_state)
        
        return states
    
    def _generate_reward(self, decision: Any, state: Dict[str, Any]) -> float:
        """生成模拟奖励"""
        # 基于决策与状态的匹配度生成奖励
        reward = 0.0
        
        # 温度控制决策奖励
        if decision.action == "action_1":  # 调整温度
            temp = state['temperature']
            if 20 <= temp <= 30:  # 理想温度范围
                reward += 1.0
            else:
                reward += 0.5
        
        # 湿度控制决策奖励
        elif decision.action == "action_2":  # 调整湿度
            humidity = state['humidity']
            if 50 <= humidity <= 70:  # 理想湿度范围
                reward += 1.0
            else:
                reward += 0.5
        
        # 资源分配决策奖励
        elif decision.action == "action_5":  # 资源分配
            reward += 0.7  # 中等奖励
        
        # 其他决策奖励
        else:
            reward += 0.5  # 默认奖励
        
        # 根据决策置信度调整奖励
        reward *= decision.confidence
        
        return reward
    
    def _generate_shared_reward(self, decisions: List[Any], state: Dict[str, Any]) -> float:
        """生成共享奖励"""
        # 基于所有决策的整体质量生成共享奖励
        total_reward = 0.0
        
        # 统计不同决策类型的分布
        action_counts = {}
        for decision in decisions:
            action = decision.action
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 多样性奖励：鼓励不同AI做出不同决策
        diversity_reward = len(action_counts) / len(decisions)
        
        # 质量奖励：基于每个决策的质量
        quality_reward = 0.0
        for decision in decisions:
            quality_reward += self._generate_reward(decision, state)
        quality_reward /= len(decisions)
        
        # 共享奖励 = 质量奖励 * 0.7 + 多样性奖励 * 0.3
        shared_reward = quality_reward * 0.7 + diversity_reward * 0.3
        
        return shared_reward
    
    def _generate_next_state(self, current_state: Dict[str, Any], decision: Any) -> Dict[str, Any]:
        """生成下一个状态"""
        next_state = current_state.copy()
        
        # 根据决策调整状态
        if decision.action == "action_1":  # 调整温度
            # 温度向理想值靠近
            next_state['temperature'] += (25 - next_state['temperature']) * 0.1
        
        elif decision.action == "action_2":  # 调整湿度
            # 湿度向理想值靠近
            next_state['humidity'] += (65 - next_state['humidity']) * 0.1
        
        elif decision.action == "action_3":  # 调整CO2
            # CO2浓度向理想值靠近
            next_state['co2_level'] += (600 - next_state['co2_level']) * 0.1
        
        # 添加一些随机变化
        for key, value in next_state.items():
            if isinstance(value, (int, float)):
                noise = value * 0.05  # 5%的随机变化
                next_state[key] += np.random.uniform(-noise, noise)
        
        return next_state
    
    async def _share_knowledge(self, ai_instances: List[Any]):
        """实现简单的知识共享"""
        # 计算平均性能指标
        total_success_rate = 0.0
        total_average_reward = 0.0
        
        for ai in ai_instances:
            total_success_rate += ai.learning_system.success_rate
            total_average_reward += ai.learning_system.average_reward
        
        avg_success_rate = total_success_rate / len(ai_instances)
        avg_average_reward = total_average_reward / len(ai_instances)
        
        # 调整每个AI实例的学习参数，向平均值靠近
        for ai in ai_instances:
            # 调整学习率
            if ai.learning_system.success_rate < avg_success_rate:
                ai.learning_system.learning_rate *= 1.05  # 提高学习率
            else:
                ai.learning_system.learning_rate *= 0.95  # 降低学习率
            
            # 限制学习率范围
            ai.learning_system.learning_rate = max(0.0001, min(0.01, ai.learning_system.learning_rate))
    
    def _print_test_summary(self, summary: Dict[str, Any]):
        """打印测试汇总结果"""
        logger.info("\n=== 主控AI测试汇总 ===")
        logger.info(f"测试完成时间: {summary['completed_at']}")
        logger.info(f"测试配置: {summary['test_config']}")
        
        # 学习能力测试汇总
        if summary['results']['learning_tests']:
            learning_test = summary['results']['learning_tests'][-1]
            logger.info("\n--- 学习能力测试结果 ---")
            logger.info(f"迭代次数: {learning_test['num_iterations']}")
            logger.info(f"测试时长: {learning_test['duration_seconds']:.2f}秒")
            logger.info(f"成功率提升: {learning_test['success_rate_improvement']:.4f}")
            logger.info(f"奖励提升: {learning_test['reward_improvement']:.4f}")
            logger.info(f"最终成功率: {learning_test['final_success_rate']:.4f}")
            logger.info(f"最终平均奖励: {learning_test['final_average_reward']:.4f}")
        
        # 决策能力测试汇总
        if summary['results']['decision_tests']:
            decision_test = summary['results']['decision_tests'][-1]
            logger.info("\n--- 决策能力测试结果 ---")
            logger.info(f"测试用例数量: {decision_test['num_test_cases']}")
            logger.info(f"测试时长: {decision_test['duration_seconds']:.2f}秒")
            logger.info(f"平均决策时间: {decision_test['avg_decision_time_ms']:.2f}毫秒")
            logger.info(f"平均置信度: {decision_test['avg_confidence']:.4f}")
            logger.info(f"平均风险分数: {decision_test['avg_risk_score']:.4f}")
            logger.info(f"决策多样性: {decision_test['action_diversity']:.4f}")
            logger.info(f"唯一决策类型: {len(decision_test['unique_actions'])}")
        
        # 协同学习能力测试汇总
        if summary['results']['collaboration_tests']:
            collaboration_test = summary['results']['collaboration_tests'][-1]
            logger.info("\n--- 协同学习能力测试结果 ---")
            logger.info(f"AI实例数量: {collaboration_test['num_ai_instances']}")
            logger.info(f"迭代次数: {collaboration_test['num_iterations']}")
            logger.info(f"测试时长: {collaboration_test['duration_seconds']:.2f}秒")
            
            # 打印每个AI实例的性能提升
            for i, improvement in enumerate(collaboration_test['performance_improvement']):
                logger.info(f"AI实例 {i+1}: 成功率提升 {improvement['success_rate_improvement']:.4f}, 奖励提升 {improvement['reward_improvement']:.4f}")
        
        logger.info("\n=== 测试完成 ===")


async def main():
    """主函数"""
    logger.info("开始主控AI能力测试")
    
    # 创建测试器实例
    tester = OrganicAITester()
    
    # 初始化AI核心
    await tester.initialize()
    
    # 运行所有测试
    config = {
        'learning_iterations': 100,
        'decision_test_cases': 50,
        'collaboration_instances': 3,
        'collaboration_iterations': 50
    }
    
    results = await tester.run_all_tests(config)
    
    logger.info("主控AI能力测试完成")


if __name__ == "__main__":
    asyncio.run(main())
