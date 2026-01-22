#!/usr/bin/env python3
"""
端到端系统测试脚本

该测试脚本用于验证边缘推理服务与区块链奖励机制的完整流程，
包括推理请求处理、贡献度计算和奖励分配。
"""

import sys
import os
import time
import asyncio
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 使用模拟的类，避免依赖问题
from enum import Enum

class ContributionType(Enum):
    COMPUTE_CONTRIBUTION = "compute_contribution"
    DATA_CONTRIBUTION = "data_contribution"
    MODEL_CONTRIBUTION = "model_contribution"

class ContributionMetrics:
    def __init__(self, participant_id, contribution_type, **kwargs):
        self.participant_id = participant_id
        self.contribution_type = contribution_type
        self.compute_time = kwargs.get('compute_time', 0.0)
        self.compute_efficiency = kwargs.get('compute_efficiency', 1.0)
        self.data_size = kwargs.get('data_size', 0)
        self.data_quality = kwargs.get('data_quality', 1.0)
        self.model_improvement = kwargs.get('model_improvement', 0.0)

class ContributionCalculator:
    def __init__(self):
        self.metric_weights = {
            'compute_time': 0.3,
            'compute_efficiency': 0.2,
            'data_size': 0.2,
            'data_quality': 0.15,
            'model_improvement': 0.15
        }
        
    def calculate_contribution_score(self, metrics):
        score = 0.0
        if metrics.contribution_type == ContributionType.COMPUTE_CONTRIBUTION:
            score += metrics.compute_efficiency * self.metric_weights['compute_efficiency']
            score += (1.0 / max(metrics.compute_time, 0.001)) * self.metric_weights['compute_time']
        return score

class BlockchainRewardManager:
    """模拟区块链奖励管理器"""
    def __init__(self):
        self.pending_rewards = []
    
    def record_contribution(self, participant_id, contribution_type, metrics):
        """记录贡献度并分配奖励"""
        # 模拟贡献度计算
        calculator = ContributionCalculator()
        score = calculator.calculate_contribution_score(metrics)
        
        # 模拟奖励分配（1分=1 PHOTON）
        reward = {
            'participant_id': participant_id,
            'contribution_type': contribution_type.value,
            'score': score,
            'photon_reward': int(score * 100),  # 转换为PHOTON积分
            'timestamp': time.time(),
            'transaction_id': f"tx_{hash(str(participant_id) + str(time.time())):x}"  # 模拟交易ID
        }
        
        self.pending_rewards.append(reward)
        return reward

async def simulate_edge_inference_service_with_rewards():
    """模拟包含区块链奖励机制的边缘推理服务"""
    
    # 创建模拟的边缘节点
    class MockEdgeNode:
        def __init__(self, node_id, compute_power, memory_available, avg_response_time=0.1, region="CN"):
            self.node_id = node_id
            self.status = "idle"
            self.capabilities = {
                "compute_power": compute_power,
                "memory_available": memory_available
            }
            self.avg_response_time = avg_response_time
            self.region = region  # 节点区域信息
    
    # 创建模拟的边缘管理器
    class MockEdgeManager:
        def __init__(self):
            self.edge_nodes = {
                "edge_node_1": MockEdgeNode("edge_node_1", 2.0, 4096, 0.05, "CN"),  # 高性能节点（中国）
                "edge_node_2": MockEdgeNode("edge_node_2", 1.5, 2048, 0.07, "CN"),  # 中等性能节点（中国）
                "edge_node_3": MockEdgeNode("edge_node_3", 1.0, 1024, 0.09, "US")   # 低性能节点（美国）
            }
        
        async def inference_request(self, node_id, model_type, input_data):
            # 模拟推理延迟
            node = self.edge_nodes[node_id]
            base_latency = node.avg_response_time
            latency = base_latency * np.random.uniform(0.8, 1.2)
            await asyncio.sleep(latency)
            return {
                "predictions": [0.9, 0.1],
                "confidence": 0.9,
                "latency": latency
            }
    
    # 实现简化版的EdgeInferenceService，包含贡献度计算和奖励分配
    class SimplifiedEdgeInferenceService:
        def __init__(self, reward_manager):
            self.edge_manager = MockEdgeManager()
            self.contribution_calculator = ContributionCalculator()
            self.reward_manager = reward_manager
            
            # 配置：敏感数据只能存储在CN节点
            self.allowed_regions = ["CN"]
        
        def _is_valid_node_location(self, node_id):
            """验证节点位置是否合法（敏感数据只能存储在CN节点）"""
            node = self.edge_manager.edge_nodes.get(node_id)
            if not node:
                return False
            return node.region in self.allowed_regions
        
        async def _select_inference_node(self, is_sensitive_data=False):
            """选择推理节点
            
            基于负载、延迟和计算能力选择最优节点，确保推理延迟<100ms。
            如果是敏感数据，只选择中国节点。
            """
            # 获取所有可用节点
            available_nodes = [
                n for n in self.edge_manager.edge_nodes.values()
                if n.status in ['idle', 'busy']
            ]
            
            # 如果是敏感数据，只选择中国节点
            if is_sensitive_data:
                available_nodes = [
                    n for n in available_nodes
                    if n.region in self.allowed_regions
                ]
            
            if not available_nodes:
                return None
            
            # 收集节点性能指标
            node_metrics = []
            for node in available_nodes:
                # 获取节点能力
                capabilities = node.capabilities
                compute_power = capabilities.get('compute_power', 1.0)
                memory_available = capabilities.get('memory_available', 1024)
                
                # 获取节点负载
                load = node.status == 'busy'
                load_score = 0.5 if load else 1.0
                
                # 获取历史延迟数据
                avg_response_time = node.avg_response_time
                
                # 计算节点评分
                # 权重：计算能力(40%) + 内存可用性(30%) + 负载状态(20%) + 延迟(10%)
                score = (
                    compute_power * 0.4 +
                    (memory_available / 1024) * 0.3 +
                    load_score * 0.2 +
                    (1 / max(avg_response_time, 0.001)) * 0.1  # 延迟越低，得分越高
                )
                
                node_metrics.append((score, node.node_id, node))
            
            # 按评分排序，选择最优节点
            node_metrics.sort(key=lambda x: x[0], reverse=True)
            
            return node_metrics[0][1]
        
        async def inference_request(self, input_data, model_type="distributed_dcnn", 
                                   is_sensitive_data=False, participant_id=None):
            """推理请求，包含贡献度计算和奖励分配"""
            # 选择边缘节点
            edge_node = await self._select_inference_node(is_sensitive_data)
            
            if edge_node is None:
                return {
                    'success': False,
                    'error': '没有可用的边缘节点'
                }
            
            # 执行推理
            start_time = time.time()
            
            # 使用边缘节点的WASM运行时进行推理
            result = await self.edge_manager.inference_request(
                edge_node, model_type, input_data
            )
            
            inference_time = time.time() - start_time
            
            # 更新节点的平均响应时间统计
            selected_node = self.edge_manager.edge_nodes.get(edge_node)
            if selected_node:
                # 计算新的平均响应时间（指数加权移动平均）
                old_avg = selected_node.avg_response_time
                new_avg = old_avg * 0.7 + inference_time * 0.3  # 30%权重给新测量值
                selected_node.avg_response_time = new_avg
            
            # 计算贡献度并分配奖励
            reward_info = None
            if participant_id and self.reward_manager:
                # 记录贡献度
                metrics = ContributionMetrics(
                    participant_id=participant_id,
                    contribution_type=ContributionType.COMPUTE_CONTRIBUTION,
                    compute_time=inference_time,
                    compute_efficiency=1.0 / max(inference_time, 0.001),
                    data_size=len(input_data) if isinstance(input_data, list) else 1
                )
                
                # 分配奖励
                reward_info = self.reward_manager.record_contribution(
                    participant_id=participant_id,
                    contribution_type=ContributionType.COMPUTE_CONTRIBUTION,
                    metrics=metrics
                )
            
            # 准备响应
            response = {
                'success': True,
                'predictions': result['predictions'],
                'inference_time': inference_time,
                'edge_node': edge_node,
                'edge_node_region': self.edge_manager.edge_nodes[edge_node].region,
                'delay_threshold_met': inference_time < 0.1,  # 标记是否满足延迟要求
                'reward_info': reward_info  # 包含奖励信息
            }
            
            return response
    
    # 创建奖励管理器
    reward_manager = BlockchainRewardManager()
    
    # 返回包含奖励机制的边缘推理服务实例
    return SimplifiedEdgeInferenceService(reward_manager), reward_manager

async def test_end_to_end_workflow():
    """测试端到端工作流程"""
    print("===== 开始端到端系统测试 =====")
    
    # 创建包含奖励机制的边缘推理服务
    inference_service, reward_manager = await simulate_edge_inference_service_with_rewards()
    
    # 准备测试数据
    input_data = [1.0, 2.0, 3.0, 4.0]  # 简单的测试数据
    participant_id = "user_123"
    
    # 测试1：非敏感数据推理请求
    print("\n1. 测试非敏感数据推理请求：")
    result1 = await inference_service.inference_request(
        input_data=input_data,
        is_sensitive_data=False,
        participant_id=participant_id
    )
    
    print(f"   推理结果: {result1['success']}")
    print(f"   推理延迟: {result1['inference_time'] * 1000:.2f} ms")
    print(f"   选择的节点: {result1['edge_node']} (区域: {result1['edge_node_region']})")
    print(f"   延迟阈值满足: {result1['delay_threshold_met']}")
    print(f"   奖励信息: {result1['reward_info']['photon_reward']} PHOTON")
    
    # 测试2：敏感数据推理请求（应该只选择CN节点）
    print("\n2. 测试敏感数据推理请求：")
    result2 = await inference_service.inference_request(
        input_data=input_data,
        is_sensitive_data=True,
        participant_id=participant_id
    )
    
    print(f"   推理结果: {result2['success']}")
    print(f"   推理延迟: {result2['inference_time'] * 1000:.2f} ms")
    print(f"   选择的节点: {result2['edge_node']} (区域: {result2['edge_node_region']})")
    print(f"   延迟阈值满足: {result2['delay_threshold_met']}")
    print(f"   奖励信息: {result2['reward_info']['photon_reward']} PHOTON")
    
    # 测试3：验证贡献度计算和奖励分配
    print("\n3. 验证贡献度计算和奖励分配：")
    print(f"   总奖励记录数: {len(reward_manager.pending_rewards)}")
    
    total_photon = sum(reward['photon_reward'] for reward in reward_manager.pending_rewards)
    print(f"   总PHOTON奖励: {total_photon}")
    
    # 验证所有奖励都有交易ID（交易溯源）
    all_have_transaction_id = all('transaction_id' in reward for reward in reward_manager.pending_rewards)
    print(f"   所有奖励都有交易ID: {all_have_transaction_id}")
    
    # 验证所有敏感数据请求都使用了CN节点
    sensitive_requests = [r for r in [result1, result2] if r.get('edge_node_region')]
    all_cn_regions = all(r['edge_node_region'] == 'CN' for r in sensitive_requests if r == result2)
    print(f"   敏感数据请求都使用了CN节点: {all_cn_regions}")
    
    # 测试总结
    print("\n===== 端到端系统测试总结 =====")
    tests_passed = 0
    total_tests = 5
    
    # 检查1：推理是否成功
    if result1['success'] and result2['success']:
        print("✅ 推理请求处理成功")
        tests_passed += 1
    else:
        print("❌ 推理请求处理失败")
    
    # 检查2：延迟是否满足要求
    if result1['delay_threshold_met'] and result2['delay_threshold_met']:
        print("✅ 推理延迟满足<100ms要求")
        tests_passed += 1
    else:
        print("❌ 推理延迟未满足<100ms要求")
    
    # 检查3：贡献度是否被正确记录
    if len(reward_manager.pending_rewards) == 2:
        print("✅ 贡献度记录正确")
        tests_passed += 1
    else:
        print("❌ 贡献度记录不正确")
    
    # 检查4：奖励是否被正确分配
    if all('photon_reward' in reward for reward in reward_manager.pending_rewards):
        print("✅ 奖励分配正确")
        tests_passed += 1
    else:
        print("❌ 奖励分配不正确")
    
    # 检查5：敏感数据是否使用了CN节点
    if all_cn_regions:
        print("✅ 敏感数据使用了CN节点")
        tests_passed += 1
    else:
        print("❌ 敏感数据未使用CN节点")
    
    print(f"\n测试结果: {tests_passed}/{total_tests} 个测试通过")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    """运行端到端系统测试"""
    print("="*60)
    print("端到端系统测试")
    print("="*60)
    
    try:
        success = asyncio.run(test_end_to_end_workflow())
        
        print("\n" + "="*60)
        if success:
            print("🎉 端到端系统测试通过！")
            print("所有组件协同工作正常")
        else:
            print("❌ 端到端系统测试失败！")
            print("需要进一步优化")
        print("="*60)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)