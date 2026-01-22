"""
逻辑一致性模块集成测试
测试数据流转、接口调用和错误处理
"""

import pytest
import json
from src.core.logical_consistency import LogicalConsistencyChecker


class TestLogicalConsistencyIntegration:
    """逻辑一致性模块集成测试"""
    
    def test_data_flow_input_format(self, consistency_verifier, sample_consistent_reasoning, sample_inconsistent_reasoning):
        """测试不同格式输入数据的解析"""
        # 测试文本格式输入
        result1 = consistency_verifier.verify_reasoning_text(sample_consistent_reasoning)
        assert result1['is_consistent'] == True
        
        result2 = consistency_verifier.verify_reasoning_text(sample_inconsistent_reasoning)
        assert result2['is_consistent'] == False
        
        # 测试空输入
        result3 = consistency_verifier.verify_reasoning_text("")
        # 空输入应处理为一致或返回适当错误
        assert 'is_consistent' in result3
        
        # 测试只有一个陈述的输入
        single_statement = "温度高于30度"
        result4 = consistency_verifier.verify_reasoning_text(single_statement)
        assert result4['is_consistent'] == True
    
    def test_data_flow_transformation(self, consistency_verifier, sample_consistent_reasoning):
        """测试数据在各模块间流转时的格式一致性"""
        from src.core.logical_consistency.inference_chain import InferenceChain
        
        # 创建推理链
        chain = InferenceChain()
        chain.from_reasoning_text(sample_consistent_reasoning)
        
        # 验证链的结构
        assert len(chain.nodes) > 0
        assert len(chain.start_nodes) > 0
        assert len(chain.end_nodes) > 0
        
        # 测试拓扑排序
        order = chain.topological_sort()
        assert len(order) == len(chain.nodes)
        
        # 测试验证结果格式一致性
        result = consistency_verifier.node_wise_consistency_verification(chain)
        assert isinstance(result, dict)
        assert 'is_consistent' in result
        assert 'verification_results' in result
    
    def test_data_flow_error_handling(self, consistency_verifier):
        """测试系统对非法输入的处理能力"""
        from src.core.logical_consistency.inference_chain import InferenceChain
        
        # 创建包含循环的推理链
        chain = InferenceChain()
        node1 = chain.add_node("n1", "陈述1")
        node2 = chain.add_node("n2", "陈述2")
        node3 = chain.add_node("n3", "陈述3")
        
        # 创建循环
        chain.add_edge("n1", "n2")
        chain.add_edge("n2", "n3")
        chain.add_edge("n3", "n1")
        
        # 验证循环检测
        assert chain.detect_cycles()
        
        # 测试验证结果中的错误处理
        result = consistency_verifier.node_wise_consistency_verification(chain)
        assert result['is_consistent'] == False
        assert 'error' in result
    
    def test_api_health_check(self, test_client):
        """测试健康检查接口返回正确的状态信息"""
        # 测试根路径
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        
        # 测试基本健康检查
        response = test_client.get("/health")
        assert response.status_code == 200
        
        # 测试详细健康检查
        response = test_client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_logical_consistency_checker_integration(self, sample_decision_request):
        """测试逻辑一致性检查器的集成功能"""
        # 创建逻辑一致性检查器实例
        checker = LogicalConsistencyChecker()
        
        # 测试集成检查
        history = []
        result = checker.check_consistency(sample_decision_request, history)
        
        # 验证结果结构
        assert 'is_consistent' in result
        assert 'conflicts' in result
        assert 'consistency_score' in result
        assert 'rule_check' in result
        assert 'model_check' in result
        assert 'chain_check' in result
        
        # 测试禁用推理链验证
        checker = LogicalConsistencyChecker({'use_chain_verification': False})
        result = checker.check_consistency(sample_decision_request, history)
        assert 'chain_check' in result
    
    def test_error_handling_missing_parameters(self, test_client):
        """测试缺少必要参数时的错误提示"""
        # 测试逻辑一致性检查的简化接口（如果存在）
        # 这里假设存在一个简化的一致性检查接口
        simplified_request = {
            "reasoning": "温度高于30度。湿度低于40%。"
        }
        
        # 测试空请求
        response = test_client.post("/api/logical-consistency/check", json={})
        assert response.status_code in [400, 422]  # 应该返回客户端错误
        
        # 测试缺少关键参数
        response = test_client.post("/api/logical-consistency/check", json={"invalid_param": "value"})
        assert response.status_code in [400, 422]  # 应该返回客户端错误
    
    def test_error_handling_type_errors(self, test_client):
        """测试参数类型错误时的处理"""
        # 测试参数类型错误（reasoning应该是字符串）
        invalid_request = {
            "reasoning": 12345  # 类型错误，应该是字符串
        }
        
        response = test_client.post("/api/logical-consistency/check", json=invalid_request)
        assert response.status_code in [400, 422]  # 应该返回客户端错误
    
    def test_functional_metrics_accuracy(self, consistency_verifier):
        """测试逻辑一致性检查的准确率"""
        # 准备测试用例
        test_cases = [
            # (reasoning_text, expected_is_consistent)
            ("温度高于30度。湿度低于40%。需要增加灌溉。", True),
            ("温度高于30度。温度低于20度。需要增加灌溉。", False),
            ("湿度高于80%。需要通风。执行通风操作。", True),
            ("湿度高于80%。湿度低于20%。需要通风。", False),
            ("光照强度低于5000lux。需要补光。开启补光灯。", True),
            ("光照强度低于5000lux。光照强度高于10000lux。需要补光。", False),
        ]
        
        # 执行测试
        correct_count = 0
        for reasoning_text, expected in test_cases:
            result = consistency_verifier.verify_reasoning_text(reasoning_text)
            if result['is_consistent'] == expected:
                correct_count += 1
        
        # 计算准确率
        accuracy = correct_count / len(test_cases)
        print(f"逻辑一致性检查准确率: {accuracy:.2f}")
        
        # 准确率应大于0.85
        assert accuracy > 0.85, f"准确率 {accuracy:.2f} 低于预期阈值0.85"
    
    def test_functional_metrics_precision_recall(self, consistency_verifier):
        """测试逻辑一致性检查的精确率和召回率"""
        # 准备测试用例，包含正例（矛盾）和负例（一致）
        test_cases = [
            # (reasoning_text, is_contradiction)
            ("温度高于30度。湿度低于40%。需要增加灌溉。", False),  # 一致
            ("温度高于30度。温度低于20度。需要增加灌溉。", True),   # 矛盾
            ("湿度高于80%。需要通风。执行通风操作。", False),      # 一致
            ("湿度高于80%。湿度低于20%。需要通风。", True),         # 矛盾
            ("光照强度低于5000lux。需要补光。开启补光灯。", False),  # 一致
            ("光照强度低于5000lux。光照强度高于10000lux。需要补光。", True),  # 矛盾
            ("土壤pH值低于6.0。需要施加石灰。", False),          # 一致
            ("土壤pH值低于6.0。土壤pH值高于7.5。需要施加石灰。", True),  # 矛盾
            ("CO2浓度高于1000ppm。需要通风。", False),           # 一致
            ("CO2浓度高于1000ppm。CO2浓度低于300ppm。需要通风。", True),  # 矛盾
        ]
        
        # 执行测试
        true_positives = 0  # 正确识别的矛盾
        false_positives = 0  # 误判为矛盾的一致
        false_negatives = 0  # 误判为一致的矛盾
        true_negatives = 0  # 正确识别的一致
        
        for reasoning_text, is_contradiction in test_cases:
            result = consistency_verifier.verify_reasoning_text(reasoning_text)
            predicted_contradiction = not result['is_consistent']
            
            if is_contradiction and predicted_contradiction:
                true_positives += 1
            elif not is_contradiction and predicted_contradiction:
                false_positives += 1
            elif is_contradiction and not predicted_contradiction:
                false_negatives += 1
            elif not is_contradiction and not predicted_contradiction:
                true_negatives += 1
        
        # 计算精确率和召回率
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        # 计算F1分数
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        print(f"精确率: {precision:.2f}, 召回率: {recall:.2f}, F1分数: {f1_score:.2f}")
        
        # 验证指标
        assert precision > 0.8, f"精确率 {precision:.2f} 低于预期阈值0.8"
        assert recall > 0.8, f"召回率 {recall:.2f} 低于预期阈值0.8"
        assert f1_score > 0.8, f"F1分数 {f1_score:.2f} 低于预期阈值0.8"
    
    def test_contradiction_localization_accuracy(self, consistency_verifier):
        """测试矛盾定位准确率"""
        # 准备测试用例，包含不同位置的矛盾
        test_cases = [
            # (reasoning_text, expected_error_node)
            ("温度高于30度。温度低于20度。需要增加灌溉。执行灌溉操作。", "n_2"),  # 第2个节点矛盾
            ("湿度高于80%。湿度低于20%。需要通风。执行通风操作。", "n_2"),         # 第2个节点矛盾
            ("光照强度低于5000lux。需要补光。光照强度高于10000lux。开启补光灯。", "n_3"),  # 第3个节点矛盾
        ]
        
        # 执行测试
        correct_localizations = 0
        total_contradictions = len(test_cases)
        
        for reasoning_text, expected_error_node in test_cases:
            result = consistency_verifier.verify_reasoning_text(reasoning_text)
            
            if not result['is_consistent'] and 'error_node' in result:
                if result['error_node'] == expected_error_node:
                    correct_localizations += 1
        
        # 计算矛盾定位准确率
        if total_contradictions > 0:
            localization_accuracy = correct_localizations / total_contradictions
        else:
            localization_accuracy = 0.0
        
        print(f"矛盾定位准确率: {localization_accuracy:.2f}")
        
        # 矛盾定位准确率应大于0.8
        assert localization_accuracy > 0.8, f"矛盾定位准确率 {localization_accuracy:.2f} 低于预期阈值0.8"
    
    def test_rule_engine_integration(self, sample_decision_request):
        """测试规则引擎与NCV的集成"""
        checker = LogicalConsistencyChecker()
        
        # 测试规则引擎的一致性检查
        history = []
        result = checker.check_consistency(sample_decision_request, history)
        
        # 验证规则检查结果
        assert 'rule_check' in result
        assert 'consistency_score' in result['rule_check']
        assert 'conflicts' in result['rule_check']
        
        # 测试带有冲突的决策
        conflicting_decision = sample_decision_request.copy()
        conflicting_decision['reasoning'] = "温度高于30度。温度低于20度。需要增加灌溉。执行灌溉操作。"
        
        result = checker.check_consistency(conflicting_decision, history)
        assert result['is_consistent'] == False
        
    def test_integrated_consistency_check(self, sample_decision_request):
        """测试集成一致性检查功能"""
        checker = LogicalConsistencyChecker()
        
        # 准备历史数据
        history = [
            {
                'action': 'monitor',
                'parameters': {},
                'reasoning': "正常监测",
                'timestamp': "2026-01-13T09:00:00"
            }
        ]
        
        # 执行集成检查
        result = checker.check_consistency(sample_decision_request, history)
        
        # 验证综合结果
        assert 'is_consistent' in result
        assert 'consistency_score' in result
        assert 0.0 <= result['consistency_score'] <= 1.0
        
        # 验证各模块结果
        assert 'rule_check' in result
        assert 'model_check' in result
        assert 'chain_check' in result
        
        # 验证权重配置
        assert 'weight_config' in result
        assert 'rule_weight' in result['weight_config']
        assert 'model_weight' in result['weight_config']


class TestLogicalConsistencyPerformance:
    """逻辑一致性模块性能测试"""
    
    def test_performance_response_time(self, consistency_verifier):
        """测试响应时间性能"""
        import time
        
        # 准备测试用例
        reasoning_text = "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。光照强度低于5000lux。需要补光。开启补光灯。CO2浓度高于1000ppm。需要通风。执行通风操作。"
        
        # 多次执行测量平均响应时间
        num_executions = 10
        total_time = 0.0
        
        for _ in range(num_executions):
            start_time = time.time()
            consistency_verifier.verify_reasoning_text(reasoning_text)
            end_time = time.time()
            total_time += (end_time - start_time)
        
        average_time = total_time / num_executions
        print(f"平均响应时间: {average_time:.4f}秒")
        
        # 平均响应时间应小于0.1秒
        assert average_time < 0.1, f"平均响应时间 {average_time:.4f}秒 超过阈值0.1秒"
    
    def test_performance_throughput(self, consistency_verifier):
        """测试吞吐量性能"""
        import time
        
        # 准备测试用例
        test_cases = [
            "温度高于30度。湿度低于40%。需要增加灌溉。",
            "湿度高于80%。需要通风。执行通风操作。",
            "光照强度低于5000lux。需要补光。开启补光灯。",
            "CO2浓度高于1000ppm。需要通风。执行通风操作。",
            "土壤pH值低于6.0。需要施加石灰。执行施肥操作。",
        ]
        
        # 测量处理时间
        num_requests = 100
        start_time = time.time()
        
        for i in range(num_requests):
            reasoning_text = test_cases[i % len(test_cases)]
            consistency_verifier.verify_reasoning_text(reasoning_text)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算吞吐量
        throughput = num_requests / total_time
        print(f"吞吐量: {throughput:.2f} 请求/秒")
        
        # 吞吐量应大于10请求/秒
        assert throughput > 10, f"吞吐量 {throughput:.2f} 请求/秒 低于阈值10请求/秒"
    
    def test_performance_stability(self, consistency_verifier):
        """测试系统稳定性"""
        # 准备测试用例
        reasoning_text = "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。"
        
        # 长时间运行测试
        num_iterations = 100
        success_count = 0
        
        for _ in range(num_iterations):
            try:
                result = consistency_verifier.verify_reasoning_text(reasoning_text)
                if 'is_consistent' in result:
                    success_count += 1
            except Exception as e:
                print(f"测试过程中发生错误: {e}")
        
        # 计算稳定性
        stability = success_count / num_iterations
        print(f"系统稳定性: {stability:.2f}")
        
        # 稳定性应大于0.99
        assert stability > 0.99, f"系统稳定性 {stability:.2f} 低于阈值0.99"
