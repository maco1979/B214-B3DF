# Test cases for logical consistency module
# Including inference chain verification and forward chaining
import pytest
from src.core.logical_consistency import (
    LogicalConsistencyChecker,
    NodeWiseConsistencyVerifier,
    InferenceChain
)
from src.core.logical_consistency.rule_engine import (
    forward_chaining,
    DecisionFactRule,
    EnhancedRuleEngine
)


class TestInferenceChain:
    """测试推理链功能"""
    
    def test_create_linear_chain(self):
        """测试创建线性推理链"""
        chain = InferenceChain()
        statements = [
            "温度高于30度",
            "湿度低于40%",
            "需要增加灌溉",
            "执行灌溉操作"
        ]
        chain.build_linear_chain(statements)
        
        assert len(chain.nodes) == 4
        assert len(chain.start_nodes) == 1
        assert len(chain.end_nodes) == 1
        assert not chain.detect_cycles()
    
    def test_topological_sort(self):
        """测试拓扑排序"""
        chain = InferenceChain()
        statements = [
            "温度高于30度",
            "湿度低于40%",
            "需要增加灌溉",
            "执行灌溉操作"
        ]
        chain.build_linear_chain(statements)
        
        sorted_nodes = chain.topological_sort()
        assert len(sorted_nodes) == 4
        # 验证排序顺序
        assert sorted_nodes[0].node_id == "n_1"
        assert sorted_nodes[1].node_id == "n_2"
        assert sorted_nodes[2].node_id == "n_3"
        assert sorted_nodes[3].node_id == "n_4"
    
    def test_detect_cycles(self):
        """测试循环检测"""
        chain = InferenceChain()
        # 创建包含循环的链
        node1 = chain.add_node("n1", "陈述1")
        node2 = chain.add_node("n2", "陈述2")
        node3 = chain.add_node("n3", "陈述3")
        
        # 创建循环
        chain.add_edge("n1", "n2")
        chain.add_edge("n2", "n3")
        chain.add_edge("n3", "n1")
        
        assert chain.detect_cycles()
    
    def test_from_reasoning_text(self):
        """测试从推理文本构建推理链"""
        chain = InferenceChain()
        reasoning_text = "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。"
        chain.from_reasoning_text(reasoning_text)
        
        assert len(chain.nodes) == 4


class TestNodeWiseConsistencyVerifier:
    """测试节点级一致性验证器"""
    
    def test_verify_consistent_chain(self):
        """测试验证一致的推理链"""
        verifier = NodeWiseConsistencyVerifier()
        reasoning_text = "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。"
        
        result = verifier.verify_reasoning_text(reasoning_text)
        
        assert result['is_consistent'] == True
        assert result['message'] == "All steps are consistent"
        assert len(result['verification_results']) == 4
    
    def test_verify_inconsistent_chain(self):
        """测试验证不一致的推理链"""
        verifier = NodeWiseConsistencyVerifier()
        reasoning_text = "温度高于30度。温度低于20度。需要增加灌溉。执行灌溉操作。"
        
        result = verifier.verify_reasoning_text(reasoning_text)
        
        assert result['is_consistent'] == False
        assert "Error at node" in result['error']
        assert len(result['verification_results']) > 0


class TestLogicalConsistencyChecker:
    """测试逻辑一致性检查器"""
    
    def test_integrated_check(self):
        """测试集成检查，包括推理链验证"""
        checker = LogicalConsistencyChecker()
        
        decision = {
            'action': 'irrigate',
            'parameters': {
                'duration': 30,
                'amount': 50
            },
            'reasoning': "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。",
            'timestamp': "2026-01-13T10:00:00"
        }
        
        history = [
            {
                'action': 'monitor',
                'parameters': {},
                'reasoning': "正常监测",
                'timestamp': "2026-01-13T09:00:00"
            }
        ]
        
        result = checker.check_consistency(decision, history)
        
        assert 'chain_check' in result
        assert 'rule_check' in result
        assert 'model_check' in result
        assert 'consistency_score' in result
    
    def test_disable_chain_verification(self):
        """测试禁用推理链验证"""
        checker = LogicalConsistencyChecker({
            'use_chain_verification': False
        })
        
        decision = {
            'action': 'irrigate',
            'parameters': {
                'duration': 30,
                'amount': 50
            },
            'reasoning': "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。",
            'timestamp': "2026-01-13T10:00:00"
        }
        
        history = []
        result = checker.check_consistency(decision, history)
        
        # 验证chain_check结果为空
        assert 'chain_check' in result
        assert not result['chain_check'] or 'verification_results' not in result['chain_check']


class TestForwardChaining:
    """测试正向链推理算法"""
    
    def test_basic_forward_chaining(self):
        """测试基本的正向链推理"""
        # 创建规则
        rules = [
            DecisionFactRule(
                name="rule1",
                description="高温低湿需要增加灌溉",
                conditions=["temperature_high", "humidity_low"],
                conclusions=["need_increase_irrigation"]
            ),
            DecisionFactRule(
                name="rule2",
                description="需要灌溉且土壤干燥则执行灌溉",
                conditions=["need_increase_irrigation", "soil_dry"],
                conclusions=["execute_irrigation"]
            )
        ]
        
        # 已知事实
        facts = {"temperature_high", "humidity_low", "soil_dry"}
        
        # 执行正向链推理
        inferred_facts = forward_chaining(facts, rules)
        
        # 验证结果
        assert "need_increase_irrigation" in inferred_facts
        assert "execute_irrigation" in inferred_facts
        assert len(inferred_facts) == 5  # 3个初始事实 + 2个推断事实
    
    def test_empty_facts(self):
        """测试空事实集"""
        rules = [
            DecisionFactRule(
                name="rule1",
                description="高温低湿需要增加灌溉",
                conditions=["temperature_high", "humidity_low"],
                conclusions=["need_increase_irrigation"]
            )
        ]
        
        facts = set()
        inferred_facts = forward_chaining(facts, rules)
        
        assert inferred_facts == set()
    
    def test_no_rules(self):
        """测试无规则情况"""
        rules = []
        facts = {"temperature_high", "humidity_low"}
        inferred_facts = forward_chaining(facts, rules)
        
        assert inferred_facts == facts


class TestEnhancedRuleEngine:
    """测试增强型规则引擎"""
    
    def test_enhanced_rule_engine_init(self):
        """测试增强型规则引擎初始化"""
        engine = EnhancedRuleEngine()
        assert len(engine.rules) > 0  # 默认规则已初始化
        assert hasattr(engine, 'forward_rules')
        assert len(engine.forward_rules) > 0  # 正向链规则已初始化
    
    def test_check_consistency_with_reasoning(self):
        """测试使用正向链推理增强一致性检查"""
        engine = EnhancedRuleEngine()
        
        decision = {
            'action': 'reduce_irrigation',
            'parameters': {
                'temperature': 36.0,
                'humidity': 15.0
            },
            'reasoning': "温度高于35度且湿度低于20%，需要增加灌溉"
        }
        
        history = []
        result = engine.check_consistency_with_reasoning(decision, history)
        
        assert 'forward_chaining_result' in result
        assert 'inferred_facts' in result['forward_chaining_result']
        assert 'is_consistent' in result['forward_chaining_result']
        assert 'need_increase_irrigation' in result['forward_chaining_result']['inferred_facts']
        assert result['forward_chaining_result']['is_consistent'] == False  # 决策与推理结果不一致
    
    def test_consistent_decision(self):
        """测试一致的决策"""
        engine = EnhancedRuleEngine()
        
        decision = {
            'action': 'increase_irrigation',
            'parameters': {
                'temperature': 36.0,
                'humidity': 15.0
            },
            'reasoning': "温度高于35度且湿度低于20%，需要增加灌溉"
        }
        
        history = []
        result = engine.check_consistency_with_reasoning(decision, history)
        
        assert result['forward_chaining_result']['is_consistent'] == True  # 决策与推理结果一致
        assert 'need_increase_irrigation' in result['forward_chaining_result']['inferred_facts']
