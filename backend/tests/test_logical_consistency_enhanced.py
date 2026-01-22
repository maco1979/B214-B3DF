import pytest
import sys
import os
import numpy as np

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from core.logical_consistency.inference_chain import Node, InferenceChain, NodeWiseConsistencyVerifier
from core.logical_consistency.consistency_checker import LogicalConsistencyChecker


def test_node_embedding_generation():
    """测试节点嵌入生成功能"""
    node = Node("n1", "温度高于30摄氏度")
    node.generate_embedding()
    
    # 验证嵌入已生成
    assert node.embedding is not None
    # 验证嵌入是numpy数组
    assert isinstance(node.embedding, np.ndarray)
    # 验证嵌入有正确的维度（基于MD5哈希）
    assert node.embedding.shape == (16,)  # MD5哈希生成16字节，即16维向量


def test_atomic_statement_extraction():
    """测试原子语句提取功能"""
    # 测试包含多个句子的内容
    node = Node("n1", "温度高于30摄氏度。湿度低于60%。压力正常。")
    assert len(node.atomic_statements) == 3
    
    # 测试包含分号的内容
    node2 = Node("n2", "温度高于30摄氏度；湿度低于60%")
    assert len(node2.atomic_statements) == 2
    
    # 测试单行内容
    node3 = Node("n3", "温度高于30摄氏度")
    assert len(node3.atomic_statements) == 1


def test_hierarchical_verification():
    """测试分层验证架构"""
    # 创建推理链
    chain = InferenceChain()
    chain.build_linear_chain([
        "今天天气晴朗。",
        "温度高于30摄氏度。",
        "湿度低于60%。",
        "适合户外活动。"
    ])
    
    # 创建验证器
    verifier = NodeWiseConsistencyVerifier()
    
    # 运行节点级一致性验证
    result = verifier.node_wise_consistency_verification(chain)
    
    # 验证结果
    assert result['is_consistent'] == True
    assert result['message'] == "All steps are consistent"
    assert len(result['verification_results']) == 4
    
    # 检查每个节点的验证结果
    for i, result in enumerate(result['verification_results']):
        assert result['result'] == "Correct"
        # 检查验证级别信息
        assert 'verification_levels' in result
        assert result['verification_levels']['atomic'] == 'passed'
        assert result['verification_levels']['dependency'] == 'passed'
        assert result['verification_levels']['global'] == 'passed'


def test_inconsistent_chain_detection():
    """测试不一致链检测"""
    # 创建包含矛盾的推理链
    chain = InferenceChain()
    chain.build_linear_chain([
        "今天天气晴朗。",
        "温度高于30摄氏度。",
        "温度低于20摄氏度。",  # 这里出现矛盾
        "适合户外活动。"
    ])
    
    # 创建验证器
    verifier = NodeWiseConsistencyVerifier()
    
    # 运行节点级一致性验证
    result = verifier.node_wise_consistency_verification(chain)
    
    # 验证结果
    assert result['is_consistent'] == False
    assert result['error_node'] == "n_3"
    assert "温度低于20摄氏度" in result['error_content']
    
    # 检查失败的验证级别
    for result in result['verification_results']:
        if result['node_id'] == "n_3":
            assert result['result'] == "Incorrect"
            # 应该是在依赖验证或全局验证级别失败
            assert result['verification_levels']['dependency'] == 'failed' or result['verification_levels']['global'] == 'failed'


def test_consistency_checker_integration():
    """测试一致性检查器集成"""
    # 创建一致性检查器
    checker = LogicalConsistencyChecker()
    
    # 测试一致的决策
    consistent_decision = {
        'reasoning': "今天天气晴朗。温度高于30摄氏度。湿度低于60%。适合户外活动。",
        'parameters': {
            'temperature': 35,
            'humidity': 50,
            'weather': 'sunny'
        }
    }
    
    result = checker.check_consistency(consistent_decision, [])
    assert result['is_consistent'] == True
    assert result['consistency_score'] >= 0.9
    
    # 测试不一致的决策
    inconsistent_decision = {
        'reasoning': "今天天气晴朗。温度高于30摄氏度。温度低于20摄氏度。适合户外活动。",
        'parameters': {
            'temperature': 35,
            'humidity': 50,
            'weather': 'sunny'
        }
    }
    
    result = checker.check_consistency(inconsistent_decision, [])
    assert result['is_consistent'] == False
    assert result['consistency_score'] < 0.5


def test_cross_node_dependencies():
    """测试节点间依赖关系"""
    # 创建推理链
    chain = InferenceChain()
    chain.build_linear_chain([
        "如果温度高于30摄氏度，那么天气炎热。",
        "温度高于30摄氏度。",
        "因此，天气炎热。"
    ])
    
    # 创建验证器
    verifier = NodeWiseConsistencyVerifier()
    
    # 运行节点级一致性验证
    result = verifier.node_wise_consistency_verification(chain)
    
    # 验证结果
    assert result['is_consistent'] == True
    assert result['message'] == "All steps are consistent"


def test_embedding_similarity_check():
    """测试嵌入相似性检查"""
    # 创建两个完全相同的节点
    node1 = Node("n1", "温度很高")
    node2 = Node("n2", "温度很高")

    # 生成嵌入
    node1.generate_embedding()
    node2.generate_embedding()

    # 计算相似度
    from numpy import dot
    from numpy.linalg import norm

    cosine_similarity = dot(node1.embedding, node2.embedding) / (norm(node1.embedding) * norm(node2.embedding))

    # 完全相同的节点应该有极高的相似度
    assert cosine_similarity > 0.95
    
    # 测试嵌入生成成功
    assert node1.embedding is not None
    assert node2.embedding is not None
    assert isinstance(node1.embedding, np.ndarray)
    assert isinstance(node2.embedding, np.ndarray)


def test_global_context_verification():
    """测试全局上下文验证"""
    # 创建推理链
    chain = InferenceChain()
    chain.build_linear_chain([
        "今天是周一。",
        "明天是周二。",
        "后天是周三。",
        "今天是周五。"  # 与第一条矛盾
    ])
    
    # 创建验证器
    verifier = NodeWiseConsistencyVerifier()
    
    # 运行节点级一致性验证
    result = verifier.node_wise_consistency_verification(chain)
    
    # 验证结果
    assert result['is_consistent'] == False
    assert result['error_node'] == "n_4"


if __name__ == "__main__":
    import numpy as np
    test_node_embedding_generation()
    test_atomic_statement_extraction()
    test_hierarchical_verification()
    test_inconsistent_chain_detection()
    test_consistency_checker_integration()
    test_cross_node_dependencies()
    test_embedding_similarity_check()
    test_global_context_verification()
    print("所有逻辑一致性测试通过！")
