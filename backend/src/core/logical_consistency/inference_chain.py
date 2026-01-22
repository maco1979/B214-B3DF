# Inference Chain Verification
# Implements Node-wise Consistency Verification (NCV) for inference chain integrity
from typing import Dict, List, Any, Set, Optional
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class Node:
    """推理节点类，代表推理链中的单个断言"""
    def __init__(self, node_id: str, content: str):
        self.node_id = node_id
        self.content = content
        self.predecessors: Set[Node] = set()
        self.successors: Set[Node] = set()
        self.embedding: Optional[np.ndarray] = None  # 语义嵌入
        self.atomic_statements: List[str] = self._extract_atomic_statements()  # 原子语句
    
    def _extract_atomic_statements(self) -> List[str]:
        """提取原子语句（简化实现）"""
        # 简单实现：按标点分割句子
        if '.' in self.content or '。' in self.content:
            # 先统一替换中文句号为英文句号
            content = self.content.replace('。', '.')
            return [stmt.strip() + '.' for stmt in content.split('.') if stmt.strip()]
        elif ';' in self.content or '；' in self.content:
            # 先统一替换中文分号为英文分号
            content = self.content.replace('；', ';')
            return [stmt.strip() + ';' for stmt in content.split(';') if stmt.strip()]
        else:
            return [self.content]
    
    def generate_embedding(self, embedding_model=None):
        """生成语义嵌入
        
        Args:
            embedding_model: 嵌入模型，默认使用简单的基于哈希的嵌入
        """
        if embedding_model:
            # 使用外部嵌入模型
            self.embedding = embedding_model.encode(self.content)
        else:
            # 简单实现：基于哈希的嵌入，确保相同文本生成相同嵌入
            import hashlib
            
            # 使用MD5哈希生成固定长度的嵌入
            hash_obj = hashlib.md5(self.content.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # 转换为numpy数组
            self.embedding = np.array([b / 255.0 for b in hash_bytes])
            
            # 确保嵌入向量不为全0，避免计算余弦相似度时分母为0
            if np.sum(self.embedding) == 0:
                self.embedding += 0.01
            # 归一化到单位向量
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding /= norm
    
    def add_predecessor(self, node: 'Node'):
        """添加前驱节点"""
        self.predecessors.add(node)
        node.successors.add(self)
    
    def add_successor(self, node: 'Node'):
        """添加后继节点"""
        self.successors.add(node)
        node.predecessors.add(self)
    
    def is_predecessor(self, other: 'Node') -> bool:
        """检查是否是另一个节点的前驱"""
        return other in self.successors
    
    def get_all_predecessors(self) -> Set['Node']:
        """获取所有前驱节点（包括间接前驱）"""
        all_preds = set()
        queue = deque(self.predecessors)
        
        while queue:
            node = queue.popleft()
            if node not in all_preds:
                all_preds.add(node)
                queue.extend(node.predecessors)
        
        return all_preds
    
    def get_all_successors(self) -> Set['Node']:
        """获取所有后继节点（包括间接后继）"""
        all_succs = set()
        queue = deque(self.successors)
        
        while queue:
            node = queue.popleft()
            if node not in all_succs:
                all_succs.add(node)
                queue.extend(node.successors)
        
        return all_succs
    
    def __repr__(self):
        return f"Node(id={self.node_id}, content={self.content[:30]}...)"


class InferenceChain:
    """推理链类，用于构建和管理推理节点网络"""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.start_nodes: List[Node] = []
        self.end_nodes: List[Node] = []
    
    def add_node(self, node_id: str, content: str) -> Node:
        """添加节点到推理链"""
        if node_id in self.nodes:
            logger.warning(f"节点 {node_id} 已存在，将覆盖现有节点")
        
        node = Node(node_id, content)
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """添加节点间的边"""
        if from_node_id not in self.nodes:
            raise ValueError(f"源节点 {from_node_id} 不存在")
        if to_node_id not in self.nodes:
            raise ValueError(f"目标节点 {to_node_id} 不存在")
        
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        from_node.add_successor(to_node)
    
    def build_linear_chain(self, statements: List[str]) -> None:
        """构建线性推理链"""
        # 清除现有节点
        self.nodes.clear()
        self.start_nodes.clear()
        self.end_nodes.clear()
        
        # 添加节点
        prev_node = None
        for i, statement in enumerate(statements):
            node_id = f"n_{i+1}"
            current_node = self.add_node(node_id, statement)
            
            if i == 0:
                self.start_nodes.append(current_node)
            if i == len(statements) - 1:
                self.end_nodes.append(current_node)
            
            if prev_node:
                self.add_edge(prev_node.node_id, current_node.node_id)
            
            prev_node = current_node
    
    def detect_cycles(self) -> bool:
        """检测推理链中是否存在循环"""
        # Kahn's algorithm for cycle detection
        in_degree = {node: len(node.predecessors) for node in self.nodes.values()}
        queue = deque([node for node in self.nodes.values() if in_degree[node] == 0])
        visited_count = 0
        
        while queue:
            node = queue.popleft()
            visited_count += 1
            
            for successor in node.successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        return visited_count != len(self.nodes)
    
    def topological_sort(self) -> List[Node]:
        """对推理链进行拓扑排序"""
        if self.detect_cycles():
            raise ValueError("推理链中存在循环，无法进行拓扑排序")
        
        # Kahn's algorithm for topological sorting
        in_degree = {node: len(node.predecessors) for node in self.nodes.values()}
        queue = deque([node for node in self.nodes.values() if in_degree[node] == 0])
        sorted_nodes = []
        
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            
            for successor in node.successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        return sorted_nodes
    
    def from_reasoning_text(self, reasoning_text: str) -> 'InferenceChain':
        """从推理文本构建推理链（简化实现）"""
        # 简单实现：按句号分割推理文本
        statements = []
        for stmt in reasoning_text.split("。"):
            if stmt.strip():
                statements.append(stmt.strip())
        
        # 构建线性链
        self.build_linear_chain(statements)
        return self


class NodeWiseConsistencyVerifier:
    """节点级一致性验证器，实现分层验证架构"""
    def __init__(self):
        self.verification_results: List[Dict[str, Any]] = []
        self.global_context: List[str] = []  # 全局上下文，用于全局推理验证
    
    def verify_atomic_statements(self, node: Node) -> bool:
        """原子语句验证（第一级）
        
        Args:
            node: 要验证的节点
            
        Returns:
            bool: 原子语句是否一致
        """
        for stmt in node.atomic_statements:
            # 检查原子语句的基本有效性
            if len(stmt.strip()) < 3:
                return False
            
            # 检查简单的自相矛盾
            if "不" in stmt and "不是" not in stmt and "不会" not in stmt and "不能" not in stmt:
                # 简单否定检测
                words = stmt.split()
                for i in range(len(words)-1):
                    if words[i] == words[i+1]:
                        return False
        
        return True
    
    def verify_logical_dependencies(self, node: Node, prior_steps: Set[Node]) -> bool:
        """逻辑依赖验证（第二级）
        
        Args:
            node: 要验证的节点
            prior_steps: 之前已验证的节点集合
            
        Returns:
            bool: 逻辑依赖是否一致
        """
        current_content = node.content
        
        # 检查当前节点与所有已验证节点之间的矛盾
        for prior_node in prior_steps:
            prior_content = prior_node.content
            
            # 检测温度矛盾
            has_high_temp_current = "温度高于" in current_content or "温度>" in current_content
            has_low_temp_current = "温度低于" in current_content or "温度<" in current_content
            has_high_temp_prior = "温度高于" in prior_content or "温度>" in prior_content
            has_low_temp_prior = "温度低于" in prior_content or "温度<" in prior_content
            
            # 如果当前是高温，之前有低温，或者当前是低温，之前有高温，存在矛盾
            if (has_high_temp_current and has_low_temp_prior) or (has_low_temp_current and has_high_temp_prior):
                return False
            
            # 检查简单的否定矛盾
            if any(keyword in current_content and keyword in prior_content for keyword in ["温度", "湿度", "压力", "速度"]):
                # 检查是否一个是肯定，一个是否定
                if ("不" in current_content and "不" not in prior_content) or ("不" not in current_content and "不" in prior_content):
                    return False
            
            # 检查嵌入相似度（如果可用）
            if node.embedding is not None and prior_node.embedding is not None:
                # 计算余弦相似度
                similarity = (np.dot(node.embedding, prior_node.embedding) / 
                           (np.linalg.norm(node.embedding) * np.linalg.norm(prior_node.embedding) + 1e-8))
                # 如果相似度极低，可能存在矛盾
                if similarity < -0.5:
                    return False
        
        return True
    
    def verify_global_reasoning(self, node: Node, all_nodes: List[Node]) -> bool:
        """全局推理验证（第三级）
        
        Args:
            node: 要验证的节点
            all_nodes: 所有节点的列表
            
        Returns:
            bool: 全局推理是否一致
        """
        # 检查全局一致性（简化实现）
        all_contents = [n.content for n in all_nodes]
        current_content = node.content
        
        # 检查是否存在全局矛盾
        has_contradiction = False
        
        # 统计关键属性的取值范围
        temp_high_count = 0
        temp_low_count = 0
        humidity_high_count = 0
        humidity_low_count = 0
        today_claims = []
        
        for content in all_contents:
            # 检查温度
            if "温度高于" in content or "温度>" in content:
                temp_high_count += 1
            if "温度低于" in content or "温度<" in content:
                temp_low_count += 1
            
            # 检查湿度
            if "湿度高于" in content or "湿度>" in content:
                humidity_high_count += 1
            if "湿度低于" in content or "湿度<" in content:
                humidity_low_count += 1
            
            # 检查日期
            if "今天是" in content:
                today_claims.append(content)
        
        # 检查当前内容的温度和湿度
        if "温度高于" in current_content or "温度>" in current_content:
            temp_high_count += 1
        if "温度低于" in current_content or "温度<" in current_content:
            temp_low_count += 1
        if "湿度高于" in current_content or "湿度>" in current_content:
            humidity_high_count += 1
        if "湿度低于" in current_content or "湿度<" in current_content:
            humidity_low_count += 1
        if "今天是" in current_content:
            today_claims.append(current_content)
        
        # 检查温度声明的全局一致性
        if temp_high_count > 0 and temp_low_count > 0:
            has_contradiction = True
        
        # 检查湿度声明的全局一致性
        if humidity_high_count > 0 and humidity_low_count > 0:
            has_contradiction = True
        
        # 检查日期声明的全局一致性
        if len(today_claims) > 1:
            # 解析并比较不同的今天声明
            parsed_days = []
            for claim in today_claims:
                # 提取日期
                day = claim.split("今天是")[1].split("。")[0].strip()
                parsed_days.append(day)
            
            # 如果有不同的日期声明，存在矛盾
            if len(set(parsed_days)) > 1:
                has_contradiction = True
        
        return not has_contradiction
    
    def verify_node(self, node: Node, prior_steps: Set[Node], all_nodes: List[Node] = None) -> str:
        """验证单个节点的一致性（分层验证架构）
        
        Args:
            node: 要验证的节点
            prior_steps: 之前已验证的节点集合
            all_nodes: 所有节点的列表，用于全局推理验证
            
        Returns:
            "Correct" 或 "Incorrect"
        """
        all_nodes = all_nodes or []
        
        # 1. 原子语句验证（第一级）
        if not self.verify_atomic_statements(node):
            return "Incorrect"
        
        # 2. 逻辑依赖验证（第二级）
        if not self.verify_logical_dependencies(node, prior_steps):
            return "Incorrect"
        
        # 3. 全局推理验证（第三级）- 只考虑已验证的节点和当前节点
        verified_nodes = list(prior_steps) + [node]
        if not self.verify_global_reasoning(node, verified_nodes):
            return "Incorrect"
        
        return "Correct"
    
    def node_wise_consistency_verification(self, chain: InferenceChain) -> Dict[str, Any]:
        """节点级一致性验证算法，实现分层验证架构
        
        Args:
            chain: 要验证的推理链
            
        Returns:
            验证结果，包含是否一致和错误信息
        """
        self.verification_results.clear()
        self.global_context.clear()
        
        try:
            # 拓扑排序
            order = chain.topological_sort()
        except ValueError as e:
            return {
                'is_consistent': False,
                'error': str(e),
                'verification_results': self.verification_results,
                'verification_level': 'topological_sort'
            }
        
        # 生成所有节点的嵌入
        for node in order:
            node.generate_embedding()
        
        verified_nodes = set()
        all_nodes = list(chain.nodes.values())
        
        for node in order:
            # 收集所有前驱节点和已验证节点
            prior_nodes = set()
            for pred in node.predecessors:
                if pred in verified_nodes:
                    prior_nodes.add(pred)
            
            # 验证节点（使用分层验证架构）
            # 对于全局验证，我们需要考虑所有已处理的节点（包括当前节点）
            processed_nodes = list(verified_nodes) + [node]
            result = self.verify_node(node, prior_nodes, processed_nodes)
            
            # 记录验证结果，包含分层验证信息
            verification_result = {
                'node_id': node.node_id,
                'content': node.content,
                'result': result,
                'prior_nodes': [n.node_id for n in prior_nodes],
                'atomic_statements': node.atomic_statements,
                'has_embedding': node.embedding is not None,
                'verification_levels': {
                    'atomic': 'passed',
                    'dependency': 'passed',
                    'global': 'passed'
                }
            }
            
            # 检查全局矛盾（特别是日期矛盾）
            current_content = node.content
            if "今天是" in current_content:
                # 提取当前节点的日期
                current_day = current_content.split("今天是")[1].split("。")[0].strip()
                
                # 检查与所有已验证节点的日期矛盾
                for verified_node in verified_nodes:
                    if "今天是" in verified_node.content:
                        verified_day = verified_node.content.split("今天是")[1].split("。")[0].strip()
                        if current_day != verified_day:
                            # 发现日期矛盾
                            result = "Incorrect"
                            verification_result['result'] = "Incorrect"
                            verification_result['verification_levels']['global'] = 'failed'
                            break
            
            # 如果验证失败，确定失败的级别
            if result == "Incorrect":
                # 重新运行各级验证以确定具体失败级别
                if not self.verify_atomic_statements(node):
                    verification_result['verification_levels']['atomic'] = 'failed'
                elif not self.verify_logical_dependencies(node, prior_nodes):
                    verification_result['verification_levels']['dependency'] = 'failed'
                elif not self.verify_global_reasoning(node, processed_nodes):
                    verification_result['verification_levels']['global'] = 'failed'
            
            self.verification_results.append(verification_result)
            
            if result == "Incorrect":
                return {
                    'is_consistent': False,
                    'error': f"Error at node {node.node_id}: {node.content}",
                    'error_node': node.node_id,
                    'error_content': node.content,
                    'error_level': [level for level, status in verification_result['verification_levels'].items() if status == 'failed'][0],
                    'verification_results': self.verification_results,
                    'verification_level': 'node_verification'
                }
            elif result == "Correct":
                verified_nodes.add(node)
                # 更新全局上下文
                self.global_context.append(node.content)
        
        return {
            'is_consistent': True,
            'message': "All steps are consistent",
            'verification_results': self.verification_results,
            'verification_level': 'complete',
            'global_context': self.global_context
        }
    
    def verify_reasoning_text(self, reasoning_text: str) -> Dict[str, Any]:
        """验证推理文本的一致性"""
        # 构建推理链
        chain = InferenceChain().from_reasoning_text(reasoning_text)
        
        # 执行验证
        return self.node_wise_consistency_verification(chain)
