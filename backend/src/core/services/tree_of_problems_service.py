"""Tree of Problems Framework Service

实现问题树（Tree of Problems）框架，用于通用问题解决
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProblemNode:
    """问题节点数据类"""
    id: str
    description: str
    parent_id: Optional[str] = None
    depth: int = 0
    is_solved: bool = False
    solution: Optional[str] = None
    confidence: float = 0.0
    children: List['ProblemNode'] = None
    
    def __post_init__(self):
        self.children = self.children or []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "description": self.description,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "is_solved": self.is_solved,
            "solution": self.solution,
            "confidence": self.confidence,
            "children": [child.to_dict() for child in self.children]
        }

class TreeOfProblemsService:
    """问题树框架服务类"""
    
    def __init__(self):
        logger.info("✅ 问题树框架服务初始化成功")
    
    def build_problem_tree(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建问题树
        
        Args:
            problem: 原始问题描述
            context: 问题上下文信息
            
        Returns:
            包含问题树、根节点和节点映射的字典
        """
        if not problem:
            return {
                "root_id": None,
                "nodes": {},
                "tree": None
            }
        
        try:
            # 生成唯一ID
            import uuid
            
            # 创建根节点
            root_id = str(uuid.uuid4())
            root_node = ProblemNode(
                id=root_id,
                description=problem,
                parent_id=None,
                depth=0
            )
            
            # 节点映射
            nodes = {root_id: root_node}
            
            # 初始分解问题（简单示例，实际应使用更复杂的算法或API）
            self._decompose_problem(root_node, nodes, context)
            
            # 构建树结构
            tree = self._build_tree_structure(root_id, nodes)
            
            return {
                "root_id": root_id,
                "nodes": {node_id: node.to_dict() for node_id, node in nodes.items()},
                "tree": tree
            }
        except Exception as e:
            logger.error(f"构建问题树失败: {e}")
            return {
                "root_id": None,
                "nodes": {},
                "tree": None
            }
    
    def _decompose_problem(self, parent_node: ProblemNode, nodes: Dict[str, ProblemNode], context: Optional[Dict[str, Any]] = None):
        """分解问题为子问题
        
        Args:
            parent_node: 父问题节点
            nodes: 节点映射字典
            context: 问题上下文
        """
        # 简单的问题分解逻辑示例
        # 实际应用中应使用更复杂的算法或调用专业API
        import uuid
        
        # 限制分解深度，避免无限递归
        if parent_node.depth >= 3:
            return
        
        # 示例分解策略：根据问题类型分解
        problem_text = parent_node.description.lower()
        
        # 分解示例1：如果问题包含"如何"，分解为步骤
        if "如何" in problem_text or "怎样" in problem_text:
            steps = ["分析问题", "制定方案", "执行方案", "评估结果"]
            for step in steps:
                child_id = str(uuid.uuid4())
                child_node = ProblemNode(
                    id=child_id,
                    description=f"{step}: {parent_node.description}",
                    parent_id=parent_node.id,
                    depth=parent_node.depth + 1
                )
                parent_node.children.append(child_node)
                nodes[child_id] = child_node
                # 递归分解子问题
                self._decompose_problem(child_node, nodes, context)
        
        # 分解示例2：如果问题包含"为什么"，分解为原因分析
        elif "为什么" in problem_text or "为何" in problem_text:
            causes = ["根本原因", "直接原因", "间接原因", "外部因素"]
            for cause in causes:
                child_id = str(uuid.uuid4())
                child_node = ProblemNode(
                    id=child_id,
                    description=f"{cause}: {parent_node.description}",
                    parent_id=parent_node.id,
                    depth=parent_node.depth + 1
                )
                parent_node.children.append(child_node)
                nodes[child_id] = child_node
                # 递归分解子问题
                self._decompose_problem(child_node, nodes, context)
        
        # 分解示例3：通用分解为三个层次
        else:
            if parent_node.depth == 0:
                # 第一层分解：是什么、为什么、怎么做
                decomposition = [
                    f"问题定义：{parent_node.description}",
                    f"原因分析：{parent_node.description}",
                    f"解决方案：{parent_node.description}"
                ]
            elif parent_node.depth == 1:
                # 第二层分解：具体子问题
                decomposition = [
                    f"子问题1：{parent_node.description}",
                    f"子问题2：{parent_node.description}",
                    f"子问题3：{parent_node.description}"
                ]
            else:
                # 第三层分解：详细步骤
                decomposition = [
                    f"步骤1：{parent_node.description}",
                    f"步骤2：{parent_node.description}",
                    f"步骤3：{parent_node.description}"
                ]
            
            for subproblem in decomposition:
                child_id = str(uuid.uuid4())
                child_node = ProblemNode(
                    id=child_id,
                    description=subproblem,
                    parent_id=parent_node.id,
                    depth=parent_node.depth + 1
                )
                parent_node.children.append(child_node)
                nodes[child_id] = child_node
                # 递归分解子问题
                self._decompose_problem(child_node, nodes, context)
    
    def _build_tree_structure(self, root_id: str, nodes: Dict[str, ProblemNode]) -> Dict[str, Any]:
        """构建树结构字典
        
        Args:
            root_id: 根节点ID
            nodes: 节点映射字典
            
        Returns:
            树结构字典
        """
        def build_subtree(node_id: str) -> Dict[str, Any]:
            node = nodes[node_id]
            subtree = node.to_dict()
            return subtree
        
        if root_id not in nodes:
            return {}
        
        return build_subtree(root_id)
    
    def solve_problem_tree(self, problem_tree: Dict[str, Any]) -> Dict[str, Any]:
        """解决问题树
        
        Args:
            problem_tree: 问题树结构
            
        Returns:
            包含解决方案和解决路径的字典
        """
        if not problem_tree or "tree" not in problem_tree:
            return {
                "solution": "",
                "solving_path": [],
                "confidence": 0.0
            }
        
        try:
            tree = problem_tree["tree"]
            
            # 自下而上解决问题
            solution_parts = []
            solving_path = []
            
            # 递归解决子树
            def solve_subtree(node: Dict[str, Any]) -> Dict[str, Any]:
                # 先解决所有子节点
                for child in node["children"]:
                    solve_subtree(child)
                
                # 标记当前节点为已解决
                node["is_solved"] = True
                
                # 生成当前节点的解决方案
                if not node["children"]:
                    # 叶子节点解决方案
                    node["solution"] = f"解决: {node['description']}"
                    node["confidence"] = 0.8
                else:
                    # 非叶子节点解决方案
                    child_solutions = [child["solution"] for child in node["children"] if child["solution"]]
                    if child_solutions:
                        node["solution"] = f"综合解决方案: {node['description']}\n" + "\n".join(child_solutions)
                        node["confidence"] = min(1.0, sum(child["confidence"] for child in node["children"]) / len(node["children"]))
                    else:
                        node["solution"] = f"解决方案: {node['description']}"
                        node["confidence"] = 0.6
                
                solving_path.append(node["id"])
                return node
            
            # 解决整个树
            solved_tree = solve_subtree(tree)
            
            # 构建最终解决方案
            final_solution = solved_tree["solution"]
            
            return {
                "solution": final_solution,
                "solving_path": solving_path,
                "confidence": solved_tree["confidence"],
                "solved_tree": solved_tree
            }
        except Exception as e:
            logger.error(f"解决问题树失败: {e}")
            return {
                "solution": "",
                "solving_path": [],
                "confidence": 0.0
            }
    
    def generate_problem_solution(self, problem: str, context: Optional[Dict[str, Any]] = None) -> str:
        """生成问题解决方案
        
        Args:
            problem: 问题描述
            context: 上下文信息
            
        Returns:
            解决方案文本
        """
        if not problem:
            return ""
        
        # 构建问题树
        problem_tree = self.build_problem_tree(problem, context)
        
        # 解决问题树
        solution_result = self.solve_problem_tree(problem_tree)
        
        # 格式化解决方案
        if solution_result["solution"]:
            return f"问题解决分析:\n\n{solution_result['solution']}\n\n解决置信度: {solution_result['confidence']:.2f}"
        else:
            return "未能生成有效的解决方案"
    
    def evaluate_solution(self, problem: str, solution: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估解决方案
        
        Args:
            problem: 原始问题
            solution: 解决方案
            context: 上下文信息
            
        Returns:
            评估结果
        """
        if not problem or not solution:
            return {
                "relevance": 0.0,
                "completeness": 0.0,
                "feasibility": 0.0,
                "effectiveness": 0.0,
                "overall_score": 0.0,
                "feedback": ""
            }
        
        try:
            # 简单评估逻辑（实际应使用更复杂的算法）
            relevance = 0.8 if problem.lower() in solution.lower() else 0.5
            completeness = 0.9 if "解决" in solution else 0.6
            feasibility = 0.7 if "步骤" in solution or "方案" in solution else 0.5
            effectiveness = 0.8 if "结果" in solution or "评估" in solution else 0.6
            
            overall_score = (relevance + completeness + feasibility + effectiveness) / 4
            
            feedback = f"解决方案评估: 相关性 {relevance:.2f}, 完整性 {completeness:.2f}, 可行性 {feasibility:.2f}, 有效性 {effectiveness:.2f}, 综合评分 {overall_score:.2f}"
            
            return {
                "relevance": relevance,
                "completeness": completeness,
                "feasibility": feasibility,
                "effectiveness": effectiveness,
                "overall_score": overall_score,
                "feedback": feedback
            }
        except Exception as e:
            logger.error(f"评估解决方案失败: {e}")
            return {
                "relevance": 0.0,
                "completeness": 0.0,
                "feasibility": 0.0,
                "effectiveness": 0.0,
                "overall_score": 0.0,
                "feedback": "评估失败"
            }

# 创建单例实例
tree_of_problems_service = TreeOfProblemsService()
