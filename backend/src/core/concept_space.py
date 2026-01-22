"""Concept Space Exploration Module

实现概念空间探索，支持探索性和变革性创造力
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import uuid
import random

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """概念数据类"""
    id: str
    name: str
    type: str  # 概念类型
    description: str
    attributes: Dict[str, Any]
    confidence: float
    timestamp: datetime


@dataclass
class ConceptRelation:
    """概念关系数据类"""
    id: str
    source_concept: str  # 源概念ID
    target_concept: str  # 目标概念ID
    relation_type: str  # 关系类型
    weight: float  # 关系权重
    timestamp: datetime


@dataclass
class ConceptSpace:
    """概念空间数据类"""
    id: str
    name: str
    description: str
    concepts: Dict[str, Concept]  # 概念字典
    relations: List[ConceptRelation]  # 关系列表
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExplorationResult:
    """概念空间探索结果"""
    id: str
    concept_space_id: str
    exploration_path: List[Tuple[str, str]]  # (概念ID, 关系类型)
    discovered_concepts: List[Concept]  # 发现的新概念
    creative_connections: List[ConceptRelation]  # 创建的新连接
    timestamp: datetime
    explanation: str  # 探索解释


class ConceptSpaceExplorationSystem:
    """概念空间探索系统"""
    
    def __init__(self):
        self.concept_spaces: Dict[str, ConceptSpace] = {}  # 概念空间字典
        self._initialize_default_concept_space()
        
        logger.info("✅ 概念空间探索系统初始化完成")
    
    def _initialize_default_concept_space(self):
        """初始化默认概念空间"""
        # 创建默认概念空间
        default_space = ConceptSpace(
            id="space_default",
            name="默认概念空间",
            description="包含基本概念和关系的默认空间",
            concepts={},
            relations=[],
            timestamp=datetime.now(),
            metadata={"type": "default", "source": "system"}
        )
        
        # 添加默认概念
        default_concepts = [
            Concept(
                id="concept_agriculture",
                name="农业",
                type="domain",
                description="农业是种植农作物和饲养动物的科学、艺术和业务",
                attributes={"domain": "primary", "importance": 0.9},
                confidence=1.0,
                timestamp=datetime.now()
            ),
            Concept(
                id="concept_technology",
                name="技术",
                type="domain",
                description="技术是应用科学知识解决问题的方法和工具",
                attributes={"domain": "secondary", "importance": 0.9},
                confidence=1.0,
                timestamp=datetime.now()
            ),
            Concept(
                id="concept_intelligence",
                name="智能",
                type="abstract",
                description="智能是学习、理解、推理和适应环境的能力",
                attributes={"type": "abstract", "complexity": 0.95},
                confidence=0.95,
                timestamp=datetime.now()
            ),
            Concept(
                id="concept_environment",
                name="环境",
                type="domain",
                description="环境是生物体周围的自然和人造条件",
                attributes={"domain": "primary", "importance": 0.95},
                confidence=1.0,
                timestamp=datetime.now()
            ),
            Concept(
                id="concept_sustainability",
                name="可持续性",
                type="abstract",
                description="可持续性是满足当前需求而不损害未来世代满足其需求的能力",
                attributes={"type": "abstract", "complexity": 0.85},
                confidence=0.9,
                timestamp=datetime.now()
            )
        ]
        
        for concept in default_concepts:
            default_space.concepts[concept.id] = concept
        
        # 添加默认关系
        default_relations = [
            ConceptRelation(
                id="rel_1",
                source_concept="concept_agriculture",
                target_concept="concept_environment",
                relation_type="interacts_with",
                weight=0.9,
                timestamp=datetime.now()
            ),
            ConceptRelation(
                id="rel_2",
                source_concept="concept_agriculture",
                target_concept="concept_sustainability",
                relation_type="requires",
                weight=0.85,
                timestamp=datetime.now()
            ),
            ConceptRelation(
                id="rel_3",
                source_concept="concept_technology",
                target_concept="concept_agriculture",
                relation_type="enhances",
                weight=0.9,
                timestamp=datetime.now()
            ),
            ConceptRelation(
                id="rel_4",
                source_concept="concept_technology",
                target_concept="concept_intelligence",
                relation_type="enables",
                weight=0.95,
                timestamp=datetime.now()
            ),
            ConceptRelation(
                id="rel_5",
                source_concept="concept_intelligence",
                target_concept="concept_sustainability",
                relation_type="contributes_to",
                weight=0.8,
                timestamp=datetime.now()
            )
        ]
        
        default_space.relations.extend(default_relations)
        
        # 添加到概念空间字典
        self.concept_spaces[default_space.id] = default_space
        
        logger.info(f"默认概念空间初始化完成，包含 {len(default_concepts)} 个概念和 {len(default_relations)} 个关系")
    
    def create_concept_space(self, name: str, description: str) -> str:
        """创建新的概念空间
        
        Args:
            name: 概念空间名称
            description: 概念空间描述
            
        Returns:
            概念空间ID
        """
        space_id = str(uuid.uuid4())
        concept_space = ConceptSpace(
            id=space_id,
            name=name,
            description=description,
            concepts={},
            relations=[],
            timestamp=datetime.now(),
            metadata={"type": "custom", "source": "user"}
        )
        
        self.concept_spaces[space_id] = concept_space
        logger.info(f"创建新概念空间: {name} (ID: {space_id})")
        return space_id
    
    def add_concept(self, concept_space_id: str, name: str, type: str, description: str, 
                   attributes: Dict[str, Any], confidence: float = 0.9) -> str:
        """添加概念到概念空间
        
        Args:
            concept_space_id: 概念空间ID
            name: 概念名称
            type: 概念类型
            description: 概念描述
            attributes: 概念属性
            confidence: 置信度
            
        Returns:
            概念ID
        """
        if concept_space_id not in self.concept_spaces:
            logger.error(f"概念空间不存在: {concept_space_id}")
            return ""
        
        concept_id = str(uuid.uuid4())
        concept = Concept(
            id=concept_id,
            name=name,
            type=type,
            description=description,
            attributes=attributes,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.concept_spaces[concept_space_id].concepts[concept_id] = concept
        logger.debug(f"添加概念到空间 {concept_space_id}: {name}")
        return concept_id
    
    def add_concept_relation(self, concept_space_id: str, source_concept: str, target_concept: str, 
                           relation_type: str, weight: float = 0.8) -> str:
        """添加概念关系
        
        Args:
            concept_space_id: 概念空间ID
            source_concept: 源概念ID
            target_concept: 目标概念ID
            relation_type: 关系类型
            weight: 关系权重
            
        Returns:
            关系ID
        """
        if concept_space_id not in self.concept_spaces:
            logger.error(f"概念空间不存在: {concept_space_id}")
            return ""
        
        concept_space = self.concept_spaces[concept_space_id]
        
        # 检查概念是否存在
        if source_concept not in concept_space.concepts or target_concept not in concept_space.concepts:
            logger.error(f"概念不存在: {source_concept} 或 {target_concept}")
            return ""
        
        relation_id = str(uuid.uuid4())
        relation = ConceptRelation(
            id=relation_id,
            source_concept=source_concept,
            target_concept=target_concept,
            relation_type=relation_type,
            weight=weight,
            timestamp=datetime.now()
        )
        
        concept_space.relations.append(relation)
        logger.debug(f"添加概念关系: {source_concept} {relation_type} {target_concept}")
        return relation_id
    
    def explore_concept_space(self, concept_space_id: str, start_concept: str, 
                            exploration_type: str = "exploratory", 
                            depth: int = 3, max_concepts: int = 10) -> ExplorationResult:
        """探索概念空间
        
        Args:
            concept_space_id: 概念空间ID
            start_concept: 起始概念ID
            exploration_type: 探索类型: exploratory（探索性）或 transformative（变革性）
            depth: 探索深度
            max_concepts: 最大发现概念数
            
        Returns:
            探索结果
        """
        if concept_space_id not in self.concept_spaces:
            logger.error(f"概念空间不存在: {concept_space_id}")
            raise ValueError(f"概念空间不存在: {concept_space_id}")
        
        concept_space = self.concept_spaces[concept_space_id]
        
        if start_concept not in concept_space.concepts:
            logger.error(f"起始概念不存在: {start_concept}")
            raise ValueError(f"起始概念不存在: {start_concept}")
        
        logger.info(f"开始探索概念空间 {concept_space_id}，起始概念: {start_concept}，类型: {exploration_type}")
        
        visited = set()
        exploration_path = []
        discovered_concepts = []
        creative_connections = []
        
        # 探索性创造力：系统搜索概念空间
        if exploration_type == "exploratory":
            self._exploratory_exploration(concept_space, start_concept, depth, visited, 
                                        exploration_path, discovered_concepts, max_concepts)
        # 变革性创造力：修改概念空间的约束
        elif exploration_type == "transformative":
            self._transformative_exploration(concept_space, start_concept, depth, visited, 
                                           exploration_path, discovered_concepts, 
                                           creative_connections, max_concepts)
        else:
            logger.error(f"不支持的探索类型: {exploration_type}")
            raise ValueError(f"不支持的探索类型: {exploration_type}")
        
        # 生成探索结果
        result = ExplorationResult(
            id=str(uuid.uuid4()),
            concept_space_id=concept_space_id,
            exploration_path=exploration_path,
            discovered_concepts=discovered_concepts,
            creative_connections=creative_connections,
            timestamp=datetime.now(),
            explanation=self._generate_exploration_explanation(exploration_type, start_concept, 
                                                              exploration_path, discovered_concepts)
        )
        
        logger.info(f"概念空间探索完成，发现 {len(discovered_concepts)} 个新概念")
        return result
    
    def _exploratory_exploration(self, concept_space: ConceptSpace, current_concept: str, depth: int, 
                               visited: Set[str], path: List[Tuple[str, str]], 
                               discovered: List[Concept], max_concepts: int):
        """探索性探索：系统搜索概念空间
        
        Args:
            concept_space: 概念空间
            current_concept: 当前概念ID
            depth: 剩余探索深度
            visited: 已访问概念集合
            path: 探索路径
            discovered: 发现的概念列表
            max_concepts: 最大发现概念数
        """
        if depth <= 0 or len(discovered) >= max_concepts:
            return
        
        if current_concept in visited:
            return
        
        visited.add(current_concept)
        current_concept_obj = concept_space.concepts[current_concept]
        discovered.append(current_concept_obj)
        
        # 查找当前概念的所有邻居
        neighbors = []
        for relation in concept_space.relations:
            if relation.source_concept == current_concept:
                neighbors.append((relation.target_concept, relation.relation_type))
            elif relation.target_concept == current_concept:
                neighbors.append((relation.source_concept, f"inverse_{relation.relation_type}"))
        
        # 随机选择邻居继续探索
        random.shuffle(neighbors)
        
        for neighbor, relation_type in neighbors:
            if neighbor not in visited and len(discovered) < max_concepts:
                path.append((current_concept, relation_type))
                self._exploratory_exploration(concept_space, neighbor, depth - 1, visited, 
                                            path, discovered, max_concepts)
    
    def _transformative_exploration(self, concept_space: ConceptSpace, current_concept: str, depth: int, 
                                  visited: Set[str], path: List[Tuple[str, str]], 
                                  discovered: List[Concept], creative_connections: List[ConceptRelation], 
                                  max_concepts: int):
        """变革性探索：修改概念空间的约束
        
        Args:
            concept_space: 概念空间
            current_concept: 当前概念ID
            depth: 剩余探索深度
            visited: 已访问概念集合
            path: 探索路径
            discovered: 发现的概念列表
            creative_connections: 创建的新连接
            max_concepts: 最大发现概念数
        """
        if depth <= 0 or len(discovered) >= max_concepts:
            return
        
        if current_concept in visited:
            return
        
        visited.add(current_concept)
        current_concept_obj = concept_space.concepts[current_concept]
        discovered.append(current_concept_obj)
        
        # 1. 首先进行正常探索
        neighbors = []
        for relation in concept_space.relations:
            if relation.source_concept == current_concept:
                neighbors.append((relation.target_concept, relation.relation_type))
            elif relation.target_concept == current_concept:
                neighbors.append((relation.source_concept, f"inverse_{relation.relation_type}"))
        
        random.shuffle(neighbors)
        
        for neighbor, relation_type in neighbors:
            if neighbor not in visited and len(discovered) < max_concepts:
                path.append((current_concept, relation_type))
                self._transformative_exploration(concept_space, neighbor, depth - 1, visited, 
                                               path, discovered, creative_connections, max_concepts)
        
        # 2. 创建创造性连接（变革性创造力）
        if len(creative_connections) < 3:  # 限制创建的连接数量
            # 随机选择一个未直接连接的概念
            all_concepts = list(concept_space.concepts.keys())
            possible_targets = [c for c in all_concepts if c != current_concept and c not in visited]
            
            if possible_targets:
                target_concept = random.choice(possible_targets)
                
                # 检查是否已存在连接
                has_connection = False
                for relation in concept_space.relations:
                    if (relation.source_concept == current_concept and relation.target_concept == target_concept) or \
                       (relation.source_concept == target_concept and relation.target_concept == current_concept):
                        has_connection = True
                        break
                
                if not has_connection:
                    # 创建新的创造性连接
                    new_relation = ConceptRelation(
                        id=str(uuid.uuid4()),
                        source_concept=current_concept,
                        target_concept=target_concept,
                        relation_type="creative_link",
                        weight=0.5,  # 创造性连接权重较低
                        timestamp=datetime.now()
                    )
                    concept_space.relations.append(new_relation)
                    creative_connections.append(new_relation)
                    logger.debug(f"创建创造性连接: {current_concept} -> {target_concept}")
    
    def _generate_exploration_explanation(self, exploration_type: str, start_concept: str, 
                                        path: List[Tuple[str, str]], discovered: List[Concept]) -> str:
        """生成探索解释
        
        Args:
            exploration_type: 探索类型
            start_concept: 起始概念
            path: 探索路径
            discovered: 发现的概念
            
        Returns:
            探索解释
        """
        if exploration_type == "exploratory":
            explanation = f"从概念 {start_concept} 开始进行探索性探索，发现了以下概念："
        else:
            explanation = f"从概念 {start_concept} 开始进行变革性探索，发现了以下概念并创建了新连接："
        
        for concept in discovered:
            explanation += f"\n- {concept.name} ({concept.type}): {concept.description[:50]}..."
        
        return explanation
    
    def get_concept_space(self, concept_space_id: str) -> Optional[ConceptSpace]:
        """获取概念空间
        
        Args:
            concept_space_id: 概念空间ID
            
        Returns:
            概念空间
        """
        return self.concept_spaces.get(concept_space_id)
    
    def get_concept_relations(self, concept_space_id: str, concept_id: str) -> List[ConceptRelation]:
        """获取概念的所有关系
        
        Args:
            concept_space_id: 概念空间ID
            concept_id: 概念ID
            
        Returns:
            概念关系列表
        """
        if concept_space_id not in self.concept_spaces:
            return []
        
        concept_space = self.concept_spaces[concept_space_id]
        relations = []
        
        for relation in concept_space.relations:
            if relation.source_concept == concept_id or relation.target_concept == concept_id:
                relations.append(relation)
        
        return relations
    
    def generate_creative_idea(self, concept_space_id: str, concepts: List[str]) -> Dict[str, Any]:
        """基于概念生成创意想法
        
        Args:
            concept_space_id: 概念空间ID
            concepts: 概念ID列表
            
        Returns:
            创意想法
        """
        if concept_space_id not in self.concept_spaces:
            logger.error(f"概念空间不存在: {concept_space_id}")
            return {}
        
        concept_space = self.concept_spaces[concept_space_id]
        
        # 获取概念对象
        concept_objects = []
        for concept_id in concepts:
            if concept_id in concept_space.concepts:
                concept_objects.append(concept_space.concepts[concept_id])
        
        if len(concept_objects) < 2:
            logger.warning("生成创意需要至少2个概念")
            return {}
        
        # 生成创意想法
        idea = {
            "id": str(uuid.uuid4()),
            "concepts": [concept.name for concept in concept_objects],
            "title": f"创意: {concept_objects[0].name} + {concept_objects[1].name}",
            "description": self._generate_idea_description(concept_objects),
            "confidence": random.uniform(0.7, 0.95),
            "timestamp": datetime.now().isoformat(),
            "concept_space_id": concept_space_id
        }
        
        logger.info(f"生成创意: {idea['title']}")
        return idea
    
    def _generate_idea_description(self, concepts: List[Concept]) -> str:
        """生成创意描述
        
        Args:
            concepts: 概念列表
            
        Returns:
            创意描述
        """
        # 简单的创意描述生成
        concept_names = [concept.name for concept in concepts]
        
        if len(concepts) == 2:
            return f"将{concepts[0].name}和{concepts[1].name}相结合，可以创造出新的解决方案。{concepts[0].name}提供了{concepts[0].type}基础，而{concepts[1].name}则带来了{concepts[1].type}创新，这种组合有可能产生突破性的成果。"
        else:
            return f"通过整合{', '.join(concept_names[:-1])}和{concept_names[-1]}这些不同领域的概念，可以探索出全新的解决方案。每个概念都带来了独特的视角和能力，它们的结合有望创造出具有变革性的创新。"
    
    def get_creativity_statistics(self) -> Dict[str, Any]:
        """获取创造力统计信息
        
        Returns:
            统计信息字典
        """
        total_concepts = 0
        total_relations = 0
        concept_types = {}
        relation_types = {}
        
        for space in self.concept_spaces.values():
            total_concepts += len(space.concepts)
            total_relations += len(space.relations)
            
            # 统计概念类型
            for concept in space.concepts.values():
                concept_types[concept.type] = concept_types.get(concept.type, 0) + 1
            
            # 统计关系类型
            for relation in space.relations:
                relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
        
        return {
            "total_concept_spaces": len(self.concept_spaces),
            "total_concepts": total_concepts,
            "total_relations": total_relations,
            "concept_type_distribution": concept_types,
            "relation_type_distribution": relation_types
        }


# 创建全局概念空间探索系统实例
concept_space_exploration_system = ConceptSpaceExplorationSystem()
