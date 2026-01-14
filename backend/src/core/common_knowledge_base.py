"""
常识知识库模块
提供物理世界、人类社会和上下文理解的基本常识
支持知识的存储、检索和推理
"""
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """常识知识条目"""
    id: str
    type: str  # physical, social, contextual, causal, abstract
    content: str
    confidence: float
    source: str
    relations: List[str]  # 相关知识ID列表
    timestamp: datetime
    category: str
    subcategory: str
    parent_id: Optional[str] = None  # 父知识ID，用于构建层次结构
    children_ids: List[str] = None  # 子知识ID列表
    abstraction_level: int = 0  # 抽象级别，0表示具体，数值越高越抽象
    attributes: Optional[Dict[str, Any]] = None  # 知识属性
    
    def __post_init__(self):
        """初始化默认值"""
        if self.children_ids is None:
            self.children_ids = []
        if self.attributes is None:
            self.attributes = {}


@dataclass
class KnowledgeRelation:
    """知识关系数据结构"""
    source_id: str
    target_id: str
    relation_type: str  # subclass, part_of, causes, implies, associated_with
    confidence: float
    timestamp: datetime
    source: str


class CommonKnowledgeBase:
    """常识知识库"""
    
    def __init__(self):
        self.knowledge_graph: Dict[str, KnowledgeEntry] = {}
        self.relations: List[KnowledgeRelation] = []
        self.relation_index: Dict[str, List[KnowledgeRelation]] = {}  # 按关系类型索引
        self.source_relation_index: Dict[str, List[KnowledgeRelation]] = {}  # 按源知识ID索引
        self.target_relation_index: Dict[str, List[KnowledgeRelation]] = {}  # 按目标知识ID索引
        self.category_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.abstraction_index: Dict[int, List[str]] = {}  # 按抽象级别索引
        self.hierarchy_index: Dict[str, List[str]] = {}  # 按父知识ID索引
        self._initialize_default_knowledge()
        logger.info("常识知识库初始化完成，包含 %d 条默认知识", len(self.knowledge_graph))
    
    def _initialize_default_knowledge(self):
        """初始化默认常识知识"""
        default_knowledge = [
            # 物理常识
            {
                "id": "phys_001",
                "type": "physical",
                "content": "下雨需要带伞",
                "confidence": 0.99,
                "source": "common_sense",
                "relations": [],
                "category": "weather",
                "subcategory": "rain"
            },
            {
                "id": "phys_002",
                "type": "physical",
                "content": "水会从高处流向低处",
                "confidence": 0.99,
                "source": "common_sense",
                "relations": [],
                "category": "physics",
                "subcategory": "gravity"
            },
            {
                "id": "phys_003",
                "type": "physical",
                "content": "加热会使物体温度升高",
                "confidence": 0.98,
                "source": "common_sense",
                "relations": [],
                "category": "physics",
                "subcategory": "thermodynamics"
            },
            {
                "id": "phys_004",
                "type": "physical",
                "content": "冰在高温下会融化",
                "confidence": 0.99,
                "source": "common_sense",
                "relations": [],
                "category": "physics",
                "subcategory": "phase_change"
            },
            
            # 社会常识
            {
                "id": "soc_001",
                "type": "social",
                "content": "红灯停，绿灯行",
                "confidence": 0.99,
                "source": "common_sense",
                "relations": [],
                "category": "traffic",
                "subcategory": "rules"
            },
            {
                "id": "soc_002",
                "type": "social",
                "content": "见面打招呼是礼貌行为",
                "confidence": 0.95,
                "source": "common_sense",
                "relations": [],
                "category": "social",
                "subcategory": "etiquette"
            },
            {
                "id": "soc_003",
                "type": "social",
                "content": "工作时间应该专注于工作",
                "confidence": 0.90,
                "source": "common_sense",
                "relations": [],
                "category": "work",
                "subcategory": "etiquette"
            },
            
            # 上下文常识
            {
                "id": "ctx_001",
                "type": "contextual",
                "content": "在餐厅应该点餐用餐",
                "confidence": 0.98,
                "source": "common_sense",
                "relations": [],
                "category": "context",
                "subcategory": "restaurant"
            },
            {
                "id": "ctx_002",
                "type": "contextual",
                "content": "在图书馆应该保持安静",
                "confidence": 0.99,
                "source": "common_sense",
                "relations": [],
                "category": "context",
                "subcategory": "library"
            },
            
            # 因果关系
            {
                "id": "caus_001",
                "type": "causal",
                "content": "缺乏睡眠会导致疲劳",
                "confidence": 0.95,
                "source": "common_sense",
                "relations": [],
                "category": "health",
                "subcategory": "sleep"
            },
            {
                "id": "caus_002",
                "type": "causal",
                "content": "运动可以增强体质",
                "confidence": 0.90,
                "source": "common_sense",
                "relations": [],
                "category": "health",
                "subcategory": "exercise"
            },
            
            # 审美意识
            {
                "id": "aest_001",
                "type": "contextual",
                "content": "对称性是一种美学价值",
                "confidence": 0.90,
                "source": "common_sense",
                "relations": [],
                "category": "aesthetics",
                "subcategory": "symmetry"
            },
            {
                "id": "aest_002",
                "type": "contextual",
                "content": "和谐的色彩搭配具有更高的美学价值",
                "confidence": 0.85,
                "source": "common_sense",
                "relations": [],
                "category": "aesthetics",
                "subcategory": "color_harmony"
            },
            {
                "id": "aest_003",
                "type": "contextual",
                "content": "自然景观具有独特的美学价值",
                "confidence": 0.95,
                "source": "common_sense",
                "relations": [],
                "category": "aesthetics",
                "subcategory": "natural_beauty"
            },
            {
                "id": "aest_004",
                "type": "contextual",
                "content": "中等复杂度的事物更具美学价值",
                "confidence": 0.80,
                "source": "common_sense",
                "relations": [],
                "category": "aesthetics",
                "subcategory": "complexity"
            },
            {
                "id": "aest_005",
                "type": "contextual",
                "content": "艺术作品具有超越实用性的美学价值",
                "confidence": 0.90,
                "source": "common_sense",
                "relations": [],
                "category": "aesthetics",
                "subcategory": "art"
            },
            
            # 人类价值和共情
            {
                "id": "hum_001",
                "type": "social",
                "content": "人类生命是脆弱的，需要关怀和保护",
                "confidence": 0.98,
                "source": "common_sense",
                "relations": [],
                "category": "human_values",
                "subcategory": "life"
            },
            {
                "id": "hum_002",
                "type": "social",
                "content": "痛苦是生命的一部分，需要理解和同情",
                "confidence": 0.95,
                "source": "common_sense",
                "relations": [],
                "category": "human_values",
                "subcategory": "suffering"
            },
            {
                "id": "hum_003",
                "type": "social",
                "content": "共情是理解和回应他人情感的能力",
                "confidence": 0.90,
                "source": "common_sense",
                "relations": [],
                "category": "human_values",
                "subcategory": "empathy"
            },
            {
                "id": "hum_004",
                "type": "social",
                "content": "尊重他人的尊严和权利是基本的道德原则",
                "confidence": 0.97,
                "source": "common_sense",
                "relations": [],
                "category": "human_values",
                "subcategory": "dignity"
            },
            {
                "id": "hum_005",
                "type": "contextual",
                "content": "美的体验能够丰富人类的精神生活",
                "confidence": 0.88,
                "source": "common_sense",
                "relations": [],
                "category": "human_values",
                "subcategory": "beauty_experience"
            },
            {
                "id": "hum_006",
                "type": "contextual",
                "content": "无私的行为和利他主义具有道德价值",
                "confidence": 0.85,
                "source": "common_sense",
                "relations": [],
                "category": "human_values",
                "subcategory": "altruism"
            },
        ]
        
        # 添加到知识库
        for knowledge in default_knowledge:
            self.add_knowledge(**knowledge)
    
    def add_knowledge(self, **kwargs):
        """添加知识到知识库"""
        knowledge_entry = KnowledgeEntry(
            id=kwargs["id"],
            type=kwargs["type"],
            content=kwargs["content"],
            confidence=kwargs["confidence"],
            source=kwargs["source"],
            relations=kwargs["relations"],
            timestamp=datetime.now(),
            category=kwargs["category"],
            subcategory=kwargs["subcategory"],
            parent_id=kwargs.get("parent_id"),
            children_ids=kwargs.get("children_ids"),
            abstraction_level=kwargs.get("abstraction_level", 0),
            attributes=kwargs.get("attributes")
        )
        
        # 添加到知识图谱
        self.knowledge_graph[knowledge_entry.id] = knowledge_entry
        
        # 更新分类索引
        if knowledge_entry.category not in self.category_index:
            self.category_index[knowledge_entry.category] = []
        self.category_index[knowledge_entry.category].append(knowledge_entry.id)
        
        # 更新类型索引
        if knowledge_entry.type not in self.type_index:
            self.type_index[knowledge_entry.type] = []
        self.type_index[knowledge_entry.type].append(knowledge_entry.id)
        
        # 更新抽象级别索引
        if knowledge_entry.abstraction_level not in self.abstraction_index:
            self.abstraction_index[knowledge_entry.abstraction_level] = []
        self.abstraction_index[knowledge_entry.abstraction_level].append(knowledge_entry.id)
        
        # 更新层次结构
        if knowledge_entry.parent_id:
            # 添加到父知识的子列表
            parent = self.knowledge_graph.get(knowledge_entry.parent_id)
            if parent:
                if knowledge_entry.id not in parent.children_ids:
                    parent.children_ids.append(knowledge_entry.id)
            
            # 更新层次索引
            if knowledge_entry.parent_id not in self.hierarchy_index:
                self.hierarchy_index[knowledge_entry.parent_id] = []
            self.hierarchy_index[knowledge_entry.parent_id].append(knowledge_entry.id)
        
        logger.debug(f"添加知识成功: {knowledge_entry.id} - {knowledge_entry.content}")
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, 
                    confidence: float = 0.9, source: str = "system"):
        """添加知识关系"""
        # 检查源和目标知识是否存在
        if source_id not in self.knowledge_graph or target_id not in self.knowledge_graph:
            logger.error(f"添加关系失败: 源知识 {source_id} 或目标知识 {target_id} 不存在")
            return False
        
        # 创建关系
        relation = KnowledgeRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            timestamp=datetime.now(),
            source=source
        )
        
        # 添加到关系列表
        self.relations.append(relation)
        
        # 更新关系类型索引
        if relation_type not in self.relation_index:
            self.relation_index[relation_type] = []
        self.relation_index[relation_type].append(relation)
        
        # 更新源知识关系索引
        if source_id not in self.source_relation_index:
            self.source_relation_index[source_id] = []
        self.source_relation_index[source_id].append(relation)
        
        # 更新目标知识关系索引
        if target_id not in self.target_relation_index:
            self.target_relation_index[target_id] = []
        self.target_relation_index[target_id].append(relation)
        
        # 如果是subclass关系，更新层次结构
        if relation_type == "subclass":
            source_knowledge = self.knowledge_graph[source_id]
            target_knowledge = self.knowledge_graph[target_id]
            
            # 设置源知识的父ID为目标知识
            source_knowledge.parent_id = target_id
            
            # 更新目标知识的子列表
            if source_id not in target_knowledge.children_ids:
                target_knowledge.children_ids.append(source_id)
            
            # 更新层次索引
            if target_id not in self.hierarchy_index:
                self.hierarchy_index[target_id] = []
            if source_id not in self.hierarchy_index[target_id]:
                self.hierarchy_index[target_id].append(source_id)
        
        logger.debug(f"添加关系成功: {source_id} {relation_type} {target_id}")
        return True
    
    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        """根据ID获取知识"""
        return self.knowledge_graph.get(knowledge_id)
    
    def get_knowledge_by_type(self, knowledge_type: str) -> List[KnowledgeEntry]:
        """根据类型获取知识"""
        knowledge_ids = self.type_index.get(knowledge_type, [])
        return [self.knowledge_graph[id] for id in knowledge_ids if id in self.knowledge_graph]
    
    def get_knowledge_by_category(self, category: str) -> List[KnowledgeEntry]:
        """根据分类获取知识"""
        knowledge_ids = self.category_index.get(category, [])
        return [self.knowledge_graph[id] for id in knowledge_ids if id in self.knowledge_graph]
    
    def search_knowledge(self, query: str) -> List[KnowledgeEntry]:
        """搜索知识"""
        results = []
        query = query.lower()
        
        for knowledge in self.knowledge_graph.values():
            if query in knowledge.content.lower():
                results.append(knowledge)
        
        return results
    
    def get_related_knowledge(self, knowledge_id: str) -> List[KnowledgeEntry]:
        """获取相关知识"""
        knowledge = self.get_knowledge_by_id(knowledge_id)
        if not knowledge:
            return []
        
        related = []
        
        # 获取传统关系中的知识
        for relation_id in knowledge.relations:
            related_knowledge = self.get_knowledge_by_id(relation_id)
            if related_knowledge:
                related.append(related_knowledge)
        
        # 获取通过关系连接的知识
        if knowledge_id in self.source_relation_index:
            for relation in self.source_relation_index[knowledge_id]:
                if relation.target_id in self.knowledge_graph:
                    related.append(self.knowledge_graph[relation.target_id])
        
        if knowledge_id in self.target_relation_index:
            for relation in self.target_relation_index[knowledge_id]:
                if relation.source_id in self.knowledge_graph:
                    related.append(self.knowledge_graph[relation.source_id])
        
        return related
    
    def get_hierarchy_above(self, knowledge_id: str, max_levels: int = -1) -> List[KnowledgeEntry]:
        """获取知识层次结构中的上级知识"""
        hierarchy = []
        current = self.get_knowledge_by_id(knowledge_id)
        if not current:
            return hierarchy
        
        level = 0
        while current.parent_id and (max_levels == -1 or level < max_levels):
            parent = self.get_knowledge_by_id(current.parent_id)
            if parent:
                hierarchy.append(parent)
                current = parent
                level += 1
            else:
                break
        
        return hierarchy
    
    def get_hierarchy_below(self, knowledge_id: str, max_levels: int = -1) -> List[KnowledgeEntry]:
        """获取知识层次结构中的下级知识"""
        hierarchy = []
        
        def _traverse_children(node_id: str, current_level: int = 0):
            """递归遍历子节点"""
            if max_levels != -1 and current_level >= max_levels:
                return
            
            node = self.get_knowledge_by_id(node_id)
            if not node:
                return
            
            for child_id in node.children_ids:
                child = self.get_knowledge_by_id(child_id)
                if child:
                    hierarchy.append(child)
                    _traverse_children(child_id, current_level + 1)
        
        _traverse_children(knowledge_id)
        return hierarchy
    
    def get_all_hierarchy(self, knowledge_id: str) -> Dict[str, Any]:
        """获取完整的知识层次结构"""
        knowledge = self.get_knowledge_by_id(knowledge_id)
        if not knowledge:
            return {}
        
        return {
            "knowledge": knowledge,
            "ancestors": self.get_hierarchy_above(knowledge_id),
            "descendants": self.get_hierarchy_below(knowledge_id)
        }
    
    def reason_deductive(self, premise: str, conclusion: str) -> bool:
        """演绎推理: 如果前提为真，则结论必然为真"""
        # 简单的演绎推理示例
        # 查找前提和结论之间的因果关系
        premise_knowledge = self.search_knowledge(premise)
        conclusion_knowledge = self.search_knowledge(conclusion)
        
        if not premise_knowledge or not conclusion_knowledge:
            return False
        
        # 检查是否存在因果关系
        for relation in self.relations:
            if (relation.source_id in [pk.id for pk in premise_knowledge] and 
                relation.target_id in [ck.id for ck in conclusion_knowledge] and
                relation.relation_type == "causes"):
                return True
        
        return False
    
    def reason_inductive(self, specific_cases: List[str], general_rule: str) -> float:
        """归纳推理: 从具体案例中归纳出一般规则"""
        # 计算支持度
        supporting_cases = 0
        for case in specific_cases:
            case_knowledge = self.search_knowledge(case)
            if case_knowledge:
                supporting_cases += 1
        
        confidence = supporting_cases / len(specific_cases) if specific_cases else 0.0
        return confidence
    
    def reason_abductive(self, observation: str, possible_explanations: List[str]) -> List[Dict[str, Any]]:
        """溯因推理: 从观察到的结果推测最可能的原因"""
        explanations = []
        
        for explanation in possible_explanations:
            # 搜索与观察和解释相关的知识
            observation_knowledge = self.search_knowledge(observation)
            explanation_knowledge = self.search_knowledge(explanation)
            
            if observation_knowledge and explanation_knowledge:
                # 查找因果关系
                for relation in self.relations:
                    if (relation.relation_type == "causes" and 
                        relation.source_id in [ek.id for ek in explanation_knowledge] and
                        relation.target_id in [ok.id for ok in observation_knowledge]):
                        explanations.append({
                            "explanation": explanation,
                            "confidence": relation.confidence,
                            "evidence": relation
                        })
        
        # 按置信度排序
        return sorted(explanations, key=lambda x: x["confidence"], reverse=True)
    
    def get_knowledge_by_abstraction(self, level: int) -> List[KnowledgeEntry]:
        """根据抽象级别获取知识"""
        knowledge_ids = self.abstraction_index.get(level, [])
        return [self.knowledge_graph[id] for id in knowledge_ids if id in self.knowledge_graph]
    
    def get_relations_by_type(self, relation_type: str) -> List[KnowledgeRelation]:
        """根据关系类型获取关系"""
        return self.relation_index.get(relation_type, [])
    
    def reason_about_context(self, context: Dict[str, Any]) -> List[KnowledgeEntry]:
        """根据上下文进行常识推理"""
        results = []
        
        # 上下文匹配推理
        if "weather" in context:
            weather_context = context["weather"].lower()
            if "rain" in weather_context:
                # 下雨相关知识
                results.extend(self.get_knowledge_by_category("weather"))
                # 添加因果关系推理
                rain_knowledge = self.search_knowledge("下雨")
                if rain_knowledge:
                    results.extend(self.get_related_knowledge(rain_knowledge[0].id))
        
        if "location" in context:
            location = context["location"].lower()
            if location == "restaurant":
                results.extend(self.search_knowledge("餐厅"))
            elif location == "library":
                results.extend(self.search_knowledge("图书馆"))
        
        return results
    
    def infer_causality(self, event: str) -> List[KnowledgeEntry]:
        """根据事件进行因果推理"""
        # 简单的因果推理示例
        event = event.lower()
        
        if "下雨" in event:
            return self.search_knowledge("下雨") + self.get_related_knowledge("phys_001")
        elif "加热" in event:
            return self.search_knowledge("加热") + self.get_related_knowledge("phys_003")
        elif "睡眠" in event:
            sleep_knowledge = self.get_knowledge_by_id("caus_001")
            return [sleep_knowledge] if sleep_knowledge else []
        elif "运动" in event:
            exercise_knowledge = self.get_knowledge_by_id("caus_002")
            return [exercise_knowledge] if exercise_knowledge else []
        
        return []


# 创建全局常识知识库实例
common_knowledge_base = CommonKnowledgeBase()
