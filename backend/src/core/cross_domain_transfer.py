"""Cross-Domain Knowledge Transfer Module

实现跨领域知识迁移机制，支持不同领域知识的整合和迁移
"""

from typing import Dict, Any, List, Optional, Union, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# 导入逻辑一致性检查模块
from .logical_consistency.consistency_checker import LogicalConsistencyChecker


@dataclass
class KnowledgeTransferItem:
    """知识迁移项"""
    id: str
    source_domain: str
    target_domain: str
    knowledge_type: str
    content: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TransferRule:
    """知识迁移规则"""
    id: str
    name: str
    source_domain: str
    target_domain: str
    condition: str
    action: str
    priority: int
    enabled: bool = True
    confidence_threshold: float = 0.7


@dataclass
class KnowledgeMapping:
    """跨领域知识映射"""
    source_concept: str
    target_concept: str
    similarity: float
    mapping_type: str
    context: Optional[Dict[str, Any]] = None
    validation_score: Optional[float] = None


@dataclass
class DomainMapping:
    """领域映射结构"""
    source_domain: str
    target_domain: str
    mappings: List[KnowledgeMapping]
    alignment_strategy: str
    validation_metrics: Dict[str, float]
    last_updated: datetime


class CrossDomainTransferService:
    """跨领域知识迁移服务"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        self.transfer_rules: List[TransferRule] = []
        self.transfer_history: List[KnowledgeTransferItem] = []
        self.domain_alignments: Dict[str, Dict[str, List[str]]] = {}  # 领域对齐映射
        self.knowledge_mappings: Dict[str, DomainMapping] = {}  # 跨领域知识映射存储
        self.concept_embeddings: Dict[str, np.ndarray] = {}  # 概念嵌入存储
        
        # 初始化逻辑一致性检查器
        self.consistency_checker = LogicalConsistencyChecker(config.get('consistency_config', {}))
        
        # 初始化默认迁移规则
        self._init_default_rules()
        
        # 初始化默认领域映射
        self._init_default_domain_mappings()
        
        logger.info("✅ 跨领域知识迁移服务初始化成功")
    
    def _init_default_domain_mappings(self):
        """初始化默认领域映射"""
        # 农业到环境领域的映射
        agri_env_mappings = [
            KnowledgeMapping(
                source_concept="作物生长温度",
                target_concept="环境温度",
                similarity=0.95,
                mapping_type="direct",
                context={"unit": "°C", "range": [10, 35]},
                validation_score=0.92
            ),
            KnowledgeMapping(
                source_concept="土壤湿度",
                target_concept="环境湿度",
                similarity=0.88,
                mapping_type="direct",
                context={"unit": "%", "range": [40, 80]},
                validation_score=0.85
            ),
            KnowledgeMapping(
                source_concept="病虫害",
                target_concept="生物危害",
                similarity=0.82,
                mapping_type="conceptual",
                context={"taxonomy": "agricultural_pests"},
                validation_score=0.78
            )
        ]
        
        # 农业到医疗领域的映射（基于病虫害诊断与医疗诊断的相似性）
        agri_med_mappings = [
            KnowledgeMapping(
                source_concept="作物病害诊断",
                target_concept="疾病诊断",
                similarity=0.75,
                mapping_type="analogical",
                context={"methodology": "symptom-based_diagnosis"},
                validation_score=0.70
            ),
            KnowledgeMapping(
                source_concept="农药治疗",
                target_concept="药物治疗",
                similarity=0.68,
                mapping_type="functional",
                context={"mechanism": "targeted_treatment"},
                validation_score=0.65
            )
        ]
        
        # 创建领域映射
        self.knowledge_mappings["agriculture->environment"] = DomainMapping(
            source_domain="agriculture",
            target_domain="environment",
            mappings=agri_env_mappings,
            alignment_strategy="direct_mapping",
            validation_metrics={"accuracy": 0.92, "coverage": 0.85},
            last_updated=datetime.now()
        )
        
        self.knowledge_mappings["agriculture->healthcare"] = DomainMapping(
            source_domain="agriculture",
            target_domain="healthcare",
            mappings=agri_med_mappings,
            alignment_strategy="analogical_reasoning",
            validation_metrics={"accuracy": 0.75, "coverage": 0.65},
            last_updated=datetime.now()
        )
        
        logger.info("初始化默认领域映射完成")
    
    def _init_default_rules(self):
        """初始化默认迁移规则，包括LagTran框架和无监督域适应"""
        # 农业到其他领域的迁移规则
        default_rules = [
            # LagTran框架：基于语言引导的领域迁移
            TransferRule(
                id="lagtran_rule_1",
                name="LagTran: 农业-环境温度湿度迁移",
                source_domain="agriculture",
                target_domain="environment",
                condition="if knowledge_type == 'environmental_data' and contains('temperature', knowledge) or contains('humidity', knowledge)",
                action="lagtran_transfer('environmental_monitoring')",
                priority=1,
                confidence_threshold=0.85
            ),
            TransferRule(
                id="lagtran_rule_2",
                name="LagTran: 农业-健康作物健康迁移",
                source_domain="agriculture",
                target_domain="healthcare",
                condition="if knowledge_type == 'crop_health'",
                action="lagtran_transfer('plant_pathology')",
                priority=2,
                confidence_threshold=0.8
            ),
            TransferRule(
                id="lagtran_rule_3",
                name="LagTran: 通用-农业知识迁移",
                source_domain="general",
                target_domain="agriculture",
                condition="if knowledge_type == 'general_knowledge'",
                action="lagtran_transfer('agriculture_adaptation')",
                priority=3,
                confidence_threshold=0.75
            ),
            # 无监督域适应规则
            TransferRule(
                id="uda_rule_1",
                name="UDA: 图像域适应",
                source_domain="agriculture",
                target_domain="environment",
                condition="if knowledge_type == 'image'",
                action="uda_transfer('visual_domain_adaptation')",
                priority=4,
                confidence_threshold=0.7
            ),
            TransferRule(
                id="uda_rule_2",
                name="UDA: 文本域适应",
                source_domain="general",
                target_domain="agriculture",
                condition="if knowledge_type == 'text'",
                action="uda_transfer('text_domain_adaptation')",
                priority=5,
                confidence_threshold=0.7
            ),
            # 跨模态迁移规则
            TransferRule(
                id="cross_modal_rule_1",
                name="跨模态: 图像到文本迁移",
                source_domain="vision",
                target_domain="text",
                condition="if knowledge_type == 'image'",
                action="cross_modal_transfer('image_to_text')",
                priority=6,
                confidence_threshold=0.75
            )
        ]
        
        self.transfer_rules.extend(default_rules)
    
    def add_transfer_rule(self, rule: TransferRule):
        """添加迁移规则
        
        Args:
            rule: 迁移规则
        """
        self.transfer_rules.append(rule)
        logger.info(f"添加迁移规则: {rule.name} (ID: {rule.id})")
    
    def remove_transfer_rule(self, rule_id: str):
        """删除迁移规则
        
        Args:
            rule_id: 规则ID
        """
        self.transfer_rules = [rule for rule in self.transfer_rules if rule.id != rule_id]
        logger.info(f"删除迁移规则: {rule_id}")
    
    def get_rules_for_domains(self, source_domain: str, target_domain: str) -> List[TransferRule]:
        """获取特定领域对的迁移规则
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            
        Returns:
            迁移规则列表
        """
        return [
            rule for rule in self.transfer_rules 
            if rule.source_domain == source_domain 
            and rule.target_domain == target_domain
            and rule.enabled
        ]
    
    def transfer_knowledge(self, knowledge: Dict[str, Any], source_domain: str, target_domain: str, context: Optional[Dict[str, Any]] = None) -> KnowledgeTransferItem:
        """迁移知识到目标领域
        
        Args:
            knowledge: 要迁移的知识
            source_domain: 源领域
            target_domain: 目标领域
            context: 上下文信息
            
        Returns:
            知识迁移项
        """
        try:
            knowledge_type = knowledge.get("type", "general")
            
            # 1. 查找适用的迁移规则
            applicable_rules = self.get_rules_for_domains(source_domain, target_domain)
            applicable_rules.sort(key=lambda r: r.priority)
            
            # 2. 查找领域映射
            domain_mapping_key = f"{source_domain}->{target_domain}"
            domain_mapping = self.knowledge_mappings.get(domain_mapping_key)
            
            # 3. 执行迁移
            transferred_content = knowledge.copy()
            confidence = 0.5  # 默认置信度
            transfer_strategy = "default"
            applied_mappings = []
            
            # 4. 应用迁移逻辑
            if applicable_rules and domain_mapping:
                # 使用规则和领域映射结合的迁移策略
                rule = applicable_rules[0]
                transferred_content = self._apply_transfer_rule(knowledge, rule)
                
                # 应用领域映射增强迁移
                enhanced_content, used_mappings = self._apply_domain_mappings(transferred_content, domain_mapping)
                transferred_content = enhanced_content
                applied_mappings = used_mappings
                
                confidence = min(1.0, rule.confidence_threshold + 0.1 + (sum(m.similarity for m in used_mappings) / len(used_mappings) * 0.2 if used_mappings else 0))
                transfer_strategy = "rule+mapping"
            elif applicable_rules:
                # 只使用规则迁移
                rule = applicable_rules[0]
                transferred_content = self._apply_transfer_rule(knowledge, rule)
                confidence = min(1.0, rule.confidence_threshold + 0.1)
                transfer_strategy = "rule_based"
            elif domain_mapping:
                # 只使用领域映射迁移
                enhanced_content, used_mappings = self._apply_domain_mappings(knowledge, domain_mapping)
                transferred_content = enhanced_content
                applied_mappings = used_mappings
                
                if used_mappings:
                    confidence = sum(m.similarity for m in used_mappings) / len(used_mappings)
                transfer_strategy = "mapping_based"
            else:
                # 使用默认迁移策略
                transferred_content = self._default_transfer_strategy(knowledge, source_domain, target_domain)
                confidence = 0.6
                transfer_strategy = "default"
            
            # 5. 检查迁移结果的逻辑一致性
            consistency_result = self.consistency_checker.check_consistency(
                decision=transferred_content,
                history=self.transfer_history[-10:]  # 使用最近10条历史记录
            )
            
            # 6. 根据一致性结果调整置信度
            if not consistency_result['is_consistent']:
                confidence *= 0.7  # 降低置信度
                logger.warning(f"迁移结果存在逻辑不一致: {consistency_result['conflicts']}")
            
            # 7. 确保置信度在合理范围内
            confidence = max(0.1, min(1.0, confidence))
            
            # 8. 创建迁移项
            transfer_item = KnowledgeTransferItem(
                id=str(uuid.uuid4()),
                source_domain=source_domain,
                target_domain=target_domain,
                knowledge_type=knowledge_type,
                content=transferred_content,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "source_knowledge_id": knowledge.get("id"),
                    "transfer_rules_applied": [rule.id for rule in applicable_rules],
                    "transfer_strategy": transfer_strategy,
                    "applied_mappings": [{
                        "source_concept": m.source_concept,
                        "target_concept": m.target_concept,
                        "similarity": m.similarity
                    } for m in applied_mappings],
                    "consistency_check": {
                        "is_consistent": consistency_result['is_consistent'],
                        "consistency_score": consistency_result['consistency_score'],
                        "conflicts": consistency_result['conflicts']
                    },
                    "context": context
                }
            )
            
            # 9. 保存迁移历史
            self.transfer_history.append(transfer_item)
            
            logger.info(f"知识迁移成功: {source_domain} -> {target_domain}, 类型: {knowledge_type}, 置信度: {confidence:.2f}, 策略: {transfer_strategy}")
            return transfer_item
        except Exception as e:
            logger.error(f"知识迁移失败: {str(e)}")
            # 返回失败的迁移项
            return KnowledgeTransferItem(
                id=str(uuid.uuid4()),
                source_domain=source_domain,
                target_domain=target_domain,
                knowledge_type=knowledge.get("type", "general"),
                content=knowledge,
                confidence=0.1,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def _apply_domain_mappings(self, knowledge: Dict[str, Any], domain_mapping: DomainMapping) -> Tuple[Dict[str, Any], List[KnowledgeMapping]]:
        """应用领域映射增强迁移结果
        
        Args:
            knowledge: 迁移后的知识
            domain_mapping: 领域映射
            
        Returns:
            增强后的知识和使用的映射列表
        """
        enhanced_content = knowledge.copy()
        used_mappings = []
        
        # 将源领域概念映射到目标领域概念
        for mapping in domain_mapping.mappings:
            # 查找知识中是否包含源概念
            if mapping.source_concept in str(knowledge):
                used_mappings.append(mapping)
                
                # 更新知识内容
                if isinstance(enhanced_content, dict):
                    # 递归更新字典中的概念
                    enhanced_content = self._replace_concepts_in_dict(enhanced_content, mapping.source_concept, mapping.target_concept)
                elif isinstance(enhanced_content, str):
                    # 直接替换字符串中的概念
                    enhanced_content = enhanced_content.replace(mapping.source_concept, mapping.target_concept)
        
        # 添加领域映射元数据
        if used_mappings:
            enhanced_content["domain_mapping_metadata"] = {
                "source_domain": domain_mapping.source_domain,
                "target_domain": domain_mapping.target_domain,
                "alignment_strategy": domain_mapping.alignment_strategy,
                "used_mappings_count": len(used_mappings),
                "average_similarity": sum(m.similarity for m in used_mappings) / len(used_mappings)
            }
        
        return enhanced_content, used_mappings
    
    def _replace_concepts_in_dict(self, data: Dict[str, Any], old_concept: str, new_concept: str) -> Dict[str, Any]:
        """递归替换字典中的概念
        
        Args:
            data: 要处理的字典
            old_concept: 旧概念
            new_concept: 新概念
            
        Returns:
            替换后的字典
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._replace_concepts_in_dict(value, old_concept, new_concept)
            elif isinstance(value, list):
                result[key] = [
                    self._replace_concepts_in_dict(item, old_concept, new_concept) if isinstance(item, dict) 
                    else item.replace(old_concept, new_concept) if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                result[key] = value.replace(old_concept, new_concept)
            else:
                result[key] = value
        return result
    
    def _apply_transfer_rule(self, knowledge: Dict[str, Any], rule: TransferRule) -> Dict[str, Any]:
        """应用迁移规则
        
        Args:
            knowledge: 知识
            rule: 迁移规则
            
        Returns:
            迁移后的知识
        """
        transferred_content = knowledge.copy()
        
        # 根据规则动作应用不同的迁移策略
        if "lagtran_transfer" in rule.action:
            # 执行LagTran框架迁移
            transferred_content = self._lagtran_transfer(knowledge, rule)
        elif "uda_transfer" in rule.action:
            # 执行无监督域适应迁移
            transferred_content = self._uda_transfer(knowledge, rule)
        elif "cross_modal_transfer" in rule.action:
            # 执行跨模态迁移
            transferred_content = self._cross_modal_transfer(knowledge, rule)
        elif "温度湿度" in rule.name:
            # 温度湿度数据迁移（兼容旧规则）
            transferred_content["domain_specific_info"] = {
                "measurement_type": "environmental_monitoring",
                "unit_system": "metric"
            }
        elif "作物健康" in rule.name:
            # 作物健康数据迁移（兼容旧规则）
            transferred_content["domain_specific_info"] = {
                "analysis_type": "plant_pathology",
                "diagnostic_framework": "ICAR"
            }
        elif "通用知识" in rule.name:
            # 通用知识迁移到农业领域（兼容旧规则）
            transferred_content["domain_specific_info"] = {
                "agriculture_domain": "general_farming",
                "adaptation_strategy": "direct_mapping"
            }
        
        return transferred_content
    
    def _lagtran_transfer(self, knowledge: Dict[str, Any], rule: TransferRule) -> Dict[str, Any]:
        """LagTran框架：基于语言引导的领域迁移
        
        Args:
            knowledge: 知识
            rule: 迁移规则
            
        Returns:
            迁移后的知识
        """
        transferred_content = knowledge.copy()
        
        # 提取目标领域信息
        target_info = rule.action.split("'")[1]
        
        # 添加LagTran迁移元数据
        transferred_content["lagtran_metadata"] = {
            "transfer_type": "language_guided",
            "target_info": target_info,
            "source_rule": rule.id,
            "confidence_threshold": rule.confidence_threshold
        }
        
        # 根据目标领域调整内容
        if target_info == "environmental_monitoring":
            # 环境监测领域调整
            transferred_content["domain_specific_info"] = {
                "measurement_type": "environmental_monitoring",
                "unit_system": "metric",
                "monitoring_scope": "ecosystem"
            }
        elif target_info == "plant_pathology":
            # 植物病理学领域调整
            transferred_content["domain_specific_info"] = {
                "analysis_type": "plant_pathology",
                "diagnostic_framework": "ICAR",
                "disease_classification": "standard"
            }
        elif target_info == "agriculture_adaptation":
            # 农业适应性调整
            transferred_content["domain_specific_info"] = {
                "agriculture_domain": "general_farming",
                "adaptation_strategy": "lagtran_mapping",
                "context_enrichment": "enabled"
            }
        
        return transferred_content
    
    def _uda_transfer(self, knowledge: Dict[str, Any], rule: TransferRule) -> Dict[str, Any]:
        """无监督域适应（UDA）迁移
        
        Args:
            knowledge: 知识
            rule: 迁移规则
            
        Returns:
            迁移后的知识
        """
        transferred_content = knowledge.copy()
        
        # 提取适应类型
        adaptation_type = rule.action.split("'")[1]
        
        # 添加UDA迁移元数据
        transferred_content["uda_metadata"] = {
            "adaptation_type": adaptation_type,
            "source_domain": rule.source_domain,
            "target_domain": rule.target_domain,
            "confidence_threshold": rule.confidence_threshold
        }
        
        # 根据适应类型调整内容
        if adaptation_type == "visual_domain_adaptation":
            # 视觉域适应
            transferred_content["domain_specific_info"] = {
                "adaptation_technique": "unsupervised_da",
                "feature_alignment": "enabled",
                "domain_discriminator": "active"
            }
        elif adaptation_type == "text_domain_adaptation":
            # 文本域适应
            transferred_content["domain_specific_info"] = {
                "adaptation_technique": "unsupervised_text_da",
                "vocabulary_alignment": "enabled",
                "distribution_matching": "active"
            }
        
        return transferred_content
    
    def _cross_modal_transfer(self, knowledge: Dict[str, Any], rule: TransferRule) -> Dict[str, Any]:
        """跨模态迁移
        
        Args:
            knowledge: 知识
            rule: 迁移规则
            
        Returns:
            迁移后的知识
        """
        transferred_content = knowledge.copy()
        
        # 提取迁移类型
        transfer_type = rule.action.split("'")[1]
        
        # 添加跨模态迁移元数据
        transferred_content["cross_modal_metadata"] = {
            "transfer_type": transfer_type,
            "source_modality": rule.source_domain,
            "target_modality": rule.target_domain,
            "confidence_threshold": rule.confidence_threshold
        }
        
        # 根据迁移类型调整内容
        if transfer_type == "image_to_text":
            # 图像到文本迁移
            transferred_content["domain_specific_info"] = {
                "modality_mapping": "image_to_text",
                "captioning_model": "enabled",
                "visual_encoding": "active"
            }
        
        return transferred_content
    
    def _default_transfer_strategy(self, knowledge: Dict[str, Any], source_domain: str, target_domain: str) -> Dict[str, Any]:
        """默认迁移策略
        
        Args:
            knowledge: 知识
            source_domain: 源领域
            target_domain: 目标领域
            
        Returns:
            迁移后的知识
        """
        transferred_content = knowledge.copy()
        
        # 添加领域迁移元数据
        transferred_content["cross_domain_metadata"] = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "transfer_strategy": "default_mapping",
            "transfer_timestamp": datetime.now().isoformat()
        }
        
        # 简单的字段映射
        if source_domain == "general" and target_domain == "agriculture":
            # 通用知识到农业领域的映射
            if "concept" in transferred_content:
                transferred_content["agriculture_concept"] = transferred_content["concept"]
        elif target_domain == "environment":
            # 到环境领域的映射
            if "value" in transferred_content:
                transferred_content["environmental_value"] = transferred_content["value"]
        
        return transferred_content
    
    def batch_transfer_knowledge(self, knowledge_list: List[Dict[str, Any]], source_domain: str, target_domain: str) -> List[KnowledgeTransferItem]:
        """批量迁移知识
        
        Args:
            knowledge_list: 知识列表
            source_domain: 源领域
            target_domain: 目标领域
            
        Returns:
            知识迁移项列表
        """
        results = []
        for knowledge in knowledge_list:
            result = self.transfer_knowledge(knowledge, source_domain, target_domain)
            results.append(result)
        return results
    
    def get_transfer_history(self, limit: int = 100) -> List[KnowledgeTransferItem]:
        """获取迁移历史
        
        Args:
            limit: 返回的历史记录数量
            
        Returns:
            迁移历史列表
        """
        return self.transfer_history[-limit:]
    
    def create_domain_alignment(self, source_domain: str, target_domain: str, mappings: List[str]):
        """创建领域对齐映射
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            mappings: 领域映射关系
        """
        if source_domain not in self.domain_alignments:
            self.domain_alignments[source_domain] = {}
        
        self.domain_alignments[source_domain][target_domain] = mappings
        logger.info(f"创建领域对齐: {source_domain} -> {target_domain}, 映射数量: {len(mappings)}")
    
    def get_domain_alignment(self, source_domain: str, target_domain: str) -> List[str]:
        """获取领域对齐映射
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            
        Returns:
            领域映射关系列表
        """
        if source_domain in self.domain_alignments and target_domain in self.domain_alignments[source_domain]:
            return self.domain_alignments[source_domain][target_domain]
        return []
    
    def adapt_knowledge_to_domain(self, knowledge: Dict[str, Any], target_domain: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """将知识适应到目标领域
        
        Args:
            knowledge: 知识
            target_domain: 目标领域
            context: 上下文信息
            
        Returns:
            适应后的知识
        """
        # 获取知识的源领域（如果有的话）
        source_domain = knowledge.get("domain", "general")
        
        # 执行知识迁移
        transfer_result = self.transfer_knowledge(knowledge, source_domain, target_domain)
        
        return transfer_result.content
    
    def generate_concept_embedding(self, concept: str) -> np.ndarray:
        """生成概念嵌入
        
        Args:
            concept: 概念文本
            
        Returns:
            概念嵌入向量
        """
        if concept in self.concept_embeddings:
            return self.concept_embeddings[concept]
        
        # 简单实现：基于哈希的嵌入，确保相同文本生成相同嵌入
        import hashlib
        
        # 使用MD5哈希生成固定长度的嵌入
        hash_obj = hashlib.md5(concept.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # 转换为numpy数组
        embedding = np.array([b / 255.0 for b in hash_bytes])
        
        # 确保嵌入向量不为全0，避免计算余弦相似度时分母为0
        if np.sum(embedding) == 0:
            embedding += 0.01
        # 归一化到单位向量
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        # 保存嵌入
        self.concept_embeddings[concept] = embedding
        
        return embedding
    
    def calculate_cross_domain_similarity(self, source_concept: str, target_concept: str) -> float:
        """计算跨领域概念相似度
        
        Args:
            source_concept: 源领域概念
            target_concept: 目标领域概念
            
        Returns:
            相似度分数（0-1）
        """
        # 生成概念嵌入
        source_embedding = self.generate_concept_embedding(source_concept)
        target_embedding = self.generate_concept_embedding(target_concept)
        
        # 计算余弦相似度
        similarity = cosine_similarity([source_embedding], [target_embedding])[0][0]
        
        # 确保相似度在0-1范围内
        return max(0.0, min(1.0, similarity))
    
    def create_knowledge_mapping(self, source_concept: str, target_concept: str, source_domain: str, target_domain: str, mapping_type: str, context: Optional[Dict[str, Any]] = None) -> KnowledgeMapping:
        """创建跨领域知识映射
        
        Args:
            source_concept: 源领域概念
            target_concept: 目标领域概念
            source_domain: 源领域
            target_domain: 目标领域
            mapping_type: 映射类型
            context: 上下文信息
            
        Returns:
            创建的知识映射
        """
        # 计算相似度
        similarity = self.calculate_cross_domain_similarity(source_concept, target_concept)
        
        # 创建知识映射
        mapping = KnowledgeMapping(
            source_concept=source_concept,
            target_concept=target_concept,
            similarity=similarity,
            mapping_type=mapping_type,
            context=context,
            validation_score=None
        )
        
        # 验证映射
        validation_score = self.validate_knowledge_mapping(mapping)
        mapping.validation_score = validation_score
        
        # 更新领域映射
        domain_mapping_key = f"{source_domain}->{target_domain}"
        if domain_mapping_key not in self.knowledge_mappings:
            # 创建新的领域映射
            self.knowledge_mappings[domain_mapping_key] = DomainMapping(
                source_domain=source_domain,
                target_domain=target_domain,
                mappings=[mapping],
                alignment_strategy="dynamic",
                validation_metrics={"accuracy": validation_score, "coverage": 0.1},
                last_updated=datetime.now()
            )
        else:
            # 添加到现有领域映射
            domain_mapping = self.knowledge_mappings[domain_mapping_key]
            domain_mapping.mappings.append(mapping)
            domain_mapping.last_updated = datetime.now()
            
            # 更新验证指标
            avg_accuracy = sum(m.validation_score for m in domain_mapping.mappings if m.validation_score is not None) / len([m for m in domain_mapping.mappings if m.validation_score is not None])
            domain_mapping.validation_metrics["accuracy"] = avg_accuracy
            domain_mapping.validation_metrics["coverage"] = len(domain_mapping.mappings) / 100.0  # 简化计算
        
        logger.info(f"创建知识映射: {source_concept} -> {target_concept}, 相似度: {similarity:.3f}, 验证分数: {validation_score:.3f}")
        return mapping
    
    def validate_knowledge_mapping(self, mapping: KnowledgeMapping) -> float:
        """验证知识映射的有效性
        
        Args:
            mapping: 要验证的知识映射
            
        Returns:
            验证分数（0-1）
        """
        # 基础验证：相似度分数
        base_score = mapping.similarity
        
        # 上下文验证
        context_score = 0.8
        if mapping.context:
            # 检查上下文完整性
            if "unit" in mapping.context or "taxonomy" in mapping.context or "methodology" in mapping.context:
                context_score = 1.0
            elif "range" in mapping.context:
                context_score = 0.9
        
        # 映射类型验证
        type_score = 1.0
        if mapping.mapping_type == "analogical":
            # 类比映射需要更严格的验证
            type_score = 0.8
        elif mapping.mapping_type == "conceptual":
            # 概念映射需要中等验证
            type_score = 0.9
        
        # 综合验证分数
        validation_score = (base_score * 0.6) + (context_score * 0.2) + (type_score * 0.2)
        
        return max(0.0, min(1.0, validation_score))
    
    def optimize_domain_mappings(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        """优化领域映射
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            
        Returns:
            优化结果
        """
        domain_mapping_key = f"{source_domain}->{target_domain}"
        if domain_mapping_key not in self.knowledge_mappings:
            return {"status": "error", "message": "领域映射不存在"}
        
        domain_mapping = self.knowledge_mappings[domain_mapping_key]
        
        # 1. 过滤低质量映射
        initial_count = len(domain_mapping.mappings)
        domain_mapping.mappings = [m for m in domain_mapping.mappings if m.validation_score is None or m.validation_score >= 0.7]
        filtered_count = initial_count - len(domain_mapping.mappings)
        
        # 2. 去重映射
        unique_mappings = {}
        for mapping in domain_mapping.mappings:
            key = f"{mapping.source_concept}->{mapping.target_concept}"
            if key not in unique_mappings:
                unique_mappings[key] = mapping
            else:
                # 保留质量更高的映射
                existing = unique_mappings[key]
                if (mapping.validation_score or 0) > (existing.validation_score or 0):
                    unique_mappings[key] = mapping
        
        domain_mapping.mappings = list(unique_mappings.values())
        dedup_count = len(domain_mapping.mappings)
        
        # 3. 更新验证指标
        avg_accuracy = sum(m.validation_score for m in domain_mapping.mappings if m.validation_score is not None) / len([m for m in domain_mapping.mappings if m.validation_score is not None])
        domain_mapping.validation_metrics["accuracy"] = avg_accuracy
        domain_mapping.validation_metrics["coverage"] = len(domain_mapping.mappings) / 100.0  # 简化计算
        domain_mapping.last_updated = datetime.now()
        
        logger.info(f"优化领域映射: {source_domain}->{target_domain}, 过滤: {filtered_count}, 去重: {initial_count - filtered_count - dedup_count}, 剩余: {dedup_count}")
        
        return {
            "status": "success",
            "initial_count": initial_count,
            "filtered_count": filtered_count,
            "dedup_count": initial_count - filtered_count - dedup_count,
            "final_count": dedup_count,
            "avg_accuracy": avg_accuracy
        }
    
    def evaluate_transfer_quality(self, transfer_item: KnowledgeTransferItem) -> float:
        """评估迁移质量
        
        Args:
            transfer_item: 知识迁移项
            
        Returns:
            迁移质量评分（0-1）
        """
        # 1. 基础评分：迁移置信度
        base_score = transfer_item.confidence
        
        # 2. 规则置信度评估
        applicable_rules = [rule for rule in self.transfer_rules if rule.id in transfer_item.metadata.get("transfer_rules_applied", [])]
        rule_score = 1.0
        if applicable_rules:
            rule = applicable_rules[0]
            rule_score = rule.confidence_threshold
        
        # 3. 一致性评估
        consistency_score = transfer_item.metadata.get("consistency_check", {}).get("consistency_score", 1.0)
        
        # 4. 映射质量评估
        mapping_score = 1.0
        applied_mappings = transfer_item.metadata.get("applied_mappings", [])
        if applied_mappings:
            mapping_score = sum(m["similarity"] for m in applied_mappings) / len(applied_mappings)
        
        # 5. 综合评分
        total_score = (base_score * 0.4) + (rule_score * 0.2) + (consistency_score * 0.3) + (mapping_score * 0.1)
        
        # 6. 限制在0-1之间
        return max(0.0, min(1.0, total_score))
    
    def assess_transfer_risk(self, transfer_item: KnowledgeTransferItem) -> Dict[str, Any]:
        """评估迁移风险
        
        Args:
            transfer_item: 知识迁移项
            
        Returns:
            风险评估结果
        """
        # 1. 基于迁移质量评分
        quality_score = self.evaluate_transfer_quality(transfer_item)
        
        # 2. 基于一致性检查
        consistency_check = transfer_item.metadata.get("consistency_check", {})
        has_conflicts = not consistency_check.get("is_consistent", True)
        conflict_count = len(consistency_check.get("conflicts", []))
        
        # 3. 基于映射相似度
        applied_mappings = transfer_item.metadata.get("applied_mappings", [])
        avg_similarity = sum(m["similarity"] for m in applied_mappings) / len(applied_mappings) if applied_mappings else 1.0
        
        # 4. 风险等级评估
        risk_level = "low"
        risk_score = 0.0
        
        if quality_score < 0.5:
            risk_level = "high"
            risk_score = 0.8
        elif quality_score < 0.7:
            risk_level = "medium"
            risk_score = 0.5
        else:
            risk_level = "low"
            risk_score = 0.2
        
        # 5. 调整风险等级
        if has_conflicts:
            risk_score += 0.3
            if conflict_count >= 2:
                risk_level = "high"
            else:
                risk_level = "medium"
        
        if avg_similarity < 0.7:
            risk_score += 0.2
        
        # 确保风险分数在0-1范围内
        risk_score = max(0.0, min(1.0, risk_score))
        
        # 6. 风险缓解建议
        recommendations = []
        if risk_level in ["medium", "high"]:
            if has_conflicts:
                recommendations.append("检查并解决迁移结果中的逻辑冲突")
            if avg_similarity < 0.7:
                recommendations.append("优化跨领域概念映射，提高相似度")
            if quality_score < 0.7:
                recommendations.append("重新评估迁移策略，考虑使用更合适的规则或映射")
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "quality_score": quality_score,
            "has_conflicts": has_conflicts,
            "conflict_count": conflict_count,
            "avg_mapping_similarity": avg_similarity,
            "recommendations": recommendations
        }
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """获取迁移统计信息
        
        Returns:
            统计信息字典
        """
        total_transfers = len(self.transfer_history)
        
        # 按领域对统计
        domain_pair_stats = {}
        for item in self.transfer_history:
            pair = f"{item.source_domain}->{item.target_domain}"
            if pair not in domain_pair_stats:
                domain_pair_stats[pair] = {
                    "count": 0,
                    "average_confidence": 0.0,
                    "total_confidence": 0.0
                }
            domain_pair_stats[pair]["count"] += 1
            domain_pair_stats[pair]["total_confidence"] += item.confidence
        
        # 计算平均置信度
        for pair in domain_pair_stats:
            domain_pair_stats[pair]["average_confidence"] = \
                domain_pair_stats[pair]["total_confidence"] / domain_pair_stats[pair]["count"]
        
        # 按知识类型统计
        type_stats = {}
        for item in self.transfer_history:
            if item.knowledge_type not in type_stats:
                type_stats[item.knowledge_type] = 0
            type_stats[item.knowledge_type] += 1
        
        return {
            "total_transfers": total_transfers,
            "domain_pair_stats": domain_pair_stats,
            "type_stats": type_stats,
            "total_rules": len(self.transfer_rules),
            "enabled_rules": len([r for r in self.transfer_rules if r.enabled])
        }


# 创建全局跨领域知识迁移服务实例
cross_domain_transfer_service = CrossDomainTransferService()
