"""Neural-Symbolic Hybrid System
Integration of Prolog reasoning engine with neural language modeling
Implementing NELLIE-inspired architecture for explainable reasoning
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import sys

logger = logging.getLogger(__name__)


# 尝试导入PySwip进行Prolog集成
try:
    from pyswip import Prolog, registerForeign
    PROLOG_AVAILABLE = True
    logger.info("✅ PySwip库导入成功，Prolog推理可用")
except ImportError as e:
    logger.warning(f"⚠️ PySwip库导入失败，将使用简化的符号推理: {str(e)}")
    PROLOG_AVAILABLE = False

# 导入统一知识表示
from .common_knowledge_base import common_knowledge_base, KnowledgeEntry


@dataclass
class ReasoningResult:
    """推理结果数据结构"""
    conclusion: str
    confidence: float
    reasoning_path: List[str]
    timestamp: datetime
    used_knowledge: List[str]  # 使用的知识ID列表
    is_symbolic: bool  # 是否为符号推理结果
    neural_support: Optional[float] = None  # 神经网络支持度
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class NeuralSymbolicRule:
    """神经符号规则数据结构"""
    rule_id: str
    antecedents: List[str]  # 前提条件
    consequent: str  # 结论
    confidence: float  # 规则置信度
    neural_weight: float  # 神经网络权重
    source: str  # 规则来源
    timestamp: datetime


class NeuralSymbolicSystem:
    """神经符号混合系统"""
    
    def __init__(self):
        self.prolog = None
        self.rules: List[NeuralSymbolicRule] = []
        self.rule_index: Dict[str, NeuralSymbolicRule] = {}
        self.knowledge_base = common_knowledge_base
        self.initialized = False
        
        # 初始化Prolog引擎
        self._init_prolog()
        
        # 加载默认规则
        self._load_default_rules()
        
        logger.info(f"神经符号系统初始化完成，Prolog支持: {PROLOG_AVAILABLE}")
    
    def _init_prolog(self):
        """初始化Prolog引擎"""
        if PROLOG_AVAILABLE:
            try:
                self.prolog = Prolog()
                # 初始化Prolog知识库
                self._init_prolog_knowledge()
                self.initialized = True
                logger.info("✅ Prolog引擎初始化成功")
            except Exception as e:
                logger.error(f"❌ Prolog引擎初始化失败: {str(e)}")
                self.prolog = None
                self.initialized = False
    
    def _init_prolog_knowledge(self):
        """初始化Prolog知识库"""
        if not self.prolog:
            return
        
        # 初始Prolog规则和事实
        initial_facts = [
            "category(weather, natural_phenomenon)",
            "category(rain, weather)",
            "category(heating, physics)",
            "category(sleep, health)",
            "category(exercise, health)",
            "relation(causes, rain, wetness)",
            "relation(causes, heating, temperature_increase)",
            "relation(causes, sleep_deprivation, fatigue)",
            "relation(causes, exercise, physical_strength)",
            "rule(if_rain_then_umbrella, [rain], need_umbrella, 0.99)",
            "rule(if_heating_then_hot, [heating], temperature_increase, 0.98)",
            "rule(if_sleep_deprivation_then_fatigue, [sleep_deprivation], fatigue, 0.95)",
            "rule(if_exercise_then_health, [exercise], good_health, 0.90)"
        ]
        
        # 添加事实到Prolog
        for fact in initial_facts:
            try:
                self.prolog.assertz(fact)
            except Exception as e:
                logger.error(f"添加Prolog事实失败: {fact}, 错误: {str(e)}")
    
    def _load_default_rules(self):
        """加载默认神经符号规则"""
        default_rules = [
            {
                "rule_id": "weather_umbrella_rule",
                "antecedents": ["rain"],
                "consequent": "need_umbrella",
                "confidence": 0.99,
                "neural_weight": 0.95,
                "source": "system"
            },
            {
                "rule_id": "temperature_rule",
                "antecedents": ["heating"],
                "consequent": "temperature_increase",
                "confidence": 0.98,
                "neural_weight": 0.90,
                "source": "system"
            },
            {
                "rule_id": "sleep_fatigue_rule",
                "antecedents": ["sleep_deprivation"],
                "consequent": "fatigue",
                "confidence": 0.95,
                "neural_weight": 0.85,
                "source": "system"
            },
            {
                "rule_id": "exercise_health_rule",
                "antecedents": ["exercise"],
                "consequent": "good_health",
                "confidence": 0.90,
                "neural_weight": 0.80,
                "source": "system"
            }
        ]
        
        for rule_data in default_rules:
            rule = NeuralSymbolicRule(
                rule_id=rule_data["rule_id"],
                antecedents=rule_data["antecedents"],
                consequent=rule_data["consequent"],
                confidence=rule_data["confidence"],
                neural_weight=rule_data["neural_weight"],
                source=rule_data["source"],
                timestamp=datetime.now()
            )
            self.add_rule(rule)
    
    def add_rule(self, rule: NeuralSymbolicRule):
        """添加神经符号规则"""
        self.rules.append(rule)
        self.rule_index[rule.rule_id] = rule
        
        # 如果Prolog可用，将规则添加到Prolog知识库
        if self.prolog:
            try:
                # 将规则转换为Prolog格式
                antecedents_str = ", ".join(rule.antecedents)
                prolog_rule = f"rule({rule.rule_id}, [{antecedents_str}], {rule.consequent}, {rule.confidence})"
                self.prolog.assertz(prolog_rule)
            except Exception as e:
                logger.error(f"添加Prolog规则失败: {rule.rule_id}, 错误: {str(e)}")
    
    def add_prolog_fact(self, fact: str):
        """添加Prolog事实"""
        if self.prolog:
            try:
                self.prolog.assertz(fact)
                logger.debug(f"添加Prolog事实成功: {fact}")
                return True
            except Exception as e:
                logger.error(f"添加Prolog事实失败: {fact}, 错误: {str(e)}")
        return False
    
    def symbolic_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """符号推理，使用Prolog或简化的符号推理"""
        logger.debug(f"开始符号推理，查询: {query}")
        
        results = []
        
        if self.prolog and PROLOG_AVAILABLE:
            # 使用Prolog进行推理
            results.extend(self._prolog_reasoning(query, context))
        else:
            # 使用简化的符号推理
            results.extend(self._simple_symbolic_reasoning(query, context))
        
        return results
    
    def _prolog_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """使用Prolog进行推理"""
        results = []
        
        try:
            # 将自然语言查询转换为Prolog查询
            prolog_query = self._natural_to_prolog(query)
            logger.debug(f"转换后的Prolog查询: {prolog_query}")
            
            # 执行Prolog查询
            prolog_results = list(self.prolog.query(prolog_query))
            
            for result in prolog_results:
                # 解析Prolog结果
                conclusion = self._prolog_to_natural(result)
                
                reasoning_result = ReasoningResult(
                    conclusion=conclusion,
                    confidence=0.95,  # 默认置信度
                    reasoning_path=[f"Prolog查询: {prolog_query}", f"结果: {result}"],
                    timestamp=datetime.now(),
                    used_knowledge=["prolog_rule"],
                    is_symbolic=True
                )
                results.append(reasoning_result)
        except Exception as e:
            logger.error(f"Prolog推理失败: {str(e)}")
            # 回退到简化推理
            results.extend(self._simple_symbolic_reasoning(query, context))
        
        return results
    
    def _simple_symbolic_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """简化的符号推理，当Prolog不可用时使用"""
        results = []
        query_lower = query.lower()
        
        # 规则匹配推理
        matched_rules = []
        for rule in self.rules:
            # 检查规则的前提条件是否在查询中
            if all(antecedent.lower() in query_lower for antecedent in rule.antecedents):
                matched_rules.append(rule)
        
        for rule in matched_rules:
            # 生成推理结果
            conclusion = f"根据规则 {rule.rule_id}，{rule.consequent}"
            reasoning_result = ReasoningResult(
                conclusion=conclusion,
                confidence=rule.confidence,
                reasoning_path=[
                    f"前提: {', '.join(rule.antecedents)}",
                    f"规则: {rule.rule_id}",
                    f"结论: {rule.consequent}"
                ],
                timestamp=datetime.now(),
                used_knowledge=[rule.rule_id],
                is_symbolic=True
            )
            results.append(reasoning_result)
        
        # 如果没有匹配的规则，尝试从知识库中搜索相关知识
        if not results:
            # 搜索相关知识
            related_knowledge = self.knowledge_base.search_knowledge(query)
            if related_knowledge:
                for knowledge in related_knowledge:
                    conclusion = f"知识库中找到: {knowledge.content}"
                    reasoning_result = ReasoningResult(
                        conclusion=conclusion,
                        confidence=knowledge.confidence,
                        reasoning_path=[f"从知识库获取: {knowledge.id}"],
                        timestamp=datetime.now(),
                        used_knowledge=[knowledge.id],
                        is_symbolic=True
                    )
                    results.append(reasoning_result)
        
        return results
    
    def _natural_to_prolog(self, natural_query: str) -> str:
        """将自然语言查询转换为Prolog查询"""
        # 简单的查询转换示例
        query_lower = natural_query.lower()
        
        if "下雨" in query_lower or "rain" in query_lower:
            return "relation(causes, rain, X)"
        elif "加热" in query_lower or "heating" in query_lower:
            return "relation(causes, heating, X)"
        elif "睡眠" in query_lower or "sleep" in query_lower:
            return "relation(causes, sleep_deprivation, X)"
        elif "运动" in query_lower or "exercise" in query_lower:
            return "relation(causes, exercise, X)"
        else:
            # 默认查询
            return f"category(X, Y)"
    
    def _prolog_to_natural(self, prolog_result: Dict[str, Any]) -> str:
        """将Prolog结果转换为自然语言"""
        # 简单的结果转换示例
        result_str = ""
        for key, value in prolog_result.items():
            result_str += f"{key} = {value}, "
        return result_str.rstrip(", ")
    
    def neural_symbolic_integration(self, neural_result: str, symbolic_context: List[str]) -> ReasoningResult:
        """神经符号集成，结合神经网络结果和符号推理"""
        logger.debug(f"开始神经符号集成，神经结果: {neural_result}, 符号上下文: {symbolic_context}")
        
        # 计算符号支持度
        symbolic_support = self._calculate_symbolic_support(neural_result, symbolic_context)
        
        # 计算综合置信度
        # 这里使用简单的加权平均，实际应用中可以使用更复杂的融合机制
        neural_confidence = 0.85  # 假设神经网络结果的置信度
        integrated_confidence = (neural_confidence * 0.7) + (symbolic_support * 0.3)
        
        # 生成推理结果
        result = ReasoningResult(
            conclusion=neural_result,
            confidence=integrated_confidence,
            reasoning_path=[
                f"神经网络结果: {neural_result}",
                f"符号上下文: {', '.join(symbolic_context)}",
                f"符号支持度: {symbolic_support}",
                f"综合置信度: {integrated_confidence}"
            ],
            timestamp=datetime.now(),
            used_knowledge=symbolic_context,
            is_symbolic=False,
            neural_support=neural_confidence
        )
        
        return result
    
    def _calculate_symbolic_support(self, neural_result: str, symbolic_context: List[str]) -> float:
        """计算符号支持度"""
        # 简单的支持度计算示例
        support_score = 0.0
        
        # 检查神经结果是否与符号上下文一致
        neural_lower = neural_result.lower()
        
        for context in symbolic_context:
            context_lower = context.lower()
            # 搜索相关知识
            related_knowledge = self.knowledge_base.search_knowledge(context_lower)
            
            for knowledge in related_knowledge:
                if knowledge.content.lower() in neural_lower or neural_lower in knowledge.content.lower():
                    support_score += knowledge.confidence
        
        # 归一化支持度
        if symbolic_context:
            support_score /= len(symbolic_context)
        
        return min(1.0, max(0.0, support_score))
    
    def explain_reasoning(self, result: ReasoningResult) -> str:
        """生成推理过程的自然语言解释"""
        explanation = f"结论: {result.conclusion}\n"
        explanation += f"置信度: {result.confidence:.2f}\n"
        explanation += "推理路径:\n"
        
        for i, step in enumerate(result.reasoning_path):
            explanation += f"  {i+1}. {step}\n"
        
        explanation += f"使用的知识: {', '.join(result.used_knowledge)}\n"
        explanation += f"推理类型: {'符号推理' if result.is_symbolic else '神经符号集成'}\n"
        explanation += f"推理时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return explanation
    
    def reason_about_knowledge(self, knowledge_id: str) -> List[ReasoningResult]:
        """对特定知识进行推理"""
        knowledge = self.knowledge_base.get_knowledge_by_id(knowledge_id)
        if not knowledge:
            logger.error(f"知识不存在: {knowledge_id}")
            return []
        
        # 获取知识的相关知识
        related_knowledge = self.knowledge_base.get_related_knowledge(knowledge_id)
        related_knowledge_ids = [k.id for k in related_knowledge]
        
        # 生成查询
        query = f"关于 {knowledge.content} 的推理"
        
        # 执行推理
        return self.symbolic_reasoning(query, {"knowledge_id": knowledge_id, "related_knowledge": related_knowledge_ids})
    
    def get_prolog_facts(self) -> List[str]:
        """获取所有Prolog事实"""
        if self.prolog and PROLOG_AVAILABLE:
            try:
                # 查询所有事实
                facts = []
                for fact in self.prolog.query("fact(X)"):
                    facts.append(str(fact))
                return facts
            except Exception as e:
                logger.error(f"获取Prolog事实失败: {str(e)}")
        return []
    
    def get_prolog_rules(self) -> List[str]:
        """获取所有Prolog规则"""
        if self.prolog and PROLOG_AVAILABLE:
            try:
                rules = []
                for rule in self.prolog.query("rule(X, Y, Z, W)"):
                    rules.append(str(rule))
                return rules
            except Exception as e:
                logger.error(f"获取Prolog规则失败: {str(e)}")
        return []


# 创建全局神经符号系统实例
neural_symbolic_system = NeuralSymbolicSystem()