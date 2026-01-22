# Rule Engine for Logical Consistency Checking
# Implements rule-based logic consistency detection with forward chaining
from typing import Dict, List, Any, Set
import logging

logger = logging.getLogger(__name__)

class ForwardChainRule:
    """正向链规则基类，支持条件检查和应用"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def check_conditions(self, facts: Set[str]) -> bool:
        """检查规则条件是否满足"""
        raise NotImplementedError("Subclasses must implement check_conditions method")
    
    def apply(self, facts: Set[str]) -> Set[str]:
        """应用规则，返回新推断的事实"""
        raise NotImplementedError("Subclasses must implement apply method")
    
    def is_relevant(self, fact: str) -> bool:
        """检查规则是否与给定事实相关"""
        raise NotImplementedError("Subclasses must implement is_relevant method")

class Rule:
    """逻辑规则基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def check(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查决策是否符合规则，返回冲突列表"""
        raise NotImplementedError("Subclasses must implement check method")

class ParameterConsistencyRule(Rule):
    """参数一致性规则"""
    def __init__(self, parameter_name: str, max_change: float, description: str):
        super().__init__(f"param_{parameter_name}_consistency", description)
        self.parameter_name = parameter_name
        self.max_change = max_change
    
    def check(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts = []
        
        if not history:
            return conflicts
        
        last_decision = history[-1]
        
        if (self.parameter_name in decision['parameters'] and 
            self.parameter_name in last_decision['parameters']):
            current_value = decision['parameters'][self.parameter_name]
            last_value = last_decision['parameters'][self.parameter_name]
            
            if abs(current_value - last_value) > self.max_change:
                conflicts.append({
                    'type': 'parameter_conflict',
                    'feature': self.parameter_name,
                    'current_value': current_value,
                    'previous_value': last_value,
                    'message': f"{self.parameter_name}变化过大，超过最大允许变化值{self.max_change}"
                })
        
        return conflicts

class ReasoningConsistencyRule(Rule):
    """推理依据一致性规则"""
    def __init__(self):
        super().__init__("reasoning_consistency", "检查推理依据是否存在矛盾")
    
    def check(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts = []
        
        if not history:
            return conflicts
        
        last_decision = history[-1]
        current_reasoning = decision['reasoning'].lower()
        last_reasoning = last_decision['reasoning'].lower()
        
        # 检查明显的矛盾关键词
        conflict_pairs = [
            ('高温', '低温'),
            ('下雨', '晴天'),
            ('增加', '减少'),
            ('上升', '下降'),
            ('开启', '关闭')
        ]
        
        for word1, word2 in conflict_pairs:
            if word1 in current_reasoning and word2 in last_reasoning:
                conflicts.append({
                    'type': 'reasoning_conflict',
                    'message': f"推理依据存在矛盾：当前决策基于'{word1}'，而上一个决策基于'{word2}'"
                })
        
        return conflicts

class RuleEngine:
    """规则引擎，管理和应用所有逻辑一致性规则"""
    def __init__(self):
        self.rules = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """初始化默认规则"""
        # 添加温度参数一致性规则（最大变化10度）
        self.add_rule(ParameterConsistencyRule(
            parameter_name='temperature',
            max_change=10.0,
            description='温度参数变化不应超过10度'
        ))
        
        # 添加湿度参数一致性规则（最大变化20%）
        self.add_rule(ParameterConsistencyRule(
            parameter_name='humidity',
            max_change=20.0,
            description='湿度参数变化不应超过20%'
        ))
        
        # 添加推理依据一致性规则
        self.add_rule(ReasoningConsistencyRule())
    
    def add_rule(self, rule: Rule):
        """添加新规则"""
        self.rules.append(rule)
        logger.info(f"添加规则：{rule.name}")
    
    def check_consistency(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查决策的逻辑一致性"""
        consistency_check = {
            'is_consistent': True,
            'conflicts': [],
            'consistency_score': 1.0,
            'rule_check_results': []
        }
        
        # 应用所有规则
        for rule in self.rules:
            rule_conflicts = rule.check(decision, history)
            consistency_check['conflicts'].extend(rule_conflicts)
            consistency_check['rule_check_results'].append({
                'rule_name': rule.name,
                'conflicts_found': len(rule_conflicts)
            })
        
        # 更新一致性状态
        if consistency_check['conflicts']:
            consistency_check['is_consistent'] = False
            # 计算一致性分数：每个冲突降低0.2分，最低0.0
            consistency_check['consistency_score'] = max(0.0, 1.0 - len(consistency_check['conflicts']) * 0.2)
        
        return consistency_check
    
    def get_rules(self) -> List[Rule]:
        """获取所有规则"""
        return self.rules

def forward_chaining(facts: Set[str], rules: List[ForwardChainRule]) -> Set[str]:
    """正向链推理算法
    
    Args:
        facts: 已知事实集合
        rules: 规则列表
        
    Returns:
        推断出的所有事实集合
    """
    inferred_facts = set(facts)
    agenda = list(rules)
    
    logger.debug(f"正向链推理开始，已知事实: {facts}, 规则数量: {len(rules)}")
    
    while agenda:
        rule = agenda.pop(0)
        logger.debug(f"检查规则: {rule.name}")
        
        if rule.check_conditions(inferred_facts):
            logger.debug(f"规则条件满足: {rule.name}")
            new_facts = rule.apply(inferred_facts)
            
            if new_facts:
                for fact in new_facts:
                    if fact not in inferred_facts:
                        logger.debug(f"推断出新事实: {fact}")
                        inferred_facts.add(fact)
                        # 重新检查所有与新事实相关的规则
                        relevant_rules = [r for r in rules if r.is_relevant(fact)]
                        agenda.extend(relevant_rules)
                        logger.debug(f"添加相关规则到议程: {[r.name for r in relevant_rules]}")
    
    logger.debug(f"正向链推理结束，推断出的事实: {inferred_facts}")
    return inferred_facts

class DecisionFactRule(ForwardChainRule):
    """决策事实规则，用于正向链推理"""
    def __init__(self, name: str, description: str, conditions: List[str], conclusions: List[str]):
        super().__init__(name, description)
        self.conditions = conditions
        self.conclusions = conclusions
    
    def check_conditions(self, facts: Set[str]) -> bool:
        """检查所有条件是否都满足"""
        return all(condition in facts for condition in self.conditions)
    
    def apply(self, facts: Set[str]) -> Set[str]:
        """应用规则，返回新的结论"""
        return set(self.conclusions)
    
    def is_relevant(self, fact: str) -> bool:
        """检查规则是否与给定事实相关"""
        return fact in self.conditions

class EnhancedRuleEngine(RuleEngine):
    """增强型规则引擎，集成正向链推理"""
    def __init__(self):
        super().__init__()
        self.forward_rules = []
        self._initialize_forward_rules()
    
    def _initialize_forward_rules(self):
        """初始化正向链规则"""
        # 添加农业决策相关的正向链规则
        
        # 规则1: 如果温度 > 35度且湿度 < 20%，则需要增加灌溉
        self.add_forward_rule(DecisionFactRule(
            name="irrigation_rule",
            description="高温低湿时需要增加灌溉",
            conditions=["temperature_high", "humidity_low"],
            conclusions=["need_increase_irrigation"]
        ))
        
        # 规则2: 如果湿度 > 90%且有降雨预报，则需要减少灌溉
        self.add_forward_rule(DecisionFactRule(
            name="reduce_irrigation_rule",
            description="高湿度且有降雨预报时需要减少灌溉",
            conditions=["humidity_high", "rain_forecast"],
            conclusions=["need_reduce_irrigation"]
        ))
        
        # 规则3: 如果需要增加灌溉且土壤干燥，则执行灌溉操作
        self.add_forward_rule(DecisionFactRule(
            name="execute_irrigation_rule",
            description="需要增加灌溉且土壤干燥时执行灌溉操作",
            conditions=["need_increase_irrigation", "soil_dry"],
            conclusions=["execute_irrigation"]
        ))
    
    def add_forward_rule(self, rule: ForwardChainRule):
        """添加正向链规则"""
        self.forward_rules.append(rule)
        logger.info(f"添加正向链规则：{rule.name}")
    
    def check_consistency_with_reasoning(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用正向链推理增强一致性检查"""
        # 首先执行基本的规则检查
        base_result = super().check_consistency(decision, history)
        
        # 提取决策中的事实
        decision_facts = self._extract_facts_from_decision(decision)
        logger.debug(f"从决策中提取的事实: {decision_facts}")
        
        # 执行正向链推理
        inferred_facts = forward_chaining(decision_facts, self.forward_rules)
        logger.debug(f"正向链推理结果: {inferred_facts}")
        
        # 检查推理结果与决策是否一致
        reasoning_consistent = self._check_reasoning_consistency(decision, inferred_facts)
        
        # 综合结果
        comprehensive_result = {
            **base_result,
            'forward_chaining_result': {
                'inferred_facts': list(inferred_facts),
                'is_consistent': reasoning_consistent
            }
        }
        
        # 更新整体一致性状态
        if not reasoning_consistent:
            comprehensive_result['is_consistent'] = False
            comprehensive_result['consistency_score'] = max(0.0, comprehensive_result['consistency_score'] - 0.3)
            comprehensive_result['conflicts'].append({
                'type': 'forward_chaining_conflict',
                'message': "正向链推理结果与决策不一致"
            })
        
        return comprehensive_result
    
    def _extract_facts_from_decision(self, decision: Dict[str, Any]) -> Set[str]:
        """从决策中提取事实"""
        facts = set()
        
        # 从参数中提取事实
        if 'parameters' in decision:
            params = decision['parameters']
            for key, value in params.items():
                if key == 'temperature' and isinstance(value, (int, float)):
                    if value > 35:
                        facts.add("temperature_high")
                    elif value < 5:
                        facts.add("temperature_low")
                elif key == 'humidity' and isinstance(value, (int, float)):
                    if value > 90:
                        facts.add("humidity_high")
                    elif value < 20:
                        facts.add("humidity_low")
                elif key == 'soil_moisture' and isinstance(value, (int, float)):
                    if value < 30:
                        facts.add("soil_dry")
        
        # 从推理中提取事实
        if 'reasoning' in decision:
            reasoning = decision['reasoning'].lower()
            if '高温' in reasoning or '温度高' in reasoning:
                facts.add("temperature_high")
            if '低温' in reasoning or '温度低' in reasoning:
                facts.add("temperature_low")
            if '高湿度' in reasoning or '湿度高' in reasoning:
                facts.add("humidity_high")
            if '低湿度' in reasoning or '湿度低' in reasoning:
                facts.add("humidity_low")
            if '降雨' in reasoning or '下雨' in reasoning:
                facts.add("rain_forecast")
            if '干燥' in reasoning or '缺水' in reasoning:
                facts.add("soil_dry")
        
        return facts
    
    def _check_reasoning_consistency(self, decision: Dict[str, Any], inferred_facts: Set[str]) -> bool:
        """检查决策与推理结果的一致性"""
        # 检查决策动作是否与推理结果一致
        if 'action' in decision:
            action = decision['action'].lower()
            
            # 如果推理结果是需要增加灌溉，但决策是减少灌溉，则不一致
            if "need_increase_irrigation" in inferred_facts and "reduce" in action:
                return False
            
            # 如果推理结果是需要减少灌溉，但决策是增加灌溉，则不一致
            if "need_reduce_irrigation" in inferred_facts and "increase" in action:
                return False
            
            # 如果推理结果是执行灌溉，但决策不是灌溉相关，则不一致
            if "execute_irrigation" in inferred_facts and "irrigation" not in action:
                return False
        
        return True
