# Logical Consistency Checker
# Combines rule-based, model-based, and inference chain consistency detection
from typing import Dict, List, Any
import logging
from .rule_engine import RuleEngine, EnhancedRuleEngine
from .model_detector import ModelDetector
from .inference_chain import NodeWiseConsistencyVerifier

logger = logging.getLogger(__name__)

class LogicalConsistencyChecker:
    """逻辑一致性检查器，结合规则基、模型基和推理链检测"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # 初始化增强型规则引擎，集成正向链推理
        self.rule_engine = EnhancedRuleEngine()
        
        # 初始化模型检测器，支持配置参数
        model_config = config.get('model_config', {})
        self.model_detector = ModelDetector(model_config)
        
        # 初始化推理链验证器
        self.chain_verifier = NodeWiseConsistencyVerifier()
        
        # 配置参数
        self.use_model_detection = config.get('use_model_detection', True)  # 是否使用模型检测
        self.use_chain_verification = config.get('use_chain_verification', True)  # 是否使用推理链验证
        self.consistency_weight_rule = config.get('consistency_weight_rule', 0.6)  # 规则检测权重
        self.consistency_weight_model = config.get('consistency_weight_model', 0.4)  # 模型检测权重
        self.use_forward_chaining = config.get('use_forward_chaining', True)  # 是否使用正向链推理
        
        logger.info(f"逻辑一致性检查器初始化成功，配置: {config}")
    
    def check_consistency(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查决策的逻辑一致性，返回综合检查结果"""
        # 1. 规则基检测
        rule_result = self.rule_engine.check_consistency(decision, history)
        
        # 2. 增强型规则检测（使用正向链推理）
        if self.use_forward_chaining and hasattr(self.rule_engine, 'check_consistency_with_reasoning'):
            enhanced_rule_result = self.rule_engine.check_consistency_with_reasoning(decision, history)
            # 合并冲突
            rule_result['conflicts'].extend(enhanced_rule_result['conflicts'])
            rule_result['is_consistent'] = enhanced_rule_result['is_consistent']
            rule_result['consistency_score'] = enhanced_rule_result['consistency_score']
            rule_result['forward_chaining_result'] = enhanced_rule_result['forward_chaining_result']
        
        # 3. 模型基检测
        model_result = {}
        if self.use_model_detection:
            model_result = self.model_detector.check_consistency(decision, history)
        
        # 4. 推理链一致性验证
        chain_result = {}
        if self.use_chain_verification and 'reasoning' in decision:
            reasoning_text = decision['reasoning']
            chain_result = self.chain_verifier.verify_reasoning_text(reasoning_text)
        
        # 5. 综合结果
        comprehensive_result = {
            'is_consistent': True,
            'conflicts': rule_result['conflicts'],
            'consistency_score': 1.0,
            'rule_check': rule_result,
            'model_check': model_result,
            'chain_check': chain_result,
            'timestamp': decision.get('timestamp', None),
            'weight_config': {
                'rule_weight': self.consistency_weight_rule,
                'model_weight': self.consistency_weight_model
            },
            'use_forward_chaining': self.use_forward_chaining
        }
        
        # 计算综合一致性分数
        if self.use_model_detection:
            # 加权平均规则检测和模型检测的一致性分数
            base_score = (
                rule_result['consistency_score'] * self.consistency_weight_rule +
                model_result['consistency_score'] * self.consistency_weight_model
            )
        else:
            # 只使用规则检测的一致性分数
            base_score = rule_result['consistency_score']
        
        # 考虑推理链检查结果
        chain_consistent = chain_result.get('is_consistent', True)
        if not chain_consistent:
            # 如果推理链不一致，降低一致性分数
            comprehensive_result['consistency_score'] = base_score * 0.2
        else:
            comprehensive_result['consistency_score'] = base_score
        
        # 确保分数在0-1范围内
        comprehensive_result['consistency_score'] = max(0.0, min(1.0, comprehensive_result['consistency_score']))
        
        # 综合一致性状态
        if not rule_result['is_consistent'] or (self.use_model_detection and not model_result['is_consistent']) or not chain_consistent:
            comprehensive_result['is_consistent'] = False
        
        return comprehensive_result
    
    def get_checker_info(self) -> Dict[str, Any]:
        """获取检查器信息，包括模型配置"""
        return {
            'use_model_detection': self.use_model_detection,
            'weights': {
                'rule': self.consistency_weight_rule,
                'model': self.consistency_weight_model
            },
            'model_info': self.model_detector.get_model_info() if hasattr(self.model_detector, 'get_model_info') else {}
        }
    
    def update_model_threshold(self, new_threshold: float) -> bool:
        """更新模型检测的一致性阈值"""
        if hasattr(self.model_detector, 'update_threshold'):
            return self.model_detector.update_threshold(new_threshold)
        return False
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]], decision: Dict[str, Any]) -> Dict[str, Any]:
        """解决逻辑冲突，返回修正后的决策"""
        resolved_decision = decision.copy()
        
        for conflict in conflicts:
            if conflict['type'] == 'parameter_conflict':
                # 参数冲突解决
                resolved_decision = self._resolve_parameter_conflict(resolved_decision, conflict)
            elif conflict['type'] == 'reasoning_conflict':
                # 推理冲突解决
                resolved_decision = self._resolve_reasoning_conflict(resolved_decision, conflict)
        
        return resolved_decision
    
    def _resolve_parameter_conflict(self, decision: Dict[str, Any], conflict: Dict[str, Any]) -> Dict[str, Any]:
        """解决参数冲突"""
        resolved_decision = decision.copy()
        feature = conflict['feature']
        current_value = conflict['current_value']
        previous_value = conflict['previous_value']
        
        # 简单的参数冲突解决：取中间值
        if feature in resolved_decision['parameters']:
            resolved_value = (current_value + previous_value) / 2
            resolved_decision['parameters'][feature] = resolved_value
            logger.info(f"解决{feature}参数冲突：当前值{current_value}，前值{previous_value}，修正为{resolved_value}")
        
        return resolved_decision
    
    def _resolve_reasoning_conflict(self, decision: Dict[str, Any], conflict: Dict[str, Any]) -> Dict[str, Any]:
        """解决推理冲突"""
        resolved_decision = decision.copy()
        
        # 简单的推理冲突解决：添加冲突说明
        if 'reasoning' in resolved_decision:
            resolved_decision['reasoning'] = f"{resolved_decision['reasoning']}（已解决推理冲突：{conflict['message']}）"
            logger.info(f"解决推理冲突：{conflict['message']}")
        
        return resolved_decision
    
    def add_rule(self, rule):
        """添加新规则"""
        self.rule_engine.add_rule(rule)
    
    def set_model_detection(self, use_model: bool):
        """设置是否使用模型检测"""
        self.use_model_detection = use_model
    
    def set_consistency_weights(self, rule_weight: float, model_weight: float):
        """设置一致性权重"""
        self.consistency_weight_rule = rule_weight
        self.consistency_weight_model = model_weight
