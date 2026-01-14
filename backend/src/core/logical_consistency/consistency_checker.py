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
        """检查决策的逻辑一致性 - 增强版，返回综合检查结果
        增强：
        1. 改进综合评分算法，考虑冲突严重程度
        2. 添加历史上下文分析
        3. 支持更复杂的推理链验证
        4. 实现自适应权重调整
        """
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
        
        # 5. 历史上下文分析
        context_analysis = self._analyze_context_consistency(decision, history)
        
        # 6. 冲突严重程度评估
        conflict_severity = self._assess_conflict_severity(rule_result['conflicts'])
        
        # 7. 综合结果
        comprehensive_result = {
            'is_consistent': True,
            'conflicts': rule_result['conflicts'],
            'conflict_severity': conflict_severity,
            'consistency_score': 1.0,
            'rule_check': rule_result,
            'model_check': model_result,
            'chain_check': chain_result,
            'context_analysis': context_analysis,
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
        
        # 考虑上下文一致性
        context_score = context_analysis.get('context_consistency_score', 1.0)
        base_score *= context_score
        
        # 考虑推理链检查结果
        chain_consistent = chain_result.get('is_consistent', True)
        if not chain_consistent:
            # 根据推理链错误类型调整惩罚
            error_type = chain_result.get('error_type', 'unknown')
            if error_type == 'contradiction':
                base_score *= 0.2  # 矛盾错误惩罚最重
            elif error_type == 'inconsistency':
                base_score *= 0.4  # 不一致错误惩罚
            else:
                base_score *= 0.6  # 其他错误惩罚较轻
        
        # 考虑冲突严重程度
        severity_penalty = self._calculate_severity_penalty(conflict_severity)
        base_score *= (1 - severity_penalty)
        
        # 确保分数在0-1范围内
        comprehensive_result['consistency_score'] = max(0.0, min(1.0, base_score))
        
        # 综合一致性状态
        # 采用更严格的一致性判断：冲突严重程度超过中等或一致性分数低于0.5视为不一致
        if (not rule_result['is_consistent'] or 
            (self.use_model_detection and not model_result['is_consistent']) or 
            not chain_consistent or 
            conflict_severity['level'] >= 2 or 
            comprehensive_result['consistency_score'] < 0.5):
            comprehensive_result['is_consistent'] = False
        
        return comprehensive_result
    
    def _analyze_context_consistency(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析决策与历史上下文的一致性
        
        Args:
            decision: 当前决策
            history: 历史决策记录
            
        Returns:
            上下文一致性分析结果
        """
        if not history:
            return {
                'context_consistency_score': 1.0,
                'related_decisions': 0,
                'context_issues': []
            }
        
        # 分析相关历史决策
        related_decisions = []
        context_issues = []
        action = decision.get('action', '')
        
        for past_decision in history[-20:]:  # 只考虑最近20个决策
            if past_decision.get('action', '') == action:
                related_decisions.append(past_decision)
                
                # 检查参数一致性
                past_params = past_decision.get('parameters', {})
                current_params = decision.get('parameters', {})
                
                for key in set(past_params.keys()) & set(current_params.keys()):
                    past_value = past_params[key]
                    current_value = current_params[key]
                    
                    if isinstance(past_value, (int, float)) and isinstance(current_value, (int, float)):
                        # 数值参数一致性检查
                        relative_diff = abs(past_value - current_value) / max(1.0, abs(past_value))
                        if relative_diff > 1.0:  # 参数变化超过100%，可能存在问题
                            context_issues.append({
                                'type': 'parameter_drift',
                                'feature': key,
                                'past_value': past_value,
                                'current_value': current_value,
                                'relative_diff': relative_diff
                            })
        
        # 计算上下文一致性分数
        if context_issues:
            # 上下文一致性分数 = 1 - 问题数 / 相关参数数
            total_params = sum(len(set(d.get('parameters', {}).keys())) for d in related_decisions)
            if total_params > 0:
                context_score = max(0.5, 1.0 - (len(context_issues) / total_params))
            else:
                context_score = 1.0
        else:
            context_score = 1.0
        
        return {
            'context_consistency_score': context_score,
            'related_decisions': len(related_decisions),
            'context_issues': context_issues
        }
    
    def _assess_conflict_severity(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估冲突的严重程度
        
        Args:
            conflicts: 冲突列表
            
        Returns:
            冲突严重程度评估结果
        """
        if not conflicts:
            return {
                'level': 0,  # 0: 无冲突, 1: 轻微, 2: 中等, 3: 严重
                'level_text': '无冲突',
                'conflict_types': {},
                'total_conflicts': 0
            }
        
        # 按冲突类型统计
        conflict_types = {}
        for conflict in conflicts:
            conflict_type = conflict.get('type', 'unknown')
            conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1
        
        # 计算严重程度
        severity_level = 0
        total_conflicts = len(conflicts)
        
        if total_conflicts >= 3:
            severity_level = 3  # 严重
        elif total_conflicts == 2:
            severity_level = 2  # 中等
        elif total_conflicts == 1:
            severity_level = 1  # 轻微
        
        # 特殊冲突类型提升严重程度
        if 'contradiction' in conflict_types:
            severity_level = min(3, severity_level + 1)
        
        severity_text_map = {
            0: '无冲突',
            1: '轻微',
            2: '中等',
            3: '严重'
        }
        
        return {
            'level': severity_level,
            'level_text': severity_text_map[severity_level],
            'conflict_types': conflict_types,
            'total_conflicts': total_conflicts
        }
    
    def _calculate_severity_penalty(self, conflict_severity: Dict[str, Any]) -> float:
        """计算冲突严重程度带来的分数惩罚
        
        Args:
            conflict_severity: 冲突严重程度评估结果
            
        Returns:
            惩罚因子 (0-0.5)
        """
        severity_level = conflict_severity['level']
        
        # 根据严重程度计算惩罚
        penalty_map = {
            0: 0.0,    # 无冲突，无惩罚
            1: 0.05,   # 轻微冲突，5%惩罚
            2: 0.15,   # 中等冲突，15%惩罚
            3: 0.3     # 严重冲突，30%惩罚
        }
        
        return penalty_map.get(severity_level, 0.0)
    
    def adaptive_weight_adjustment(self, performance_history: List[Dict[str, Any]]):
        """自适应调整规则和模型的权重
        
        Args:
            performance_history: 历史性能记录
        """
        if not performance_history or len(performance_history) < 10:
            return  # 数据不足，不调整
        
        # 分析规则和模型的历史表现
        rule_correct = 0
        model_correct = 0
        total = len(performance_history)
        
        for record in performance_history:
            if 'rule_check' in record and record['rule_check'].get('is_consistent') == record.get('actual_consistent'):
                rule_correct += 1
            if 'model_check' in record and record['model_check'].get('is_consistent') == record.get('actual_consistent'):
                model_correct += 1
        
        # 计算准确率
        rule_accuracy = rule_correct / total if total > 0 else 0.0
        model_accuracy = model_correct / total if total > 0 else 0.0
        
        # 调整权重：准确率越高，权重越大
        if rule_accuracy + model_accuracy > 0:
            self.consistency_weight_rule = rule_accuracy / (rule_accuracy + model_accuracy)
            self.consistency_weight_model = model_accuracy / (rule_accuracy + model_accuracy)
            
            logger.info(f"自适应调整权重：规则准确率={rule_accuracy:.3f}，模型准确率={model_accuracy:.3f}，新权重：规则={self.consistency_weight_rule:.3f}，模型={self.consistency_weight_model:.3f}")
    
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
