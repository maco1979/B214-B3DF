# Logical Consistency Module
# This module provides logical consistency checking capabilities for AGI systems
# including rule-based, model-based, and inference chain consistency detection

from .consistency_checker import LogicalConsistencyChecker
from .rule_engine import RuleEngine
from .model_detector import ModelDetector, SC_Energy
from .inference_chain import NodeWiseConsistencyVerifier, InferenceChain

__all__ = [
    'LogicalConsistencyChecker',
    'RuleEngine',
    'ModelDetector',
    'SC_Energy',
    'NodeWiseConsistencyVerifier',
    'InferenceChain'
]
