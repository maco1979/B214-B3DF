#!/usr/bin/env python3
"""
测试夹具配置文件

定义用于测试的通用夹具和设置
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch

# 将backend目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def event_loop():
    """为测试提供事件循环"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def mock_cognitive_architecture():
    """创建模拟的认知架构"""
    from src.core.cognitive_architecture import CognitiveArchitecture
    
    # 创建模拟对象
    mock_arch = Mock(spec=CognitiveArchitecture)
    mock_arch.initialized = True
    mock_arch.get_status.return_value = {
        "status": "active",
        "modules": ["decision_engine", "emotional_processing", "meta_cognitive"]
    }
    mock_arch.process_input.return_value = {
        "text": "test response",
        "emotion_analysis": {
            "dominant_emotions": ["joy"],
            "emotion_scores": {"joy": 0.8, "sadness": 0.2}
        }
    }
    
    return mock_arch

@pytest.fixture(scope="module")
async def mock_ethical_rule_engine():
    """创建模拟的伦理规则引擎"""
    from src.core.ethical_rule_engine import EthicalRuleEngine
    
    # 创建模拟对象
    mock_engine = Mock(spec=EthicalRuleEngine)
    mock_engine.evaluate_ethical_decision.return_value = {
        "ethical_evaluation": {
            "ethical_score": 0.95,
            "rule_violations": []
        },
        "rule_execution_results": [],
        "timestamp": "2023-01-01T00:00:00Z"
    }
    mock_engine.align_decision.return_value = {
        "original_action": "test action",
        "aligned_action": "test action (ethically aligned)",
        "ethical_evaluation": {
            "ethical_score": 0.95
        },
        "rule_execution_results": []
    }
    
    return mock_engine

@pytest.fixture(scope="module")
async def mock_hume_evi_service():
    """创建模拟的Hume AI EVI服务"""
    
    # 创建模拟对象
    mock_service = Mock()
    mock_service.analyze_emotions.return_value = {
        "joy": 0.8,
        "sadness": 0.2,
        "anger": 0.1,
        "fear": 0.05,
        "surprise": 0.15
    }
    mock_service.get_dominant_emotions.return_value = ["joy", "surprise"]
    mock_service.generate_emotional_response.return_value = "That sounds great! I'm happy to help you with that."
    
    return mock_service

@pytest.fixture(scope="module")
async def mock_creativity_service():
    """创建模拟的创造力服务"""
    
    # 创建模拟对象
    mock_service = Mock()
    mock_service.generate_creative_ideas.return_value = [
        {"idea": "使用AI预测病虫害", "score": 0.9},
        {"idea": "自动化灌溉系统", "score": 0.85}
    ]
    mock_service.enhance_creative_idea.return_value = "使用先进的AI模型预测病虫害，并结合自动化系统实时处理"
    mock_service.generate_creative_story.return_value = {
        "story": "未来农场的故事...",
        "genre": "科幻",
        "length": 200
    }
    
    return mock_service

@pytest.fixture(scope="module")
async def mock_performance_monitor():
    """创建模拟的性能监控器"""
    from src.performance.performance_monitor import PerformanceMonitor
    
    # 创建模拟对象
    mock_monitor = Mock(spec=PerformanceMonitor)
    mock_monitor.get_system_performance_report.return_value = {
        "overall_health": "healthy",
        "metrics_summary": {
            "total_metrics": 100,
            "healthy_metrics": 95,
            "warning_metrics": 5,
            "critical_metrics": 0,
            "health_score": 95.0
        },
        "system_resources": {
            "cpu_usage": 40.0,
            "memory_usage": 60.0,
            "memory_available": 8.0,
            "disk_usage": 30.0,
            "disk_free": 200.0
        }
    }
    
    return mock_monitor

@pytest.fixture(scope="module")
async def mock_meta_cognitive_system():
    """创建模拟的元认知系统"""
    from src.core.meta_cognitive_controller import MetaCognitiveSystem
    
    # 创建模拟对象
    mock_system = Mock(spec=MetaCognitiveSystem)
    mock_system.initialized = True
    mock_system.get_status.return_value = {
        "initialized": True,
        "last_update": "2023-01-01T00:00:00Z",
        "self_awareness_level": "SELF_REFLECTIVE",
        "assessment_summary": {
            "overall_score": 0.9,
            "capability_scores": {
                "decision_quality": 0.95,
                "learning_efficiency": 0.85,
                "cross_domain_effectiveness": 0.8,
                "logical_consistency": 0.9
            }
        },
        "capability_assessments": [],
        "cognitive_load": 0.5,
        "resource_usage": {},
        "cross_domain_performance": {}
    }
    
    return mock_system

@pytest.fixture(scope="module")
def mock_environment():
    """创建模拟的环境"""
    
    # 创建模拟环境变量
    with patch.dict(os.environ, {
        "API_KEY": "test_key",
        "ENVIRONMENT": "test",
        "HUME_API_KEY": "test_hume_key",
        "COMVAS_API_KEY": "test_comvas_key"
    }):
        yield

@pytest.fixture(scope="module")
def mock_dependencies():
    """模拟外部依赖"""
    
    # 模拟一些可能缺失的依赖
    mock_modules = {
        'pyswip': type('MockModule', (), {
            'Prolog': type('MockProlog', (), {
                '__init__': lambda self: None,
                'assertz': lambda self, fact: None,
                'query': lambda self, query: []
            })
        }),
        'flax': type('MockModule', (), {
            'linen': type('MockLinen', (), {
                'Module': type('MockModule', (), {})
            })
        })
    }
    
    for module_name, mock_module in mock_modules.items():
        if module_name not in sys.modules:
            sys.modules[module_name] = mock_module
    
    yield
    
    # 清理
    for module_name in mock_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
