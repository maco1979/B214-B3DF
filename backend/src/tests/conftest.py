"""
集成测试配置文件
包含全局测试配置和fixtures
"""

import pytest
import httpx
import sys
import os
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.main import app
from src.core.logical_consistency import NodeWiseConsistencyVerifier
from src.core.models.curiosity_model import MultiModalCuriosity


@pytest.fixture(scope="module")
def test_client():
    """FastAPI测试客户端"""
    return TestClient(app)


@pytest.fixture(scope="module")
async def async_test_client():
    """异步HTTP客户端"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="module")
def consistency_verifier():
    """逻辑一致性验证器实例"""
    return NodeWiseConsistencyVerifier()


@pytest.fixture(scope="module")
def curiosity_model():
    """好奇心模型实例"""
    state_dim = 128
    action_dim = 32
    modality_dims = {
        'vision': 10000,
        'speech': 128,
        'text': 10000
    }
    return MultiModalCuriosity(state_dim, action_dim, modality_dims)


@pytest.fixture(scope="module")
def mock_environment():
    """模拟环境"""
    env = Mock()
    env.action_space = type('MockActionSpace', (), {
        'n': 5,
        'sample': lambda self: 0
    })()
    
    def simulate(state, action):
        return Mock(), 0.0, False, {}
    
    env.simulate = simulate
    return env


@pytest.fixture(scope="function")
def sample_consistent_reasoning():
    """示例一致推理文本"""
    return "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。"


@pytest.fixture(scope="function")
def sample_inconsistent_reasoning():
    """示例不一致推理文本"""
    return "温度高于30度。温度低于20度。需要增加灌溉。执行灌溉操作。"


@pytest.fixture(scope="function")
def sample_decision_request():
    """示例决策请求"""
    return {
        "action": "irrigate",
        "parameters": {
            "duration": 30,
            "amount": 50
        },
        "reasoning": "温度高于30度。湿度低于40%。需要增加灌溉。执行灌溉操作。",
        "timestamp": "2026-01-13T10:00:00"
    }


@pytest.fixture(scope="function")
def sample_curiosity_state():
    """示例好奇心状态"""
    import torch
    return torch.randn(1, 128)


@pytest.fixture(scope="function")
def sample_curiosity_action():
    """示例好奇心动作"""
    import torch
    return torch.randn(1, 32)


@pytest.fixture(scope="function")
def sample_curiosity_next_state():
    """示例好奇心下一状态"""
    import torch
    return torch.randn(1, 128)
