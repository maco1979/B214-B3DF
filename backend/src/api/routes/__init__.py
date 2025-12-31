"""
API路由模块
包含所有API端点的定义
"""

from .models import router as models_router
from .inference import router as inference_router
from .training import router as training_router
from .system import router as system_router
from .edge import router as edge_router
from .federated import router as federated_router
from .agriculture import router as agriculture_router
from .decision import router as decision_router
# from .blockchain_decision import router as blockchain_decision_router
from .model_training_decision import router as model_training_decision_router
from .resource_decision import router as resource_decision_router
from .decision_monitoring import router as decision_monitoring_router
from .camera import router as camera_router
from .performance import router as performance_router
from .ai_control import router as ai_control_router
from .auth import router as auth_router
from .jepa_dtmpc import router as jepa_dtmpc_router
from .community import router as community_router

# 导入用户路由
try:
    from .user import router as user_router
except ImportError as e:
    print(f"用户路由导入失败: {e}")
    user_router = None

# 导入企业路由
try:
    from .enterprise import router as enterprise_router
except ImportError as e:
    print(f"企业路由导入失败: {e}")
    enterprise_router = None

# 可选导入区块链路由，处理缺少依赖的情况
try:
    from .blockchain import router as blockchain_router
except ImportError as e:
    blockchain_router = None

__all__ = ["models_router", "inference_router", "training_router", "system_router", "edge_router", "blockchain_router", "federated_router", "agriculture_router", "decision_router", "model_training_decision_router", "resource_decision_router", "decision_monitoring_router", "camera_router", "performance_router", "ai_control_router", "auth_router", "jepa_dtmpc_router", "community_router"]

# 动态添加user_router（如果导入成功）
if user_router is not None:
    __all__.append("user_router")

# 动态添加enterprise_router（如果导入成功）
if enterprise_router is not None:
    __all__.append("enterprise_router")