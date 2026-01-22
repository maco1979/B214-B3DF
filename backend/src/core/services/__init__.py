"""
核心服务模块
提供全局共享的服务实例
"""

from .model_manager import ModelManager
from .inference_engine import InferenceEngine
from .camera_controller import CameraController
from .ptz_camera_controller import (
    PTZCameraController, 
    PTZProtocol, 
    PTZAction,
    get_ptz_controller
)

import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上跳出4层目录到达项目根目录 (services -> core -> src -> backend -> root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
models_dir = os.path.join(project_root, "models")

# 创建全局共享的模型管理器实例
model_manager = ModelManager(models_dir)

# 创建全局共享的推理引擎实例
inference_engine = InferenceEngine(model_manager)
