"""
AI模型核心模块
基于JAX和Flax的最先进深度学习模型
"""

import logging

# 条件导入，处理 Flax 与 Python 3.14 的兼容性问题
TransformerModel = None
DiffusionModel = None
VisionModel = None
JEPAModel = None
JEPAEncoder = None
JEPAPredictor = None
JEPADecoder = None
jepa_energy_function = None
JepaDtMpcFusion = None

try:
    from .transformer_model import TransformerModel
except (ImportError, TypeError) as e:
    logging.warning(f"TransformerModel 不可用: {e}")

try:
    from .diffusion_model import DiffusionModel
except (ImportError, TypeError) as e:
    logging.warning(f"DiffusionModel 不可用: {e}")

try:
    from .vision_model import VisionModel
except (ImportError, TypeError) as e:
    logging.warning(f"VisionModel 不可用: {e}")

try:
    from .jepa_model import JEPAModel, JEPAEncoder, JEPAPredictor, JEPADecoder, jepa_energy_function
except (ImportError, TypeError) as e:
    logging.warning(f"JEPA 模型不可用: {e}")

try:
    from .jepa_dtmpc_fusion import JepaDtMpcFusion
except (ImportError, TypeError) as e:
    logging.warning(f"JepaDtMpcFusion 不可用: {e}")

__all__ = ["TransformerModel", "DiffusionModel", "VisionModel", 
           "JEPAModel", "JEPAEncoder", "JEPAPredictor", "JEPADecoder", "jepa_energy_function",
           "JepaDtMpcFusion"]