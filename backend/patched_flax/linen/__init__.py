# Flax linen模块兼容补丁

# 导入原始模块
import sys
import importlib.util

# 导入原始flax.linen模块的所有内容
import flax.linen as _original_linen

# 将原始模块的所有属性导入到当前模块
for attr in dir(_original_linen):
    if not attr.startswith('_'):
        globals()[attr] = getattr(_original_linen, attr)

# 导入我们的补丁版本的模块
from .kw_only_dataclasses import (
    dataclass,
    field,
    KW_ONLY,
    M,
    FieldName,
    Annotation,
    Default
)
from .normalization import WeightNorm

# 替换原始的WeightNorm类
globals()['WeightNorm'] = WeightNorm

# 确保使用我们的补丁版本
__all__ = [
    # 从原始模块导入的所有公开属性
    *[attr for attr in dir(_original_linen) if not attr.startswith('_')],
    # 我们的补丁版本的组件
    'dataclass',
    'field',
    'KW_ONLY',
    'M',
    'FieldName',
    'Annotation',
    'Default',
    'WeightNorm'
]
