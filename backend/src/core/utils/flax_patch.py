"""
Flax兼容性补丁，用于解决Python 3.14中Flax的dataclasses问题

该补丁修复了Flax中`variable_filter`字段缺少类型注解的问题
"""

import sys
import importlib
from typing import Optional, Any

# 使用更简单的方式解决问题：直接修改dataclasses模块的_process_class函数
def apply_flax_patch():
    """应用Flax兼容性补丁"""
    if sys.version_info < (3, 14):
        return True

    # 保存原始的_process_class函数
    import dataclasses
    original_process_class = dataclasses._process_class

    def patched_process_class(cls, *args, **kwargs):
        """修补后的_process_class函数，跳过variable_filter字段的类型检查"""
        # 检查是否有variable_filter字段但缺少类型注解
        has_variable_filter = False
        
        # 检查当前类和所有基类的字段
        for c in cls.__mro__:
            if 'variable_filter' in vars(c):
                has_variable_filter = True
                break
        
        if has_variable_filter:
            if not hasattr(cls, '__annotations__'):
                cls.__annotations__ = {}
            if 'variable_filter' not in cls.__annotations__:
                cls.__annotations__['variable_filter'] = Optional[Any]
        
        return original_process_class(cls, *args, **kwargs)

    # 替换dataclasses._process_class
    dataclasses._process_class = patched_process_class

    # 现在可以安全地导入Flax模块了
    try:
        import flax.linen.normalization
    except Exception as e:
        # 恢复原始函数
        dataclasses._process_class = original_process_class
        raise e

    return True
