"""
Flax兼容性补丁，用于解决Python 3.14中Flax的dataclasses问题

该补丁修复了Flax中字段缺少类型注解的问题
必须在导入flax之前调用apply_flax_patch()
"""

import sys
from typing import Optional, Any

_patch_applied = False

def apply_flax_patch():
    """应用Flax兼容性补丁 - 必须在导入flax之前调用"""
    global _patch_applied
    
    if _patch_applied:
        return True
        
    # 对于Python 3.14，我们需要修补dataclasses模块
    if sys.version_info >= (3, 14):
        try:
            import dataclasses
            
            # 保存原始的_process_class函数
            original_process_class = dataclasses._process_class
            
            def patched_process_class(cls, init, repr, eq, order, unsafe_hash,
                                      frozen, match_args, kw_only, slots,
                                      weakref_slot):
                """修补后的_process_class函数，自动为缺少类型注解的字段添加注解"""
                # 确保类有__annotations__属性
                if not hasattr(cls, '__annotations__'):
                    cls.__annotations__ = {}
                
                # 检查类中所有属性，为缺少类型注解的字段添加注解
                for name in list(vars(cls).keys()):
                    if name.startswith('_'):
                        continue
                    value = getattr(cls, name, None)
                    # 跳过方法和类方法
                    if callable(value) or isinstance(value, (classmethod, staticmethod, property)):
                        continue
                    # 如果字段没有类型注解，添加一个
                    if name not in cls.__annotations__:
                        cls.__annotations__[name] = Optional[Any]
                
                # 调用原始函数
                return original_process_class(cls, init, repr, eq, order, unsafe_hash,
                                             frozen, match_args, kw_only, slots,
                                             weakref_slot)
            
            # 替换原始函数
            dataclasses._process_class = patched_process_class
            _patch_applied = True
            print("Flax兼容性补丁应用成功")
            return True
            
        except Exception as e:
            print(f"Flax补丁应用失败: {e}")
            return False
    
    return True
