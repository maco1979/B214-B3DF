"""
Flax兼容性补丁，用于解决Python 3.14中Flax的dataclasses问题

该补丁修复了Flax中字段缺少类型注解的问题
必须在导入flax之前调用apply_flax_patch()
"""

import sys
import importlib
import logging
from typing import Optional, Any, TypeVar

# 获取日志记录器
logger = logging.getLogger("flax_patch")
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别，便于调试

_patch_applied = False

# 定义类型变量
T = TypeVar('T')

def apply_flax_patch():
    """应用Flax兼容性补丁 - 必须在导入flax之前调用"""
    global _patch_applied
    
    if _patch_applied:
        logger.debug("Flax兼容性补丁已应用，跳过重复应用")
        return True
        
    # 对于Python 3.14，我们需要修补dataclasses模块
    if sys.version_info >= (3, 14):
        try:
            # 1. 修补dataclasses模块，解决缺少类型注解问题
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
                        cls.__annotations__[name] = Any
                
                # 调用原始函数
                return original_process_class(cls, init, repr, eq, order, unsafe_hash,
                                             frozen, match_args, kw_only, slots,
                                             weakref_slot)
            
            # 替换原始函数
            dataclasses._process_class = patched_process_class
            
            # 2. 检查并修补_init_subclass_方法（如果需要）
            try:
                # 动态导入flax模块来检查Module类
                flax = importlib.import_module('flax')
                flax_nn = importlib.import_module('flax.nn')
                flax_linen = importlib.import_module('flax.linen')
                
                # 检查flax.linen.Module是否存在_init_subclass_方法
                if hasattr(flax_linen.Module, '__init_subclass__'):
                    logger.debug("Flax linen.Module已包含__init_subclass__方法，无需修补")
                else:
                    # 动态添加_init_subclass_方法，确保参数正确
                    def _init_subclass_(subcls):
                        """为flax.linen.Module添加的__init_subclass__方法"""
                        super(flax_linen.Module, subcls).__init_subclass__()
                    
                    flax_linen.Module.__init_subclass__ = classmethod(_init_subclass_)
                    logger.debug("已为flax.linen.Module动态添加__init_subclass__方法")
                    
                # 检查flax.nn.Module是否存在_init_subclass_方法
                if hasattr(flax_nn.Module, '__init_subclass__'):
                    logger.debug("Flax nn.Module已包含__init_subclass__方法，无需修补")
                else:
                    # 动态添加_init_subclass_方法，确保参数正确
                    def _init_subclass_(subcls):
                        """为flax.nn.Module添加的__init_subclass__方法"""
                        super(flax_nn.Module, subcls).__init_subclass__()
                    
                    flax_nn.Module.__init_subclass__ = classmethod(_init_subclass_)
                    logger.debug("已为flax.nn.Module动态添加__init_subclass__方法")
                    
            except (ImportError, AttributeError) as e:
                # 如果无法导入flax或找不到Module类，记录调试信息
                logger.debug(f"跳过Flax Module.__init_subclass__检查: {e}")
            
            _patch_applied = True
            logger.info("Flax兼容性补丁应用成功")
            return True
            
        except Exception as e:
            logger.error(f"Flax补丁应用失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    logger.debug("当前Python版本无需Flax兼容性补丁")
    return True
