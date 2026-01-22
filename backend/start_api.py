#!/usr/bin/env python3
"""
快速启动AI平台API服务
"""

# 应用Flax兼容性补丁
import sys
import os
import dataclasses

# 修改默认的dataclasses行为，使其在没有类型注释时不抛出错误
original_process_class = dataclasses._process_class

def patched_process_class(cls, init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only, slots, weakref_slot):
    """修补dataclasses._process_class，允许没有类型注释的字段"""
    try:
        return original_process_class(cls, init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only, slots, weakref_slot)
    except TypeError as e:
        if "is a field but has no type annotation" in str(e):
            # 获取类的所有字段
            fields = []
            for name, value in cls.__dict__.items():
                if isinstance(value, dataclasses.Field):
                    fields.append(name)
            
            # 为没有类型注释的字段添加类型注释
            if not hasattr(cls, '__annotations__'):
                cls.__annotations__ = {}
            
            for field_name in fields:
                if field_name not in cls.__annotations__:
                    cls.__annotations__[field_name] = type(None)
            
            # 再次尝试处理类
            return original_process_class(cls, init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only, slots, weakref_slot)
        raise

# 应用补丁
dataclasses._process_class = patched_process_class
print("已应用dataclasses兼容性补丁")

# 配置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 导入并启动应用
    import uvicorn
    from src.api import create_app
    
    if __name__ == "__main__":
        print("🚀 启动AI平台API服务...")
        print("🌍 端口: 8002") 
        print("📚 文档: http://localhost:8002/docs")
        
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
except Exception as e:
    print(f"启动失败: {e}")
    import traceback
    traceback.print_exc()