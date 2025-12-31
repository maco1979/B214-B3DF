# 猴子补丁：修复Flax与Python 3.14的兼容性问题
import sys
import os

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

print("正在应用Flax兼容性补丁...")

# 修改默认的dataclasses行为，使其在没有类型注释时不抛出错误
import dataclasses
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

print("Flax兼容性补丁已应用")

# 配置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 正常导入应用模块
import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from api import create_app
import uvicorn

# 创建FastAPI应用
app = create_app()

if __name__ == "__main__":
    # 直接硬编码使用端口8001，确保与前端配置一致
    port = 8001
    print(f"DEBUG: Using hardcoded port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)