#!/usr/bin/env python3
"""
启动后端API服务
"""

import uvicorn
import sys
import os

# 首先应用Flax补丁
if True:
    print("🧩 应用Flax兼容性补丁...")
    import sys
    import importlib
    from typing import Optional, Any
    
    # 使用更简单的方式解决问题：直接修改dataclasses模块的_process_class函数
    def apply_flax_patch():
        """应用Flax兼容性补丁"""
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

    # 应用补丁
    apply_flax_patch()
    print("✅ Flax补丁应用成功")

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # 添加根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))  # 添加backend目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/src'))  # 添加src目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/src/core'))  # 添加core目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/src/core/utils'))  # 添加utils目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/src/core/services'))  # 添加services目录

def main():
    try:
        print("🚀 启动AI平台后端API服务...")
        print("🌍 端口: 8001")
        print("📚 文档: http://localhost:8001/docs")
        
        # 导入并创建应用
        from src.api import create_app
        
        app = create_app()
        
        # 启动服务
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在项目根目录运行此脚本")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()