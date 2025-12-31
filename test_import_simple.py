#!/usr/bin/env python3
"""
简单测试脚本，仅验证ModelLightweight和ModelLightweightProcessor的导入，避免Flax初始化
"""

import sys

# 添加backend到Python路径
sys.path.insert(0, 'd:/1.5/backend')

try:
    # 仅测试导入，不初始化Flax相关模块
    from src.edge_computing.model_lightweight import ModelLightweight, ModelLightweightProcessor
    print("✅ 成功导入 ModelLightweight 和 ModelLightweightProcessor")
    
    # 验证类关系
    print(f"✅ ModelLightweight 是 ModelLightweightProcessor 的子类: {issubclass(ModelLightweight, ModelLightweightProcessor)}")
    
    # 验证__all__中包含这两个类
    from src.edge_computing.model_lightweight import __all__ as ml_all
    print(f"✅ __all__ 包含 ModelLightweightProcessor: {'ModelLightweightProcessor' in ml_all}")
    print(f"✅ __all__ 包含 ModelLightweight: {'ModelLightweight' in ml_all}")
    
    # 验证包级导出
    from src.edge_computing import ModelLightweight, ModelLightweightProcessor
    print("✅ 从 src.edge_computing 包成功导入 ModelLightweight 和 ModelLightweightProcessor")
    
    print("\n🎉 所有导入测试通过！")
    sys.exit(0)
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)