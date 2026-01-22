#!/usr/bin/env python3
"""
简单测试脚本，验证ModelLightweight的导入是否成功
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.src.edge_computing.model_lightweight import ModelLightweight
    print("✅ 成功导入 ModelLightweight")
    print(f"   ModelLightweight 类: {ModelLightweight}")
    
    # 验证ModelLightweightProcessor也能导入
    from backend.src.edge_computing.model_lightweight import ModelLightweightProcessor
    print("✅ 成功导入 ModelLightweightProcessor")
    print(f"   ModelLightweightProcessor 类: {ModelLightweightProcessor}")
    
    # 验证别名关系
    print(f"✅ ModelLightweight 是 ModelLightweightProcessor 的子类: {issubclass(ModelLightweight, ModelLightweightProcessor)}")
    
    # 测试创建实例
    lightweight = ModelLightweight()
    print("✅ 成功创建 ModelLightweight 实例")
    print(f"   实例: {lightweight}")
    
    print("\n🎉 所有导入测试通过！")
    sys.exit(0)
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
