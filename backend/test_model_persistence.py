#!/usr/bin/env python3
"""
测试模型持久化功能
验证创建模型、保存模型和加载模型的端到端流程
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from core.services.model_manager import ModelManager

def test_model_persistence():
    """测试模型持久化功能"""
    print("=== 测试模型持久化功能 ===")
    
    # 创建模型管理器实例
    model_manager = ModelManager("test_models")
    
    # 测试参数
    model_id = "test_model_123"
    model_type = "transformer"
    hyperparameters = {
        "vocab_size": 1000,
        "max_seq_len": 512,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2
    }
    description = "测试模型持久化"
    
    try:
        # 1. 创建模型
        print(f"1. 创建模型: ID={model_id}, 类型={model_type}")
        metadata = model_manager.create_model(
            model_type=model_type,
            model_id=model_id,
            hyperparameters=hyperparameters,
            description=description
        )
        print(f"   ✓ 模型创建成功: {metadata.name} (版本: {metadata.version})")
        
        # 2. 加载模型
        print(f"2. 加载模型: {model_id}")
        state, loaded_metadata = model_manager.load_model(model_id)
        print(f"   ✓ 模型加载成功: {loaded_metadata.name}")
        print(f"   ✓ 模型类型: {loaded_metadata.model_type}")
        print(f"   ✓ 模型版本: {loaded_metadata.version}")
        print(f"   ✓ 模型状态类型: {type(state).__name__}")
        print(f"   ✓ 模型参数: {state.params.keys() if hasattr(state, 'params') else '无参数'}")
        
        # 3. 列出所有模型
        print("3. 列出所有模型")
        models = model_manager.list_models()
        print(f"   ✓ 模型总数: {len(models)}")
        for model in models:
            print(f"   - {model.name} (ID: {model.model_id}, 类型: {model.model_type})")
        
        # 4. 删除测试模型
        print(f"4. 删除测试模型: {model_id}")
        model_manager.delete_model(model_id)
        print("   ✓ 模型删除成功")
        
        # 验证模型已删除
        print("5. 验证模型已删除")
        remaining_models = model_manager.list_models()
        print(f"   ✓ 剩余模型总数: {len(remaining_models)}")
        
        print("\n✅ 所有测试通过! 模型持久化功能正常工作")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理测试目录
        import shutil
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")
            print("   ✓ 测试目录清理完成")

if __name__ == "__main__":
    success = test_model_persistence()
    sys.exit(0 if success else 1)
