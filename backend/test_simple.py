# 简单的代码验证脚本，用于检查重构后的代码是否存在语法错误

# 尝试导入必要的模块
print("尝试导入模块...")
try:
    import jax
    import jax.numpy as jnp
    print("✅ 成功导入 jax 和 jax.numpy")
except ImportError as e:
    print(f"❌ 导入 jax 失败: {e}")

try:
    # 尝试导入我们的Transformer模型
    from src.core.models.transformer_model import TransformerModel
    print("✅ 成功导入 TransformerModel")
    
    # 检查类方法是否存在
    print("\n检查类方法是否存在...")
    methods = [
        "_apply_no_repeat_ngram",
        "_apply_top_k", 
        "_apply_top_p",
        "generate",
        "generate_with_quantization"
    ]
    
    for method in methods:
        if hasattr(TransformerModel, method):
            print(f"✅ 方法 {method} 存在")
        else:
            print(f"❌ 方法 {method} 不存在")
            
except Exception as e:
    print(f"❌ 导入或检查 TransformerModel 失败: {e}")

print("\n验证完成!")
