# 简单测试Flax Dense层，不使用任何补丁

# 直接导入Flax
import flax.linen as nn
import jax.numpy as jnp

# 打印Flax版本
import flax
print(f"Flax版本: {flax.__version__}")

# 测试Dense层的不同参数名称
try:
    # 尝试使用不同参数组合
    print("\n测试1: 无参数")
    dense = nn.Dense()
    print("成功创建Dense层")
except Exception as e:
    print(f"错误: {e}")

try:
    print("\n测试2: 使用features参数")
    dense = nn.Dense(features=64)
    print("成功创建Dense层")
except Exception as e:
    print(f"错误: {e}")

try:
    print("\n测试3: 使用out_features参数")
    dense = nn.Dense(out_features=64)
    print("成功创建Dense层")
except Exception as e:
    print(f"错误: {e}")

try:
    print("\n测试4: 查看Dense类的属性")
    print(f"Dense类的__init__方法: {nn.Dense.__init__}")
    print(f"Dense类的属性: {dir(nn.Dense)}")
except Exception as e:
    print(f"错误: {e}")