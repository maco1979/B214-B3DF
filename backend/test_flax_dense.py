import flax.linen as nn

# 测试Flax Dense层的参数名称
try:
    # 尝试使用features参数
    dense_features = nn.Dense(features=64)
    print("Dense层接受features参数")
except TypeError as e:
    print(f"features参数错误: {e}")

try:
    # 尝试使用out_features参数
    dense_out_features = nn.Dense(out_features=64)
    print("Dense层接受out_features参数")
except TypeError as e:
    print(f"out_features参数错误: {e}")

try:
    # 尝试使用out_dim参数
    dense_out_dim = nn.Dense(out_dim=64)
    print("Dense层接受out_dim参数")
except TypeError as e:
    print(f"out_dim参数错误: {e}")

# 查看Dense类的定义
print("\nDense类的__init__方法签名:")
import inspect
print(inspect.signature(nn.Dense.__init__))