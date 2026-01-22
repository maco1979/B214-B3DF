#!/usr/bin/env python3
"""
测试Transformer模型的重复惩罚功能
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from src.core.models.transformer_model import TransformerModel


def create_test_model():
    """创建一个测试用的Transformer模型"""
    model = TransformerModel(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=256,
        max_seq_len=100,
        dropout_rate=0.1
    )
    
    # 初始化随机参数
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # 初始化模型参数
    input_shape = (1, 10)  # (batch, seq_len)
    input_ids = jax.random.randint(init_rng, input_shape, 0, 1000)
    
    # 创建初始参数
    params = model.init(init_rng, input_ids)
    
    return model, params, rng


def test_repetition_penalty():
    """测试重复惩罚功能"""
    print("创建测试模型...")
    model, params, rng = create_test_model()
    
    # 创建一个简单的提示
    prompt = jnp.array([10, 20, 30], dtype=jnp.int32)
    max_length = 20
    temperature = 1.0
    
    print("\n测试1: 无重复惩罚 (repetition_penalty=1.0)")
    rng1 = jax.random.PRNGKey(42)
    generated1 = model.generate(params, rng1, prompt, max_length, temperature, repetition_penalty=1.0)
    print(f"生成序列: {generated1}")
    print(f"序列长度: {len(generated1)}")
    
    # 计算重复率
    unique_tokens1 = len(set(generated1.tolist()))
    repetition_rate1 = 1 - unique_tokens1 / len(generated1)
    print(f"重复率: {repetition_rate1:.2f}")
    
    print("\n测试2: 有重复惩罚 (repetition_penalty=2.0)")
    rng2 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated2 = model.generate(params, rng2, prompt, max_length, temperature, repetition_penalty=2.0)
    print(f"生成序列: {generated2}")
    print(f"序列长度: {len(generated2)}")
    
    # 计算重复率
    unique_tokens2 = len(set(generated2.tolist()))
    repetition_rate2 = 1 - unique_tokens2 / len(generated2)
    print(f"重复率: {repetition_rate2:.2f}")
    
    print("\n测试3: 强重复惩罚 (repetition_penalty=5.0)")
    rng3 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated3 = model.generate(params, rng3, prompt, max_length, temperature, repetition_penalty=5.0)
    print(f"生成序列: {generated3}")
    print(f"序列长度: {len(generated3)}")
    
    # 计算重复率
    unique_tokens3 = len(set(generated3.tolist()))
    repetition_rate3 = 1 - unique_tokens3 / len(generated3)
    print(f"重复率: {repetition_rate3:.2f}")
    
    print("\n结论:")
    print(f"无惩罚重复率: {repetition_rate1:.2f}")
    print(f"2.0惩罚重复率: {repetition_rate2:.2f}")
    print(f"5.0惩罚重复率: {repetition_rate3:.2f}")
    
    if repetition_rate3 < repetition_rate2 < repetition_rate1:
        print("✅ 重复惩罚功能正常工作!")
    else:
        print("❌ 重复惩罚功能可能存在问题!")


if __name__ == "__main__":
    test_repetition_penalty()
