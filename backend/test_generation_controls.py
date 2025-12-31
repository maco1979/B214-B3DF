#!/usr/bin/env python3
"""
测试Transformer模型的所有生成控制功能
包括重复惩罚、束搜索、n-gram惩罚、Top-k/Top-p采样等
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


def calculate_repetition_rate(sequence):
    """计算序列的重复率"""
    if len(sequence) == 0:
        return 0.0
    unique_tokens = len(set(sequence.tolist()))
    return 1 - unique_tokens / len(sequence)


def count_ngram_repetitions(sequence, ngram_size):
    """计算序列中n-gram的重复次数"""
    if ngram_size <= 0 or len(sequence) < ngram_size:
        return 0
    
    ngrams = []
    for i in range(len(sequence) - ngram_size + 1):
        ngram = tuple(sequence[i:i+ngram_size].tolist())
        ngrams.append(ngram)
    
    # 统计重复的n-gram
    unique_ngrams = set(ngrams)
    return len(ngrams) - len(unique_ngrams)


def test_repetition_penalty():
    """测试重复惩罚功能"""
    print("=== 测试1: 重复惩罚功能 ===")
    model, params, rng = create_test_model()
    
    # 创建一个简单的提示
    prompt = jnp.array([10, 20, 30], dtype=jnp.int32)
    max_length = 20
    temperature = 1.0
    
    print("\n测试1.1: 无重复惩罚 (repetition_penalty=1.0)")
    rng1 = jax.random.PRNGKey(42)
    generated1 = model.generate(params, rng1, prompt, max_length, temperature, repetition_penalty=1.0)
    print(f"生成序列: {generated1}")
    repetition_rate1 = calculate_repetition_rate(generated1)
    print(f"重复率: {repetition_rate1:.2f}")
    
    print("\n测试1.2: 中等重复惩罚 (repetition_penalty=2.0)")
    rng2 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated2 = model.generate(params, rng2, prompt, max_length, temperature, repetition_penalty=2.0)
    print(f"生成序列: {generated2}")
    repetition_rate2 = calculate_repetition_rate(generated2)
    print(f"重复率: {repetition_rate2:.2f}")
    
    print("\n测试1.3: 强重复惩罚 (repetition_penalty=5.0)")
    rng3 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated3 = model.generate(params, rng3, prompt, max_length, temperature, repetition_penalty=5.0)
    print(f"生成序列: {generated3}")
    repetition_rate3 = calculate_repetition_rate(generated3)
    print(f"重复率: {repetition_rate3:.2f}")
    
    # 验证重复惩罚是否有效
    if repetition_rate3 < repetition_rate2 < repetition_rate1:
        print("✅ 重复惩罚功能正常工作!")
    else:
        print("❌ 重复惩罚功能可能存在问题!")


def test_beam_search():
    """测试束搜索功能"""
    print("\n\n=== 测试2: 束搜索功能 ===")
    model, params, rng = create_test_model()
    
    # 创建一个简单的提示
    prompt = jnp.array([10, 20, 30], dtype=jnp.int32)
    max_length = 20
    temperature = 1.0
    repetition_penalty = 1.0
    
    print("\n测试2.1: 贪心搜索 (beam_search=False)")
    rng1 = jax.random.PRNGKey(42)
    generated1 = model.generate(
        params, rng1, prompt, max_length, temperature, repetition_penalty,
        beam_search=False
    )
    print(f"生成序列: {generated1}")
    print(f"序列长度: {len(generated1)}")
    
    print("\n测试2.2: 束搜索 (beam_width=3)")
    rng2 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated2 = model.generate(
        params, rng2, prompt, max_length, temperature, repetition_penalty,
        beam_search=True, beam_width=3
    )
    print(f"生成序列: {generated2}")
    print(f"序列长度: {len(generated2)}")
    
    print("\n测试2.3: 更大束宽的束搜索 (beam_width=10)")
    rng3 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated3 = model.generate(
        params, rng3, prompt, max_length, temperature, repetition_penalty,
        beam_search=True, beam_width=10
    )
    print(f"生成序列: {generated3}")
    print(f"序列长度: {len(generated3)}")


def test_no_repeat_ngram():
    """测试n-gram重复惩罚功能"""
    print("\n\n=== 测试3: n-gram重复惩罚功能 ===")
    model, params, rng = create_test_model()
    
    # 创建一个简单的提示
    prompt = jnp.array([10, 20, 30], dtype=jnp.int32)
    max_length = 30
    temperature = 1.0
    repetition_penalty = 1.0
    
    print("\n测试3.1: 无n-gram惩罚 (no_repeat_ngram_size=0)")
    rng1 = jax.random.PRNGKey(42)
    generated1 = model.generate(
        params, rng1, prompt, max_length, temperature, repetition_penalty,
        no_repeat_ngram_size=0
    )
    print(f"生成序列: {generated1}")
    ngram_repeats1 = count_ngram_repetitions(generated1, ngram_size=2)
    print(f"2-gram重复次数: {ngram_repeats1}")
    
    print("\n测试3.2: 2-gram重复惩罚 (no_repeat_ngram_size=2)")
    rng2 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated2 = model.generate(
        params, rng2, prompt, max_length, temperature, repetition_penalty,
        no_repeat_ngram_size=2
    )
    print(f"生成序列: {generated2}")
    ngram_repeats2 = count_ngram_repetitions(generated2, ngram_size=2)
    print(f"2-gram重复次数: {ngram_repeats2}")
    
    print("\n测试3.3: 3-gram重复惩罚 (no_repeat_ngram_size=3)")
    rng3 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated3 = model.generate(
        params, rng3, prompt, max_length, temperature, repetition_penalty,
        no_repeat_ngram_size=3
    )
    print(f"生成序列: {generated3}")
    ngram_repeats3 = count_ngram_repetitions(generated3, ngram_size=3)
    print(f"3-gram重复次数: {ngram_repeats3}")
    
    # 验证n-gram惩罚是否有效
    if ngram_repeats2 < ngram_repeats1:
        print("✅ 2-gram重复惩罚功能正常工作!")
    else:
        print("❌ 2-gram重复惩罚功能可能存在问题!")


def test_sampling_strategies():
    """测试采样策略"""
    print("\n\n=== 测试4: 采样策略 ===")
    model, params, rng = create_test_model()
    
    # 创建一个简单的提示
    prompt = jnp.array([10, 20, 30], dtype=jnp.int32)
    max_length = 20
    temperature = 1.0
    repetition_penalty = 1.0
    
    print("\n测试4.1: 贪心采样 (do_sample=False)")
    rng1 = jax.random.PRNGKey(42)
    generated1 = model.generate(
        params, rng1, prompt, max_length, temperature, repetition_penalty,
        do_sample=False
    )
    print(f"生成序列: {generated1}")
    repetition_rate1 = calculate_repetition_rate(generated1)
    print(f"重复率: {repetition_rate1:.2f}")
    
    print("\n测试4.2: Top-k采样 (do_sample=True, top_k=10)")
    rng2 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated2 = model.generate(
        params, rng2, prompt, max_length, temperature, repetition_penalty,
        do_sample=True, top_k=10
    )
    print(f"生成序列: {generated2}")
    repetition_rate2 = calculate_repetition_rate(generated2)
    print(f"重复率: {repetition_rate2:.2f}")
    
    print("\n测试4.3: Top-p采样 (do_sample=True, top_p=0.9)")
    rng3 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated3 = model.generate(
        params, rng3, prompt, max_length, temperature, repetition_penalty,
        do_sample=True, top_p=0.9
    )
    print(f"生成序列: {generated3}")
    repetition_rate3 = calculate_repetition_rate(generated3)
    print(f"重复率: {repetition_rate3:.2f}")
    
    print("\n测试4.4: Top-k + Top-p混合采样 (do_sample=True, top_k=5, top_p=0.8)")
    rng4 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated4 = model.generate(
        params, rng4, prompt, max_length, temperature, repetition_penalty,
        do_sample=True, top_k=5, top_p=0.8
    )
    print(f"生成序列: {generated4}")
    repetition_rate4 = calculate_repetition_rate(generated4)
    print(f"重复率: {repetition_rate4:.2f}")


def test_combined_controls():
    """测试组合控制参数"""
    print("\n\n=== 测试5: 组合控制参数 ===")
    model, params, rng = create_test_model()
    
    # 创建一个简单的提示
    prompt = jnp.array([10, 20, 30], dtype=jnp.int32)
    max_length = 30
    temperature = 0.7  # 降低温度增加确定性
    
    print("\n测试5.1: 标准设置 (无特殊控制)")
    rng1 = jax.random.PRNGKey(42)
    generated1 = model.generate(
        params, rng1, prompt, max_length, temperature, repetition_penalty=1.0,
        beam_search=False, no_repeat_ngram_size=0, do_sample=False
    )
    print(f"生成序列: {generated1}")
    repetition_rate1 = calculate_repetition_rate(generated1)
    ngram_repeats1 = count_ngram_repetitions(generated1, ngram_size=2)
    print(f"重复率: {repetition_rate1:.2f}, 2-gram重复次数: {ngram_repeats1}")
    
    print("\n测试5.2: 组合控制参数 (重复惩罚+束搜索+n-gram惩罚)")
    rng2 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated2 = model.generate(
        params, rng2, prompt, max_length, temperature, repetition_penalty=2.0,
        beam_search=True, beam_width=5, early_stopping=True,
        no_repeat_ngram_size=2
    )
    print(f"生成序列: {generated2}")
    repetition_rate2 = calculate_repetition_rate(generated2)
    ngram_repeats2 = count_ngram_repetitions(generated2, ngram_size=2)
    print(f"重复率: {repetition_rate2:.2f}, 2-gram重复次数: {ngram_repeats2}")
    
    print("\n测试5.3: 高级采样控制 (采样+Top-k+Top-p+重复惩罚)")
    rng3 = jax.random.PRNGKey(42)  # 使用相同的随机种子以便比较
    generated3 = model.generate(
        params, rng3, prompt, max_length, temperature, repetition_penalty=1.5,
        do_sample=True, top_k=10, top_p=0.9
    )
    print(f"生成序列: {generated3}")
    repetition_rate3 = calculate_repetition_rate(generated3)
    ngram_repeats3 = count_ngram_repetitions(generated3, ngram_size=2)
    print(f"重复率: {repetition_rate3:.2f}, 2-gram重复次数: {ngram_repeats3}")


if __name__ == "__main__":
    print("开始测试Transformer模型生成控制功能...\n")
    
    try:
        test_repetition_penalty()
        test_beam_search()
        test_no_repeat_ngram()
        test_sampling_strategies()
        test_combined_controls()
        
        print("\n\n=== 所有测试完成 ===")
        print("请查看测试结果，验证所有功能是否正常工作。")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()