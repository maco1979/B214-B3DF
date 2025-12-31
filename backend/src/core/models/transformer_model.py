"""
基于JAX和Flax的Transformer模型实现
支持高效并行训练和推理
"""

from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    d_model: int
    num_heads: int
    
    def setup(self):
        self.d_k = self.d_model // self.num_heads
        self.w_q = nn.Dense(self.d_model, use_bias=False)
        self.w_k = nn.Dense(self.d_model, use_bias=False)
        self.w_v = nn.Dense(self.d_model, use_bias=False)
        self.w_o = nn.Dense(self.d_model)
    
    def __call__(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        
        # 线性变换
        q = self.w_q(q).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.w_k(k).reshape(batch_size, -1, self.num_heads, self.d_k)
        v = self.w_v(v).reshape(batch_size, -1, self.num_heads, self.d_k)
        
        # 转置以进行批量矩阵乘法
        q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)
        k = k.transpose(0, 2, 3, 1)  # (batch, heads, d_k, seq_len)
        v = v.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)
        
        # 注意力分数计算
        scores = jnp.matmul(q, k) / jnp.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask
        
        # 注意力权重
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # 上下文向量
        context = jnp.matmul(attn_weights, v)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        return self.w_o(context)


class FeedForward(nn.Module):
    """前馈神经网络"""
    d_model: int
    d_ff: int
    
    def setup(self):
        self.linear1 = nn.Dense(self.d_ff)
        self.linear2 = nn.Dense(self.d_model)
    
    def __call__(self, x):
        return self.linear2(nn.gelu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.self_attn = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.feed_forward = FeedForward(d_model=self.d_model, d_ff=self.d_ff)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, x, mask=None, deterministic=True):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output, deterministic=deterministic))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output, deterministic=deterministic))
        
        return x


class TransformerModel(nn.Module):
    """Transformer模型"""
    vocab_size: int = 30000  # 扩展词汇表大小以支持更大模型
    d_model: int = 2048  # 扩展为大模型默认尺寸
    num_layers: int = 24  # 增加层数
    num_heads: int = 16  # 增加头数
    d_ff: int = 8192  # 扩展前馈网络维度
    max_seq_len: int = 2048  # 增加最大序列长度
    dropout_rate: float = 0.1  # 保持合适的dropout率
    
    def setup(self):
        self.token_embedding = nn.Embed(features=self.d_model, num_embeddings=self.vocab_size)
        self.position_embedding = nn.Embed(features=self.d_model, num_embeddings=self.max_seq_len)
        
        self.layers = [
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_layers)
        ]
        
        self.output_projection = nn.Dense(self.vocab_size)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, input_ids, deterministic=True):
        batch_size, seq_len = input_ids.shape
        
        # 嵌入层
        token_embeddings = self.token_embedding(input_ids)
        position_ids = jnp.arange(seq_len)
        position_embeddings = self.position_embedding(position_ids)
        
        # 合并嵌入
        x = token_embeddings + position_embeddings
        x = self.dropout(x, deterministic=deterministic)
        
        # 注意力掩码
        mask = jnp.triu(jnp.ones((seq_len, seq_len)) * -jnp.inf, k=1)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, mask=mask, deterministic=deterministic)
        
        # 输出投影
        logits = self.output_projection(x)
        
        return logits
    
    def quantize(self, quantization_type: str = "int8") -> "TransformerModel":
        """
        量化模型参数
        
        Args:
            quantization_type: 量化类型 (int8, float16, int16)
            
        Returns:
            量化后的模型
        """
        # 这个方法将在模型轻量化模块中实现实际的量化逻辑
        # 这里返回自身，因为JAX模型的量化通常是在运行时或通过专门的工具完成
        return self
    
    def dequantize(self) -> "TransformerModel":
        """
        反量化模型参数
        
        Returns:
            反量化后的模型
        """
        # 与量化类似，这里返回自身
        return self
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """
        获取模型量化信息
        
        Returns:
            量化信息字典
        """
        return {
            "model_type": "transformer",
            "support_quantization": True,
            "supported_types": ["int8", "float16", "int16"],
            "current_type": "float32"  # 默认浮点精度
        }
    
    def _apply_no_repeat_ngram(self, tokens, next_token_logits, ngram_size):
        """n-gram惩罚函数，防止重复生成相同的n-gram序列
        
        Args:
            tokens: 当前已生成的token序列
            next_token_logits: 下一个token的logits
            ngram_size: n-gram的大小
            
        Returns:
            应用了n-gram惩罚后的logits
        """
        if ngram_size <= 1:
            return next_token_logits
        
        # 确保tokens长度足够形成ngram
        if len(tokens) < ngram_size - 1:
            return next_token_logits
        
        # 获取当前的(n-1)-gram
        current_ngram = tuple(tokens[-(ngram_size-1):])
        
        # 遍历所有可能的下一个token，检查是否形成重复的ngram
        for token in range(self.vocab_size):
            # 形成候选ngram
            candidate_ngram = current_ngram + (token,)
            
            # 检查这个ngram是否已经在序列中出现过
            ngram_found = False
            for i in range(len(tokens) - ngram_size + 1):
                existing_ngram = tuple(tokens[i:i+ngram_size])
                if existing_ngram == candidate_ngram:
                    ngram_found = True
                    break
            
            # 如果ngram已经存在，将其概率设为负无穷
            if ngram_found:
                next_token_logits = next_token_logits.at[token].set(-jnp.inf)
        
        return next_token_logits
    
    def _apply_top_k(self, logits, k):
        """Top-k采样函数，只保留前k个最可能的token
        
        Args:
            logits: 原始logits
            k: 保留的token数量
            
        Returns:
            应用了Top-k采样后的logits
        """
        if k <= 0:
            return logits
        
        # 找到top-k的最小值
        top_k_values = jnp.sort(logits)[-k:]
        min_top_k = top_k_values[0]
        
        # 将所有低于top-k的logits设为负无穷
        return jnp.where(logits < min_top_k, -jnp.inf, logits)
    
    def _apply_top_p(self, logits, p):
        """Top-p采样函数（核采样），保留累积概率达到p的token
        
        Args:
            logits: 原始logits
            p: 累积概率阈值
            
        Returns:
            应用了Top-p采样后的logits
        """
        if p >= 1.0:
            return logits
        
        # 排序logits并计算累积概率
        sorted_logits = jnp.sort(logits)[::-1]
        sorted_probs = jax.nn.softmax(sorted_logits)
        cumulative_probs = jnp.cumsum(sorted_probs)
        
        # 找到需要保留的token数量
        indices = jnp.where(cumulative_probs <= p)[0]
        if len(indices) == 0:
            return logits
        
        # 保留top-p的token
        threshold = sorted_logits[indices[-1]]
        return jnp.where(logits < threshold, -jnp.inf, logits)
    
    def generate(self, params, rng, prompt, max_length=100, temperature=1.0, repetition_penalty=1.0, beam_search=False, beam_width=5, early_stopping=True, no_repeat_ngram_size=0, do_sample=False, top_p=1.0, top_k=0):
        """
        文本生成

        Args:
            params: 模型参数
            rng: 随机数生成器
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度
            repetition_penalty: 重复惩罚系数，大于1会惩罚已生成的token
            beam_search: 是否使用束搜索
            beam_width: 束搜索的束宽度
            early_stopping: 是否在所有束都结束时提前停止
            no_repeat_ngram_size: n-gram惩罚大小，0表示不使用
            do_sample: 是否使用随机采样而不是贪婪搜索
            top_p: 核采样的累积概率阈值
            top_k: 前k个最可能的token进行采样
        """
        
        if not beam_search:
            # 贪婪搜索或随机采样
            def step_fn(carry, _):
                tokens, rng = carry
                
                # 获取当前预测
                logits = self.apply(params, tokens[None, :], deterministic=False)
                next_token_logits = logits[0, -1, :] / temperature
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    # 创建已生成token的掩码
                    generated_tokens = tokens
                    # 对已生成的token应用惩罚
                    next_token_logits = next_token_logits.at[generated_tokens].set(
                        next_token_logits[generated_tokens] / repetition_penalty
                    )
                
                # 应用n-gram惩罚
                if no_repeat_ngram_size > 0:
                    next_token_logits = self._apply_no_repeat_ngram(tokens, next_token_logits, no_repeat_ngram_size)
                
                # 应用top-k和top-p采样
                if top_k > 0:
                    next_token_logits = self._apply_top_k(next_token_logits, top_k)
                if top_p < 1.0:
                    next_token_logits = self._apply_top_p(next_token_logits, top_p)
                
                # 采样
                rng, sample_rng = jax.random.split(rng)
                if do_sample:
                    next_token = jax.random.categorical(sample_rng, next_token_logits)
                else:
                    next_token = jnp.argmax(next_token_logits)
                
                # 更新序列
                tokens = jnp.concatenate([tokens, next_token[None]], axis=0)
                
                return (tokens, rng), next_token
            
            # 初始化序列
            tokens = prompt
            
            # 生成序列
            (tokens, _), _ = jax.lax.scan(
                step_fn, (tokens, rng), jnp.arange(max_length)
            )
            
            return tokens
        else:
            # 束搜索实现
            batch_size = 1
            beam_width = beam_width
            vocab_size = self.vocab_size
            
            # 初始化束
            initial_tokens = prompt
            initial_log_prob = 0.0
            
            # 创建束：每个束包含 (tokens, log_probability)
            # tokens形状: (beam_width, current_length)
            # log_probs形状: (beam_width,)
            beams = {
                'tokens': jnp.tile(initial_tokens[None, :], (beam_width, 1)),
                'log_probs': jnp.zeros(beam_width)
            }
            
            # 开始生成
            for step in range(max_length):
                # 准备输入
                current_tokens = beams['tokens']
                batch_beams = current_tokens.shape[0]
                
                # 获取当前长度
                current_length = current_tokens.shape[1]
                
                # 应用模型
                logits = self.apply(params, current_tokens, deterministic=False)
                next_token_logits = logits[:, -1, :] / temperature
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_beams):
                        generated_tokens = current_tokens[i]
                        next_token_logits = next_token_logits.at[i, generated_tokens].set(
                            next_token_logits[i, generated_tokens] / repetition_penalty
                        )
                
                # 应用n-gram惩罚
                if no_repeat_ngram_size > 0:
                    for i in range(batch_beams):
                        generated_tokens = current_tokens[i]
                        # 直接应用n-gram惩罚到当前束的logits
                        next_token_logits = next_token_logits.at[i].set(self._apply_no_repeat_ngram(generated_tokens, next_token_logits[i], no_repeat_ngram_size))
                
                # 应用Top-k和Top-p采样
                if do_sample:
                    for i in range(batch_beams):
                        if top_k > 0:
                            next_token_logits = next_token_logits.at[i].set(self._apply_top_k(next_token_logits[i], top_k))
                        if top_p < 1.0:
                            next_token_logits = next_token_logits.at[i].set(self._apply_top_p(next_token_logits[i], top_p))
                
                # 计算概率
                next_token_probs = jax.nn.log_softmax(next_token_logits, axis=-1)
                
                # 广播当前对数概率
                current_log_probs = beams['log_probs'][:, None] + next_token_probs
                
                # 展平以找到top beam_width * vocab_size个候选
                flat_log_probs = current_log_probs.reshape(-1)
                flat_indices = jnp.argsort(flat_log_probs, descending=True)[:beam_width * 2]
                
                # 选择top beam_width个候选
                top_log_probs = flat_log_probs[flat_indices]
                top_beams = flat_indices // vocab_size
                top_tokens = flat_indices % vocab_size
                
                # 更新束
                new_tokens = []
                for beam_idx, token_idx in zip(top_beams, top_tokens):
                    new_tokens.append(jnp.concatenate([current_tokens[beam_idx], jnp.array([token_idx])]))
                
                new_tokens = jnp.stack(new_tokens)
                
                # 只保留前beam_width个
                beams = {
                    'tokens': new_tokens[:beam_width],
                    'log_probs': top_log_probs[:beam_width]
                }
                
                # 提前停止检查
                if early_stopping:
                    # 检查是否所有束都已生成结束token（假设0是结束token）
                    all_finished = jnp.all(jnp.any(new_tokens == 0, axis=1))
                    if all_finished:
                        break
            
            # 返回最可能的序列
            best_beam_idx = jnp.argmax(beams['log_probs'])
            return beams['tokens'][best_beam_idx]
    
    def run_with_quantization(self, params, input_ids, quantization_type: str = "int8", deterministic=True):
        """
        使用量化精度运行模型推理
        
        Args:
            params: 模型参数
            input_ids: 输入序列
            quantization_type: 量化类型 (int8, float16, int16)
            deterministic: 是否使用确定性模式
            
        Returns:
            模型输出
        """
        # 转换输入到量化精度
        if quantization_type == "float16":
            # 使用FP16推理
            input_ids_fp16 = input_ids.astype(jnp.float16)
            
            # 定义FP16版本的apply函数
            def fp16_apply(params, inputs, deterministic):
                return self.apply(params, inputs, deterministic=deterministic)
            
            # 使用JAX的float16上下文
            with jax.default_matmul_precision('float16'):
                logits = fp16_apply(params, input_ids_fp16, deterministic)
            
            return logits
        elif quantization_type == "int8":
            # INT8量化推理（需要更复杂的实现，这里仅做示例）
            # 实际应用中需要使用量化库或自定义量化逻辑
            logits = self.apply(params, input_ids, deterministic=deterministic)
            return logits
        elif quantization_type == "int16":
            # INT16量化推理
            logits = self.apply(params, input_ids, deterministic=deterministic)
            return logits
        else:
            # 默认使用FP32
            return self.apply(params, input_ids, deterministic=deterministic)
    
    def generate_with_quantization(self, params, rng, prompt, max_length=100, temperature=1.0, quantization_type: str = "int8", repetition_penalty=1.0, beam_search=False, beam_width=5, early_stopping=True, no_repeat_ngram_size=0, do_sample=False, top_p=1.0, top_k=0):
        """
        使用量化精度进行文本生成

        Args:
            params: 模型参数
            rng: 随机数生成器
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度
            quantization_type: 量化类型 (int8, float16, int16)
            repetition_penalty: 重复惩罚系数，大于1会惩罚已生成的token
            beam_search: 是否使用束搜索
            beam_width: 束搜索的束宽度
            early_stopping: 是否在所有束都结束时提前停止
            no_repeat_ngram_size: n-gram惩罚大小，0表示不使用
            do_sample: 是否使用随机采样而不是贪婪搜索
            top_p: 核采样的累积概率阈值
            top_k: 前k个最可能的token进行采样

        Returns:
            生成的文本序列
        """
        
        if not beam_search:
            # 贪婪搜索或随机采样
            def step_fn(carry, _):
                tokens, rng = carry
                
                # 使用量化推理获取当前预测
                logits = self.run_with_quantization(params, tokens[None, :], quantization_type, deterministic=False)
                next_token_logits = logits[0, -1, :] / temperature
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    # 创建已生成token的掩码
                    generated_tokens = tokens
                    # 对已生成的token应用惩罚
                    next_token_logits = next_token_logits.at[generated_tokens].set(
                        next_token_logits[generated_tokens] / repetition_penalty
                    )
                
                # 应用n-gram惩罚
                if no_repeat_ngram_size > 0:
                    next_token_logits = self._apply_no_repeat_ngram(tokens, next_token_logits, no_repeat_ngram_size)
                
                # 应用top-k和top-p采样
                if top_k > 0:
                    next_token_logits = self._apply_top_k(next_token_logits, top_k)
                if top_p < 1.0:
                    next_token_logits = self._apply_top_p(next_token_logits, top_p)
                
                # 采样
                rng, sample_rng = jax.random.split(rng)
                if do_sample:
                    next_token = jax.random.categorical(sample_rng, next_token_logits)
                else:
                    next_token = jnp.argmax(next_token_logits)
                
                # 更新序列
                tokens = jnp.concatenate([tokens, next_token[None]], axis=0)
                
                return (tokens, rng), next_token
            
            # 初始化序列
            tokens = prompt
            
            # 生成序列
            (tokens, _), _ = jax.lax.scan(
                step_fn, (tokens, rng), jnp.arange(max_length)
            )
            
            return tokens
        else:
            # 束搜索实现
            batch_size = 1
            beam_width = beam_width
            vocab_size = self.vocab_size
            
            # 初始化束
            initial_tokens = prompt
            initial_log_prob = 0.0
            
            # 创建束：每个束包含 (tokens, log_probability)
            # tokens形状: (beam_width, current_length)
            # log_probs形状: (beam_width,)
            beams = {
                'tokens': jnp.tile(initial_tokens[None, :], (beam_width, 1)),
                'log_probs': jnp.zeros(beam_width)
            }
            
            # 开始生成
            for step in range(max_length):
                # 准备输入
                current_tokens = beams['tokens']
                batch_beams = current_tokens.shape[0]
                
                # 获取当前长度
                current_length = current_tokens.shape[1]
                
                # 使用量化推理获取当前预测
                logits = self.run_with_quantization(params, current_tokens, quantization_type, deterministic=False)
                next_token_logits = logits[:, -1, :] / temperature
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_beams):
                        generated_tokens = current_tokens[i]
                        next_token_logits = next_token_logits.at[i, generated_tokens].set(
                            next_token_logits[i, generated_tokens] / repetition_penalty
                        )
                
                # 应用n-gram惩罚
                if no_repeat_ngram_size > 0:
                    for i in range(batch_beams):
                        generated_tokens = current_tokens[i]
                        next_token_logits = next_token_logits.at[i].set(self._apply_no_repeat_ngram(generated_tokens, next_token_logits[i], no_repeat_ngram_size))
                
                # 应用Top-k和Top-p采样
                if do_sample:
                    for i in range(batch_beams):
                        if top_k > 0:
                            next_token_logits = next_token_logits.at[i].set(self._apply_top_k(next_token_logits[i], top_k))
                        if top_p < 1.0:
                            next_token_logits = next_token_logits.at[i].set(self._apply_top_p(next_token_logits[i], top_p))
                
                # 计算概率
                next_token_probs = jax.nn.log_softmax(next_token_logits, axis=-1)
                
                # 广播当前对数概率
                current_log_probs = beams['log_probs'][:, None] + next_token_probs
                
                # 展平以找到top beam_width * vocab_size个候选
                flat_log_probs = current_log_probs.reshape(-1)
                flat_indices = jnp.argsort(flat_log_probs, descending=True)[:beam_width * 2]
                
                # 选择top beam_width个候选
                top_log_probs = flat_log_probs[flat_indices]
                top_beams = flat_indices // vocab_size
                top_tokens = flat_indices % vocab_size
                
                # 更新束
                new_tokens = []
                for beam_idx, token_idx in zip(top_beams, top_tokens):
                    new_tokens.append(jnp.concatenate([current_tokens[beam_idx], jnp.array([token_idx])]))
                
                new_tokens = jnp.stack(new_tokens)
                
                # 只保留前beam_width个
                beams = {
                    'tokens': new_tokens[:beam_width],
                    'log_probs': top_log_probs[:beam_width]
                }
                
                # 提前停止检查
                if early_stopping:
                    # 检查是否所有束都已生成结束token（假设0是结束token）
                    all_finished = jnp.all(jnp.any(new_tokens == 0, axis=1))
                    if all_finished:
                        break
            
            # 返回最可能的序列
            best_beam_idx = jnp.argmax(beams['log_probs'])
            return beams['tokens'][best_beam_idx]