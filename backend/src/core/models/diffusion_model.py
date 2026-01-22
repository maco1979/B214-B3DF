"""
基于JAX和Flax的扩散模型实现
支持图像生成、超分辨率、风格迁移等任务
"""

from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class TimeEmbedding(nn.Module):
    """时间步长嵌入"""
    dim: int
    
    def setup(self):
        self.linear1 = nn.Dense(self.dim)
        self.linear2 = nn.Dense(self.dim)
    
    def __call__(self, t):
        # 正弦位置编码
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        if self.dim % 2 == 1:
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        
        emb = self.linear1(emb)
        emb = nn.gelu(emb)
        emb = self.linear2(emb)
        
        return emb


class ResNetBlock(nn.Module):
    """扩散模型专用残差块"""
    channels: int
    time_emb_dim: int
    
    def setup(self):
        # 第一层卷积
        self.conv1 = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            padding='SAME'
        )
        self.norm1 = nn.GroupNorm(num_groups=32)
        
        # 时间嵌入投影
        self.time_emb_proj = nn.Dense(self.channels)
        
        # 第二层卷积
        self.conv2 = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            padding='SAME'
        )
        self.norm2 = nn.GroupNorm(num_groups=32)
        
        # 残差连接
        self.residual_conv = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1)
        )
    
    def __call__(self, x, t_emb):
        residual = x
        
        # 第一层
        x = self.conv1(x)
        x = self.norm1(x)
        
        # 添加时间嵌入
        t_emb = self.time_emb_proj(nn.gelu(t_emb))
        t_emb = t_emb[:, None, None, :]  # 扩展到空间维度
        x = x + t_emb
        
        # 第二层
        x = nn.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        # 残差连接
        if residual.shape[-1] != self.channels:
            residual = self.residual_conv(residual)
        
        return x + residual


class AttentionBlock(nn.Module):
    """注意力块"""
    channels: int
    num_heads: int = 8
    
    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32)
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.channels
        )
    
    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        
        # 层归一化
        x_norm = self.norm(x)
        
        # 重塑为序列
        x_flat = x_norm.reshape(batch_size, height * width, channels)
        
        # 自注意力
        attended = self.attention(x_flat)
        
        # 重塑回空间维度
        attended = attended.reshape(batch_size, height, width, channels)
        
        return x + attended


class UNetBlock(nn.Module):
    """U-Net块"""
    channels: int
    time_emb_dim: int
    has_attention: bool = False
    
    def setup(self):
        self.resnet1 = ResNetBlock(self.channels, self.time_emb_dim)
        if self.has_attention:
            self.attention = AttentionBlock(self.channels)
        self.resnet2 = ResNetBlock(self.channels, self.time_emb_dim)
    
    def __call__(self, x, t_emb):
        x = self.resnet1(x, t_emb)
        if self.has_attention:
            x = self.attention(x)
        x = self.resnet2(x, t_emb)
        return x


class DiffusionModel(nn.Module):
    """扩散模型"""
    image_channels: int = 3
    base_channels: int = 64
    time_emb_dim: int = 256
    num_timesteps: int = 1000
    
    def setup(self):
        # 时间嵌入
        self.time_embedding = TimeEmbedding(self.time_emb_dim)
        
        # 输入卷积
        self.input_conv = nn.Conv(
            features=self.base_channels,
            kernel_size=(3, 3),
            padding='SAME'
        )
        
        # 下采样块
        self.down_blocks = [
            UNetBlock(self.base_channels, self.time_emb_dim),
            UNetBlock(self.base_channels * 2, self.time_emb_dim, has_attention=True),
            UNetBlock(self.base_channels * 4, self.time_emb_dim, has_attention=True),
            UNetBlock(self.base_channels * 8, self.time_emb_dim, has_attention=True),
        ]
        
        # 中间块
        self.mid_block = UNetBlock(self.base_channels * 8, self.time_emb_dim, has_attention=True)
        
        # 上采样块
        self.up_blocks = [
            UNetBlock(self.base_channels * 8, self.time_emb_dim, has_attention=True),
            UNetBlock(self.base_channels * 4, self.time_emb_dim, has_attention=True),
            UNetBlock(self.base_channels * 2, self.time_emb_dim),
            UNetBlock(self.base_channels, self.time_emb_dim),
        ]
        
        # 输出卷积
        self.output_conv = nn.Conv(
            features=self.image_channels,
            kernel_size=(3, 3),
            padding='SAME'
        )
    
    def __call__(self, x, t):
        # 时间嵌入
        t_emb = self.time_embedding(t)
        
        # 输入卷积
        x = self.input_conv(x)
        
        # 下采样路径
        down_features = []
        for block in self.down_blocks:
            x = block(x, t_emb)
            down_features.append(x)
            # 下采样
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # 中间块
        x = self.mid_block(x, t_emb)
        
        # 上采样路径
        for i, block in enumerate(self.up_blocks):
            # 上采样
            x = jax.image.resize(
                x,
                shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]),
                method='nearest'
            )
            # 跳跃连接
            if i < len(down_features):
                skip_conn = down_features[-(i+1)]
                # 确保通道数匹配
                if skip_conn.shape[-1] != x.shape[-1]:
                    skip_conn = nn.Conv(features=x.shape[-1], kernel_size=(1, 1))(skip_conn)
                x = x + skip_conn
            
            x = block(x, t_emb)
        
        # 输出
        x = self.output_conv(x)
        
        return x
    
    def forward_diffusion(self, rng, x0, t):
        """前向扩散过程"""
        # 计算噪声
        noise_rng, _ = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, x0.shape)
        
        # 计算alpha和beta
        alpha = jnp.cos((t / self.num_timesteps) * jnp.pi / 2) ** 2
        beta = 1.0 - alpha
        
        # 加噪
        xt = jnp.sqrt(alpha) * x0 + jnp.sqrt(beta) * noise
        
        return xt, noise
    
    def training_loss(self, params, rng, x0):
        """训练损失"""
        batch_size = x0.shape[0]
        
        # 随机时间步
        t_rng, model_rng = jax.random.split(rng)
        t = jax.random.randint(t_rng, (batch_size,), 0, self.num_timesteps)
        
        # 前向扩散
        xt, target_noise = self.forward_diffusion(model_rng, x0, t)
        
        # 模型预测
        predicted_noise = self.apply(params, xt, t)
        
        # 损失
        loss = jnp.mean((predicted_noise - target_noise) ** 2)
        
        return loss
    
    def sample(self, params, rng, num_samples, image_size):
        """采样生成图像"""
        # 初始化噪声
        x = jax.random.normal(rng, (num_samples, image_size, image_size, self.image_channels))
        
        # 反向扩散过程
        for t in reversed(range(self.num_timesteps)):
            # 时间步
            t_batch = jnp.full((num_samples,), t)
            
            # 预测噪声
            predicted_noise = self.apply(params, x, t_batch)
            
            # 计算alpha和beta
            alpha = jnp.cos((t / self.num_timesteps) * jnp.pi / 2) ** 2
            beta = 1.0 - alpha
            
            # 去噪
            if t > 0:
                noise = jax.random.normal(rng, x.shape)
                x = (x - jnp.sqrt(beta) * predicted_noise) / jnp.sqrt(alpha)
                x = x + jnp.sqrt(beta) * noise
            else:
                x = (x - jnp.sqrt(beta) * predicted_noise) / jnp.sqrt(alpha)
        
        return x