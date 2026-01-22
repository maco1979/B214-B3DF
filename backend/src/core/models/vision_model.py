"""
基于JAX和Flax的计算机视觉模型
支持图像分类、目标检测、语义分割等任务
"""

from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class ConvBlock(nn.Module):
    """卷积块"""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    use_batch_norm: bool = True
    
    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
            use_bias=not self.use_batch_norm
        )
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm()
    
    def __call__(self, x, deterministic=True):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x, deterministic=deterministic)
        x = nn.gelu(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    features: int
    
    def setup(self):
        self.conv1 = ConvBlock(self.features)
        self.conv2 = ConvBlock(self.features)
        
        # 如果输入输出通道数不一致，需要投影
        self.projection = None
    
    def __call__(self, x, deterministic=True):
        residual = x
        
        # 主路径
        x = self.conv1(x, deterministic=deterministic)
        x = self.conv2(x, deterministic=deterministic)
        
        # 残差连接
        if residual.shape[-1] != x.shape[-1]:
            if self.projection is None:
                self.projection = ConvBlock(
                    self.features, 
                    kernel_size=(1, 1), 
                    use_batch_norm=True
                )
            residual = self.projection(residual, deterministic=deterministic)
        
        return x + residual


class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    num_classes: int
    patch_size: Tuple[int, int] = (16, 16)
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1
    
    def setup(self):
        # 图像分块
        self.patch_embedding = nn.Conv(
            features=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='VALID'
        )
        
        # 分类标记
        self.cls_token = self.param(
            'cls_token', 
            jax.nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_dim)
        )
        
        # 位置编码
        self.position_embedding = nn.Embed(
            num_embeddings=1000,  # 最大序列长度
            features=self.hidden_dim
        )
        
        # Transformer编码器层
        self.encoder_layers = [
            self._create_encoder_layer() for _ in range(self.num_layers)
        ]
        
        # 层归一化
        self.layer_norm = nn.LayerNorm()
        
        # 分类头
        self.classifier = nn.Dense(self.num_classes)
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def _create_encoder_layer(self):
        """创建Transformer编码器层"""
        return nn.TransformerEncoderBlock(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate
        )
    
    def __call__(self, x, deterministic=True):
        batch_size = x.shape[0]
        
        # 图像分块嵌入
        x = self.patch_embedding(x)
        x = x.reshape(batch_size, -1, self.hidden_dim)
        
        # 添加分类标记
        cls_tokens = jnp.broadcast_to(
            self.cls_token, (batch_size, 1, self.hidden_dim)
        )
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # 添加位置编码
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        x = self.dropout(x, deterministic=deterministic)
        
        # Transformer编码器
        for layer in self.encoder_layers:
            x = layer(x, deterministic=deterministic)
        
        # 分类标记输出
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits


class VisionModel(nn.Module):
    """通用视觉模型"""
    num_classes: int = 10
    model_type: str = "vit"  # "vit" 或 "cnn"
    
    def setup(self):
        if self.model_type == "vit":
            self.model = VisionTransformer(num_classes=self.num_classes)
        else:
            # CNN架构
            self.conv_layers = [
                ConvBlock(64),  # 64, 64x64
                ConvBlock(128), # 128, 32x32
                ConvBlock(256), # 256, 16x16
                ConvBlock(512), # 512, 8x8
            ]
            self.global_pool = nn.avg_pool
            self.classifier = nn.Dense(self.num_classes)
    
    def __call__(self, x, deterministic=True):
        if self.model_type == "vit":
            return self.model(x, deterministic=deterministic)
        else:
            # CNN前向传播
            for conv in self.conv_layers:
                x = conv(x, deterministic=deterministic)
            
            # 全局平均池化
            x = self.global_pool(x, window_shape=(x.shape[1], x.shape[2]))
            x = x.reshape(x.shape[0], -1)
            
            return self.classifier(x)
    
    def extract_features(self, x, deterministic=True):
        """提取特征"""
        if self.model_type == "vit":
            # Vision Transformer特征提取
            batch_size = x.shape[0]
            x = self.model.patch_embedding(x)
            x = x.reshape(batch_size, -1, self.model.hidden_dim)
            
            cls_tokens = jnp.broadcast_to(
                self.model.cls_token, (batch_size, 1, self.model.hidden_dim)
            )
            x = jnp.concatenate([cls_tokens, x], axis=1)
            
            seq_len = x.shape[1]
            positions = jnp.arange(seq_len)
            pos_emb = self.model.position_embedding(positions)
            x = x + pos_emb
            
            for layer in self.model.encoder_layers:
                x = layer(x, deterministic=deterministic)
            
            return self.model.layer_norm(x)
        else:
            # CNN特征提取
            features = []
            for conv in self.conv_layers:
                x = conv(x, deterministic=deterministic)
                features.append(x)
            
            return features