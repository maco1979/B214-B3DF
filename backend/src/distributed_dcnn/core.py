"""
分布式DCNN核心实现
基于JAX和Flax构建的分布式卷积神经网络
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class DCNNArchitecture(Enum):
    """DCNN架构类型"""
    STANDARD = "standard"  # 标准卷积网络
    RESIDUAL = "residual"  # 残差网络
    DENSE = "dense"        # 密集连接网络
    LIGHTWEIGHT = "lightweight"  # 轻量化网络


@dataclass
class DCNNConfig:
    """DCNN配置参数"""
    architecture: DCNNArchitecture = DCNNArchitecture.STANDARD
    num_classes: int = 1000
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    
    # 卷积层配置
    conv_channels: List[int] = None
    kernel_sizes: List[Tuple[int, int]] = None
    strides: List[Tuple[int, int]] = None
    
    # 池化配置
    pooling_types: List[str] = None
    pooling_sizes: List[Tuple[int, int]] = None
    
    # 正则化配置
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256, 512]
        if self.kernel_sizes is None:
            self.kernel_sizes = [(3, 3)] * len(self.conv_channels)
        if self.strides is None:
            self.strides = [(1, 1)] * len(self.conv_channels)
        if self.pooling_types is None:
            self.pooling_types = ['max'] * (len(self.conv_channels) - 1)
        if self.pooling_sizes is None:
            self.pooling_sizes = [(2, 2)] * (len(self.conv_channels) - 1)


class ConvBlock(nn.Module):
    """卷积块"""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    use_batch_norm: bool = True
    activation: str = "relu"
    
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
        
        # 激活函数
        if self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "swish":
            x = nn.swish(x)
        
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    
    def setup(self):
        self.conv1 = ConvBlock(self.features, self.kernel_size)
        self.conv2 = ConvBlock(self.features, self.kernel_size)
        
        # 投影层（如果需要）
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
                    strides=(1, 1)
                )
            residual = self.projection(residual, deterministic=deterministic)
        
        return x + residual


class DenseBlock(nn.Module):
    """密集连接块"""
    growth_rate: int
    num_layers: int
    
    def setup(self):
        self.layers = [
            ConvBlock(self.growth_rate, kernel_size=(3, 3)) 
            for _ in range(self.num_layers)
        ]
    
    def __call__(self, x, deterministic=True):
        features = [x]
        
        for layer in self.layers:
            # 前一层特征与当前层输出连接
            new_features = layer(x, deterministic=deterministic)
            features.append(new_features)
            x = jnp.concatenate(features, axis=-1)
        
        return x


class DistributedDCNN(nn.Module):
    """分布式DCNN核心模型"""
    config: DCNNConfig
    
    def setup(self):
        self.conv_blocks = []
        self.pooling_layers = []
        
        # 构建卷积层
        for i, (channels, kernel_size, stride) in enumerate(
            zip(self.config.conv_channels, self.config.kernel_sizes, self.config.strides)
        ):
            if self.config.architecture == DCNNArchitecture.RESIDUAL:
                block = ResidualBlock(features=channels, kernel_size=kernel_size)
            elif self.config.architecture == DCNNArchitecture.DENSE:
                block = DenseBlock(growth_rate=channels//4, num_layers=4)
            else:
                block = ConvBlock(features=channels, kernel_size=kernel_size, strides=stride)
            
            self.conv_blocks.append(block)
            
            # 添加池化层（除了最后一层）
            if i < len(self.config.conv_channels) - 1:
                pooling_type = self.config.pooling_types[i]
                if pooling_type == 'max':
                    pool_fn = nn.max_pool
                else:
                    pool_fn = nn.avg_pool
                
                self.pooling_layers.append(pool_fn)
        
        # 全局平均池化
        self.global_pool = nn.avg_pool
        
        # 分类器
        self.classifier = nn.Dense(self.config.num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def __call__(self, x, deterministic=True):
        """前向传播"""
        batch_size = x.shape[0]
        
        # 卷积层和池化层
        for i, (conv_block, pool_layer) in enumerate(zip(self.conv_blocks, self.pooling_layers)):
            x = conv_block(x, deterministic=deterministic)
            
            # 池化
            pool_size = self.config.pooling_sizes[i]
            x = pool_layer(x, window_shape=pool_size, strides=pool_size)
        
        # 最后一层卷积（无池化）
        if len(self.conv_blocks) > len(self.pooling_layers):
            x = self.conv_blocks[-1](x, deterministic=deterministic)
        
        # 全局平均池化
        x = self.global_pool(x, window_shape=(x.shape[1], x.shape[2]))
        x = x.reshape(batch_size, -1)
        
        # Dropout
        x = self.dropout(x, deterministic=deterministic)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def extract_features(self, x, deterministic=True):
        """提取特征（用于迁移学习）"""
        features = []
        
        for i, (conv_block, pool_layer) in enumerate(zip(self.conv_blocks, self.pooling_layers)):
            x = conv_block(x, deterministic=deterministic)
            features.append(x)  # 记录特征
            
            # 池化
            pool_size = self.config.pooling_sizes[i]
            x = pool_layer(x, window_shape=pool_size, strides=pool_size)
        
        if len(self.conv_blocks) > len(self.pooling_layers):
            x = self.conv_blocks[-1](x, deterministic=deterministic)
            features.append(x)
        
        return features


class DistributedDCNNTrainer:
    """分布式DCNN训练器"""
    
    def __init__(self, config: DCNNConfig):
        self.config = config
        self.model = DistributedDCNN(config)
        
        # 初始化参数
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, *config.input_shape))
        self.params = self.model.init(rng, dummy_input)
        
        # 优化器
        self.optimizer = self._create_optimizer()
    
    def _create_optimizer(self):
        """创建优化器"""
        import optax
        
        # 学习率调度
        learning_rate = 0.001
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=1000,
            decay_rate=0.96
        )
        
        # 优化器
        optimizer = optax.adam(learning_rate=lr_schedule)
        
        return optimizer
    
    def train_step(self, params, optimizer_state, batch, rng):
        """单步训练"""
        
        def loss_fn(params):
            images, labels = batch
            logits = self.model.apply(params, images)
            
            # 交叉熵损失
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            
            # 正则化损失
            l2_loss = 0.001 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
            
            total_loss = loss + l2_loss
            
            # 计算准确率
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(predictions == jnp.argmax(labels, axis=-1))
            
            return total_loss, (loss, accuracy)
        
        # 计算梯度和损失
        (total_loss, (loss, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # 更新参数
        updates, optimizer_state = self.optimizer.update(grads, optimizer_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, optimizer_state, {
            'total_loss': total_loss,
            'loss': loss,
            'accuracy': accuracy,
            'grad_norm': jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        }
    
    def federated_aggregation(self, client_updates: List[Dict], data_sizes: List[int]):
        """联邦学习参数聚合"""
        total_data_size = sum(data_sizes)
        
        # 加权平均
        aggregated_update = {}
        
        for key in client_updates[0].keys():
            weighted_sum = None
            
            for i, update in enumerate(client_updates):
                weight = data_sizes[i] / total_data_size
                param_update = update[key]
                
                if weighted_sum is None:
                    weighted_sum = param_update * weight
                else:
                    weighted_sum += param_update * weight
            
            aggregated_update[key] = weighted_sum
        
        return aggregated_update


class DCNNModelManager:
    """DCNN模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
    
    def register_model(self, model_id: str, config: DCNNConfig, params: Dict):
        """注册模型"""
        self.models[model_id] = {
            'config': config,
            'params': params,
            'created_at': jax.random.PRNGKey(int(jax.device_put(0).device_buffer)),
            'version': '1.0.0'
        }
    
    def get_model(self, model_id: str) -> Tuple[DistributedDCNN, Dict]:
        """获取模型"""
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model_info = self.models[model_id]
        model = DistributedDCNN(model_info['config'])
        
        return model, model_info['params']
    
    def export_model(self, model_id: str, format: str = "jax") -> Dict[str, Any]:
        """导出模型"""
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model_info = self.models[model_id]
        
        if format == "jax":
            return {
                'config': model_info['config'],
                'params': model_info['params'],
                'metadata': {
                    'model_id': model_id,
                    'created_at': model_info['created_at'],
                    'version': model_info['version']
                }
            }
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 预训练配置
PRETRAINED_CONFIGS = {
    "resnet18": DCNNConfig(
        architecture=DCNNArchitecture.RESIDUAL,
        num_classes=1000,
        conv_channels=[64, 64, 128, 256, 512],
        kernel_sizes=[(7,7), (3,3), (3,3), (3,3), (3,3)],
        strides=[(2,2), (1,1), (2,2), (2,2), (2,2)]
    ),
    "efficientnet": DCNNConfig(
        architecture=DCNNArchitecture.STANDARD,
        num_classes=1000,
        conv_channels=[32, 16, 24, 40, 80, 112, 192, 320, 1280],
        kernel_sizes=[(3,3)] * 9
    ),
    "mobilenet": DCNNConfig(
        architecture=DCNNArchitecture.LIGHTWEIGHT,
        num_classes=1000,
        conv_channels=[32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024],
        kernel_sizes=[(3,3)] * 14
    )
}