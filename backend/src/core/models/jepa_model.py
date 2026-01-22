"""
JEPA (Joint Embedding Predictive Architecture) 核心组件实现

包含编码器、预测器、解码器和能量函数，用于抽象嵌入空间的预测任务
"""

# 首先应用Flax兼容性补丁
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from flax_patch import apply_flax_patch
apply_flax_patch()

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Dict, Any, Optional


class JEPAEncoder(nn.Module):
    """JEPA编码器 - 将输入数据转换为抽象嵌入"""
    embedding_dim: int = 512
    hidden_dim: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """将输入数据编码为抽象嵌入
        
        Args:
            x: 输入数据，形状为 [batch_size, ..., input_dim]
            training: 是否为训练模式
            
        Returns:
            embedding: 抽象嵌入，形状为 [batch_size, embedding_dim]
        """
        # 扁平化输入
        x = jnp.reshape(x, (x.shape[0], -1))
        
        # 全连接层
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # 嵌入层
        embedding = nn.Dense(features=self.embedding_dim)(x)
        
        return embedding


class JEPAPredictor(nn.Module):
    """JEPA预测器 - 根据当前嵌入预测未来嵌入"""
    embedding_dim: int = 512
    hidden_dim: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, current_embedding: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """根据当前嵌入预测未来嵌入
        
        Args:
            current_embedding: 当前抽象嵌入，形状为 [batch_size, embedding_dim]
            training: 是否为训练模式
            
        Returns:
            predicted_embedding: 预测的未来嵌入，形状为 [batch_size, embedding_dim]
        """
        # 预测网络
        x = nn.Dense(features=self.hidden_dim)(current_embedding)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        predicted_embedding = nn.Dense(features=self.embedding_dim)(x)
        
        return predicted_embedding


class JEPADecoder(nn.Module):
    """JEPA解码器 - 将嵌入解码回原始数据空间"""
    output_dim: int
    hidden_dim: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, embedding: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """将嵌入解码回原始数据空间
        
        Args:
            embedding: 抽象嵌入，形状为 [batch_size, embedding_dim]
            training: 是否为训练模式
            
        Returns:
            reconstructed: 重构的数据，形状为 [batch_size, output_dim]
        """
        # 解码网络
        x = nn.Dense(features=self.hidden_dim)(embedding)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        reconstructed = nn.Dense(features=self.output_dim)(x)
        
        return reconstructed


def jepa_energy_function(predicted_embedding: jnp.ndarray, actual_embedding: jnp.ndarray, 
                         temperature: float = 0.1) -> jnp.ndarray:
    """JEPA能量函数 - 使用余弦相似度评估预测嵌入与实际嵌入的一致性
    
    Args:
        predicted_embedding: 预测的嵌入，形状为 [batch_size, embedding_dim]
        actual_embedding: 实际的嵌入，形状为 [batch_size, embedding_dim]
        temperature: 温度参数，控制能量函数的平滑度
        
    Returns:
        energy: 能量值，形状为 [batch_size]
               能量越低表示预测越准确
    """
    # 归一化嵌入
    predicted_embedding = predicted_embedding / jnp.linalg.norm(predicted_embedding, axis=1, keepdims=True)
    actual_embedding = actual_embedding / jnp.linalg.norm(actual_embedding, axis=1, keepdims=True)
    
    # 计算余弦相似度
    cosine_similarity = jnp.sum(predicted_embedding * actual_embedding, axis=1)
    
    # 计算能量（使用温度缩放）
    energy = -cosine_similarity / temperature
    
    return energy


class JEPAModel(nn.Module):
    """完整的JEPA模型，整合编码器、预测器、解码器和能量函数"""
    input_dim: int
    embedding_dim: int = 512
    hidden_dim: int = 1024
    dropout_rate: float = 0.1
    temperature: float = 0.1
    
    def setup(self):
        """初始化模型组件"""
        self.encoder = JEPAEncoder(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )
        
        self.predictor = JEPAPredictor(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )
        
        self.decoder = JEPADecoder(
            output_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )
    
    def __call__(self, current_x: jnp.ndarray, future_x: jnp.ndarray, 
                 training: bool = False) -> Dict[str, Any]:
        """执行JEPA模型的前向传播
        
        Args:
            current_x: 当前输入数据，形状为 [batch_size, ..., input_dim]
            future_x: 未来输入数据，形状为 [batch_size, ..., input_dim]
            training: 是否为训练模式
            
        Returns:
            Dict包含：
            - current_embedding: 当前嵌入
            - predicted_embedding: 预测的未来嵌入
            - future_embedding: 实际的未来嵌入
            - energy: 能量值
            - reconstructed_current: 当前数据的重构
            - reconstructed_future: 未来数据的重构
        """
        # 编码当前和未来数据
        current_embedding = self.encoder(current_x, training=training)
        future_embedding = self.encoder(future_x, training=training)
        
        # 预测未来嵌入
        predicted_embedding = self.predictor(current_embedding, training=training)
        
        # 计算能量
        energy = jepa_energy_function(predicted_embedding, future_embedding, self.temperature)
        
        # 解码（可选）
        reconstructed_current = self.decoder(current_embedding, training=training)
        reconstructed_future = self.decoder(future_embedding, training=training)
        
        return {
            "current_embedding": current_embedding,
            "predicted_embedding": predicted_embedding,
            "future_embedding": future_embedding,
            "energy": energy,
            "reconstructed_current": reconstructed_current,
            "reconstructed_future": reconstructed_future
        }
    
    def predict(self, current_x: jnp.ndarray, steps: int = 1, 
                training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """使用JEPA模型进行多步预测
        
        Args:
            current_x: 当前输入数据，形状为 [batch_size, ..., input_dim]
            steps: 预测步数
            training: 是否为训练模式
            
        Returns:
            predicted_embeddings: 预测的嵌入序列，形状为 [steps, batch_size, embedding_dim]
            predicted_data: 预测的数据序列，形状为 [steps, batch_size, output_dim]
        """
        batch_size = current_x.shape[0]
        
        # 初始化预测结果
        predicted_embeddings = []
        predicted_data = []
        
        # 编码当前数据
        current_embedding = self.encoder(current_x, training=training)
        
        # 多步预测
        for _ in range(steps):
            # 预测下一个嵌入
            next_embedding = self.predictor(current_embedding, training=training)
            predicted_embeddings.append(next_embedding)
            
            # 解码为数据
            next_data = self.decoder(next_embedding, training=training)
            predicted_data.append(next_data)
            
            # 更新当前嵌入
            current_embedding = next_embedding
        
        # 转换为数组
        predicted_embeddings = jnp.stack(predicted_embeddings)
        predicted_data = jnp.stack(predicted_data)
        
        return predicted_embeddings, predicted_data


# 测试代码
if __name__ == "__main__":
    # 创建JEPA模型
    input_dim = 100
    jepa_model = JEPAModel(input_dim=input_dim)
    
    # 初始化模型参数
    rng = jax.random.PRNGKey(0)
    current_x = jax.random.normal(rng, (2, input_dim))
    future_x = jax.random.normal(rng, (2, input_dim))
    
    params = jepa_model.init(rng, current_x, future_x, training=True)
    
    # 测试前向传播
    outputs = jepa_model.apply(params, current_x, future_x, training=False)
    
    print("JEPA模型测试结果：")
    print(f"当前嵌入形状: {outputs['current_embedding'].shape}")
    print(f"预测嵌入形状: {outputs['predicted_embedding'].shape}")
    print(f"未来嵌入形状: {outputs['future_embedding'].shape}")
    print(f"能量值: {outputs['energy']}")
    print(f"重构当前数据形状: {outputs['reconstructed_current'].shape}")
    print(f"重构未来数据形状: {outputs['reconstructed_future'].shape}")
    
    # 测试多步预测
    predicted_embeddings, predicted_data = jepa_model.apply(
        params, current_x, steps=3, training=False, method=jepa_model.predict
    )
    
    print(f"\n多步预测结果：")
    print(f"预测嵌入序列形状: {predicted_embeddings.shape}")
    print(f"预测数据序列形状: {predicted_data.shape}")
