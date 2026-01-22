"""Bi-ATEN Module (Bi-directional Attention Integration)

实现双层注意力集成模块，用于领域泛化和跨领域知识迁移
"""

from typing import Dict, Any, List, Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BiATENConfig:
    """Bi-ATEN模块配置"""
    hidden_size: int = 256
    num_attention_heads: int = 8
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    feed_forward_size: int = 1024
    num_layers: int = 3  # 增加层数以增强跨领域建模能力
    num_domains: int = 500  # 扩展领域支持数量
    domain_adaptation_factor: float = 0.3  # 域自适应因子
    use_domain_adaptive_layer: bool = True  # 是否使用域自适应层
    

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config: BiATENConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # 确保hidden_size可以被num_attention_heads整除
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size {self.hidden_size} 必须能被 num_attention_heads {self.num_attention_heads} 整除"
        
        # 线性变换层
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 输出层
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            query: 查询张量，形状 [batch_size, seq_len_q, hidden_size]
            key: 键张量，形状 [batch_size, seq_len_k, hidden_size]
            value: 值张量，形状 [batch_size, seq_len_v, hidden_size]
            attention_mask: 注意力掩码，形状 [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            注意力输出，形状 [batch_size, seq_len_q, hidden_size]
        """
        batch_size = query.size(0)
        
        # 线性变换
        query = self.query_proj(query)  # [batch_size, seq_len_q, hidden_size]
        key = self.key_proj(key)        # [batch_size, seq_len_k, hidden_size]
        value = self.value_proj(value)  # [batch_size, seq_len_v, hidden_size]
        
        # 重塑为多头注意力格式
        query = query.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        key = key.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)      # [batch_size, num_heads, seq_len_k, head_dim]
        value = value.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 调整掩码形状以匹配注意力分数：[batch_size, 1, seq_len_q, seq_len_k]
            attention_mask = attention_mask.unsqueeze(1)  # 添加注意力头维度
            attention_scores = attention_scores + attention_mask
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_weights = self.dropout(attention_weights)
        
        # 保存注意力权重用于可视化和分析
        self.attention_weights = attention_weights
        
        # 加权求和
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # 重塑为原始形状
        attention_output = attention_output.transpose(1, 2).contiguous()  # [batch_size, seq_len_q, num_heads, head_dim]
        attention_output = attention_output.view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len_q, hidden_size]
        
        # 输出层
        attention_output = self.out_proj(attention_output)  # [batch_size, seq_len_q, hidden_size]
        
        return attention_output


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, config: BiATENConfig):
        super().__init__()
        self.config = config
        
        self.linear1 = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.linear2 = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状 [batch_size, seq_len, hidden_size]
        
        Returns:
            输出张量，形状 [batch_size, seq_len, hidden_size]
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AttentionFusion(nn.Module):
    """注意力融合机制，用于更有效的域内和域间注意力集成"""
    
    def __init__(self, config: BiATENConfig):
        super().__init__()
        self.config = config
        
        # 融合权重学习
        self.fusion_weights = nn.Parameter(torch.randn(2))
        self.fusion_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 可学习的融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, intra_attention_output: torch.Tensor, inter_attention_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            intra_attention_output: 域内注意力输出，形状 [batch_size, seq_len, hidden_size]
            inter_attention_output: 域间注意力输出，形状 [batch_size, seq_len, hidden_size]
            x: 原始输入，形状 [batch_size, seq_len, hidden_size]
        
        Returns:
            融合后的注意力输出，形状 [batch_size, seq_len, hidden_size]
        """
        # 计算融合门控权重
        gate_input = torch.cat([intra_attention_output, inter_attention_output], dim=-1)
        gate_weights = self.fusion_gate(gate_input)  # [batch_size, seq_len, 2]
        
        # 融合域内和域间注意力
        fused_attention = gate_weights[..., 0:1] * intra_attention_output + gate_weights[..., 1:2] * inter_attention_output
        fused_attention = self.fusion_norm(fused_attention)
        
        return fused_attention


class BiATENLayer(nn.Module):
    """双层注意力集成层"""
    
    def __init__(self, config: BiATENConfig):
        super().__init__()
        self.config = config
        
        # 域内注意力
        self.intra_domain_attention = MultiHeadAttention(config)
        self.intra_domain_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 域间注意力
        self.inter_domain_attention = MultiHeadAttention(config)
        self.inter_domain_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 注意力融合机制
        self.attention_fusion = AttentionFusion(config)
        
        # 前馈神经网络
        self.feed_forward = FeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, domain_embedding: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状 [batch_size, seq_len, hidden_size]
            domain_embedding: 域嵌入，形状 [batch_size, 1, hidden_size]
            attention_mask: 注意力掩码，形状 [batch_size, seq_len, seq_len]
        
        Returns:
            输出张量，形状 [batch_size, seq_len, hidden_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 1. 域内注意力
        intra_attention_output = self.intra_domain_attention(x, x, x, attention_mask)
        
        # 2. 域间注意力
        # 复制域嵌入以匹配序列长度
        expanded_domain_embedding = domain_embedding.expand(batch_size, seq_len, -1)
        inter_attention_output = self.inter_domain_attention(x, expanded_domain_embedding, expanded_domain_embedding)
        
        # 3. 注意力融合
        residual = x
        fused_attention = self.attention_fusion(intra_attention_output, inter_attention_output, x)
        x = residual + self.dropout(fused_attention)
        x = self.intra_domain_norm(x)  # 复用现有的归一化层
        
        # 4. 前馈神经网络
        residual = x
        feed_forward_output = self.feed_forward(x)
        x = residual + self.dropout(feed_forward_output)
        x = self.feed_forward_norm(x)
        
        return x


class DomainAdaptiveLayer(nn.Module):
    """域自适应层，用于增强跨领域知识迁移"""
    
    def __init__(self, config: BiATENConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 多层域自适应变换，支持更复杂的域特征学习
        self.domain_adaptive_transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU()
        )
        
        # 增强的域感知门控，支持多层次特征融合
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 域特定的特征转换矩阵
        self.domain_specific_projection = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, domain_emb: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征，形状 [batch_size, seq_len, hidden_size]
            domain_emb: 域嵌入，形状 [batch_size, 1, hidden_size]
        
        Returns:
            域自适应特征，形状 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 扩展域嵌入以匹配序列长度
        expanded_domain_emb = domain_emb.expand(batch_size, seq_len, -1)
        
        # 生成域自适应特征
        domain_adaptive_features = self.domain_adaptive_transform(expanded_domain_emb)
        
        # 应用域特定的特征转换
        domain_specific_features = torch.matmul(x, self.domain_specific_projection)
        
        # 计算门控权重
        gate_input = torch.cat([x, domain_adaptive_features], dim=-1)
        gate_weight = self.gate(gate_input)
        
        # 多层次特征融合：原始特征 + 域自适应特征 + 域特定转换特征
        adaptive_output = (
            gate_weight * domain_adaptive_features + 
            (1 - gate_weight) * x + 
            0.5 * domain_specific_features
        )
        
        adaptive_output = self.dropout(adaptive_output)
        adaptive_output = self.layer_norm(adaptive_output)
        
        return adaptive_output


class BiATENModule(nn.Module):
    """双层注意力集成模块"""
    
    def __init__(self, config: BiATENConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([BiATENLayer(config) for _ in range(config.num_layers)])
        
        # 扩展域嵌入层，支持更多域
        self.domain_embedding = nn.Embedding(config.num_domains, config.hidden_size)
        
        # 添加域自适应层
        if config.use_domain_adaptive_layer:
            self.domain_adaptive_layer = DomainAdaptiveLayer(config)
        
        # 动态跨域知识迁移网络
        self.cross_domain_transfer_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size * 2, config.hidden_size * config.hidden_size),
            nn.GELU()
        )
        
        # 跨域迁移门控
        self.transfer_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Bi-ATEN模块初始化完成，配置: {config.__dict__}")
    
    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征，形状 [batch_size, seq_len, hidden_size]
            domain_ids: 域ID，形状 [batch_size]
            attention_mask: 注意力掩码，形状 [batch_size, seq_len, seq_len]
        
        Returns:
            输出特征，形状 [batch_size, seq_len, hidden_size]
        """
        # 获取域嵌入
        domain_emb = self.domain_embedding(domain_ids).unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 应用域自适应层（如果启用）
        if self.config.use_domain_adaptive_layer:
            x = self.domain_adaptive_layer(x, domain_emb)
        
        # 遍历所有层
        for layer in self.layers:
            x = layer(x, domain_emb, attention_mask)
        
        # 应用动态跨域知识迁移
        # 1. 生成基于域嵌入的动态迁移矩阵
        batch_size = x.size(0)
        domain_emb_flat = domain_emb.squeeze(1)  # [batch_size, hidden_size]
        transfer_matrices = self.cross_domain_transfer_net(domain_emb_flat)  # [batch_size, hidden_size*hidden_size]
        transfer_matrices = transfer_matrices.view(batch_size, self.config.hidden_size, self.config.hidden_size)  # [batch_size, hidden_size, hidden_size]
        
        # 2. 计算跨域迁移特征
        # 对每个样本应用对应的迁移矩阵
        cross_domain_features = []
        for i in range(batch_size):
            sample_feature = x[i]  # [seq_len, hidden_size]
            sample_matrix = transfer_matrices[i]  # [hidden_size, hidden_size]
            transferred_feature = torch.matmul(sample_feature, sample_matrix)  # [seq_len, hidden_size]
            cross_domain_features.append(transferred_feature.unsqueeze(0))
        cross_domain_features = torch.cat(cross_domain_features, dim=0)  # [batch_size, seq_len, hidden_size]
        
        # 3. 应用迁移门控
        gate_weights = self.transfer_gate(domain_emb_flat).unsqueeze(1)  # [batch_size, 1, 1]
        x = x + gate_weights * cross_domain_features * self.config.domain_adaptation_factor
        
        return x
    
    def get_domain_generalized_features(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """获取领域泛化特征
        
        Args:
            x: 输入特征，形状 [batch_size, seq_len, hidden_size]
            domain_ids: 域ID，形状 [batch_size]
        
        Returns:
            领域泛化特征，形状 [batch_size, seq_len, hidden_size]
        """
        return self.forward(x, domain_ids)
    
    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 模型保存路径
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Bi-ATEN模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型
        
        Args:
            path: 模型加载路径
        """
        self.load_state_dict(torch.load(path))
        logger.info(f"Bi-ATEN模型已从: {path} 加载")
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """获取注意力权重，用于可视化和分析
        
        Returns:
            注意力权重字典，包含各层的域内和域间注意力权重
        """
        attention_weights = {}
        
        for i, layer in enumerate(self.layers):
            # 获取域内注意力权重
            if hasattr(layer.intra_domain_attention, 'attention_weights'):
                attention_weights[f'layer_{i}_intra_attention'] = layer.intra_domain_attention.attention_weights
            
            # 获取域间注意力权重
            if hasattr(layer.inter_domain_attention, 'attention_weights'):
                attention_weights[f'layer_{i}_inter_attention'] = layer.inter_domain_attention.attention_weights
        
        return attention_weights
    
    def visualize_attention(self, layer_idx: int = 0, head_idx: int = 0) -> Dict[str, torch.Tensor]:
        """可视化注意力权重
        
        Args:
            layer_idx: 要可视化的层索引
            head_idx: 要可视化的注意力头索引
        
        Returns:
            可视化数据，包含域内和域间注意力权重热力图数据
        """
        attention_weights = self.get_attention_weights()
        visualization_data = {}
        
        # 域内注意力可视化
        intra_key = f'layer_{layer_idx}_intra_attention'
        if intra_key in attention_weights:
            intra_weights = attention_weights[intra_key]
            # 选择特定头的注意力权重
            if head_idx < intra_weights.size(1):
                visualization_data['intra_attention'] = intra_weights[0, head_idx, :, :]  # [seq_len_q, seq_len_k]
        
        # 域间注意力可视化
        inter_key = f'layer_{layer_idx}_inter_attention'
        if inter_key in attention_weights:
            inter_weights = attention_weights[inter_key]
            # 选择特定头的注意力权重
            if head_idx < inter_weights.size(1):
                visualization_data['inter_attention'] = inter_weights[0, head_idx, :, :]  # [seq_len_q, seq_len_k]
        
        return visualization_data
    
    def get_feature_importance(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """获取特征重要性分析
        
        Args:
            x: 输入特征，形状 [batch_size, seq_len, hidden_size]
            domain_ids: 域ID，形状 [batch_size]
        
        Returns:
            特征重要性，形状 [batch_size, hidden_size]
        """
        # 使用梯度作为特征重要性指标
        x.requires_grad = True
        
        # 前向传播
        output = self.forward(x, domain_ids)
        
        # 计算梯度
        output.sum().backward()
        
        # 获取特征重要性（梯度的绝对值）
        feature_importance = x.grad.abs().mean(dim=1)  # [batch_size, hidden_size]
        
        return feature_importance
    
    def get_domain_contribution(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, float]:
        """获取域贡献度分析
        
        Args:
            x: 输入特征，形状 [batch_size, seq_len, hidden_size]
            domain_ids: 域ID，形状 [batch_size]
        
        Returns:
            域贡献度字典，包含各组件的贡献度
        """
        # 保存原始输出
        original_output = self.forward(x, domain_ids).detach()
        
        # 计算各组件的贡献度
        contributions = {}
        
        # 1. 计算域自适应层的贡献
        if self.config.use_domain_adaptive_layer:
            original_use_domain_adaptive = self.config.use_domain_adaptive_layer
            self.config.use_domain_adaptive_layer = False
            
            without_adaptive_output = self.forward(x, domain_ids).detach()
            adaptive_contribution = torch.mean(torch.abs(original_output - without_adaptive_output)).item()
            contributions['domain_adaptive_layer'] = adaptive_contribution
            
            self.config.use_domain_adaptive_layer = original_use_domain_adaptive
        
        # 2. 计算跨域迁移的贡献
        original_adaptation_factor = self.config.domain_adaptation_factor
        self.config.domain_adaptation_factor = 0.0
        
        without_transfer_output = self.forward(x, domain_ids).detach()
        transfer_contribution = torch.mean(torch.abs(original_output - without_transfer_output)).item()
        contributions['cross_domain_transfer'] = transfer_contribution
        
        self.config.domain_adaptation_factor = original_adaptation_factor
        
        return contributions


@dataclass
class DomainGeneralizationResult:
    """领域泛化结果"""
    original_features: torch.Tensor
    generalized_features: torch.Tensor
    domain_ids: torch.Tensor
    confidence: float
    

# 创建全局Bi-ATEN模块实例
def get_bi_aten_module():
    """获取Bi-ATEN模块实例"""
    config = BiATENConfig()
    return BiATENModule(config)

bi_aten_module = get_bi_aten_module()
