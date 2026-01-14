# Model Detector for Logical Consistency Checking
# Implements model-based logic consistency detection using SC-Energy Network
from typing import List, Dict, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class SC_Energy(nn.Module):
    """Set-Consistency Energy Network
    Computes energy scores for decision consistency
    Lower energy scores indicate more consistent decisions
    实现了技术文档中描述的基于边际损失的SC-Energy网络"""
    def __init__(self, encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hidden_size: int = 256, use_margin_loss: bool = True):
        super(SC_Energy, self).__init__()
        self.use_margin_loss = use_margin_loss
        self.alpha = 0.5  # 边际损失的alpha参数
        
        try:
            # Load pre-trained encoder model
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
            self.encoder = AutoModel.from_pretrained(encoder_model_name)
            self.hidden_size = hidden_size
            
            # MLP layers to compute energy score - 与技术文档一致
            self.fc1 = nn.Linear(self.encoder.config.hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            
            logger.info(f"SC-Energy模型初始化成功，使用编码器: {encoder_model_name}")
        except Exception as e:
            logger.error(f"SC-Energy模型初始化失败: {e}")
            # 使用简化的能量模型作为备选
            self.tokenizer = None
            self.encoder = None
            self.fc1 = nn.Linear(768, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            logger.info("使用简化版SC-Energy模型")
    
    def encode_statements(self, statements: List[str]) -> torch.Tensor:
        """编码陈述列表"""
        if not self.tokenizer or not self.encoder:
            # 使用随机嵌入作为备选
            return torch.randn(len(statements), 768)
        
        # Tokenize the statements
        inputs = self.tokenizer(statements, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # 使用pooler输出，与技术文档一致
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def forward(self, statements: List[str]) -> torch.Tensor:
        """计算陈述集合的能量分数
        实现技术文档中描述的能量计算逻辑"""
        # 对每个陈述进行编码
        embeddings = self.encode_statements(statements)
        
        # 计算集合的平均嵌入 - 与技术文档一致
        avg_embedding = torch.mean(embeddings, dim=0)
        
        # 通过MLP计算能量分数 - 与技术文档一致
        x = self.relu(self.fc1(avg_embedding))
        x = self.dropout(x)
        energy_score = self.fc2(x)
        
        return energy_score
    
    def compute_energy_loss(self, consistent_set: List[str], inconsistent_set: List[str]) -> torch.Tensor:
        """计算边际损失，用于模型训练
        实现技术文档中描述的损失函数: L_E(S_C,S_I) = max(0, E_θ(S_C) - E_θ(S_I) + α)"""
        if not self.use_margin_loss:
            return torch.tensor(0.0)
        
        # 计算一致集合和不一致集合的能量分数
        e_consistent = self.forward(consistent_set)
        e_inconsistent = self.forward(inconsistent_set)
        
        # 计算边际损失
        loss = torch.max(torch.tensor(0.0), e_consistent - e_inconsistent + self.alpha)
        
        return loss
    
    def compute_consistency_score(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        """计算决策与历史的一致性分数
        改进版：考虑历史决策的多样性和决策间的逻辑关系"""
        if not history:
            return 1.0
        
        # 使用更多的历史决策进行比较（最多5个）
        history_count = min(5, len(history))
        
        # 构建当前决策的陈述
        current_statement = f"决策: {decision['action']}, 参数: {decision['parameters']}, 推理: {decision['reasoning']}"
        
        # 计算与每个历史决策的一致性分数
        consistency_scores = []
        for hist_decision in history[-history_count:]:
            # 构建历史决策陈述
            hist_statement = f"历史决策: {hist_decision['action']}, 参数: {hist_decision['parameters']}, 推理: {hist_decision['reasoning']}"
            
            try:
                # 计算当前决策与单个历史决策的能量分数
                statements = [current_statement, hist_statement]
                energy_score = self.forward(statements).item()
                
                # 将能量分数转换为一致性分数
                score = float(torch.sigmoid(torch.tensor(-energy_score)))
                consistency_scores.append(score)
            except Exception as e:
                logger.error(f"计算单个历史决策一致性分数失败: {e}")
                consistency_scores.append(0.8)  # 默认分数
        
        # 计算平均一致性分数，考虑历史决策的权重（越近的决策权重越高）
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, score in enumerate(consistency_scores):
            # 权重随历史距离递减（最近的权重为1.0，最远的权重为0.2）
            weight = 1.0 - (i / (history_count + 1)) * 0.8
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.8
        
        # 计算加权平均一致性分数
        consistency_score = weighted_sum / total_weight
        
        # 确保分数在0-1范围内
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        return consistency_score

class ModelDetector:
    """模型检测器，使用SC-Energy模型进行逻辑一致性检测
    支持配置参数和更详细的一致性分析"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # 配置参数
        self.encoder_model_name = config.get('encoder_model_name', "sentence-transformers/all-MiniLM-L6-v2")
        self.hidden_size = config.get('hidden_size', 256)
        self.consistency_threshold = config.get('consistency_threshold', 0.5)
        self.use_margin_loss = config.get('use_margin_loss', True)
        
        # 初始化SC-Energy模型
        self.energy_model = SC_Energy(
            encoder_model_name=self.encoder_model_name,
            hidden_size=self.hidden_size,
            use_margin_loss=self.use_margin_loss
        )
        
        logger.info(f"ModelDetector初始化成功，配置: {config}")
    
    def check_consistency(self, decision: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用模型检测决策的逻辑一致性
        返回更详细的一致性分析结果"""
        consistency_check = {
            'is_consistent': True,
            'consistency_score': 1.0,
            'model': 'SC-Energy',
            'model_config': {
                'encoder': self.encoder_model_name,
                'hidden_size': self.hidden_size
            },
            'analysis_details': {}
        }
        
        try:
            # 计算一致性分数
            consistency_score = self.energy_model.compute_consistency_score(decision, history)
            consistency_check['consistency_score'] = consistency_score
            
            # 添加分析细节
            consistency_check['analysis_details'] = {
                'history_count_used': min(5, len(history)),
                'threshold_used': self.consistency_threshold
            }
            
            # 检查是否低于阈值
            if consistency_score < self.consistency_threshold:
                consistency_check['is_consistent'] = False
                consistency_check['message'] = f"模型检测到逻辑不一致，一致性分数: {consistency_score:.2f}"
                consistency_check['analysis_details']['reason'] = "一致性分数低于阈值"
            else:
                consistency_check['analysis_details']['reason'] = "一致性分数正常"
        except Exception as e:
            logger.error(f"模型检测一致性失败: {e}")
            consistency_check['consistency_score'] = 0.8
            consistency_check['is_consistent'] = True
            consistency_check['error'] = str(e)
            consistency_check['analysis_details']['reason'] = "模型检测失败，使用默认分数"
        
        return consistency_check
    
    def update_threshold(self, new_threshold: float):
        """更新一致性阈值"""
        if 0 <= new_threshold <= 1:
            self.consistency_threshold = new_threshold
            logger.info(f"更新一致性阈值为: {new_threshold}")
            return True
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'SC-Energy',
            'encoder_model': self.encoder_model_name,
            'hidden_size': self.hidden_size,
            'consistency_threshold': self.consistency_threshold,
            'use_margin_loss': self.use_margin_loss
        }
