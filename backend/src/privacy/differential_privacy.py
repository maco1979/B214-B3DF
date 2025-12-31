"""
差分隐私模块
实现差分隐私算法，保护训练数据隐私
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional
import math


class DifferentialPrivacy:
    """差分隐私实现类"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        初始化差分隐私参数
        
        Args:
            epsilon: 隐私预算，值越小隐私保护越强
            delta: 失败概率，通常设置为很小的值
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def laplace_mechanism(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Laplace机制实现
        
        Args:
            data: 原始数据
            sensitivity: 查询敏感度
            
        Returns:
            添加Laplace噪声后的数据
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def gaussian_mechanism(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Gaussian机制实现
        
        Args:
            data: 原始数据
            sensitivity: 查询敏感度
            
        Returns:
            添加Gaussian噪声后的数据
        """
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def clip_gradients(self, gradients: List[np.ndarray], clip_norm: float) -> List[np.ndarray]:
        """
        梯度裁剪
        
        Args:
            gradients: 梯度列表
            clip_norm: 裁剪阈值
            
        Returns:
            裁剪后的梯度
        """
        total_norm = 0.0
        for grad in gradients:
            total_norm += jnp.sum(grad ** 2)
        total_norm = jnp.sqrt(total_norm)
        
        if total_norm > clip_norm:
            scale = clip_norm / total_norm
            gradients = [grad * scale for grad in gradients]
        
        return gradients
    
    def add_noise_to_gradients(self, gradients: List[np.ndarray], sensitivity: float) -> List[np.ndarray]:
        """
        向梯度添加噪声
        
        Args:
            gradients: 梯度列表
            sensitivity: 敏感度
            
        Returns:
            添加噪声后的梯度
        """
        noisy_gradients = []
        for grad in gradients:
            noisy_grad = self.gaussian_mechanism(grad, sensitivity)
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def dp_sgd_step(self, 
                   gradients: List[np.ndarray], 
                   clip_norm: float,
                   noise_multiplier: float) -> List[np.ndarray]:
        """
        差分隐私SGD步骤
        
        Args:
            gradients: 梯度列表
            clip_norm: 梯度裁剪阈值
            noise_multiplier: 噪声乘子
            
        Returns:
            差分隐私处理后的梯度
        """
        # 1. 梯度裁剪
        clipped_gradients = self.clip_gradients(gradients, clip_norm)
        
        # 2. 添加噪声
        sensitivity = clip_norm
        noisy_gradients = self.add_noise_to_gradients(clipped_gradients, sensitivity * noise_multiplier)
        
        return noisy_gradients
    
    def compute_privacy_spent(self, steps: int, batch_size: int, dataset_size: int) -> Dict[str, float]:
        """
        计算隐私消耗
        
        Args:
            steps: 训练步数
            batch_size: 批次大小
            dataset_size: 数据集大小
            
        Returns:
            隐私消耗信息
        """
        # 使用高级组合定理计算隐私消耗
        q = batch_size / dataset_size  # 采样概率
        
        # 简化版的隐私消耗计算
        epsilon_spent = self.epsilon * math.sqrt(2 * steps * math.log(1 / self.delta)) * q
        
        return {
            'epsilon_spent': epsilon_spent,
            'delta': self.delta,
            'steps': steps,
            'remaining_epsilon': max(0, self.epsilon - epsilon_spent)
        }


class PrivacyAccountant:
    """隐私会计器，跟踪隐私消耗"""
    
    def __init__(self, target_epsilon: float, target_delta: float):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.epsilon_spent = 0.0
        self.steps_completed = 0
    
    def add_step(self, epsilon_cost: float):
        """添加一步的隐私消耗"""
        self.epsilon_spent += epsilon_cost
        self.steps_completed += 1
    
    def can_continue(self) -> bool:
        """检查是否可以继续训练"""
        return self.epsilon_spent < self.target_epsilon
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'epsilon_spent': self.epsilon_spent,
            'target_epsilon': self.target_epsilon,
            'remaining_epsilon': self.target_epsilon - self.epsilon_spent,
            'steps_completed': self.steps_completed,
            'can_continue': self.can_continue()
        }