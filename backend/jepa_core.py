import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAEncoder(nn.Module):
    """
    JEPA编码器组件，用于将输入数据编码为抽象表示
    """
    
    def __init__(self, input_dim, embedding_dim, hidden_dims=[64, 32]):
        """
        初始化JEPA编码器
        
        Args:
            input_dim: 输入维度
            embedding_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(JEPAEncoder, self).__init__()
        
        # 构建编码器网络
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # 输出嵌入层
        layers.append(nn.Linear(current_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, input_dim)
            
        Returns:
            embedding: 抽象嵌入表示 (batch_size, embedding_dim)
        """
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """
    JEPA预测器组件，用于根据当前嵌入预测未来嵌入
    """
    
    def __init__(self, embedding_dim, prediction_horizon, hidden_dims=[32, 64]):
        """
        初始化JEPA预测器
        
        Args:
            embedding_dim: 嵌入维度
            prediction_horizon: 预测时域
            hidden_dims: 隐藏层维度列表
        """
        super(JEPAPredictor, self).__init__()
        
        self.prediction_horizon = prediction_horizon
        
        # 构建预测器网络
        layers = []
        current_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # 输出层：预测未来多个时间步的嵌入
        layers.append(nn.Linear(current_dim, embedding_dim * prediction_horizon))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, embedding):
        """
        前向传播
        
        Args:
            embedding: 当前嵌入表示 (batch_size, embedding_dim)
            
        Returns:
            future_embeddings: 未来嵌入预测 (batch_size, prediction_horizon, embedding_dim)
        """
        predictions = self.predictor(embedding)
        return predictions.view(-1, self.prediction_horizon, embedding_dim)


class JEPADecoder(nn.Module):
    """
    JEPA解码器组件，用于将嵌入解码为原始输出空间
    """
    
    def __init__(self, embedding_dim, output_dim, hidden_dims=[32, 64]):
        """
        初始化JEPA解码器
        
        Args:
            embedding_dim: 嵌入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表
        """
        super(JEPADecoder, self).__init__()
        
        # 构建解码器网络
        layers = []
        current_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, embedding):
        """
        前向传播
        
        Args:
            embedding: 嵌入表示 (batch_size, embedding_dim) 或 (batch_size, prediction_horizon, embedding_dim)
            
        Returns:
            output: 解码输出
        """
        # 处理时间维度
        if len(embedding.shape) == 3:
            batch_size, horizon, embedding_dim = embedding.shape
            embedding = embedding.view(-1, embedding_dim)
            output = self.decoder(embedding)
            return output.view(batch_size, horizon, -1)
        else:
            return self.decoder(embedding)


class EnergyFunction(nn.Module):
    """
    能量函数，用于评估预测嵌入与实际嵌入之间的一致性
    """
    
    def __init__(self, embedding_dim, temperature=0.1):
        """
        初始化能量函数
        
        Args:
            embedding_dim: 嵌入维度
            temperature: 温度参数
        """
        super(EnergyFunction, self).__init__()
        
        self.temperature = temperature
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, predicted_embedding, actual_embedding):
        """
        计算能量值
        
        Args:
            predicted_embedding: 预测的嵌入表示
            actual_embedding: 实际的嵌入表示
            
        Returns:
            energy: 能量值
        """
        # 投影到同一空间
        predicted_proj = self.projection(predicted_embedding)
        actual_proj = self.projection(actual_embedding)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(predicted_proj, actual_proj, dim=-1)
        
        # 计算能量值
        energy = -similarity / self.temperature
        
        return energy


class JEPA(nn.Module):
    """
    完整的JEPA架构实现
    """
    
    def __init__(self, input_dim, embedding_dim, output_dim, prediction_horizon):
        """
        初始化JEPA
        
        Args:
            input_dim: 输入维度
            embedding_dim: 嵌入维度
            output_dim: 输出维度
            prediction_horizon: 预测时域
        """
        super(JEPA, self).__init__()
        
        # 编码器
        self.encoder = JEPAEncoder(input_dim, embedding_dim)
        
        # 预测器
        self.predictor = JEPAPredictor(embedding_dim, prediction_horizon)
        
        # 解码器
        self.decoder = JEPADecoder(embedding_dim, output_dim)
        
        # 能量函数
        self.energy_function = EnergyFunction(embedding_dim)
    
    def encode(self, x):
        """
        编码输入数据
        
        Args:
            x: 输入数据 (batch_size, input_dim)
            
        Returns:
            embedding: 嵌入表示 (batch_size, embedding_dim)
        """
        return self.encoder(x)
    
    def predict_embedding(self, embedding):
        """
        预测未来嵌入
        
        Args:
            embedding: 当前嵌入 (batch_size, embedding_dim)
            
        Returns:
            future_embeddings: 未来嵌入预测 (batch_size, prediction_horizon, embedding_dim)
        """
        return self.predictor(embedding)
    
    def decode(self, embedding):
        """
        解码嵌入到输出空间
        
        Args:
            embedding: 嵌入表示 (batch_size, embedding_dim) 或 (batch_size, prediction_horizon, embedding_dim)
            
        Returns:
            output: 解码输出
        """
        return self.decoder(embedding)
    
    def compute_energy(self, predicted_embedding, actual_embedding):
        """
        计算能量值
        
        Args:
            predicted_embedding: 预测嵌入
            actual_embedding: 实际嵌入
            
        Returns:
            energy: 能量值
        """
        return self.energy_function(predicted_embedding, actual_embedding)
    
    def forward(self, x, future_x=None):
        """
        完整的前向传播
        
        Args:
            x: 当前输入数据 (batch_size, input_dim)
            future_x: 未来输入数据 (batch_size, prediction_horizon, input_dim)，用于训练
            
        Returns:
            result: 包含预测、嵌入等的字典
        """
        # 编码当前输入
        current_embedding = self.encode(x)
        
        # 预测未来嵌入
        predicted_embeddings = self.predict_embedding(current_embedding)
        
        # 解码为输出预测
        output_prediction = self.decode(predicted_embeddings)
        
        result = {
            'current_embedding': current_embedding,
            'predicted_embeddings': predicted_embeddings,
            'output_prediction': output_prediction
        }
        
        # 如果提供了未来数据，计算能量值
        if future_x is not None:
            # 编码未来数据
            batch_size, horizon, _ = future_x.shape
            future_x_flat = future_x.view(-1, future_x.shape[-1])
            actual_embeddings = self.encode(future_x_flat).view(batch_size, horizon, -1)
            
            # 计算能量值
            energy = self.compute_energy(predicted_embeddings, actual_embeddings)
            result['actual_embeddings'] = actual_embeddings
            result['energy'] = energy
        
        return result


class JEPATrainer:
    """
    JEPA训练器
    """
    
    def __init__(self, jepa_model, learning_rate=0.001):
        """
        初始化训练器
        
        Args:
            jepa_model: JEPA模型实例
            learning_rate: 学习率
        """
        self.jepa_model = jepa_model
        self.optimizer = torch.optim.Adam(jepa_model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.jepa_model.to(self.device)
    
    def train_step(self, x, future_x):
        """
        单步训练
        
        Args:
            x: 当前输入数据 (batch_size, input_dim)
            future_x: 未来输入数据 (batch_size, prediction_horizon, input_dim)
            
        Returns:
            loss: 训练损失
        """
        # 将数据移到设备上
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        future_x = torch.tensor(future_x, dtype=torch.float32).to(self.device)
        
        # 前向传播
        result = self.jepa_model(x, future_x)
        
        # 计算损失（能量值的平均值）
        loss = result['energy'].mean()
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, x):
        """
        使用训练好的模型进行预测
        
        Args:
            x: 输入数据 (input_dim) 或 (batch_size, input_dim)
            
        Returns:
            prediction: 输出预测
        """
        # 确保输入是二维的
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # 转换为张量并移到设备上
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        # 关闭梯度计算
        with torch.no_grad():
            result = self.jepa_model(x)
        
        # 返回预测结果
        return result['output_prediction'].cpu().numpy()


# 示例用法
if __name__ == "__main__":
    # 测试JEPA组件
    input_dim = 5      # 输入维度
    embedding_dim = 10 # 嵌入维度
    output_dim = 1     # 输出维度
    prediction_horizon = 5  # 预测时域
    
    # 创建JEPA模型
    jepa = JEPA(input_dim, embedding_dim, output_dim, prediction_horizon)
    
    # 创建训练器
    trainer = JEPATrainer(jepa)
    
    # 模拟训练数据
    batch_size = 32
    x = np.random.rand(batch_size, input_dim)
    future_x = np.random.rand(batch_size, prediction_horizon, input_dim)
    
    # 训练一步
    loss = trainer.train_step(x, future_x)
    print(f"训练损失: {loss:.4f}")
    
    # 模拟预测
    test_x = np.random.rand(1, input_dim)
    prediction = trainer.predict(test_x)
    print(f"预测结果: {prediction.shape}")
