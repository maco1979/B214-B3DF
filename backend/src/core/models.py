"""
核心AI模型定义
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """AI任务状态枚举"""
    PENDING = "pending"  # 任务待处理
    RUNNING = "running"  # 任务执行中
    COMPLETED = "completed"  # 任务已完成
    FAILED = "failed"  # 任务执行失败
    CANCELLED = "cancelled"  # 任务已取消


class AITask(BaseModel):
    """AI任务模型"""
    task_id: str = Field(..., description="任务唯一标识符")
    task_type: str = Field(..., description="任务类型")
    params: Dict[str, Any] = Field(default_factory=dict, description="任务参数")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    result: Optional[Dict[str, Any]] = Field(default=None, description="任务结果")
    error: Optional[str] = Field(default=None, description="任务错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="任务元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="任务创建时间")
    started_at: Optional[datetime] = Field(default=None, description="任务开始执行时间")
    completed_at: Optional[datetime] = Field(default=None, description="任务完成时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EnterpriseAIConfig(BaseModel):
    """企业级AI配置"""
    # 模型存储路径
    model_storage_path: str = Field(default="./models", description="模型存储路径")
    
    # 模型池配置
    max_model_pool_size: int = Field(default=10, description="模型池最大容量")
    model_ttl: int = Field(default=3600, description="模型在池中保留的时间（秒）")
    
    # 任务配置
    max_concurrent_tasks: int = Field(default=5, description="最大并发任务数")
    task_timeout: int = Field(default=3600, description="任务超时时间（秒）")
    
    # 推理配置
    default_max_length: int = Field(default=100, description="文本生成默认最大长度")
    default_temperature: float = Field(default=1.0, description="文本生成默认温度参数")
    default_batch_size: int = Field(default=16, description="批量推理默认批次大小")
    
    # 训练配置
    default_learning_rate: float = Field(default=1e-4, description="默认学习率")
    default_batch_size_training: int = Field(default=32, description="默认训练批次大小")
    default_num_epochs: int = Field(default=10, description="默认训练轮数")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    
    class Config:
        extra = "allow"


class InferenceResult(BaseModel):
    """AI推理结果模型"""
    predictions: List[Dict[str, Any]] | Dict[str, Any] | str = Field(..., description="推理预测结果")
    confidence: Optional[float] = Field(default=None, description="推理置信度")
    processing_time: float = Field(default=0.0, description="推理处理时间（秒）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="推理元数据")
    
    class Config:
        extra = "allow"


class TrainingMetrics(BaseModel):
    """模型训练指标"""
    loss: float = Field(default=0.0, description="训练损失")
    accuracy: float = Field(default=0.0, description="训练准确率")
    val_loss: float = Field(default=0.0, description="验证集损失")
    val_accuracy: float = Field(default=0.0, description="验证集准确率")
    learning_rate: float = Field(default=0.0, description="学习率")
    epoch: int = Field(default=0, description="训练轮数")
    step: int = Field(default=0, description="训练步数")
    
    class Config:
        extra = "allow"


class ModelMetadata(BaseModel):
    """模型元数据"""
    model_id: str = Field(..., description="模型唯一标识符")
    name: str = Field(..., description="模型名称")
    type: str = Field(..., description="模型类型")
    framework: str = Field(..., description="模型框架")
    version: str = Field(default="1.0.0", description="模型版本")
    status: str = Field(default="ready", description="模型状态")
    is_enterprise_model: bool = Field(default=False, description="是否为企业级模型")
    created_at: datetime = Field(default_factory=datetime.now, description="模型创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="模型更新时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="模型扩展元数据")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="模型评估指标")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
