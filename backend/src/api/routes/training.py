"""
训练服务API路由
提供模型训练、分布式训练、训练监控等功能
"""

from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

# 使用绝对导入
from src.core.services.training_service import TrainingService
from src.core.services import model_manager

router = APIRouter(prefix="/training", tags=["training"])

# 创建训练服务实例
training_service = TrainingService(model_manager)

# 训练任务状态跟踪
_training_tasks: Dict[str, Dict[str, Any]] = {}


class TrainingRequest(BaseModel):
    """训练请求"""
    model_id: str
    dataset: Dict[str, Any]  # 训练数据集
    config: Optional[Dict[str, Any]] = None
    distributed: Optional[bool] = False


class TrainingResponse(BaseModel):
    """训练响应"""
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    model_id: str
    message: str
    progress: Optional[float] = 0.0
    metrics: Optional[Dict[str, Any]] = None


class TrainingConfig(BaseModel):
    """训练配置"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    save_checkpoint_frequency: int = 1000


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """开始训练任务"""
    try:
        # 检查模型是否存在
        metadata = model_manager.get_model_info(request.model_id)
        
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # 初始化任务状态
        _training_tasks[task_id] = {
            "status": "pending",
            "model_id": request.model_id,
            "progress": 0.0,
            "message": "任务已创建，等待执行",
            "metrics": None
        }
        
        # 在后台执行训练任务
        background_tasks.add_task(
            _execute_training_task,
            task_id,
            request.model_id,
            request.dataset,
            request.config or {},
            request.distributed
        )
        
        return TrainingResponse(
            task_id=task_id,
            status="pending",
            model_id=request.model_id,
            message="训练任务已开始",
            progress=0.0
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动训练任务失败: {str(e)}"
        )


async def _execute_training_task(
    task_id: str,
    model_id: str,
    dataset: Dict[str, Any],
    config: Dict[str, Any],
    distributed: bool
):
    """执行训练任务（后台）"""
    try:
        # 更新任务状态为运行中
        _training_tasks[task_id]["status"] = "running"
        _training_tasks[task_id]["message"] = "训练进行中"
        
        # 获取模型元数据
        metadata = model_manager.get_model_info(model_id)
        
        # 准备数据集
        train_data, train_labels = _prepare_dataset(dataset["train"], metadata.model_type)
        val_data, val_labels = None, None
        
        if "val" in dataset:
            val_data, val_labels = _prepare_dataset(dataset["val"], metadata.model_type)
        
        # 执行训练
        if distributed:
            # 分布式训练
            metrics = training_service.distributed_training(
                model_id=model_id,
                dataset={
                    "train": train_data,
                    "train_labels": train_labels,
                    "val": val_data,
                    "val_labels": val_labels
                },
                config=config
            )
        else:
            # 单设备训练
            if metadata.model_type == "transformer":
                metrics = training_service.train_transformer_model(
                    model_id=model_id,
                    train_data=train_data,
                    train_labels=train_labels,
                    val_data=val_data,
                    val_labels=val_labels,
                    config=config
                )
            elif metadata.model_type == "vision":
                metrics = training_service.train_vision_model(
                    model_id=model_id,
                    train_images=train_data,
                    train_labels=train_labels,
                    val_images=val_data,
                    val_labels=val_labels,
                    config=config
                )
            else:
                raise ValueError(f"不支持的训练模型类型: {metadata.model_type}")
        
        # 更新任务状态为完成
        _training_tasks[task_id]["status"] = "completed"
        _training_tasks[task_id]["progress"] = 100.0
        _training_tasks[task_id]["message"] = "训练完成"
        _training_tasks[task_id]["metrics"] = metrics
        
    except Exception as e:
        # 更新任务状态为失败
        _training_tasks[task_id]["status"] = "failed"
        _training_tasks[task_id]["message"] = f"训练失败: {str(e)}"


def _prepare_dataset(dataset: Dict[str, Any], model_type: str) -> tuple:
    """准备数据集"""
    
    if model_type == "transformer":
        # 文本数据
        data = jnp.array(dataset["data"], dtype=jnp.int32)
        labels = jnp.array(dataset["labels"], dtype=jnp.int32)
        
    elif model_type == "vision":
        # 图像数据
        data = jnp.array(dataset["images"], dtype=jnp.float32)
        labels = jnp.array(dataset["labels"], dtype=jnp.int32)
        
        # 确保正确的形状: [batch, height, width, channels]
        if len(data.shape) == 3:
            data = data[..., None]  # 添加通道维度
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return data, labels


@router.get("/status/{task_id}", response_model=TrainingResponse)
async def get_training_status(task_id: str):
    """获取训练任务状态"""
    
    if task_id not in _training_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务 {task_id} 不存在"
        )
    
    task_info = _training_tasks[task_id]
    
    return TrainingResponse(
        task_id=task_id,
        status=task_info["status"],
        model_id=task_info["model_id"],
        message=task_info["message"],
        progress=task_info["progress"],
        metrics=task_info.get("metrics")
    )


@router.get("/tasks")
async def list_training_tasks():
    """获取所有训练任务列表"""
    
    tasks = []
    for task_id, task_info in _training_tasks.items():
        tasks.append({
            "task_id": task_id,
            "status": task_info["status"],
            "model_id": task_info["model_id"],
            "progress": task_info["progress"],
            "message": task_info["message"]
        })
    
    return {
        "tasks": tasks,
        "total_count": len(tasks)
    }


@router.delete("/{task_id}")
async def cancel_training_task(task_id: str):
    """取消训练任务"""
    
    if task_id not in _training_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务 {task_id} 不存在"
        )
    
    task_info = _training_tasks[task_id]
    
    if task_info["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法取消状态为 {task_info['status']} 的任务"
        )
    
    # 更新任务状态为取消
    _training_tasks[task_id]["status"] = "cancelled"
    _training_tasks[task_id]["message"] = "任务已被取消"
    
    return {"message": f"训练任务 {task_id} 已取消"}


@router.get("/config/default")
async def get_default_training_config():
    """获取默认训练配置"""
    
    return training_service.default_config


@router.put("/config/default")
async def update_default_training_config(config: TrainingConfig):
    """更新默认训练配置"""
    
    training_service.default_config.update(config.dict())
    
    return {"message": "默认训练配置已更新", "config": training_service.default_config}