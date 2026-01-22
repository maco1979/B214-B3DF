"""
分布式DCNN API路由
提供联邦学习、边缘计算和区块链奖励的完整接口
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.distributed_dcnn.core import DistributedDCNN, DCNNConfig, DCNNArchitecture
from src.distributed_dcnn.federated_edge import (
    FederatedDCNNConfig, FederatedDCNNCoordinator, TrainingMode, EdgeInferenceService
)
from src.distributed_dcnn.blockchain_rewards import DCNNRewardSystem


router = APIRouter(prefix="/distributed-dcnn", tags=["分布式DCNN"])


# 全局实例（在实际应用中应该使用依赖注入）
_coordinator: Optional[FederatedDCNNCoordinator] = None
_reward_system: Optional[DCNNRewardSystem] = None
_inference_service: Optional[EdgeInferenceService] = None


# Pydantic模型
class DCNNConfigRequest(BaseModel):
    """DCNN配置请求"""
    architecture: DCNNArchitecture = DCNNArchitecture.STANDARD
    num_classes: int = Field(1000, ge=1, le=10000)
    input_shape: List[int] = Field([224, 224, 3])
    
    # 卷积层配置
    conv_channels: Optional[List[int]] = None
    kernel_sizes: Optional[List[List[int]]] = None
    
    # 正则化
    dropout_rate: float = Field(0.5, ge=0.0, le=1.0)
    use_batch_norm: bool = True


class FederatedConfigRequest(BaseModel):
    """联邦学习配置请求"""
    training_mode: TrainingMode = TrainingMode.FEDERATED
    client_fraction: float = Field(0.1, ge=0.01, le=1.0)
    rounds: int = Field(100, ge=1, le=1000)
    local_epochs: int = Field(5, ge=1, le=50)
    batch_size: int = Field(32, ge=1, le=256)
    
    # 隐私保护
    differential_privacy: bool = True
    epsilon: float = Field(1.0, ge=0.1, le=10.0)
    clip_norm: float = Field(1.0, ge=0.1, le=10.0)
    
    # 边缘计算
    model_compression: bool = True
    compression_ratio: float = Field(0.5, ge=0.1, le=0.9)


class TrainingStartRequest(BaseModel):
    """训练启动请求"""
    dcnn_config: DCNNConfigRequest
    federated_config: FederatedConfigRequest
    model_id: str = "global_dcnn"


class InferenceRequest(BaseModel):
    """推理请求"""
    input_data: List[List[List[float]]]  # 图像数据
    model_id: str = "global"
    edge_node: Optional[str] = None


class ParticipantContribution(BaseModel):
    """参与者贡献"""
    participant_id: str
    data_size: int = 0
    compute_time: float = 0.0
    accuracy_gain: float = 0.0
    data_quality: float = Field(1.0, ge=0.0, le=1.0)
    compute_efficiency: float = Field(1.0, ge=0.0, le=1.0)


class FederatedRoundRequest(BaseModel):
    """联邦学习轮次请求"""
    round_id: str
    model_id: str
    participants: List[str]
    participant_contributions: Dict[str, ParticipantContribution]
    round_number: int
    aggregated_model_hash: str


# 响应模型
class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    status: str
    current_round: Optional[int]
    total_rounds: int
    metrics: Dict[str, Any]
    edge_nodes_status: Dict[str, Any]
    federated_status: Dict[str, Any]


class InferenceResponse(BaseModel):
    """推理响应"""
    success: bool
    predictions: Optional[List[Any]]
    inference_time: float
    edge_node: str
    model_id: str
    error: Optional[str] = None


class SystemStatisticsResponse(BaseModel):
    """系统统计响应"""
    registered_models: int
    total_rewards_distributed: float
    total_participants: int
    average_reward_per_participant: float
    reward_history_summary: Dict[str, Any]


# 工具函数
def get_coordinator() -> FederatedDCNNCoordinator:
    """获取协调器实例"""
    global _coordinator
    if _coordinator is None:
        raise HTTPException(status_code=500, detail="分布式DCNN系统未初始化")
    return _coordinator


def get_reward_system() -> DCNNRewardSystem:
    """获取奖励系统实例"""
    global _reward_system
    if _reward_system is None:
        raise HTTPException(status_code=500, detail="奖励系统未初始化")
    return _reward_system


def get_inference_service() -> EdgeInferenceService:
    """获取推理服务实例"""
    global _inference_service
    if _inference_service is None:
        raise HTTPException(status_code=500, detail="推理服务未初始化")
    return _inference_service


# API路由
@router.post("/initialize", summary="初始化分布式DCNN系统")
async def initialize_system(request: TrainingStartRequest):
    """初始化分布式DCNN系统"""
    try:
        global _coordinator, _reward_system, _inference_service
        
        # 创建DCNN配置
        dcnn_config = DCNNConfig(
            architecture=request.dcnn_config.architecture,
            num_classes=request.dcnn_config.num_classes,
            input_shape=tuple(request.dcnn_config.input_shape),
            conv_channels=request.dcnn_config.conv_channels,
            dropout_rate=request.dcnn_config.dropout_rate,
            use_batch_norm=request.dcnn_config.use_batch_norm
        )
        
        # 创建联邦配置
        federated_config = FederatedDCNNConfig(
            dcnn_config=dcnn_config,
            training_mode=request.federated_config.training_mode,
            client_fraction=request.federated_config.client_fraction,
            rounds=request.federated_config.rounds,
            local_epochs=request.federated_config.local_epochs,
            batch_size=request.federated_config.batch_size,
            differential_privacy=request.federated_config.differential_privacy,
            epsilon=request.federated_config.epsilon,
            clip_norm=request.federated_config.clip_norm,
            model_compression=request.federated_config.model_compression,
            compression_ratio=request.federated_config.compression_ratio
        )
        
        # 初始化协调器
        _coordinator = FederatedDCNNCoordinator(federated_config)
        
        # 初始化系统
        success = await _coordinator.initialize_training()
        
        if not success:
            raise HTTPException(status_code=500, detail="系统初始化失败")
        
        # 初始化推理服务
        _inference_service = EdgeInferenceService(_coordinator)
        
        # 初始化奖励系统（简化实现，实际应用中需要区块链客户端）
        # _reward_system = DCNNRewardSystem(blockchain_manager)
        
        return {
            "success": True,
            "message": "分布式DCNN系统初始化成功",
            "model_id": request.model_id,
            "config": {
                "dcnn": dcnn_config.__dict__,
                "federated": federated_config.__dict__
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统初始化异常: {str(e)}")


@router.post("/training/start", summary="开始训练")
async def start_training(background_tasks: BackgroundTasks, total_rounds: Optional[int] = None):
    """开始分布式训练"""
    try:
        coordinator = get_coordinator()
        
        # 在后台启动训练
        if total_rounds is None:
            total_rounds = coordinator.config.rounds
        
        background_tasks.add_task(coordinator.run_training, total_rounds)
        
        return {
            "success": True,
            "message": f"分布式训练已启动，总轮次: {total_rounds}",
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练启动失败: {str(e)}")


@router.get("/training/status", response_model=TrainingStatusResponse, summary="获取训练状态")
async def get_training_status():
    """获取当前训练状态"""
    try:
        coordinator = get_coordinator()
        status = coordinator.get_training_status()
        
        return TrainingStatusResponse(**status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")


@router.post("/inference/predict", response_model=InferenceResponse, summary="执行推理")
async def predict(request: InferenceRequest):
    """执行分布式推理"""
    try:
        inference_service = get_inference_service()
        
        # 转换输入数据
        input_array = request.input_data
        
        # 执行推理
        result = await inference_service.inference_request(
            input_data=input_array,
            model_id=request.model_id,
            edge_node=request.edge_node
        )
        
        return InferenceResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理执行失败: {str(e)}")


@router.post("/federated/round", summary="处理联邦学习轮次")
async def process_federated_round(request: FederatedRoundRequest):
    """处理联邦学习轮次并分配奖励"""
    try:
        reward_system = get_reward_system()
        
        # 转换请求数据
        round_data = {
            "round_id": request.round_id,
            "model_id": request.model_id,
            "participants": request.participants,
            "participant_contributions": {
                p_id: {
                    "data_size": contrib.data_size,
                    "compute_time": contrib.compute_time,
                    "accuracy_gain": contrib.accuracy_gain,
                    "data_quality": contrib.data_quality,
                    "compute_efficiency": contrib.compute_efficiency
                }
                for p_id, contrib in request.participant_contributions.items()
            },
            "round_number": request.round_number,
            "aggregated_model_hash": request.aggregated_model_hash
        }
        
        # 处理轮次
        result = await reward_system.process_federated_round(round_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理联邦学习轮次失败: {str(e)}")


@router.get("/rewards/participant/{participant_id}", summary="获取参与者奖励信息")
async def get_participant_rewards(participant_id: str):
    """获取特定参与者的奖励信息"""
    try:
        reward_system = get_reward_system()
        rewards_info = await reward_system.get_participant_rewards(participant_id)
        
        return rewards_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取参与者奖励失败: {str(e)}")


@router.get("/statistics", response_model=SystemStatisticsResponse, summary="获取系统统计")
async def get_system_statistics():
    """获取系统统计信息"""
    try:
        reward_system = get_reward_system()
        statistics = await reward_system.get_system_statistics()
        
        if "error" in statistics:
            raise HTTPException(status_code=500, detail=statistics["error"])
        
        return SystemStatisticsResponse(**statistics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统统计失败: {str(e)}")


@router.get("/models/registered", summary="获取已注册模型")
async def get_registered_models():
    """获取所有已注册的模型"""
    try:
        coordinator = get_coordinator()
        
        # 在实际系统中，这里应该从模型管理器获取
        models = {
            "global_dcnn": {
                "architecture": coordinator.config.dcnn_config.architecture.value,
                "num_classes": coordinator.config.dcnn_config.num_classes,
                "input_shape": coordinator.config.dcnn_config.input_shape
            }
        }
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取注册模型失败: {str(e)}")


@router.get("/edge/nodes", summary="获取边缘节点状态")
async def get_edge_nodes():
    """获取所有边缘节点状态"""
    try:
        coordinator = get_coordinator()
        edge_status = coordinator.edge_manager.get_system_overview()
        
        return {
            "success": True,
            "edge_nodes": edge_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取边缘节点失败: {str(e)}")


@router.post("/edge/nodes/register", summary="注册边缘节点")
async def register_edge_node(node_id: str, address: str, capabilities: Dict[str, Any]):
    """注册新的边缘节点"""
    try:
        coordinator = get_coordinator()
        
        success = await coordinator.edge_manager.register_edge_node(
            node_id, address, capabilities
        )
        
        if success:
            return {
                "success": True,
                "message": f"边缘节点 {node_id} 注册成功",
                "node_id": node_id
            }
        else:
            raise HTTPException(status_code=400, detail=f"边缘节点 {node_id} 注册失败")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册边缘节点失败: {str(e)}")


@router.get("/health", summary="系统健康检查")
async def health_check():
    """分布式DCNN系统健康检查"""
    try:
        coordinator = get_coordinator()
        
        # 检查各组件状态
        coordinator_status = coordinator.get_training_status()
        edge_status = coordinator.edge_manager.get_system_overview()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "coordinator": "running" if coordinator_status["current_round"] is not None else "idle",
                "edge_manager": "running" if edge_status["online_nodes"] > 0 else "idle",
                "federated_learning": "active" if coordinator_status["federated_status"]["active_clients"] > 0 else "inactive"
            },
            "metrics": {
                "total_rounds": coordinator_status["total_rounds"],
                "online_nodes": edge_status["online_nodes"],
                "active_clients": coordinator_status["federated_status"]["active_clients"]
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# 启动时初始化（简化实现）
@router.on_event("startup")
async def startup_event():
    """应用启动时初始化默认配置"""
    try:
        # 创建默认配置
        dcnn_config = DCNNConfig()
        federated_config = FederatedDCNNConfig(dcnn_config=dcnn_config)
        
        global _coordinator, _inference_service
        _coordinator = FederatedDCNNCoordinator(federated_config)
        _inference_service = EdgeInferenceService(_coordinator)
        
        # 初始化系统（但不启动训练）
        await _coordinator.initialize_training()
        
        print("分布式DCNN系统初始化完成")
        
    except Exception as e:
        print(f"分布式DCNN系统初始化失败: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    try:
        global _coordinator
        if _coordinator:
            # 关闭边缘管理器
            await _coordinator.edge_manager.shutdown()
            print("分布式DCNN系统已关闭")
    except Exception as e:
        print(f"分布式DCNN系统关闭异常: {e}")