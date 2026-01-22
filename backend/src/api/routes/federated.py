"""
联邦学习API路由
提供联邦学习相关的REST API接口
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import json
from datetime import datetime

from src.federated.federated_learning import FederatedLearningServer, FederatedLearningClient

router = APIRouter(prefix="/federated", tags=["联邦学习"])

# 全局联邦学习服务器实例
federated_server = None


def get_federated_server() -> FederatedLearningServer:
    """获取或创建联邦学习服务器实例"""
    global federated_server
    if federated_server is None:
        # 默认模型架构
        model_architecture = {
            "name": "default_model",
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ]
        }
        federated_server = FederatedLearningServer(model_architecture)
    return federated_server


@router.post("/clients/register")
async def register_client(client_info: Dict[str, Any]) -> Dict[str, Any]:
    """注册联邦学习客户端"""
    server = get_federated_server()
    
    client_id = client_info.get('client_id')
    if not client_id:
        return {
            "success": False,
            "error": "客户端ID不能为空"
        }
    
    success = server.register_client(client_id, client_info)
    
    if not success:
        return {
            "success": False,
            "error": "客户端已存在"
        }
    
    return {
        "success": True,
        "client_id": client_id,
        "message": "客户端注册成功"
    }


@router.get("/clients")
async def get_clients() -> Dict[str, Any]:
    """获取所有客户端信息"""
    server = get_federated_server()
    
    # 将客户端字典转换为数组格式
    clients_list = [{"client_id": client_id, **client_info} for client_id, client_info in server.clients.items()]
    
    return {
        "success": True,
        "data": clients_list,
        "total_count": len(server.clients)
    }


@router.post("/rounds/start")
async def start_training_round(round_config: Dict[str, Any]) -> Dict[str, Any]:
    """开始新的训练轮次"""
    server = get_federated_server()
    
    try:
        round_info = server.start_training_round(round_config)
        
        return {
            "success": True,
            "round_info": round_info,
            "message": "训练轮次已开始"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练轮次失败: {str(e)}")


@router.post("/rounds/{round_id}/updates")
async def submit_client_update(
    round_id: str,
    update_data: Dict[str, Any]
) -> Dict[str, Any]:
    """提交客户端模型更新"""
    server = get_federated_server()
    
    client_id = update_data.get('client_id')
    if not client_id:
        raise HTTPException(status_code=400, detail="客户端ID不能为空")
    
    update = update_data.get('update')
    if not update:
        raise HTTPException(status_code=400, detail="更新数据不能为空")
    
    success = server.receive_client_update(client_id, round_id, update)
    
    if not success:
        raise HTTPException(status_code=400, detail="提交更新失败")
    
    return {
        "success": True,
        "message": "模型更新已提交"
    }


@router.post("/rounds/{round_id}/aggregate")
async def aggregate_round_updates(round_id: str) -> Dict[str, Any]:
    """聚合指定轮次的客户端更新"""
    server = get_federated_server()
    
    try:
        success = server.aggregate_updates(round_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="聚合更新失败")
        
        return {
            "success": True,
            "message": f"轮次 {round_id} 更新聚合完成",
            "rounds_completed": server.rounds_completed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚合失败: {str(e)}")


@router.get("/rounds")
async def get_training_rounds() -> Dict[str, Any]:
    """获取所有训练轮次信息"""
    server = get_federated_server()
    
    return {
        "success": True,
        "data": server.training_history,
        "total_rounds": server.rounds_completed
    }


@router.get("/model")
async def get_global_model() -> Dict[str, Any]:
    """获取当前全局模型"""
    server = get_federated_server()
    
    return {
        "success": True,
        "model": server.global_model,
        "metadata": {
            "rounds_completed": server.rounds_completed,
            "last_updated": datetime.now().isoformat()
        }
    }


@router.get("/status")
async def get_server_status() -> Dict[str, Any]:
    """获取联邦学习服务器状态"""
    server = get_federated_server()
    
    status = server.get_server_status()
    
    return {
        "success": True,
        "data": status
    }


@router.post("/privacy/config")
async def update_privacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """更新差分隐私配置"""
    server = get_federated_server()
    
    epsilon = config.get('epsilon')
    delta = config.get('delta')
    enabled = config.get('enabled')
    
    if epsilon is not None:
        server.dp_mechanism.epsilon = epsilon
    
    if delta is not None:
        server.dp_mechanism.delta = delta
    
    if enabled is not None:
        server.dp_enabled = enabled
    
    return {
        "success": True,
        "message": "隐私配置已更新",
        "current_config": {
            "epsilon": server.dp_mechanism.epsilon,
            "delta": server.dp_mechanism.delta,
            "enabled": server.dp_enabled
        }
    }


@router.get("/privacy/status")
async def get_privacy_status() -> Dict[str, Any]:
    """获取隐私保护状态"""
    server = get_federated_server()
    
    # 计算隐私消耗
    privacy_spent = server.dp_mechanism.compute_privacy_spent(
        steps=server.rounds_completed * 100,  # 假设每轮100步
        batch_size=32,
        dataset_size=10000
    )
    
    return {
        "success": True,
        "data": {
            "enabled": server.dp_enabled,
            "epsilon": server.dp_mechanism.epsilon,
            "delta": server.dp_mechanism.delta,
            "privacy_spent": privacy_spent,
            "rounds_with_privacy": server.rounds_completed
        }
    }


# 客户端模拟端点
@router.post("/clients/{client_id}/train")
async def client_local_training(
    client_id: str,
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """客户端本地训练模拟"""
    server = get_federated_server()
    
    if client_id not in server.clients:
        raise HTTPException(status_code=404, detail="客户端不存在")
    
    # 模拟客户端训练
    # 在实际系统中，这应该在客户端执行
    client_data = []  # 模拟客户端数据
    client = FederatedLearningClient(client_id, client_data)
    
    # 使用当前全局模型初始化
    global_model = server._prepare_client_model()
    client.initialize_with_global_model(global_model)
    
    # 执行本地训练
    local_update = client.local_training(training_config)
    
    # 提交更新到服务器
    success = server.receive_client_update(
        client_id, 
        training_config.get('round_id', 'current'), 
        local_update
    )
    
    return {
        "success": success,
        "update_info": local_update,
        "message": "本地训练完成" if success else "提交更新失败"
    }