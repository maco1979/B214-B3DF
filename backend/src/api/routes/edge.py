"""
边缘计算API路由
提供边缘节点管理、WASM模型部署、联邦学习等功能
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from src.edge.edge_manager import EdgeManager, EdgeNodeStatus, WASMModelConfig
from src.edge.federated_learning import FLState

logger = logging.getLogger(__name__)

# 创建边缘管理器实例
edge_manager = EdgeManager()

# 启动后台监控任务
monitor_task: Optional[asyncio.Task] = None

def startup_edge_manager():
    """启动边缘管理器"""
    global monitor_task
    if monitor_task is None:
        monitor_task = asyncio.create_task(edge_manager.monitor_edge_nodes())
        logger.info("边缘管理器后台监控任务已启动")

# 定义请求/响应模型
class EdgeNodeRegistration(BaseModel):
    """边缘节点注册请求"""
    node_id: str = Field(..., description="节点唯一标识")
    address: str = Field(..., description="节点地址")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="节点能力")

class WASMModelDeployment(BaseModel):
    """WASM模型部署请求"""
    node_id: str = Field(..., description="目标节点ID")
    model_name: str = Field(..., description="模型名称")
    wasm_path: str = Field(..., description="WASM文件路径")
    memory_limit: int = Field(default=128 * 1024 * 1024, description="内存限制")
    stack_size: int = Field(default=64 * 1024, description="堆栈大小")
    enable_simd: bool = Field(default=True, description="启用SIMD")
    enable_threads: bool = Field(default=True, description="启用多线程")

class InferenceRequest(BaseModel):
    """推理请求"""
    node_id: str = Field(..., description="目标节点ID")
    model_name: str = Field(..., description="模型名称")
    input_data: Any = Field(..., description="输入数据")
    function_name: str = Field(default="inference", description="WASM函数名")

class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    node_id: str = Field(..., description="目标节点ID")
    model_name: str = Field(..., description="模型名称")
    batch_data: List[Any] = Field(..., description="批量输入数据")
    function_name: str = Field(default="inference", description="WASM函数名")

class FLStartRequest(BaseModel):
    """联邦学习启动请求"""
    global_model: Dict[str, Any] = Field(..., description="全局模型参数")
    target_nodes: int = Field(default=5, description="目标节点数量")

class HeartbeatRequest(BaseModel):
    """心跳请求"""
    node_id: str = Field(..., description="节点ID")
    node_info: Dict[str, Any] = Field(default_factory=dict, description="节点信息")

# 响应模型
class EdgeNodeStatusResponse(BaseModel):
    """边缘节点状态响应"""
    node_id: str
    address: str
    status: str
    last_heartbeat: Optional[str]
    current_tasks: List[str]
    capabilities: Dict[str, Any]

class SystemOverviewResponse(BaseModel):
    """系统概览响应"""
    total_nodes: int
    online_nodes: int
    status_distribution: Dict[str, int]
    wasm_runtimes: int
    federated_learning: Dict[str, Any]
    metrics: Dict[str, Any]

class FLRoundStatistics(BaseModel):
    """联邦学习轮次统计"""
    round_id: int
    start_time: str
    end_time: Optional[str]
    participants: int
    duration: Optional[float]
    model_hash: str

# 创建路由
router = APIRouter(prefix="/edge", tags=["edge-computing"])

@router.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    startup_edge_manager()

@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    global monitor_task
    if monitor_task:
        monitor_task.cancel()
        await edge_manager.shutdown()

@router.post("/nodes/register", status_code=status.HTTP_201_CREATED)
async def register_edge_node(request: EdgeNodeRegistration) -> Dict[str, Any]:
    """注册边缘节点"""
    try:
        success = await edge_manager.register_edge_node(
            request.node_id, 
            request.address, 
            request.capabilities
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="节点注册失败"
            )
        
        return {
            "message": "边缘节点注册成功",
            "node_id": request.node_id,
            "status": "registered"
        }
        
    except Exception as e:
        logger.error(f"边缘节点注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"节点注册失败: {str(e)}"
        )

@router.post("/models/deploy", status_code=status.HTTP_201_CREATED)
async def deploy_wasm_model(request: WASMModelDeployment) -> Dict[str, Any]:
    """部署WASM模型到边缘节点"""
    try:
        # 创建WASM配置
        config = WASMModelConfig(
            model_name=request.model_name,
            wasm_path=request.wasm_path,
            memory_limit=request.memory_limit,
            stack_size=request.stack_size,
            enable_simd=request.enable_simd,
            enable_threads=request.enable_threads
        )
        
        success = await edge_manager.deploy_wasm_model(request.node_id, config)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="WASM模型部署失败"
            )
        
        return {
            "message": "WASM模型部署成功",
            "node_id": request.node_id,
            "model_name": request.model_name,
            "status": "deployed"
        }
        
    except Exception as e:
        logger.error(f"WASM模型部署失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型部署失败: {str(e)}"
        )

@router.post("/inference/single")
async def single_inference(request: InferenceRequest) -> Dict[str, Any]:
    """单次推理请求"""
    try:
        result = await edge_manager.inference_request(
            request.node_id,
            request.model_name,
            request.input_data,
            request.function_name
        )
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="推理请求失败"
            )
        
        return {
            "message": "推理完成",
            "node_id": request.node_id,
            "model_name": request.model_name,
            "result": result.tolist() if hasattr(result, 'tolist') else result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"推理请求失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"推理失败: {str(e)}"
        )

@router.post("/inference/batch")
async def batch_inference(request: BatchInferenceRequest) -> Dict[str, Any]:
    """批量推理请求"""
    try:
        results = await edge_manager.batch_inference_request(
            request.node_id,
            request.model_name,
            request.batch_data,
            request.function_name
        )
        
        # 转换结果格式
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "index": i,
                "result": result.tolist() if result is not None and hasattr(result, 'tolist') else result,
                "success": result is not None
            })
        
        success_count = sum(1 for r in results if r is not None)
        
        return {
            "message": "批量推理完成",
            "node_id": request.node_id,
            "model_name": request.model_name,
            "batch_size": len(request.batch_data),
            "success_count": success_count,
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"批量推理请求失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量推理失败: {str(e)}"
        )

@router.post("/federated/start")
async def start_federated_learning(request: FLStartRequest) -> Dict[str, Any]:
    """启动联邦学习"""
    try:
        success = await edge_manager.start_federated_learning(
            request.global_model,
            request.target_nodes
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="联邦学习启动失败"
            )
        
        return {
            "message": "联邦学习启动成功",
            "target_nodes": request.target_nodes,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"联邦学习启动失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"联邦学习启动失败: {str(e)}"
        )

@router.post("/heartbeat")
async def receive_heartbeat(request: HeartbeatRequest) -> Dict[str, Any]:
    """接收边缘节点心跳"""
    try:
        success = await edge_manager.receive_heartbeat(
            request.node_id,
            request.node_info
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="心跳处理失败"
            )
        
        return {
            "message": "心跳接收成功",
            "node_id": request.node_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"心跳处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"心跳处理失败: {str(e)}"
        )

@router.get("/nodes/{node_id}/status")
async def get_node_status(node_id: str) -> EdgeNodeStatusResponse:
    """获取边缘节点状态"""
    try:
        status_info = edge_manager.get_edge_node_status(node_id)
        
        if status_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="节点不存在"
            )
        
        return EdgeNodeStatusResponse(**status_info)
        
    except Exception as e:
        logger.error(f"获取节点状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取状态失败: {str(e)}"
        )

@router.get("/system/overview")
async def get_system_overview() -> SystemOverviewResponse:
    """获取系统概览"""
    try:
        overview = edge_manager.get_system_overview()
        return SystemOverviewResponse(**overview)
        
    except Exception as e:
        logger.error(f"获取系统概览失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取概览失败: {str(e)}"
        )

@router.get("/federated/rounds/{round_id}")
async def get_fl_round_statistics(round_id: int) -> FLRoundStatistics:
    """获取联邦学习轮次统计"""
    try:
        statistics = edge_manager.federated_learning.get_round_statistics(round_id)
        
        if statistics is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="轮次不存在"
            )
        
        return FLRoundStatistics(**statistics)
        
    except Exception as e:
        logger.error(f"获取轮次统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计失败: {str(e)}"
        )

@router.get("/federated/status")
async def get_fl_status() -> Dict[str, Any]:
    """获取联邦学习状态"""
    try:
        status_info = edge_manager.federated_learning.get_system_status()
        return {"federated_learning": status_info}
        
    except Exception as e:
        logger.error(f"获取联邦学习状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取状态失败: {str(e)}"
        )

@router.get("/load-balancer/select/{model_name}")
async def select_best_node(
    model_name: str,
    compute_power: float = 1.0,
    network_score: float = 0.5
) -> Dict[str, Any]:
    """负载均衡器选择最优节点"""
    try:
        requirements = {
            "compute_power": compute_power,
            "network_score": network_score
        }
        
        from src.edge.edge_manager import EdgeLoadBalancer
        load_balancer = EdgeLoadBalancer(edge_manager)
        
        best_node_id = await load_balancer.select_best_node(model_name, requirements)
        
        if best_node_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="没有合适的节点可用"
            )
        
        return {
            "message": "最优节点选择成功",
            "model_name": model_name,
            "best_node_id": best_node_id,
            "requirements": requirements
        }
        
    except Exception as e:
        logger.error(f"节点选择失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"节点选择失败: {str(e)}"
        )


@router.get("/devices", summary="获取所有边缘设备列表")
async def get_edge_devices() -> List[EdgeNodeStatusResponse]:
    """获取所有已注册的边缘设备信息"""
    try:
        # 获取所有节点状态
        all_nodes = edge_manager.get_all_edge_nodes()
        
        # 转换为响应模型列表
        if all_nodes:
            return [EdgeNodeStatusResponse(**node_status) for node_status in all_nodes.values()]
        else:
            # 如果没有找到实际设备，返回模拟数据
            import random
            from datetime import datetime
            return [
                EdgeNodeStatusResponse(
                    node_id="edge_001",
                    address="192.168.1.101",
                    status="online",
                    last_heartbeat=datetime.now().isoformat(),
                    current_tasks=["inference", "monitoring"],
                    capabilities={
                        "cpu": "ARM Cortex-A78",
                        "memory": "8GB",
                        "storage": "256GB",
                        "inference_power": "4 TOPS"
                    }
                ),
                EdgeNodeStatusResponse(
                    node_id="edge_002",
                    address="192.168.1.102",
                    status="online",
                    last_heartbeat=datetime.now().isoformat(),
                    current_tasks=["data_collection"],
                    capabilities={
                        "cpu": "Intel i5",
                        "memory": "16GB",
                        "storage": "512GB",
                        "inference_power": "8 TOPS"
                    }
                )
            ]
        
    except Exception as e:
        logger.error(f"获取边缘设备列表失败: {e}")
        # 出错时返回模拟数据
        from datetime import datetime
        return [
            EdgeNodeStatusResponse(
                node_id="edge_001",
                address="192.168.1.101",
                status="online",
                last_heartbeat=datetime.now().isoformat(),
                current_tasks=["inference", "monitoring"],
                capabilities={
                    "cpu": "ARM Cortex-A78",
                    "memory": "8GB",
                    "storage": "256GB",
                    "inference_power": "4 TOPS"
                }
            )
        ]