"""
边缘计算管理器
协调WebAssembly边缘节点和联邦学习客户端
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .webassembly_runtime import WebAssemblyRuntime, WASMModelConfig
from .federated_learning import FederatedLearning, FLClient

logger = logging.getLogger(__name__)

class EdgeNodeStatus(Enum):
    """边缘节点状态"""
    OFFLINE = "offline"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"

@dataclass
class EdgeNode:
    """边缘节点信息"""
    node_id: str
    address: str
    capabilities: Dict[str, Any]
    status: EdgeNodeStatus = EdgeNodeStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    current_tasks: Set[str] = None
    
    def __post_init__(self):
        if self.current_tasks is None:
            self.current_tasks = set()

class EdgeManager:
    """边缘计算管理器"""
    
    def __init__(self, coordinator_address: str = "localhost:8000"):
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.wasm_runtimes: Dict[str, WebAssemblyRuntime] = {}
        self.federated_learning = FederatedLearning(coordinator_address)
        
        # 任务调度
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 监控指标
        self.metrics = {
            "total_inferences": 0,
            "failed_inferences": 0,
            "avg_inference_time": 0.0,
            "edge_node_uptime": {}
        }
        
    async def register_edge_node(self, node_id: str, address: str, 
                               capabilities: Dict[str, Any]) -> bool:
        """注册边缘节点"""
        try:
            if node_id in self.edge_nodes:
                logger.warning(f"边缘节点 {node_id} 已存在")
                return False
                
            node = EdgeNode(
                node_id=node_id,
                address=address,
                capabilities=capabilities,
                status=EdgeNodeStatus.IDLE,
                last_heartbeat=datetime.now()
            )
            
            self.edge_nodes[node_id] = node
            
            # 同时注册为联邦学习客户端
            data_size = capabilities.get("data_size", 0)
            self.federated_learning.register_client(node_id, address, data_size)
            
            logger.info(f"边缘节点 {node_id} 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"边缘节点注册失败: {e}")
            return False
    
    async def deploy_wasm_model(self, node_id: str, model_config: WASMModelConfig) -> bool:
        """在边缘节点部署WASM模型"""
        try:
            if node_id not in self.edge_nodes:
                logger.error(f"边缘节点 {node_id} 不存在")
                return False
                
            # 检查节点状态
            node = self.edge_nodes[node_id]
            if node.status != EdgeNodeStatus.IDLE:
                logger.warning(f"边缘节点 {node_id} 当前状态为 {node.status.value}，无法部署模型")
                return False
                
            # 创建WASM运行时
            wasm_runtime = WebAssemblyRuntime(model_config)
            
            # 初始化运行时
            success = await wasm_runtime.initialize()
            if not success:
                logger.error(f"WASM运行时初始化失败: {node_id}")
                return False
                
            # 存储运行时实例
            runtime_key = f"{node_id}_{model_config.model_name}"
            self.wasm_runtimes[runtime_key] = wasm_runtime
            
            # 更新节点状态
            node.status = EdgeNodeStatus.BUSY
            node.current_tasks.add(model_config.model_name)
            
            logger.info(f"WASM模型 {model_config.model_name} 在节点 {node_id} 部署成功")
            return True
            
        except Exception as e:
            logger.error(f"WASM模型部署失败: {e}")
            return False
    
    async def inference_request(self, node_id: str, model_name: str, 
                               input_data: Any, function_name: str = "inference") -> Optional[Any]:
        """向边缘节点发送推理请求"""
        try:
            runtime_key = f"{node_id}_{model_name}"
            
            if runtime_key not in self.wasm_runtimes:
                logger.error(f"WASM运行时 {runtime_key} 不存在")
                return None
                
            wasm_runtime = self.wasm_runtimes[runtime_key]
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行推理
            result = await wasm_runtime.inference(input_data, function_name)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            
            # 更新指标
            self._update_inference_metrics(inference_time, result is not None)
            
            if result is not None:
                logger.info(f"推理请求完成 - 节点: {node_id}, 模型: {model_name}, 耗时: {inference_time:.3f}s")
            else:
                logger.warning(f"推理请求失败 - 节点: {node_id}, 模型: {model_name}")
                
            return result
            
        except Exception as e:
            logger.error(f"推理请求执行失败: {e}")
            return None
    
    async def batch_inference_request(self, node_id: str, model_name: str,
                                    batch_data: List[Any], function_name: str = "inference") -> List[Optional[Any]]:
        """批量推理请求"""
        try:
            runtime_key = f"{node_id}_{model_name}"
            
            if runtime_key not in self.wasm_runtimes:
                logger.error(f"WASM运行时 {runtime_key} 不存在")
                return [None] * len(batch_data)
                
            wasm_runtime = self.wasm_runtimes[runtime_key]
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行批量推理
            results = await wasm_runtime.batch_inference(batch_data, function_name)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            
            # 更新指标
            success_count = sum(1 for r in results if r is not None)
            self._update_inference_metrics(inference_time, success_count > 0, len(batch_data))
            
            logger.info(f"批量推理完成 - 节点: {node_id}, 模型: {model_name}, "
                       f"批次大小: {len(batch_data)}, 成功: {success_count}, 耗时: {inference_time:.3f}s")
                
            return results
            
        except Exception as e:
            logger.error(f"批量推理请求执行失败: {e}")
            return [None] * len(batch_data)
    
    def _update_inference_metrics(self, inference_time: float, success: bool, batch_size: int = 1):
        """更新推理指标"""
        self.metrics["total_inferences"] += batch_size
        
        if not success:
            self.metrics["failed_inferences"] += batch_size
        
        # 更新平均推理时间（指数移动平均）
        alpha = 0.1  # 平滑因子
        current_avg = self.metrics["avg_inference_time"]
        new_avg = alpha * inference_time + (1 - alpha) * current_avg
        self.metrics["avg_inference_time"] = new_avg
    
    async def start_federated_learning(self, global_model: Dict[str, Any], 
                                    target_nodes: int = 5) -> bool:
        """启动联邦学习"""
        try:
            # 选择参与联邦学习的边缘节点
            available_nodes = [n for n in self.edge_nodes.values() 
                             if n.status == EdgeNodeStatus.IDLE]
            
            if len(available_nodes) < target_nodes:
                logger.warning(f"可用节点数量不足: {len(available_nodes)}/{target_nodes}")
                return False
                
            # 初始化联邦学习轮次
            success = await self.federated_learning.initialize_round(
                global_model, target_nodes
            )
            
            if success:
                logger.info(f"联邦学习启动成功，参与节点: {target_nodes}")
            
            return success
            
        except Exception as e:
            logger.error(f"联邦学习启动失败: {e}")
            return False
    
    async def process_model_update(self, node_id: str, model_update: Dict[str, Any]) -> bool:
        """处理模型更新"""
        try:
            # 这里应该将模型更新转发给联邦学习协调器
            # 由于环境限制，这里简化处理
            
            logger.info(f"收到节点 {node_id} 的模型更新")
            
            # 模拟处理延迟
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"模型更新处理失败: {e}")
            return False
    
    async def monitor_edge_nodes(self):
        """监控边缘节点状态"""
        while True:
            try:
                current_time = datetime.now()
                
                for node_id, node in self.edge_nodes.items():
                    # 检查节点心跳
                    if node.last_heartbeat:
                        time_diff = current_time - node.last_heartbeat
                        
                        if time_diff > timedelta(minutes=5):
                            # 节点离线
                            node.status = EdgeNodeStatus.OFFLINE
                            self.federated_learning.update_client_status(node_id, False)
                            logger.warning(f"边缘节点 {node_id} 心跳超时，标记为离线")
                        elif time_diff > timedelta(minutes=1):
                            # 节点可能有问题
                            if node.status != EdgeNodeStatus.ERROR:
                                node.status = EdgeNodeStatus.ERROR
                                logger.warning(f"边缘节点 {node_id} 心跳延迟，标记为错误状态")
                    
                    # 更新节点运行时间统计
                    if node.status == EdgeNodeStatus.IDLE or node.status == EdgeNodeStatus.BUSY:
                        if node_id not in self.metrics["edge_node_uptime"]:
                            self.metrics["edge_node_uptime"][node_id] = 0
                        self.metrics["edge_node_uptime"][node_id] += 1
                
                # 每30秒检查一次
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"边缘节点监控失败: {e}")
                await asyncio.sleep(30)  # 出错后继续监控
    
    async def receive_heartbeat(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """接收边缘节点心跳"""
        try:
            if node_id not in self.edge_nodes:
                logger.warning(f"收到未知节点 {node_id} 的心跳")
                return False
                
            node = self.edge_nodes[node_id]
            
            # 更新节点信息
            node.last_heartbeat = datetime.now()
            
            # 根据心跳信息更新节点状态
            if node_info.get("status") == "idle" and node.status != EdgeNodeStatus.IDLE:
                node.status = EdgeNodeStatus.IDLE
                node.current_tasks.clear()
            elif node_info.get("status") == "busy" and node.status != EdgeNodeStatus.BUSY:
                node.status = EdgeNodeStatus.BUSY
            
            # 更新联邦学习客户端状态
            self.federated_learning.update_client_status(node_id, True)
            
            return True
            
        except Exception as e:
            logger.error(f"处理心跳失败: {e}")
            return False
    
    def get_edge_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取边缘节点状态"""
        if node_id not in self.edge_nodes:
            return None
            
        node = self.edge_nodes[node_id]
        
        return {
            "node_id": node.node_id,
            "address": node.address,
            "status": node.status.value,
            "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
            "current_tasks": list(node.current_tasks),
            "capabilities": node.capabilities
        }
    
    def get_all_edge_nodes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有边缘节点信息"""
        result = {}
        for node_id in self.edge_nodes:
            node_status = self.get_edge_node_status(node_id)
            if node_status:
                result[node_id] = node_status
        return result
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        total_nodes = len(self.edge_nodes)
        online_nodes = sum(1 for n in self.edge_nodes.values() 
                         if n.status in [EdgeNodeStatus.IDLE, EdgeNodeStatus.BUSY])
        
        # 计算节点状态分布
        status_distribution = {}
        for status in EdgeNodeStatus:
            status_distribution[status.value] = sum(
                1 for n in self.edge_nodes.values() if n.status == status
            )
        
        return {
            "total_nodes": total_nodes,
            "online_nodes": online_nodes,
            "status_distribution": status_distribution,
            "wasm_runtimes": len(self.wasm_runtimes),
            "federated_learning": self.federated_learning.get_system_status(),
            "metrics": self.metrics
        }
    
    async def shutdown(self):
        """关闭边缘管理器"""
        try:
            # 关闭所有WASM运行时
            for runtime_key, runtime in self.wasm_runtimes.items():
                await runtime.shutdown()
            
            # 取消所有运行中的任务
            for task in self.running_tasks.values():
                task.cancel()
            
            # 等待任务完成
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
            
            logger.info("边缘管理器已关闭")
            
        except Exception as e:
            logger.error(f"边缘管理器关闭失败: {e}")

class EdgeLoadBalancer:
    """边缘计算负载均衡器"""
    
    def __init__(self, edge_manager: EdgeManager):
        self.edge_manager = edge_manager
    
    async def select_best_node(self, model_name: str, requirements: Dict[str, Any]) -> Optional[str]:
        """选择最优的边缘节点"""
        try:
            available_nodes = [
                n for n in self.edge_manager.edge_nodes.values()
                if n.status in [EdgeNodeStatus.IDLE, EdgeNodeStatus.BUSY]
            ]
            
            if not available_nodes:
                return None
            
            # 基于多种因素评分
            scored_nodes = []
            
            for node in available_nodes:
                score = self._calculate_node_score(node, model_name, requirements)
                scored_nodes.append((score, node.node_id))
            
            # 选择评分最高的节点
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            best_node_id = scored_nodes[0][1]
            
            return best_node_id
            
        except Exception as e:
            logger.error(f"选择最优节点失败: {e}")
            return None
    
    def _calculate_node_score(self, node: EdgeNode, model_name: str, 
                            requirements: Dict[str, Any]) -> float:
        """计算节点评分"""
        score = 0.0
        
        # 1. 节点状态权重
        if node.status == EdgeNodeStatus.IDLE:
            score += 100
        elif node.status == EdgeNodeStatus.BUSY:
            score += 50
        
        # 2. 计算能力匹配度
        required_compute = requirements.get("compute_power", 1.0)
        node_compute = node.capabilities.get("compute_power", 1.0)
        compute_score = min(node_compute / required_compute, 1.0) * 50
        score += compute_score
        
        # 3. 网络延迟考虑
        network_score = node.capabilities.get("network_score", 0.5) * 30
        score += network_score
        
        # 4. 当前负载考虑
        current_load = len(node.current_tasks)
        load_penalty = current_load * 10
        score -= load_penalty
        
        return max(score, 0)