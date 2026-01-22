#!/usr/bin/env python3
"""
边缘节点客户端示例
演示如何作为边缘节点与中心协调器交互
"""

import asyncio
import logging
import time
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("edge_node")

@dataclass
class EdgeNodeConfig:
    """边缘节点配置"""
    node_id: str
    coordinator_url: str = "http://localhost:8000"
    heartbeat_interval: int = 30  # 秒
    capabilities: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {
                "compute_power": 1.0,
                "memory": 1024,  # MB
                "storage": 5000,  # MB
                "network_score": 0.8,
                "data_size": 1000,  # 数据样本数量
                "supported_models": ["linear_regression", "cnn_classifier"]
            }

class EdgeNodeClient:
    """边缘节点客户端"""
    
    def __init__(self, config: EdgeNodeConfig):
        self.config = config
        self.is_running = False
        self.registered = False
        self.current_tasks = set()
        
    async def register_with_coordinator(self) -> bool:
        """向协调器注册节点"""
        try:
            registration_data = {
                "node_id": self.config.node_id,
                "address": f"edge_node_{self.config.node_id}:8001",
                "capabilities": self.config.capabilities
            }
            
            response = requests.post(
                f"{self.config.coordinator_url}/api/v1/edge/nodes/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 201:
                logger.info(f"节点注册成功: {self.config.node_id}")
                self.registered = True
                return True
            else:
                logger.error(f"节点注册失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"节点注册异常: {e}")
            return False
    
    async def send_heartbeat(self) -> bool:
        """发送心跳到协调器"""
        try:
            heartbeat_data = {
                "node_id": self.config.node_id,
                "node_info": {
                    "status": "idle" if not self.current_tasks else "busy",
                    "current_tasks": list(self.current_tasks),
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "cpu_usage": 0.3,  # 模拟CPU使用率
                        "memory_usage": 0.5,  # 模拟内存使用率
                        "network_throughput": 100.0  # Mbps
                    }
                }
            }
            
            response = requests.post(
                f"{self.config.coordinator_url}/api/v1/edge/heartbeat",
                json=heartbeat_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"心跳发送成功: {self.config.node_id}")
                return True
            else:
                logger.warning(f"心跳发送失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"心跳发送异常: {e}")
            return False
    
    async def perform_inference(self, model_name: str, input_data: Any) -> Optional[Any]:
        """执行推理任务"""
        try:
            # 模拟推理过程
            logger.info(f"开始推理任务 - 模型: {model_name}, 输入大小: {len(input_data) if hasattr(input_data, '__len__') else 1}")
            
            # 添加任务到当前任务集
            task_id = f"inference_{model_name}_{int(time.time())}"
            self.current_tasks.add(task_id)
            
            # 模拟推理延迟
            await asyncio.sleep(0.5)
            
            # 模拟推理结果
            if model_name == "linear_regression":
                # 简单的线性回归模拟
                result = [x * 2.0 + 1.0 for x in input_data] if isinstance(input_data, list) else input_data * 2.0 + 1.0
            elif model_name == "cnn_classifier":
                # 简单的分类模拟
                result = [{"class": "cat", "confidence": 0.85}] if isinstance(input_data, list) else {"class": "cat", "confidence": 0.85}
            else:
                result = {"error": "模型不支持"}
            
            # 移除任务
            self.current_tasks.remove(task_id)
            
            logger.info(f"推理任务完成: {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"推理任务异常: {e}")
            if task_id in self.current_tasks:
                self.current_tasks.remove(task_id)
            return None
    
    async def participate_federated_learning(self, global_model: Dict[str, Any]) -> Dict[str, Any]:
        """参与联邦学习"""
        try:
            logger.info("开始参与联邦学习")
            
            # 模拟本地训练
            task_id = f"fl_training_{int(time.time())}"
            self.current_tasks.add(task_id)
            
            # 模拟训练过程
            await asyncio.sleep(2.0)  # 模拟训练时间
            
            # 生成模型更新（模拟）
            model_update = {}
            for key, param in global_model.items():
                # 添加小的随机更新
                import random
                update = param + random.uniform(-0.1, 0.1)
                model_update[key] = update
            
            # 移除任务
            self.current_tasks.remove(task_id)
            
            logger.info("联邦学习参与完成")
            return model_update
            
        except Exception as e:
            logger.error(f"联邦学习参与异常: {e}")
            if task_id in self.current_tasks:
                self.current_tasks.remove(task_id)
            return {}
    
    async def run(self):
        """运行边缘节点"""
        logger.info(f"启动边缘节点: {self.config.node_id}")
        
        # 注册节点
        if not await self.register_with_coordinator():
            logger.error("节点注册失败，无法启动")
            return
        
        self.is_running = True
        last_heartbeat = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # 发送心跳
                if current_time - last_heartbeat >= self.config.heartbeat_interval:
                    await self.send_heartbeat()
                    last_heartbeat = current_time
                
                # 模拟一些工作负载
                if not self.current_tasks and current_time % 60 < 10:  # 每分钟的前10秒有工作
                    # 随机选择任务类型
                    import random
                    task_type = random.choice(["inference", "training"])
                    
                    if task_type == "inference":
                        model_name = random.choice(self.config.capabilities["supported_models"])
                        input_data = [random.uniform(0, 1) for _ in range(10)]
                        await self.perform_inference(model_name, input_data)
                    elif task_type == "training":
                        global_model = {"weight1": 1.0, "weight2": 2.0, "bias": 0.5}
                        await self.participate_federated_learning(global_model)
                
                # 短暂休眠
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭节点...")
        except Exception as e:
            logger.error(f"节点运行异常: {e}")
        finally:
            self.is_running = False
            logger.info("边缘节点已停止")
    
    def stop(self):
        """停止边缘节点"""
        self.is_running = False
        logger.info("正在停止边缘节点...")

async def main():
    """主函数"""
    # 创建多个边缘节点示例
    nodes = []
    
    for i in range(3):  # 创建3个边缘节点
        config = EdgeNodeConfig(
            node_id=f"edge_node_{i}",
            coordinator_url="http://localhost:8000",
            capabilities={
                "compute_power": 0.5 + i * 0.2,  # 节点0: 0.5, 节点1: 0.7, 节点2: 0.9
                "memory": 512 + i * 256,
                "storage": 2000 + i * 1000,
                "network_score": 0.7 + i * 0.1,
                "data_size": 500 + i * 250,
                "supported_models": ["linear_regression", "cnn_classifier"]
            }
        )
        
        node = EdgeNodeClient(config)
        nodes.append(node)
    
    # 并行运行所有节点
    tasks = [asyncio.create_task(node.run()) for node in nodes]
    
    try:
        # 等待所有节点运行
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止所有节点...")
        for node in nodes:
            node.stop()
        # 等待任务完成
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    # 运行边缘节点客户端
    asyncio.run(main())