"""
云边协同管理模块

负责云中心与边缘节点之间的数据同步和模型更新，包括：
- 数据同步策略
- 模型更新机制
- 一致性保证
- 冲突解决策略
"""

import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import hashlib


class SyncStrategy(Enum):
    """同步策略"""
    PUSH = "push"  # 推送模式
    PULL = "pull"  # 拉取模式
    HYBRID = "hybrid"  # 混合模式
    EVENT_DRIVEN = "event_driven"  # 事件驱动


class ConsistencyLevel(Enum):
    """一致性级别"""
    STRONG = "strong"  # 强一致性
    EVENTUAL = "eventual"  # 最终一致性
    WEAK = "weak"  # 弱一致性


class ConflictResolution(Enum):
    """冲突解决策略"""
    LAST_WRITE_WINS = "last_write_wins"  # 最后写入获胜
    VECTOR_CLOCKS = "vector_clocks"  # 向量时钟
    MANUAL = "manual"  # 手动解决
    AUTOMATIC = "automatic"  # 自动解决


@dataclass
class SyncTask:
    """同步任务"""
    task_id: str
    sync_type: str  # data, model, config
    source: str
    targets: List[str]
    payload: Any
    priority: int
    created_at: datetime
    status: str  # pending, running, completed, failed
    retry_count: int = 0
    last_attempt: Optional[datetime] = None


@dataclass
class SyncResult:
    """同步结果"""
    task_id: str
    success_count: int
    failure_count: int
    total_duration: timedelta
    details: Dict[str, Any]


class CloudEdgeSyncManager:
    """云边协同管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 同步任务队列
        self.sync_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, SyncTask] = {}
        self.completed_tasks: Dict[str, SyncResult] = {}
        
        # 节点状态管理
        self.edge_nodes: Dict[str, Dict[str, Any]] = {}
        
        # 数据版本管理
        self.data_versions: Dict[str, str] = {}  # data_key -> version_hash
        
        # 数据本地化相关
        self.data_localization_logs: List[Dict[str, Any]] = []  # 数据本地化审计日志
        self.sensitive_data_keys: Set[str] = set()  # 敏感数据标识
        
        # 同步统计
        self.sync_statistics = self._initialize_statistics()
        
        # 启动同步工作器
        self.sync_workers = []
        self.is_running = False
        
        # 初始化数据本地化配置
        self._initialize_localization()
    
    def _initialize_localization(self):
        """初始化数据本地化配置"""
        localization_config = self.config.get("localization", {})
        if localization_config.get("enabled", True):
            # 预定义敏感数据标识
            self.sensitive_data_keys = {
                "user_id", "device_id", "location", "personal_data", "biometric",
                "health_data", "agriculture_data", "farm_location"
            }
            self.logger.info("数据本地化保护机制已初始化")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "sync_strategy": SyncStrategy.HYBRID,
            "consistency_level": ConsistencyLevel.EVENTUAL,
            "conflict_resolution": ConflictResolution.LAST_WRITE_WINS,
            "max_retries": 3,
            "retry_delay": 5,  # 重试延迟(秒)
            "batch_size": 100,
            "sync_timeout": 300,  # 同步超时(秒)
            "worker_count": 3,  # 同步工作器数量
            "heartbeat_interval": 30,  # 心跳间隔(秒)
            "localization": {
                "enabled": True,  # 启用数据本地化
                "allowed_regions": ["CN"],  # 仅允许中国地区的数据存储
                "audit_enabled": True,  # 启用数据存储审计
                "encryption_enabled": True  # 启用本地存储加密
            }
        }
    
    async def start_sync_service(self) -> bool:
        """启动同步服务"""
        try:
            if self.is_running:
                self.logger.warning("同步服务已经在运行")
                return False
            
            self.is_running = True
            
            # 启动同步工作器
            for i in range(self.config["worker_count"]):
                worker = asyncio.create_task(self._sync_worker(f"worker_{i}"))
                self.sync_workers.append(worker)
            
            # 启动心跳检测
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self.sync_workers.append(heartbeat_task)
            
            self.logger.info("云边同步服务启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动同步服务失败: {e}")
            self.is_running = False
            return False
    
    async def stop_sync_service(self) -> bool:
        """停止同步服务"""
        try:
            if not self.is_running:
                return True
            
            self.is_running = False
            
            # 取消所有工作器
            for worker in self.sync_workers:
                worker.cancel()
            
            # 等待工作器完成
            await asyncio.gather(*self.sync_workers, return_exceptions=True)
            self.sync_workers.clear()
            
            self.logger.info("云边同步服务已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止同步服务失败: {e}")
            return False
    
    def _is_valid_node_location(self, node_id: str) -> bool:
        """检查节点位置是否符合数据本地化要求"""
        localization_config = self.config.get("localization", {})
        if not localization_config.get("enabled", True):
            return True
            
        # 获取节点信息
        node_info = self.edge_nodes.get(node_id, {})
        node_region = node_info.get("region", "CN")  # 默认中国地区
        
        # 检查节点地区是否在允许列表中
        allowed_regions = localization_config.get("allowed_regions", ["CN"])
        return node_region in allowed_regions
    
    def _is_sensitive_data(self, data_key: str) -> bool:
        """检查数据是否属于敏感数据"""
        return any(sensitive_key in data_key.lower() for sensitive_key in self.sensitive_data_keys)
    
    def _log_localization_audit(self, data_key: str, target_nodes: List[str], result: str, details: Dict[str, Any] = None):
        """记录数据本地化审计日志"""
        localization_config = self.config.get("localization", {})
        if localization_config.get("audit_enabled", True):
            audit_log = {
                "timestamp": datetime.now().isoformat(),
                "data_key": data_key,
                "target_nodes": target_nodes,
                "sensitive_data": self._is_sensitive_data(data_key),
                "result": result,
                "details": details or {},
                "sync_type": "data"
            }
            self.data_localization_logs.append(audit_log)
            
            # 限制日志大小
            if len(self.data_localization_logs) > 1000:
                self.data_localization_logs.pop(0)
    
    async def sync_data(self,
                      data_key: str,
                      data: Any,
                      target_nodes: List[str],
                      priority: int = 1) -> str:
        """同步数据到边缘节点"""
        try:
            # 数据本地化验证
            localization_config = self.config.get("localization", {})
            if localization_config.get("enabled", True):
                # 过滤不符合本地化要求的节点
                valid_nodes = [node for node in target_nodes if self._is_valid_node_location(node)]
                invalid_nodes = list(set(target_nodes) - set(valid_nodes))
                
                if invalid_nodes:
                    self.logger.warning(f"以下节点不符合数据本地化要求，已过滤: {invalid_nodes}")
                    self._log_localization_audit(
                        data_key, target_nodes, "filtered", 
                        {"valid_nodes": valid_nodes, "invalid_nodes": invalid_nodes}
                    )
                    
                    if not valid_nodes:
                        self.logger.error("所有目标节点均不符合数据本地化要求，同步任务取消")
                        return ""
                    target_nodes = valid_nodes
            
            # 生成数据版本
            version_hash = self._generate_data_version(data)
            
            # 创建同步任务
            task_id = f"data_sync_{int(datetime.now().timestamp())}"
            sync_task = SyncTask(
                task_id=task_id,
                sync_type="data",
                source="cloud",
                targets=target_nodes,
                payload={
                    "data_key": data_key,
                    "data": data,
                    "version": version_hash,
                    "timestamp": datetime.now().isoformat(),
                    "localization_check": True
                },
                priority=priority,
                created_at=datetime.now(),
                status="pending"
            )
            
            # 添加到队列
            await self.sync_queue.put(sync_task)
            self.active_tasks[task_id] = sync_task
            
            # 更新数据版本
            self.data_versions[data_key] = version_hash
            
            self.logger.info(f"数据同步任务创建成功: {task_id}, 目标节点: {len(target_nodes)}个")
            
            # 记录审计日志
            self._log_localization_audit(data_key, target_nodes, "success")
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"创建数据同步任务失败: {e}")
            self._log_localization_audit(data_key, target_nodes, "failed", {"error": str(e)})
            return ""
    
    async def sync_model(self,
                       model_id: str,
                       model_data: Any,
                       target_nodes: List[str],
                       priority: int = 10) -> str:
        """同步模型到边缘节点"""
        try:
            # 创建模型同步任务
            task_id = f"model_sync_{int(datetime.now().timestamp())}"
            sync_task = SyncTask(
                task_id=task_id,
                sync_type="model",
                source="cloud",
                targets=target_nodes,
                payload={
                    "model_id": model_id,
                    "model_data": model_data,
                    "version": self._generate_model_version(model_data),
                    "timestamp": datetime.now().isoformat(),
                    "compression": "quantized"  # 模型压缩信息
                },
                priority=priority,
                created_at=datetime.now(),
                status="pending"
            )
            
            await self.sync_queue.put(sync_task)
            self.active_tasks[task_id] = sync_task
            
            self.logger.info(f"模型同步任务创建成功: {task_id}, 模型: {model_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"创建模型同步任务失败: {e}")
            return ""
    
    async def get_sync_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取同步任务状态"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status,
                "progress": self._calculate_progress(task),
                "retry_count": task.retry_count,
                "created_at": task.created_at.isoformat(),
                "last_attempt": task.last_attempt.isoformat() if task.last_attempt else None
            }
        elif task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "completed",
                "success_count": result.success_count,
                "failure_count": result.failure_count,
                "total_duration": result.total_duration.total_seconds(),
                "completed_at": datetime.now().isoformat()
            }
        
        return None
    
    def register_edge_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """注册边缘节点"""
        try:
            self.edge_nodes[node_id] = {
                "info": node_info,
                "last_heartbeat": datetime.now(),
                "status": "online",
                "sync_capabilities": node_info.get("sync_capabilities", {}),
                "data_versions": {}
            }
            
            self.logger.info(f"边缘节点注册成功: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"注册边缘节点失败: {e}")
            return False
    
    async def handle_edge_heartbeat(self, node_id: str, heartbeat_data: Dict[str, Any]) -> bool:
        """处理边缘节点心跳"""
        try:
            if node_id not in self.edge_nodes:
                self.logger.warning(f"收到未知节点的心跳: {node_id}")
                return False
            
            node = self.edge_nodes[node_id]
            node["last_heartbeat"] = datetime.now()
            node["status"] = "online"
            
            # 更新节点数据版本信息
            if "data_versions" in heartbeat_data:
                node["data_versions"].update(heartbeat_data["data_versions"])
            
            # 检查是否需要反向同步
            await self._check_reverse_sync(node_id, heartbeat_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理心跳失败: {e}")
            return False
    
    async def _sync_worker(self, worker_id: str):
        """同步工作器"""
        self.logger.info(f"同步工作器启动: {worker_id}")
        
        while self.is_running:
            try:
                # 从队列获取任务
                task = await asyncio.wait_for(
                    self.sync_queue.get(), 
                    timeout=1.0
                )
                
                if task:
                    await self._process_sync_task(task, worker_id)
                    self.sync_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"同步工作器异常: {e}")
                await asyncio.sleep(1)
    
    async def _process_sync_task(self, task: SyncTask, worker_id: str):
        """处理同步任务"""
        try:
            task.status = "running"
            task.last_attempt = datetime.now()
            
            self.logger.info(f"工作器 {worker_id} 开始处理任务: {task.task_id}")
            
            # 根据同步类型处理任务
            if task.sync_type == "data":
                result = await self._sync_data_to_nodes(task)
            elif task.sync_type == "model":
                result = await self._sync_model_to_nodes(task)
            else:
                result = SyncResult(
                    task_id=task.task_id,
                    success_count=0,
                    failure_count=len(task.targets),
                    total_duration=timedelta(0),
                    details={"error": f"未知的同步类型: {task.sync_type}"}
                )
            
            # 更新任务状态
            if result.failure_count == 0:
                task.status = "completed"
            elif result.success_count > 0:
                task.status = "partial"
            else:
                task.status = "failed"
            
            # 存储结果
            self.completed_tasks[task.task_id] = result
            
            # 从活跃任务中移除
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            self.logger.info(f"任务处理完成: {task.task_id}, "
                           f"成功: {result.success_count}, 失败: {result.failure_count}")
            
        except Exception as e:
            self.logger.error(f"处理同步任务失败: {e}")
            
            # 重试逻辑
            if task.retry_count < self.config["max_retries"]:
                task.retry_count += 1
                task.status = "pending"
                
                # 延迟后重新加入队列
                await asyncio.sleep(self.config["retry_delay"] * task.retry_count)
                await self.sync_queue.put(task)
            else:
                # 重试次数用尽，标记为失败
                task.status = "failed"
                result = SyncResult(
                    task_id=task.task_id,
                    success_count=0,
                    failure_count=len(task.targets),
                    total_duration=timedelta(0),
                    details={"error": f"重试次数用尽: {str(e)}"}
                )
                self.completed_tasks[task.task_id] = result
                
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    async def _sync_data_to_nodes(self, task: SyncTask) -> SyncResult:
        """同步数据到节点"""
        start_time = datetime.now()
        success_count = 0
        failure_count = 0
        details = {}
        
        for node_id in task.targets:
            try:
                if node_id not in self.edge_nodes:
                    details[node_id] = "节点未注册"
                    failure_count += 1
                    continue
                
                # 模拟数据同步（实际应用中需要网络通信）
                sync_success = await self._send_data_to_node(node_id, task.payload)
                
                if sync_success:
                    success_count += 1
                    details[node_id] = "同步成功"
                    
                    # 更新节点数据版本
                    data_key = task.payload["data_key"]
                    version = task.payload["version"]
                    self.edge_nodes[node_id]["data_versions"][data_key] = version
                else:
                    failure_count += 1
                    details[node_id] = "同步失败"
                    
            except Exception as e:
                failure_count += 1
                details[node_id] = f"同步异常: {str(e)}"
        
        total_duration = datetime.now() - start_time
        
        return SyncResult(
            task_id=task.task_id,
            success_count=success_count,
            failure_count=failure_count,
            total_duration=total_duration,
            details=details
        )
    
    async def _sync_model_to_nodes(self, task: SyncTask) -> SyncResult:
        """同步模型到节点"""
        start_time = datetime.now()
        success_count = 0
        failure_count = 0
        details = {}
        
        for node_id in task.targets:
            try:
                if node_id not in self.edge_nodes:
                    details[node_id] = "节点未注册"
                    failure_count += 1
                    continue
                
                # 检查节点模型部署能力
                node_capabilities = self.edge_nodes[node_id]["sync_capabilities"]
                if not node_capabilities.get("model_deployment", False):
                    details[node_id] = "节点不支持模型部署"
                    failure_count += 1
                    continue
                
                # 模拟模型同步
                sync_success = await self._send_model_to_node(node_id, task.payload)
                
                if sync_success:
                    success_count += 1
                    details[node_id] = "模型同步成功"
                else:
                    failure_count += 1
                    details[node_id] = "模型同步失败"
                    
            except Exception as e:
                failure_count += 1
                details[node_id] = f"模型同步异常: {str(e)}"
        
        total_duration = datetime.now() - start_time
        
        return SyncResult(
            task_id=task.task_id,
            success_count=success_count,
            failure_count=failure_count,
            total_duration=total_duration,
            details=details
        )
    
    async def _send_data_to_node(self, node_id: str, payload: Any) -> bool:
        """发送数据到节点（模拟实现）"""
        # 实际应用中需要实现网络通信
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return True
    
    async def _send_model_to_node(self, node_id: str, payload: Any) -> bool:
        """发送模型到节点（模拟实现）"""
        # 实际应用中需要实现模型传输和部署
        await asyncio.sleep(0.5)  # 模拟模型传输时间
        return True
    
    async def _heartbeat_monitor(self):
        """心跳监控"""
        while self.is_running:
            try:
                current_time = datetime.now()
                offline_threshold = timedelta(seconds=self.config["heartbeat_interval"] * 3)
                
                for node_id, node_info in self.edge_nodes.items():
                    last_heartbeat = node_info["last_heartbeat"]
                    
                    if current_time - last_heartbeat > offline_threshold:
                        node_info["status"] = "offline"
                        self.logger.warning(f"边缘节点 {node_id} 心跳超时，标记为离线")
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                self.logger.error(f"心跳监控异常: {e}")
                await asyncio.sleep(self.config["heartbeat_interval"])
    
    async def _check_reverse_sync(self, node_id: str, heartbeat_data: Dict[str, Any]):
        """检查是否需要反向同步"""
        try:
            if "data_updates" not in heartbeat_data:
                return
            
            data_updates = heartbeat_data["data_updates"]
            
            for data_key, edge_version in data_updates.items():
                cloud_version = self.data_versions.get(data_key)
                
                # 检查版本冲突
                if cloud_version and edge_version != cloud_version:
                    await self._resolve_data_conflict(node_id, data_key, edge_version, cloud_version)
                
        except Exception as e:
            self.logger.error(f"检查反向同步失败: {e}")
    
    async def _resolve_data_conflict(self, node_id: str, data_key: str, 
                                   edge_version: str, cloud_version: str):
        """解决数据冲突"""
        conflict_strategy = self.config["conflict_resolution"]
        
        if conflict_strategy == ConflictResolution.LAST_WRITE_WINS:
            # 最后写入获胜策略
            self.logger.info(f"数据冲突解决: {data_key}, 采用最后写入获胜策略")
            # 这里可以获取时间戳信息来决定哪个版本更新
            
        elif conflict_strategy == ConflictResolution.MANUAL:
            # 手动解决策略
            self.logger.warning(f"数据冲突需要手动解决: {data_key}")
            # 记录冲突信息，等待人工干预
            
        # 其他冲突解决策略...
    
    def _generate_data_version(self, data: Any) -> str:
        """生成数据版本"""
        data_str = str(data).encode('utf-8')
        return hashlib.md5(data_str).hexdigest()
    
    def _generate_model_version(self, model_data: Any) -> str:
        """生成模型版本"""
        # 简化实现：使用时间戳作为版本
        return str(int(datetime.now().timestamp()))
    
    async def get_performance_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """获取边缘计算性能指标"""
        try:
            # 模拟实现：返回性能指标数据
            performance_data = {
                "avg_latency": 85.2,  # 平均延迟(ms)
                "throughput": 1250,    # 吞吐量(请求/秒)
                "resource_utilization": {
                    "cpu": 0.65,       # CPU利用率
                    "memory": 0.42,    # 内存利用率
                    "disk": 0.38       # 磁盘利用率
                },
                "health_status": "healthy",  # 健康状态
                "metrics_timestamp": datetime.now().isoformat()
            }
            
            return performance_data
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {e}")
            return {
                "avg_latency": 0,
                "throughput": 0,
                "resource_utilization": {
                    "cpu": 0,
                    "memory": 0,
                    "disk": 0
                },
                "health_status": "unknown",
                "metrics_timestamp": datetime.now().isoformat()
            }
    
    def _calculate_progress(self, task: SyncTask) -> float:
        """计算任务进度"""
        # 简化实现
        if task.status == "completed":
            return 1.0
        elif task.status == "running":
            return 0.5
        else:
            return 0.0
    
    def _initialize_statistics(self) -> Dict[str, Any]:
        """初始化统计信息"""
        return {
            "total_sync_tasks": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "total_data_transferred_mb": 0,
            "average_sync_duration": timedelta(0),
            "node_availability": {}
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "registered_nodes": len(self.edge_nodes),
            "online_nodes": len([n for n in self.edge_nodes.values() if n["status"] == "online"]),
            "queue_size": self.sync_queue.qsize(),
            "statistics": self.sync_statistics
        }