"""
连接池管理器 - 管理各种设备连接的池化管理
支持连接复用、状态监控、自动重连等功能
"""
import asyncio
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    """连接信息"""
    device_id: str
    connection_type: str
    controller: Any
    status: ConnectionStatus
    last_activity: float
    created_at: float
    max_idle_time: int = 300  # 5分钟无活动自动断开


class ConnectionPoolManager:
    """连接池管理器"""
    
    def __init__(self, max_connections: int = 1000, cleanup_interval: int = 60):
        self.max_connections = max_connections
        self.cleanup_interval = cleanup_interval
        self.connections: Dict[str, ConnectionInfo] = {}
        self.active_connections: int = 0
        self._cleanup_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """初始化连接池管理器"""
        logger.info(f"初始化连接池管理器，最大连接数: {self.max_connections}")
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
    async def register_connection(self, device_id: str, controller: Any) -> bool:
        """注册新的连接"""
        if self.active_connections >= self.max_connections:
            logger.warning(f"连接池已满，无法注册设备 {device_id}")
            return False
            
        connection_info = ConnectionInfo(
            device_id=device_id,
            connection_type=controller.get_connection_type(),
            controller=controller,
            status=ConnectionStatus.CONNECTING,
            last_activity=time.time(),
            created_at=time.time()
        )
        
        self.connections[device_id] = connection_info
        self.active_connections += 1
        logger.info(f"成功注册设备连接: {device_id}")
        return True
    
    async def get_connection(self, device_id: str) -> Optional[Any]:
        """获取连接控制器"""
        if device_id not in self.connections:
            logger.warning(f"设备连接不存在: {device_id}")
            return None
            
        connection_info = self.connections[device_id]
        connection_info.last_activity = time.time()
        
        # 检查连接状态，如果断开则尝试重连
        if connection_info.status == ConnectionStatus.DISCONNECTED:
            await self._attempt_reconnect(connection_info)
            
        return connection_info.controller
    
    async def update_connection_status(self, device_id: str, status: ConnectionStatus):
        """更新连接状态"""
        if device_id in self.connections:
            self.connections[device_id].status = status
            self.connections[device_id].last_activity = time.time()
    
    async def _attempt_reconnect(self, connection_info: ConnectionInfo):
        """尝试重连"""
        try:
            logger.info(f"尝试重连设备: {connection_info.device_id}")
            # 使用控制器的重连方法
            if hasattr(connection_info.controller, 'reconnect'):
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor, 
                    connection_info.controller.reconnect
                )
                if result.get('success', False):
                    connection_info.status = ConnectionStatus.CONNECTED
                    logger.info(f"重连成功: {connection_info.device_id}")
                else:
                    connection_info.status = ConnectionStatus.ERROR
                    logger.error(f"重连失败: {connection_info.device_id}")
            else:
                # 如果控制器没有重连方法，则尝试重新连接
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    connection_info.controller.connect,
                    {}
                )
                if result.get('success', False):
                    connection_info.status = ConnectionStatus.CONNECTED
                    logger.info(f"连接成功: {connection_info.device_id}")
                else:
                    connection_info.status = ConnectionStatus.ERROR
                    logger.error(f"连接失败: {connection_info.device_id}")
        except Exception as e:
            logger.error(f"重连过程中发生错误: {e}")
            connection_info.status = ConnectionStatus.ERROR
    
    async def remove_connection(self, device_id: str) -> bool:
        """移除连接"""
        if device_id in self.connections:
            connection_info = self.connections[device_id]
            # 断开连接
            if hasattr(connection_info.controller, 'disconnect'):
                await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    connection_info.controller.disconnect
                )
            
            del self.connections[device_id]
            self.active_connections -= 1
            logger.info(f"已移除设备连接: {device_id}")
            return True
        return False
    
    async def _periodic_cleanup(self):
        """定期清理空闲连接"""
        while True:
            try:
                current_time = time.time()
                to_remove = []
                
                for device_id, connection_info in self.connections.items():
                    idle_time = current_time - connection_info.last_activity
                    if idle_time > connection_info.max_idle_time:
                        to_remove.append(device_id)
                
                for device_id in to_remove:
                    logger.info(f"清理空闲连接: {device_id}")
                    await self.remove_connection(device_id)
                
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"清理连接池时发生错误: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        connected_count = sum(1 for info in self.connections.values() 
                            if info.status == ConnectionStatus.CONNECTED)
        connecting_count = sum(1 for info in self.connections.values() 
                             if info.status == ConnectionStatus.CONNECTING)
        error_count = sum(1 for info in self.connections.values() 
                        if info.status == ConnectionStatus.ERROR)
        
        return {
            "total_connections": len(self.connections),
            "active_connections": self.active_connections,
            "max_connections": self.max_connections,
            "connected": connected_count,
            "connecting": connecting_count,
            "error": error_count,
            "disconnected": len(self.connections) - connected_count - connecting_count - error_count
        }
    
    async def shutdown(self):
        """关闭连接池管理器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 断开所有连接
        for device_id in list(self.connections.keys()):
            await self.remove_connection(device_id)
        
        self._executor.shutdown(wait=True)
        logger.info("连接池管理器已关闭")


# 全局连接池管理器实例
connection_pool_manager = ConnectionPoolManager()