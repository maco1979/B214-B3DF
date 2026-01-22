"""设备DAO类
提供设备数据的访问接口
"""

import logging
from typing import List, Dict, Any, Optional
from .base_dao import BaseDAO

# 导入缓存服务
from ..services.async_cache_service import memory_cache

# 导入安全工具
from ..utils.security_utils import security_utils

logger = logging.getLogger(__name__)


class DeviceDAO(BaseDAO):
    """设备数据访问对象"""
    
    def __init__(self, initial_data: List[Dict[str, Any]] = None):
        """初始化设备DAO"""
        super().__init__()
        self._data_store = {}  # 主数据存储，ID -> 设备数据
        
        # 添加索引字典加速查询
        self._indexes = {
            "type": {},  # 设备类型索引，type -> [device_id]
            "connected": {}  # 连接状态索引，connected -> [device_id]
        }
        
        # 初始化数据
        if initial_data:
            for device in initial_data:
                # 加密敏感数据
                encrypted_device = self._encrypt_sensitive_data(device.copy())
                device_id = encrypted_device["id"]
                self._data_store[device_id] = encrypted_device
                
                # 更新索引
                self._update_indexes(device_id, encrypted_device)
            
            logger.info(f"设备DAO初始化完成，加载了 {len(initial_data)} 个设备")
    
    def _encrypt_sensitive_data(self, device: Dict[str, Any]) -> Dict[str, Any]:
        """加密设备中的敏感数据"""
        for key, value in device.items():
            if security_utils.is_sensitive_data(key) and isinstance(value, str):
                device[key] = security_utils.encrypt_data(value)
            elif isinstance(value, dict):
                # 递归加密嵌套字典中的敏感数据
                for sub_key, sub_value in value.items():
                    if security_utils.is_sensitive_data(sub_key) and isinstance(sub_value, str):
                        value[sub_key] = security_utils.encrypt_data(sub_value)
        return device
    
    def _decrypt_sensitive_data(self, device: Dict[str, Any]) -> Dict[str, Any]:
        """解密设备中的敏感数据"""
        for key, value in device.items():
            if security_utils.is_sensitive_data(key) and isinstance(value, dict) and "encrypted_data" in value and "iv" in value:
                # 解密敏感数据
                try:
                    device[key] = security_utils.decrypt_data(value["encrypted_data"], value["iv"])
                except Exception as e:
                    logger.error(f"解密敏感数据 {key} 失败: {e}")
                    device[key] = "[解密失败]"
            elif isinstance(value, dict):
                # 递归解密嵌套字典中的敏感数据
                for sub_key, sub_value in value.items():
                    if security_utils.is_sensitive_data(sub_key) and isinstance(sub_value, dict) and "encrypted_data" in sub_value and "iv" in sub_value:
                        try:
                            value[sub_key] = security_utils.decrypt_data(sub_value["encrypted_data"], sub_value["iv"])
                        except Exception as e:
                            logger.error(f"解密嵌套敏感数据 {sub_key} 失败: {e}")
                            value[sub_key] = "[解密失败]"
        return device
    
    def _update_indexes(self, device_id: int, device_data: Dict[str, Any]):
        """更新设备索引"""
        # 更新类型索引
        device_type = device_data.get("type", "")
        if device_type not in self._indexes["type"]:
            self._indexes["type"][device_type] = []
        if device_id not in self._indexes["type"][device_type]:
            self._indexes["type"][device_type].append(device_id)
        
        # 更新连接状态索引
        connected = device_data.get("connected", False)
        connected_key = str(connected)
        if connected_key not in self._indexes["connected"]:
            self._indexes["connected"][connected_key] = []
        if device_id not in self._indexes["connected"][connected_key]:
            self._indexes["connected"][connected_key].append(device_id)
    
    def _remove_from_indexes(self, device_id: int, device_data: Dict[str, Any]):
        """从索引中移除设备"""
        # 从类型索引移除
        device_type = device_data.get("type", "")
        if device_type in self._indexes["type"]:
            if device_id in self._indexes["type"][device_type]:
                self._indexes["type"][device_type].remove(device_id)
            # 清理空索引
            if not self._indexes["type"][device_type]:
                del self._indexes["type"][device_type]
        
        # 从连接状态索引移除
        connected = device_data.get("connected", False)
        connected_key = str(connected)
        if connected_key in self._indexes["connected"]:
            if device_id in self._indexes["connected"][connected_key]:
                self._indexes["connected"][connected_key].remove(device_id)
            # 清理空索引
            if not self._indexes["connected"][connected_key]:
                del self._indexes["connected"][connected_key]
    
    async def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取设备"""
        # 生成缓存键
        cache_key = f"device:{id}"
        
        # 尝试从缓存获取
        cached_device = await memory_cache.get(cache_key)
        if cached_device:
            logger.debug(f"从缓存获取设备 {id} 成功")
            # 解密敏感数据后返回
            return self._decrypt_sensitive_data(cached_device.copy())
        
        # 从数据存储获取
        device = self._data_store.get(id)
        if device:
            logger.debug(f"根据ID {id} 获取设备成功")
            # 解密敏感数据
            decrypted_device = self._decrypt_sensitive_data(device.copy())
            # 缓存结果
            await memory_cache.set(cache_key, device, ttl=300)
            return decrypted_device
        else:
            logger.debug(f"根据ID {id} 未找到设备")
        return None
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """获取所有设备"""
        # 生成缓存键
        cache_key = "devices:all"
        
        # 尝试从缓存获取
        cached_devices = await memory_cache.get(cache_key)
        if cached_devices:
            logger.debug(f"从缓存获取所有设备成功，共 {len(cached_devices)} 个")
            # 解密每个设备的敏感数据
            return [self._decrypt_sensitive_data(device.copy()) for device in cached_devices]
        
        # 从数据存储获取
        devices = list(self._data_store.values())
        logger.debug(f"获取所有设备，共 {len(devices)} 个")
        # 缓存结果
        await memory_cache.set(cache_key, devices, ttl=300)
        # 解密每个设备的敏感数据
        return [self._decrypt_sensitive_data(device.copy()) for device in devices]
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建设备"""
        # 生成唯一ID
        device_id = self._generate_id()
        device = {
            "id": device_id,
            **data
        }
        
        # 加密敏感数据
        encrypted_device = self._encrypt_sensitive_data(device)
        self._data_store[device_id] = encrypted_device
        
        # 更新索引
        self._update_indexes(device_id, encrypted_device)
        
        logger.info(f"创建设备成功，ID: {device_id}")
        
        # 清除相关缓存
        await memory_cache.delete("devices:all")
        await memory_cache.delete(f"devices:online")
        if "type" in device:
            await memory_cache.delete(f"devices:type:{device['type']}")
        
        # 返回解密后的设备数据
        return self._decrypt_sensitive_data(device)

    
    async def update(self, id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新设备"""
        if id not in self._data_store:
            logger.warning(f"更新设备失败，ID: {id} 不存在")
            return None
        
        # 获取现有设备数据
        existing_device = self._data_store[id].copy()
        
        # 从旧索引中移除
        self._remove_from_indexes(id, existing_device)
        
        # 解密现有设备数据
        decrypted_device = self._decrypt_sensitive_data(existing_device)
        
        # 合并更新数据
        decrypted_device.update(data)
        
        # 加密敏感数据
        encrypted_device = self._encrypt_sensitive_data(decrypted_device.copy())
        self._data_store[id] = encrypted_device
        
        # 添加到新索引
        self._update_indexes(id, encrypted_device)
        
        logger.info(f"更新设备成功，ID: {id}")
        
        # 清除相关缓存
        await memory_cache.delete(f"device:{id}")
        await memory_cache.delete("devices:all")
        await memory_cache.delete(f"devices:online")
        
        # 清除旧类型缓存
        old_type = existing_device.get("type", "")
        if old_type:
            await memory_cache.delete(f"devices:type:{old_type}")
        
        # 清除新类型缓存
        new_type = encrypted_device.get("type", "")
        if new_type:
            await memory_cache.delete(f"devices:type:{new_type}")
        
        # 返回解密后的设备数据
        return decrypted_device
    
    async def delete(self, id: int) -> bool:
        """删除设备"""
        if id not in self._data_store:
            logger.warning(f"删除设备失败，ID: {id} 不存在")
            return False
        
        # 获取设备信息
        device = self._data_store[id]
        device_type = device.get("type", "")
        
        # 从索引中移除
        self._remove_from_indexes(id, device)
        
        # 删除设备数据
        del self._data_store[id]
        logger.info(f"删除设备成功，ID: {id}")
        
        # 清除相关缓存
        await memory_cache.delete(f"device:{id}")
        await memory_cache.delete("devices:all")
        await memory_cache.delete(f"devices:online")
        if device_type:
            await memory_cache.delete(f"devices:type:{device_type}")
        
        return True
    
    async def get_online_devices(self) -> List[Dict[str, Any]]:
        """获取在线设备"""
        # 生成缓存键
        cache_key = "devices:online"
        
        # 尝试从缓存获取
        cached_devices = await memory_cache.get(cache_key)
        if cached_devices:
            logger.debug(f"从缓存获取在线设备成功，共 {len(cached_devices)} 个")
            # 解密每个设备的敏感数据
            return [self._decrypt_sensitive_data(device.copy()) for device in cached_devices]
        
        # 使用索引加速查询
        online_device_ids = self._indexes["connected"].get("True", [])
        online_devices = []
        
        for device_id in online_device_ids:
            device = self._data_store.get(device_id)
            if device and device.get("status", "offline") == "online":
                online_devices.append(device)
        
        logger.debug(f"获取在线设备，共 {len(online_devices)} 个，使用索引加速查询")
        # 缓存结果
        await memory_cache.set(cache_key, online_devices, ttl=300)
        # 解密每个设备的敏感数据
        return [self._decrypt_sensitive_data(device.copy()) for device in online_devices]
    
    async def get_devices_by_type(self, device_type: str) -> List[Dict[str, Any]]:
        """根据设备类型获取设备"""
        # 生成缓存键
        cache_key = f"devices:type:{device_type}"
        
        # 尝试从缓存获取
        cached_devices = await memory_cache.get(cache_key)
        if cached_devices:
            logger.debug(f"从缓存获取类型为 {device_type} 的设备成功，共 {len(cached_devices)} 个")
            # 解密每个设备的敏感数据
            return [self._decrypt_sensitive_data(device.copy()) for device in cached_devices]
        
        # 使用索引加速查询
        device_ids = self._indexes["type"].get(device_type, [])
        devices = []
        
        for device_id in device_ids:
            device = self._data_store.get(device_id)
            if device:
                devices.append(device)
        
        logger.debug(f"根据类型 {device_type} 获取设备，共 {len(devices)} 个，使用索引加速查询")
        # 缓存结果
        await memory_cache.set(cache_key, devices, ttl=300)
        # 解密每个设备的敏感数据
        return [self._decrypt_sensitive_data(device.copy()) for device in devices]
    
    async def update_device_connection(self, id: int, connected: bool) -> Optional[Dict[str, Any]]:
        """更新设备连接状态"""
        result = await self.update(id, {
            "connected": connected,
            "status": "online" if connected else "offline"
        })
        return result