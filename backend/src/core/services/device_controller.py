"""设备控制服务
提供设备发现、设备管理、命令路由、状态监控等功能
"""

from typing import Optional, Dict, List, Any
import logging
import uuid
import time
from enum import Enum
import json

# 配置日志
logger = logging.getLogger(__name__)


class DeviceStatus(str, Enum):
    """设备状态枚举"""
    ONLINE = "online"       # 在线
    OFFLINE = "offline"     # 离线
    BUSY = "busy"          # 忙碌
    ERROR = "error"         # 错误
    UNKNOWN = "unknown"     # 未知


class DeviceType(str, Enum):
    """设备类型枚举"""
    LIGHT = "light"               # 灯光
    THERMOSTAT = "thermostat"     # 恒温器
    CAMERA = "camera"             # 摄像头
    SPEAKER = "speaker"           # 音箱
    SENSOR = "sensor"             # 传感器
    DOOR_LOCK = "door_lock"       # 门锁
    CURTAIN = "curtain"           # 窗帘
    SWITCH = "switch"             # 开关
    OTHERS = "others"             # 其他


class Device:
    """设备类"""
    
    def __init__(self, device_id: str, name: str, device_type: str, 
                 protocol: str, endpoint: str, status: str = DeviceStatus.UNKNOWN):
        """初始化设备
        
        Args:
            device_id: 设备ID
            name: 设备名称
            device_type: 设备类型
            protocol: 通信协议
            endpoint: 设备端点
            status: 设备状态
        """
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.protocol = protocol
        self.endpoint = endpoint
        self.status = status
        self.last_seen = time.time()
        self.properties = {}
        self.capabilities = []
        self.group_id = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "status": self.status,
            "last_seen": self.last_seen,
            "properties": self.properties,
            "capabilities": self.capabilities,
            "group_id": self.group_id
        }
    
    def update_status(self, status: DeviceStatus):
        """更新设备状态"""
        self.status = status
        self.last_seen = time.time()
    
    def update_properties(self, properties: Dict[str, Any]):
        """更新设备属性"""
        self.properties.update(properties)
        self.last_seen = time.time()


class DeviceController:
    """设备控制器类"""
    
    def __init__(self):
        """初始化设备控制器"""
        self.devices = {}  # 设备注册表
        self.groups = {}   # 设备组注册表
        self.device_discovery_callbacks = []  # 设备发现回调
        self.status_update_callbacks = []     # 状态更新回调
        logger.info("设备控制器初始化完成")
    
    def register_device(self, device: Device) -> bool:
        """注册设备
        
        Args:
            device: 设备对象
            
        Returns:
            注册是否成功
        """
        try:
            self.devices[device.device_id] = device
            logger.info(f"设备注册成功: {device.name} ({device.device_id})")
            
            # 调用设备发现回调
            for callback in self.device_discovery_callbacks:
                try:
                    callback(device)
                except Exception as e:
                    logger.error(f"设备发现回调失败: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"设备注册失败: {str(e)}")
            return False
    
    def unregister_device(self, device_id: str) -> bool:
        """注销设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            注销是否成功
        """
        try:
            if device_id in self.devices:
                del self.devices[device_id]
                logger.info(f"设备注销成功: {device_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"设备注销失败: {str(e)}")
            return False
    
    def discover_devices(self, protocol: Optional[str] = None) -> List[Device]:
        """发现设备
        
        Args:
            protocol: 通信协议
            
        Returns:
            发现的设备列表
        """
        discovered_devices = []
        logger.info(f"开始设备发现，协议: {protocol}")
        
        # 模拟设备发现，实际项目中应实现真实的设备发现逻辑
        # 例如：使用SSDP、mDNS、MQTT等协议进行设备发现
        
        # 模拟发现几个设备
        mock_devices = [
            Device(
                device_id=f"device_{uuid.uuid4()}",
                name="客厅灯",
                device_type=DeviceType.LIGHT,
                protocol="mqtt",
                endpoint="mqtt://localhost:1883/device/light/living",
                status=DeviceStatus.ONLINE
            ),
            Device(
                device_id=f"device_{uuid.uuid4()}",
                name="卧室温度传感器",
                device_type=DeviceType.SENSOR,
                protocol="http",
                endpoint="http://192.168.1.100/api",
                status=DeviceStatus.ONLINE
            ),
            Device(
                device_id=f"device_{uuid.uuid4()}",
                name="前门锁",
                device_type=DeviceType.DOOR_LOCK,
                protocol="zigbee",
                endpoint="zigbee://0x12345678",
                status=DeviceStatus.ONLINE
            )
        ]
        
        # 注册发现的设备
        for device in mock_devices:
            if not protocol or device.protocol == protocol:
                self.register_device(device)
                discovered_devices.append(device)
        
        logger.info(f"设备发现完成，发现 {len(discovered_devices)} 个设备")
        return discovered_devices
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """获取设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            设备对象或None
        """
        return self.devices.get(device_id)
    
    def get_devices_by_type(self, device_type: str) -> List[Device]:
        """根据设备类型获取设备列表
        
        Args:
            device_type: 设备类型
            
        Returns:
            设备列表
        """
        return [device for device in self.devices.values() 
                if device.device_type == device_type]
    
    def get_devices_by_status(self, status: DeviceStatus) -> List[Device]:
        """根据设备状态获取设备列表
        
        Args:
            status: 设备状态
            
        Returns:
            设备列表
        """
        return [device for device in self.devices.values() 
                if device.status == status]
    
    def get_online_devices(self) -> List[Device]:
        """获取在线设备列表
        
        Returns:
            在线设备列表
        """
        return self.get_devices_by_status(DeviceStatus.ONLINE)
    
    def send_command(self, device_id: str, command: str, 
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """向设备发送命令
        
        Args:
            device_id: 设备ID
            command: 命令
            params: 命令参数
            
        Returns:
            命令执行结果
        """
        try:
            if device_id not in self.devices:
                logger.error(f"设备不存在: {device_id}")
                return {
                    "success": False,
                    "error": f"设备不存在: {device_id}"
                }
            
            device = self.devices[device_id]
            
            if device.status != DeviceStatus.ONLINE:
                logger.error(f"设备不在线: {device_id}")
                return {
                    "success": False,
                    "error": f"设备不在线: {device_id}"
                }
            
            logger.info(f"向设备发送命令: {device_id} - {command} {params}")
            
            # 根据设备协议发送命令
            # 这里是模拟实现，实际项目中应根据不同协议实现真实的命令发送逻辑
            result = {
                "success": True,
                "device_id": device_id,
                "command": command,
                "params": params,
                "timestamp": time.time(),
                "message": f"命令已发送到设备: {device.name}"
            }
            
            # 更新设备状态为忙碌
            device.update_status(DeviceStatus.BUSY)
            
            # 模拟命令执行延迟
            time.sleep(0.5)
            
            # 更新设备状态为在线
            device.update_status(DeviceStatus.ONLINE)
            
            # 更新设备属性
            if command == "turn_on" and device.device_type == DeviceType.LIGHT:
                device.update_properties({"power": "on"})
            elif command == "turn_off" and device.device_type == DeviceType.LIGHT:
                device.update_properties({"power": "off"})
            
            logger.info(f"命令执行成功: {device_id}")
            return result
            
        except Exception as e:
            logger.error(f"命令发送失败: {str(e)}")
            return {
                "success": False,
                "error": f"命令发送失败: {str(e)}"
            }
    
    def broadcast_command(self, command: str, params: Optional[Dict[str, Any]] = None, 
                         device_type: Optional[str] = None) -> Dict[str, Any]:
        """广播命令到多个设备
        
        Args:
            command: 命令
            params: 命令参数
            device_type: 设备类型（可选，指定类型则只发送给该类型设备）
            
        Returns:
            命令执行结果
        """
        try:
            results = {}
            
            # 获取目标设备列表
            if device_type:
                target_devices = self.get_devices_by_type(device_type)
            else:
                target_devices = list(self.devices.values())
            
            logger.info(f"广播命令: {command} 到 {len(target_devices)} 个设备")
            
            # 向每个设备发送命令
            for device in target_devices:
                if device.status == DeviceStatus.ONLINE:
                    result = self.send_command(device.device_id, command, params)
                    results[device.device_id] = result
            
            return {
                "success": True,
                "total_devices": len(target_devices),
                "online_devices": len([d for d in target_devices if d.status == DeviceStatus.ONLINE]),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"广播命令失败: {str(e)}")
            return {
                "success": False,
                "error": f"广播命令失败: {str(e)}"
            }
    
    def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """获取设备状态
        
        Args:
            device_id: 设备ID
            
        Returns:
            设备状态信息
        """
        try:
            device = self.get_device(device_id)
            if not device:
                return {
                    "success": False,
                    "error": f"设备不存在: {device_id}"
                }
            
            return {
                "success": True,
                "device_id": device_id,
                "status": device.status,
                "properties": device.properties,
                "last_seen": device.last_seen,
                "name": device.name,
                "type": device.device_type
            }
            
        except Exception as e:
            logger.error(f"获取设备状态失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取设备状态失败: {str(e)}"
            }
    
    def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """更新设备状态
        
        Args:
            device_id: 设备ID
            status: 设备状态
            
        Returns:
            更新是否成功
        """
        try:
            device = self.get_device(device_id)
            if not device:
                return False
            
            old_status = device.status
            device.update_status(status)
            
            # 调用状态更新回调
            for callback in self.status_update_callbacks:
                try:
                    callback(device, old_status, status)
                except Exception as e:
                    logger.error(f"状态更新回调失败: {str(e)}")
            
            logger.info(f"设备状态更新: {device_id} - {old_status} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"更新设备状态失败: {str(e)}")
            return False
    
    def register_device_discovery_callback(self, callback):
        """注册设备发现回调
        
        Args:
            callback: 回调函数
        """
        self.device_discovery_callbacks.append(callback)
    
    def register_status_update_callback(self, callback):
        """注册状态更新回调
        
        Args:
            callback: 回调函数
        """
        self.status_update_callbacks.append(callback)
    
    def create_device_group(self, name: str, device_ids: List[str]) -> str:
        """创建设备组
        
        Args:
            name: 组名称
            device_ids: 设备ID列表
            
        Returns:
            组ID
        """
        try:
            group_id = f"group_{uuid.uuid4()}"
            
            # 验证设备是否存在
            valid_device_ids = []
            for device_id in device_ids:
                if device_id in self.devices:
                    valid_device_ids.append(device_id)
                    # 将设备添加到组
                    self.devices[device_id].group_id = group_id
            
            self.groups[group_id] = {
                "group_id": group_id,
                "name": name,
                "device_ids": valid_device_ids,
                "created_at": time.time()
            }
            
            logger.info(f"设备组创建成功: {name} ({group_id})")
            return group_id
            
        except Exception as e:
            logger.error(f"创建设备组失败: {str(e)}")
            return ""
    
    def send_group_command(self, group_id: str, command: str, 
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """向设备组发送命令
        
        Args:
            group_id: 组ID
            command: 命令
            params: 命令参数
            
        Returns:
            命令执行结果
        """
        try:
            if group_id not in self.groups:
                return {
                    "success": False,
                    "error": f"设备组不存在: {group_id}"
                }
            
            group = self.groups[group_id]
            
            # 向组内所有设备发送命令
            results = {}
            for device_id in group["device_ids"]:
                result = self.send_command(device_id, command, params)
                results[device_id] = result
            
            return {
                "success": True,
                "group_id": group_id,
                "group_name": group["name"],
                "total_devices": len(group["device_ids"]),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"发送组命令失败: {str(e)}")
            return {
                "success": False,
                "error": f"发送组命令失败: {str(e)}"
            }
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """获取所有设备
        
        Returns:
            设备列表
        """
        return [device.to_dict() for device in self.devices.values()]
    
    def health_check(self) -> Dict[str, Any]:
        """设备健康检查
        
        Returns:
            健康检查结果
        """
        total_devices = len(self.devices)
        online_devices = len(self.get_online_devices())
        offline_devices = total_devices - online_devices
        
        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "timestamp": time.time()
        }
    
    async def control_device(self, device_id: str, command: str, 
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """控制设备（异步方法，供场景管理器调用）
        
        Args:
            device_id: 设备ID
            command: 命令
            params: 命令参数
            
        Returns:
            命令执行结果
        """
        return self.send_command(device_id, command, params)


# 单例模式
device_controller = DeviceController()