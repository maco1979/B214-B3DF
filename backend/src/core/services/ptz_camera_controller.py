"""
PTZ云台摄像头控制器
支持真实的云台摄像头物理转动、变焦、对焦等操作
适用于农业监控、无人机控制、智能安防等场景
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import serial
import socket
import json

logger = logging.getLogger(__name__)


class PTZProtocol(Enum):
    """云台控制协议"""
    PELCO_D = "pelco_d"  # Pelco-D协议（最常用）
    PELCO_P = "pelco_p"  # Pelco-P协议
    VISCA = "visca"      # VISCA协议（Sony等）
    ONVIF = "onvif"      # ONVIF标准协议
    HTTP_API = "http"    # HTTP API接口


class PTZAction(Enum):
    """云台动作"""
    PAN_LEFT = "pan_left"          # 向左转
    PAN_RIGHT = "pan_right"        # 向右转
    TILT_UP = "tilt_up"            # 向上转
    TILT_DOWN = "tilt_down"        # 向下转
    ZOOM_IN = "zoom_in"            # 拉近
    ZOOM_OUT = "zoom_out"          # 拉远
    FOCUS_NEAR = "focus_near"      # 近焦
    FOCUS_FAR = "focus_far"        # 远焦
    IRIS_OPEN = "iris_open"        # 光圈开
    IRIS_CLOSE = "iris_close"      # 光圈关
    PRESET_SET = "preset_set"      # 设置预置位
    PRESET_GOTO = "preset_goto"    # 转到预置位
    AUTO_SCAN = "auto_scan"        # 自动扫描
    PATROL = "patrol"              # 巡航
    STOP = "stop"                  # 停止


class PTZCameraController:
    """PTZ云台摄像头控制器"""
    
    def __init__(self, 
                 protocol: PTZProtocol = PTZProtocol.PELCO_D,
                 connection_type: str = "serial",
                 **connection_params):
        """
        初始化云台控制器
        
        Args:
            protocol: 云台控制协议
            connection_type: 连接类型 (serial, network, http)
            connection_params: 连接参数
                - serial: port, baudrate, address
                - network: host, port, address
                - http: base_url, username, password
        """
        self.protocol = protocol
        self.connection_type = connection_type
        self.connection_params = connection_params
        self.connection = None
        self.is_connected = False
        
        # 云台状态
        self.current_pan = 0.0    # 水平角度 (-180 to 180)
        self.current_tilt = 0.0   # 垂直角度 (-90 to 90)
        self.current_zoom = 1.0   # 变焦倍数 (1.0 to max_zoom)
        self.max_zoom = 20.0
        
        # 预置位
        self.presets = {}  # {preset_id: {"pan": float, "tilt": float, "zoom": float}}
        
        logger.info(f"PTZ控制器初始化: protocol={protocol.value}, type={connection_type}")
    
    async def connect(self) -> Dict[str, Any]:
        """
        连接云台摄像头
        
        Returns:
            Dict: {"success": bool, "message": str}
        """
        try:
            if self.connection_type == "serial":
                return await self._connect_serial()
            elif self.connection_type == "network":
                return await self._connect_network()
            elif self.connection_type == "http":
                return await self._connect_http()
            else:
                return {
                    "success": False,
                    "message": f"不支持的连接类型: {self.connection_type}"
                }
        except Exception as e:
            logger.error(f"连接云台失败: {e}")
            return {"success": False, "message": f"连接失败: {str(e)}"}
    
    async def _connect_serial(self) -> Dict[str, Any]:
        """通过串口连接云台"""
        try:
            port = self.connection_params.get("port", "/dev/ttyUSB0")
            baudrate = self.connection_params.get("baudrate", 9600)
            
            # 异步打开串口
            def open_serial():
                return serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1
                )
            
            self.connection = await asyncio.to_thread(open_serial)
            self.is_connected = True
            
            logger.info(f"串口连接成功: {port}@{baudrate}")
            return {
                "success": True,
                "message": f"串口连接成功: {port}",
                "connection_info": {
                    "port": port,
                    "baudrate": baudrate,
                    "protocol": self.protocol.value
                }
            }
        except Exception as e:
            logger.error(f"串口连接失败: {e}")
            return {"success": False, "message": f"串口连接失败: {str(e)}"}
    
    async def _connect_network(self) -> Dict[str, Any]:
        """通过网络连接云台"""
        try:
            host = self.connection_params.get("host", "192.168.1.100")
            port = self.connection_params.get("port", 5000)
            
            # 异步创建TCP连接
            reader, writer = await asyncio.open_connection(host, port)
            self.connection = {"reader": reader, "writer": writer}
            self.is_connected = True
            
            logger.info(f"网络连接成功: {host}:{port}")
            return {
                "success": True,
                "message": f"网络连接成功: {host}:{port}",
                "connection_info": {
                    "host": host,
                    "port": port,
                    "protocol": self.protocol.value
                }
            }
        except Exception as e:
            logger.error(f"网络连接失败: {e}")
            return {"success": False, "message": f"网络连接失败: {str(e)}"}
    
    async def _connect_http(self) -> Dict[str, Any]:
        """通过HTTP API连接云台"""
        try:
            base_url = self.connection_params.get("base_url", "http://192.168.1.100")
            username = self.connection_params.get("username", "admin")
            password = self.connection_params.get("password", "admin")
            
            # 测试连接
            import aiohttp
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(username, password)
                async with session.get(f"{base_url}/api/status", auth=auth) as resp:
                    if resp.status == 200:
                        self.connection = {
                            "base_url": base_url,
                            "auth": {"username": username, "password": password}
                        }
                        self.is_connected = True
                        
                        logger.info(f"HTTP连接成功: {base_url}")
                        return {
                            "success": True,
                            "message": f"HTTP连接成功: {base_url}",
                            "connection_info": {
                                "base_url": base_url,
                                "protocol": self.protocol.value
                            }
                        }
                    else:
                        return {"success": False, "message": f"HTTP连接失败: {resp.status}"}
        except Exception as e:
            logger.error(f"HTTP连接失败: {e}")
            return {"success": False, "message": f"HTTP连接失败: {str(e)}"}
    
    async def disconnect(self) -> Dict[str, Any]:
        """断开连接"""
        try:
            if self.connection_type == "serial" and self.connection:
                await asyncio.to_thread(self.connection.close)
            elif self.connection_type == "network" and self.connection:
                self.connection["writer"].close()
                await self.connection["writer"].wait_closed()
            
            self.connection = None
            self.is_connected = False
            
            logger.info("云台连接已断开")
            return {"success": True, "message": "连接已断开"}
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
            return {"success": False, "message": f"断开连接失败: {str(e)}"}
    
    async def execute_action(self, 
                            action: PTZAction, 
                            speed: int = 50,
                            **params) -> Dict[str, Any]:
        """
        执行云台动作
        
        Args:
            action: 云台动作
            speed: 速度 (0-100)
            params: 其他参数 (如预置位编号等)
        
        Returns:
            Dict: {"success": bool, "message": str, "new_state": dict}
        """
        if not self.is_connected:
            return {"success": False, "message": "云台未连接"}
        
        try:
            # 根据协议生成命令
            if self.protocol == PTZProtocol.PELCO_D:
                command = await self._build_pelco_d_command(action, speed, **params)
            elif self.protocol == PTZProtocol.VISCA:
                command = await self._build_visca_command(action, speed, **params)
            elif self.protocol == PTZProtocol.ONVIF:
                command = await self._build_onvif_command(action, speed, **params)
            elif self.protocol == PTZProtocol.HTTP_API:
                command = await self._build_http_command(action, speed, **params)
            else:
                return {"success": False, "message": f"不支持的协议: {self.protocol.value}"}
            
            # 发送命令
            result = await self._send_command(command)
            
            # 更新状态
            self._update_state(action, speed, **params)
            
            return {
                "success": True,
                "message": f"{action.value} 执行成功",
                "action": action.value,
                "new_state": {
                    "pan": self.current_pan,
                    "tilt": self.current_tilt,
                    "zoom": self.current_zoom
                }
            }
        except Exception as e:
            logger.error(f"执行云台动作失败: {e}")
            return {"success": False, "message": f"执行失败: {str(e)}"}
    
    async def _build_pelco_d_command(self, action: PTZAction, speed: int, **params) -> bytes:
        """构建Pelco-D协议命令"""
        address = self.connection_params.get("address", 1)
        
        # Pelco-D命令格式: 0xFF + 地址 + 命令1 + 命令2 + 数据1 + 数据2 + 校验和
        sync = 0xFF
        cmd1 = 0x00
        cmd2 = 0x00
        data1 = speed  # 水平速度
        data2 = speed  # 垂直速度
        
        # 根据动作设置命令字节
        if action == PTZAction.PAN_LEFT:
            cmd2 = 0x04
        elif action == PTZAction.PAN_RIGHT:
            cmd2 = 0x02
        elif action == PTZAction.TILT_UP:
            cmd2 = 0x08
        elif action == PTZAction.TILT_DOWN:
            cmd2 = 0x10
        elif action == PTZAction.ZOOM_IN:
            cmd2 = 0x20
        elif action == PTZAction.ZOOM_OUT:
            cmd2 = 0x40
        elif action == PTZAction.STOP:
            cmd1 = 0x00
            cmd2 = 0x00
            data1 = 0x00
            data2 = 0x00
        elif action == PTZAction.PRESET_SET:
            cmd1 = 0x00
            cmd2 = 0x03
            data2 = params.get("preset_id", 1)
        elif action == PTZAction.PRESET_GOTO:
            cmd1 = 0x00
            cmd2 = 0x07
            data2 = params.get("preset_id", 1)
        
        # 计算校验和
        checksum = (address + cmd1 + cmd2 + data1 + data2) % 256
        
        return bytes([sync, address, cmd1, cmd2, data1, data2, checksum])
    
    async def _build_visca_command(self, action: PTZAction, speed: int, **params) -> bytes:
        """构建VISCA协议命令"""
        # VISCA命令格式: 0x81 + 命令 + 0xFF
        # 这里只提供基本实现
        address = 1
        
        if action == PTZAction.PAN_RIGHT:
            # Pan Right
            pan_speed = min(speed, 24)
            tilt_speed = 0
            return bytes([0x81, 0x01, 0x06, 0x01, pan_speed, tilt_speed, 0x02, 0x03, 0xFF])
        elif action == PTZAction.ZOOM_IN:
            # Zoom In
            zoom_speed = min(speed // 10, 7)
            return bytes([0x81, 0x01, 0x04, 0x07, 0x20 + zoom_speed, 0xFF])
        elif action == PTZAction.STOP:
            # Stop
            return bytes([0x81, 0x01, 0x06, 0x01, 0x00, 0x00, 0x03, 0x03, 0xFF])
        
        # 默认停止命令
        return bytes([0x81, 0x01, 0x06, 0x01, 0x00, 0x00, 0x03, 0x03, 0xFF])
    
    async def _build_onvif_command(self, action: PTZAction, speed: int, **params) -> Dict[str, Any]:
        """构建ONVIF协议命令（SOAP XML）"""
        # ONVIF需要SOAP XML命令，这里返回字典表示
        velocity = speed / 100.0
        
        command = {
            "action": action.value,
            "velocity": {
                "pan": 0.0,
                "tilt": 0.0,
                "zoom": 0.0
            }
        }
        
        if action == PTZAction.PAN_LEFT:
            command["velocity"]["pan"] = -velocity
        elif action == PTZAction.PAN_RIGHT:
            command["velocity"]["pan"] = velocity
        elif action == PTZAction.TILT_UP:
            command["velocity"]["tilt"] = velocity
        elif action == PTZAction.TILT_DOWN:
            command["velocity"]["tilt"] = -velocity
        elif action == PTZAction.ZOOM_IN:
            command["velocity"]["zoom"] = velocity
        elif action == PTZAction.ZOOM_OUT:
            command["velocity"]["zoom"] = -velocity
        
        return command
    
    async def _build_http_command(self, action: PTZAction, speed: int, **params) -> Dict[str, Any]:
        """构建HTTP API命令"""
        return {
            "action": action.value,
            "speed": speed,
            **params
        }
    
    async def _send_command(self, command: Any) -> Dict[str, Any]:
        """发送命令到云台"""
        try:
            if self.connection_type == "serial":
                # 发送串口命令
                await asyncio.to_thread(self.connection.write, command)
                return {"success": True}
            
            elif self.connection_type == "network":
                # 发送网络命令
                writer = self.connection["writer"]
                writer.write(command)
                await writer.drain()
                return {"success": True}
            
            elif self.connection_type == "http":
                # 发送HTTP命令
                import aiohttp
                base_url = self.connection["base_url"]
                auth_data = self.connection["auth"]
                
                async with aiohttp.ClientSession() as session:
                    auth = aiohttp.BasicAuth(auth_data["username"], auth_data["password"])
                    async with session.post(
                        f"{base_url}/api/ptz/control",
                        json=command,
                        auth=auth
                    ) as resp:
                        return {"success": resp.status == 200}
            
            return {"success": False}
        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_state(self, action: PTZAction, speed: int, **params):
        """更新云台状态"""
        # 根据动作更新位置（简化模拟）
        speed_factor = speed / 100.0
        
        if action == PTZAction.PAN_LEFT:
            self.current_pan -= 5 * speed_factor
        elif action == PTZAction.PAN_RIGHT:
            self.current_pan += 5 * speed_factor
        elif action == PTZAction.TILT_UP:
            self.current_tilt += 3 * speed_factor
        elif action == PTZAction.TILT_DOWN:
            self.current_tilt -= 3 * speed_factor
        elif action == PTZAction.ZOOM_IN:
            self.current_zoom = min(self.current_zoom + 0.5 * speed_factor, self.max_zoom)
        elif action == PTZAction.ZOOM_OUT:
            self.current_zoom = max(self.current_zoom - 0.5 * speed_factor, 1.0)
        elif action == PTZAction.PRESET_GOTO:
            preset_id = params.get("preset_id")
            if preset_id in self.presets:
                preset = self.presets[preset_id]
                self.current_pan = preset["pan"]
                self.current_tilt = preset["tilt"]
                self.current_zoom = preset["zoom"]
        
        # 限制范围
        self.current_pan = max(-180, min(180, self.current_pan))
        self.current_tilt = max(-90, min(90, self.current_tilt))
    
    async def move_to_position(self, 
                              pan: float, 
                              tilt: float, 
                              zoom: Optional[float] = None,
                              speed: int = 50) -> Dict[str, Any]:
        """
        移动到指定位置（绝对位置控制）
        
        Args:
            pan: 目标水平角度 (-180 to 180)
            tilt: 目标垂直角度 (-90 to 90)
            zoom: 目标变焦倍数 (可选)
            speed: 移动速度 (0-100)
        
        Returns:
            Dict: 执行结果
        """
        try:
            # 计算需要移动的方向和距离
            pan_diff = pan - self.current_pan
            tilt_diff = tilt - self.current_tilt
            
            # 执行水平移动
            if abs(pan_diff) > 1:
                action = PTZAction.PAN_RIGHT if pan_diff > 0 else PTZAction.PAN_LEFT
                await self.execute_action(action, speed)
                await asyncio.sleep(abs(pan_diff) / 10)  # 模拟移动时间
                await self.execute_action(PTZAction.STOP, 0)
            
            # 执行垂直移动
            if abs(tilt_diff) > 1:
                action = PTZAction.TILT_UP if tilt_diff > 0 else PTZAction.TILT_DOWN
                await self.execute_action(action, speed)
                await asyncio.sleep(abs(tilt_diff) / 10)
                await self.execute_action(PTZAction.STOP, 0)
            
            # 执行变焦
            if zoom is not None:
                zoom_diff = zoom - self.current_zoom
                if abs(zoom_diff) > 0.1:
                    action = PTZAction.ZOOM_IN if zoom_diff > 0 else PTZAction.ZOOM_OUT
                    await self.execute_action(action, speed)
                    await asyncio.sleep(abs(zoom_diff))
                    await self.execute_action(PTZAction.STOP, 0)
            
            return {
                "success": True,
                "message": f"移动到位置: pan={pan}, tilt={tilt}, zoom={zoom}",
                "current_position": {
                    "pan": self.current_pan,
                    "tilt": self.current_tilt,
                    "zoom": self.current_zoom
                }
            }
        except Exception as e:
            logger.error(f"移动到位置失败: {e}")
            return {"success": False, "message": f"移动失败: {str(e)}"}
    
    async def set_preset(self, preset_id: int, name: Optional[str] = None) -> Dict[str, Any]:
        """
        设置预置位
        
        Args:
            preset_id: 预置位编号 (1-256)
            name: 预置位名称（可选）
        
        Returns:
            Dict: 执行结果
        """
        try:
            # 保存当前位置为预置位
            self.presets[preset_id] = {
                "pan": self.current_pan,
                "tilt": self.current_tilt,
                "zoom": self.current_zoom,
                "name": name or f"预置位{preset_id}"
            }
            
            # 发送设置预置位命令
            result = await self.execute_action(PTZAction.PRESET_SET, 0, preset_id=preset_id)
            
            logger.info(f"预置位{preset_id}已设置: {self.presets[preset_id]}")
            return {
                "success": True,
                "message": f"预置位{preset_id}设置成功",
                "preset": self.presets[preset_id]
            }
        except Exception as e:
            logger.error(f"设置预置位失败: {e}")
            return {"success": False, "message": f"设置失败: {str(e)}"}
    
    async def goto_preset(self, preset_id: int) -> Dict[str, Any]:
        """
        转到预置位
        
        Args:
            preset_id: 预置位编号
        
        Returns:
            Dict: 执行结果
        """
        try:
            if preset_id not in self.presets:
                return {"success": False, "message": f"预置位{preset_id}不存在"}
            
            result = await self.execute_action(PTZAction.PRESET_GOTO, 0, preset_id=preset_id)
            
            return {
                "success": True,
                "message": f"已转到预置位{preset_id}",
                "preset": self.presets[preset_id],
                "current_position": {
                    "pan": self.current_pan,
                    "tilt": self.current_tilt,
                    "zoom": self.current_zoom
                }
            }
        except Exception as e:
            logger.error(f"转到预置位失败: {e}")
            return {"success": False, "message": f"转到失败: {str(e)}"}
    
    async def auto_patrol(self, 
                         presets: list, 
                         dwell_time: int = 5) -> Dict[str, Any]:
        """
        自动巡航（依次访问多个预置位）
        
        Args:
            presets: 预置位列表 [1, 2, 3, ...]
            dwell_time: 每个位置停留时间（秒）
        
        Returns:
            Dict: 执行结果
        """
        try:
            logger.info(f"开始自动巡航: {presets}")
            
            for preset_id in presets:
                # 转到预置位
                await self.goto_preset(preset_id)
                
                # 停留
                await asyncio.sleep(dwell_time)
            
            return {
                "success": True,
                "message": f"巡航完成，访问了{len(presets)}个预置位"
            }
        except Exception as e:
            logger.error(f"自动巡航失败: {e}")
            return {"success": False, "message": f"巡航失败: {str(e)}"}
    
    async def auto_track_object(self, 
                                target_bbox: Tuple[int, int, int, int],
                                frame_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        自动跟踪目标（根据视觉识别结果调整云台）
        
        Args:
            target_bbox: 目标边界框 (x, y, w, h)
            frame_size: 画面尺寸 (width, height)
        
        Returns:
            Dict: 执行结果
        """
        try:
            x, y, w, h = target_bbox
            frame_w, frame_h = frame_size
            
            # 计算目标中心点
            target_center_x = x + w / 2
            target_center_y = y + h / 2
            
            # 计算偏移量（相对于画面中心）
            offset_x = target_center_x - frame_w / 2
            offset_y = target_center_y - frame_h / 2
            
            # 转换为角度偏移
            pan_offset = offset_x / frame_w * 30  # 假设视场角30度
            tilt_offset = -offset_y / frame_h * 20  # 假设垂直视场角20度
            
            # 判断是否需要调整
            threshold = 5  # 5度阈值
            
            if abs(pan_offset) > threshold:
                action = PTZAction.PAN_RIGHT if pan_offset > 0 else PTZAction.PAN_LEFT
                speed = int(min(abs(pan_offset) * 10, 100))
                await self.execute_action(action, speed)
                await asyncio.sleep(0.1)
                await self.execute_action(PTZAction.STOP, 0)
            
            if abs(tilt_offset) > threshold:
                action = PTZAction.TILT_UP if tilt_offset > 0 else PTZAction.TILT_DOWN
                speed = int(min(abs(tilt_offset) * 10, 100))
                await self.execute_action(action, speed)
                await asyncio.sleep(0.1)
                await self.execute_action(PTZAction.STOP, 0)
            
            return {
                "success": True,
                "message": "目标跟踪调整完成",
                "offset": {
                    "pan": pan_offset,
                    "tilt": tilt_offset
                }
            }
        except Exception as e:
            logger.error(f"自动跟踪失败: {e}")
            return {"success": False, "message": f"跟踪失败: {str(e)}"}
    
    def get_status(self) -> Dict[str, Any]:
        """获取云台当前状态"""
        return {
            "connected": self.is_connected,
            "protocol": self.protocol.value,
            "connection_type": self.connection_type,
            "position": {
                "pan": self.current_pan,
                "tilt": self.current_tilt,
                "zoom": self.current_zoom
            },
            "presets": self.presets
        }


# 全局云台控制器实例
_ptz_controller: Optional[PTZCameraController] = None


def get_ptz_controller(**kwargs) -> PTZCameraController:
    """获取全局云台控制器实例"""
    global _ptz_controller
    if _ptz_controller is None:
        _ptz_controller = PTZCameraController(**kwargs)
    return _ptz_controller
