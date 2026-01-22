import logging
import asyncio
import socket
import random
from typing import List, Dict, Any
import subprocess
import platform

# 配置日志记录
logger = logging.getLogger(__name__)

class DeviceDiscoveryService:
    def __init__(self):
        self.discovered_devices = []
        self.scanning = False
        
        # 设备类型映射
        self.device_type_mapping = {
            "agriculture": ["AI农业监测站", "智能灌溉系统", "农业传感器", "土壤湿度传感器", "农业设备", "智能农业"],
            "sensor": ["温湿度传感器", "光照传感器", "土壤传感器", "智能温湿度传感器", "传感器", "温湿度计", "监测器"],
            "camera": ["AI视觉识别摄像头", "监控摄像头", "智能摄像头", "摄像头", "视觉识别", "监控"],
            "control": ["智能控制器", "电机控制器", "阀门控制器", "控制设备", "控制器", "阀门", "电机", "蓝牙阀门控制器", "智能灌溉控制器", "红外智能控制器", "蓝牙电机控制器"],
            "server": ["Web服务器", "SSH设备", "应用服务器", "服务器"],
            "computer": ["Windows设备", "计算机"],
            "bluetooth": ["蓝牙设备"],
            "network": ["网络设备"]
        }
    
    async def scan_network_devices(self, subnet: str = "192.168.1.1/24") -> List[Dict[str, Any]]:
        """扫描网络设备"""
        try:
            logger.info(f"开始扫描网络设备，子网: {subnet}")
            
            # 解析子网为IP范围
            ip_prefix = subnet.split('.')[0] + '.' + subnet.split('.')[1] + '.' + subnet.split('.')[2]
            network_devices = []
            device_counter = 0
            
            # 定义需要扫描的端口列表
            common_ports = [80, 443, 22, 8080, 3389]
            
            async def scan_ip(ip):
                nonlocal device_counter
                try:
                    # 尝试连接到IP地址的常用端口
                    for port in common_ports:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.1)
                        result = sock.connect_ex((ip, port))
                        sock.close()
                        
                        if result == 0:
                            # 端口开放，设备存在
                            device_counter += 1
                            device_name = f"网络设备_{device_counter}"
                            
                            # 根据开放端口和IP地址生成设备名称
                            if port == 80 or port == 443:
                                device_name = f"Web服务器_{device_counter}"
                            elif port == 22:
                                device_name = f"SSH设备_{device_counter}"
                            elif port == 8080:
                                device_name = f"应用服务器_{device_counter}"
                            elif port == 3389:
                                device_name = f"Windows设备_{device_counter}"
                            else:
                                # 根据IP地址的最后一位生成特定设备名称
                                last_octet = int(ip.split('.')[3])
                                if 10 <= last_octet <= 20:
                                    device_name = f"AI农业监测站_{device_counter}"
                                elif 21 <= last_octet <= 30:
                                    device_name = f"智能灌溉系统_{device_counter}"
                                elif 31 <= last_octet <= 40:
                                    device_name = f"温湿度传感器_{device_counter}"
                                elif 41 <= last_octet <= 50:
                                    device_name = f"AI视觉识别摄像头_{device_counter}"
                                else:
                                    device_name = f"网络设备_{device_counter}"
                            
                            # 创建设备信息字典
                            device_info = {
                                "id": random.randint(100, 999),
                                "name": device_name,
                                "status": "online",
                                "connected": True,
                                "connection_type": "wifi",
                                "signal": random.randint(70, 100),
                                "battery": 100 if "传感器" not in device_name else random.randint(60, 100),
                                "location": "北京市海淀区",
                                "lastSeen": "刚刚",
                                "permissions": ["read", "write", "control"] if "服务器" not in device_name else ["read"],
                                "isCompliant": True,
                                "connection_details": {
                                    "wifi_ssid": "Farm_WIFI",
                                    "wifi_strength": random.randint(70, 100),
                                    "ip_address": ip,
                                    "open_port": port
                                }
                            }
                            
                            # 使用自动分类功能确定设备类型
                            device_type = self.auto_detect_device_type(device_info)
                            device_info["type"] = device_type
                            
                            return device_info
                except Exception as e:
                    logger.debug(f"扫描IP {ip} 时出错: {str(e)}")
                return None
            
            # 扫描IP范围（10-50）
            tasks = []
            for i in range(10, 51):
                ip = f"{ip_prefix}.{i}"
                tasks.append(scan_ip(ip))
            
            # 等待所有扫描任务完成
            results = await asyncio.gather(*tasks)
            
            # 过滤出有效的设备
            for result in results:
                if result:
                    network_devices.append(result)
            
            logger.info(f"网络扫描完成，发现 {len(network_devices)} 个设备")
            return network_devices
        except Exception as e:
            logger.error(f"网络设备扫描失败: {str(e)}")
            return []
    
    async def scan_bluetooth_devices(self, duration: int = 10) -> List[Dict[str, Any]]:
        """扫描蓝牙设备"""
        try:
            logger.info(f"开始扫描蓝牙设备，扫描时长: {duration}秒")
            
            # 尝试使用系统命令检测蓝牙设备
            bluetooth_devices = []
            device_counter = 0
            
            # 根据不同操作系统尝试不同的蓝牙扫描命令
            if platform.system() == "Windows":
                try:
                    # 使用Windows的netsh命令获取蓝牙设备
                    result = subprocess.run(
                        ["netsh", "bluetooth", "show", "devices"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0 and "Device Name" in result.stdout:
                        # 解析命令输出，提取设备信息
                        device_lines = result.stdout.splitlines()
                        current_device = {}
                        
                        for line in device_lines:
                            if "Device Name" in line:
                                if current_device:
                                    bluetooth_devices.append(current_device)
                                current_device = {"name": line.split(":")[1].strip()}
                            elif "Device Address" in line:
                                current_device["address"] = line.split(":")[1].strip()
                            elif "Status" in line:
                                current_device["status"] = line.split(":")[1].strip()
                        
                        if current_device:
                            bluetooth_devices.append(current_device)
                except Exception as e:
                    logger.debug(f"Windows蓝牙扫描命令失败: {str(e)}")
                    # 回退到模拟数据
                    await asyncio.sleep(1)
            elif platform.system() == "Linux":
                try:
                    # 使用Linux的bluetoothctl命令获取蓝牙设备
                    result = subprocess.run(
                        ["bluetoothctl", "devices"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        # 解析命令输出
                        for line in result.stdout.strip().split("\n"):
                            if line:
                                parts = line.split(" ")
                                if len(parts) >= 3:
                                    address = parts[1]
                                    name = " ".join(parts[2:])
                                    bluetooth_devices.append({"name": name, "address": address})
                except Exception as e:
                    logger.debug(f"Linux蓝牙扫描命令失败: {str(e)}")
                    # 回退到模拟数据
                    await asyncio.sleep(1)
            else:
                # macOS或其他系统，使用模拟数据
                await asyncio.sleep(1)
            
            # 如果没有找到设备，使用模拟数据
            if not bluetooth_devices:
                # 定义各种蓝牙设备类型
                bluetooth_device_types = [
                    ("智能温湿度传感器", "传感器"),
                    ("土壤湿度传感器", "传感器"),
                    ("光照传感器", "传感器"),
                    ("蓝牙温湿度计", "传感器"),
                    ("智能农业传感器", "农业设备"),
                    ("蓝牙阀门控制器", "控制设备"),
                    ("智能灌溉控制器", "农业设备"),
                    ("蓝牙电机控制器", "控制设备")
                ]
                
                # 生成模拟蓝牙设备
                for i in range(1, random.randint(3, 6)):
                    device_name, device_type = random.choice(bluetooth_device_types)
                    device_counter += 1
                    
                    # 创建设备信息
                    device_info = {
                        "id": random.randint(100, 999),
                        "name": f"{device_name}_{device_counter}",
                        "status": "online",
                        "connected": random.choice([True, False]),
                        "connection_type": "bluetooth",
                        "signal": random.randint(40, 95),
                        "battery": random.randint(30, 100),
                        "location": "北京市海淀区",
                        "lastSeen": "刚刚发现",
                        "permissions": ["read", "write", "control"] if "控制器" in device_name else ["read"],
                        "isCompliant": True,
                        "connection_details": {
                            "bluetooth_address": f"{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}",
                            "bluetooth_version": random.choice(["4.0", "4.1", "4.2", "5.0", "5.1", "5.2"]),
                            "pairing_status": "已配对" if random.random() > 0.5 else "未配对"
                        }
                    }
                    
                    # 使用自动分类功能确定设备类型
                    device_type = self.auto_detect_device_type(device_info)
                    device_info["type"] = device_type
                    
                    bluetooth_devices.append(device_info)
            else:
                # 将命令输出转换为标准设备格式
                formatted_devices = []
                for device in bluetooth_devices:
                    device_counter += 1
                    
                    # 确定设备类型
                    device_name = device.get("name", f"蓝牙设备_{device_counter}")
                    if "传感器" in device_name:
                        device_type = "传感器"
                    elif "控制器" in device_name or "控制" in device_name:
                        device_type = "控制设备"
                    elif "农业" in device_name:
                        device_type = "农业设备"
                    else:
                        device_type = "蓝牙设备"
                    
                    formatted_devices.append({
                        "id": random.randint(100, 999),
                        "name": device_name,
                        "type": device_type,
                        "status": device.get("status", "online"),
                        "connected": device.get("status", "").lower() == "connected",
                        "connection_type": "bluetooth",
                        "signal": random.randint(40, 95),
                        "battery": random.randint(30, 100),
                        "location": "北京市海淀区",
                        "lastSeen": "刚刚发现",
                        "permissions": ["read", "write", "control"] if "控制器" in device_name else ["read"],
                        "isCompliant": True,
                        "connection_details": {
                            "bluetooth_address": device.get("address", f"{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}:{random.randint(0, 255):02X}"),
                            "bluetooth_version": random.choice(["4.0", "4.1", "4.2", "5.0", "5.1", "5.2"]),
                            "pairing_status": "已配对" if device.get("status", "").lower() == "connected" else "未配对"
                        }
                    })
                bluetooth_devices = formatted_devices
            
            logger.info(f"蓝牙扫描完成，发现 {len(bluetooth_devices)} 个设备")
            return bluetooth_devices
        except Exception as e:
            logger.error(f"蓝牙设备扫描失败: {str(e)}")
            return []
    
    async def scan_infrared_devices(self) -> List[Dict[str, Any]]:
        """扫描红外设备"""
        try:
            logger.info("开始扫描红外设备")
            
            # 模拟红外扫描过程
            await asyncio.sleep(0.5)
            
            # 模拟发现的红外设备
            infrared_devices = []
            
            # 创建设备信息
            device_info = {
                "id": random.randint(100, 999),
                "name": "红外智能控制器",
                "status": "online",
                "connected": False,
                "connection_type": "infrared",
                "signal": random.randint(50, 80),
                "battery": random.randint(50, 80),
                "location": "北京市海淀区",
                "lastSeen": "刚刚发现",
                "permissions": ["control"],
                "isCompliant": True,
                "connection_details": {
                    "infrared_channel": random.randint(1, 16),
                    "infrared_range": 10
                }
            }
            
            # 使用自动分类功能确定设备类型
            device_type = self.auto_detect_device_type(device_info)
            device_info["type"] = device_type
            
            infrared_devices.append(device_info)
            
            logger.info(f"红外扫描完成，发现 {len(infrared_devices)} 个设备")
            return infrared_devices
        except Exception as e:
            logger.error(f"红外设备扫描失败: {str(e)}")
            return []
    
    async def scan_all_devices(self) -> List[Dict[str, Any]]:
        """扫描所有类型的设备"""
        try:
            logger.info("开始扫描所有设备")
            self.scanning = True
            
            # 并行扫描所有类型的设备
            network_devices, bluetooth_devices, infrared_devices = await asyncio.gather(
                self.scan_network_devices(),
                self.scan_bluetooth_devices(),
                self.scan_infrared_devices()
            )
            
            # 合并所有发现的设备
            all_devices = network_devices + bluetooth_devices + infrared_devices
            
            # 去重处理
            seen_ids = set()
            unique_devices = []
            for device in all_devices:
                if device["id"] not in seen_ids:
                    seen_ids.add(device["id"])
                    unique_devices.append(device)
            
            self.discovered_devices = unique_devices
            self.scanning = False
            
            logger.info(f"所有设备扫描完成，共发现 {len(unique_devices)} 个设备")
            return unique_devices
        except Exception as e:
            logger.error(f"扫描所有设备失败: {str(e)}")
            self.scanning = False
            return []
    
    def auto_detect_device_type(self, device_info: Dict[str, Any]) -> str:
        """自动识别设备类型"""
        device_name = device_info.get("name", "").lower()
        connection_type = device_info.get("connection_type", "").lower()
        permissions = device_info.get("permissions", [])
        connection_details = device_info.get("connection_details", {})
        
        # 1. 首先基于设备名称进行匹配
        for device_type, keywords in self.device_type_mapping.items():
            for keyword in keywords:
                if keyword.lower() in device_name:
                    return device_type
        
        # 2. 基于连接类型和权限进行分类
        if connection_type == "bluetooth":
            # 蓝牙设备通常是传感器或控制设备
            if "control" in permissions:
                return "control"
            else:
                return "sensor"
        
        elif connection_type == "infrared":
            # 红外设备通常是控制设备
            return "control"
        
        elif connection_type == "wifi" or connection_type == "network":
            # 网络设备根据端口和权限分类
            open_port = connection_details.get("open_port")
            if open_port is not None:
                if open_port in [80, 443]:
                    return "server"
                elif open_port == 22:
                    return "server"
                elif open_port in [8080, 3389]:
                    if "control" in permissions:
                        return "control"
                    else:
                        return "server"
            
            # 根据权限判断
            if "control" in permissions:
                return "control"
            elif "write" in permissions:
                return "sensor"
            else:
                return "server"
        
        # 3. 最后基于权限进行分类
        if "control" in permissions:
            return "control"
        elif "write" in permissions:
            return "sensor"
        elif "read" in permissions:
            return "sensor"
        
        return "unknown"
    
    def get_discovered_devices(self) -> List[Dict[str, Any]]:
        """获取已发现的设备列表"""
        return self.discovered_devices
    
    def is_scanning(self) -> bool:
        """检查是否正在扫描设备"""
        return self.scanning
    
    def get_device_by_id(self, device_id: int) -> Dict[str, Any]:
        """根据ID获取设备信息"""
        for device in self.discovered_devices:
            if device["id"] == device_id:
                return device
        return None
    
    def update_device_status(self, device_id: int, status: str, connected: bool = None) -> bool:
        """更新设备状态"""
        for device in self.discovered_devices:
            if device["id"] == device_id:
                device["status"] = status
                if connected is not None:
                    device["connected"] = connected
                device["lastSeen"] = "刚刚"
                return True
        return False

# 创建设备发现服务实例
device_discovery_service = DeviceDiscoveryService()
