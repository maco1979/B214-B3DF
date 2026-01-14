"""
传感器适配器插件主模块
提供传感器数据的标准化接入和处理
支持多种通信协议和传感器类型
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
import asyncio
import random
from datetime import datetime
import json
import yaml
from dataclasses import dataclass, asdict
from enum import Enum

# 导入插件基类
from src.core.plugins.plugin_manager import BasePlugin, PluginStatus, BrainNavigationRegion


# 创建API路由
router = APIRouter(prefix="/sensor", tags=["sensor_adapter"])


# 通信协议类型
class CommunicationProtocol(Enum):
    """传感器通信协议"""
    HTTP = "http"
    MQTT = "mqtt"
    WEBSOCKET = "websocket"
    MODBUS = "modbus"
    OPCUA = "opcua"
    COAP = "coap"
    SIMULATION = "simulation"


# 传感器类型扩展
class SensorType(Enum):
    """传感器类型"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    SOIL_MOISTURE = "soil_moisture"
    LIGHT = "light"
    PRESSURE = "pressure"
    WIND = "wind"
    RAIN = "rain"
    SOIL_PH = "soil_ph"
    SOIL_NUTRIENT = "soil_nutrient"
    AIR_QUALITY = "air_quality"
    WATER_QUALITY = "water_quality"
    MOTION = "motion"
    SOUND = "sound"
    POWER = "power"
    CURRENT = "current"
    VOLTAGE = "voltage"
    FREQUENCY = "frequency"


@dataclass
class SensorConfig:
    """传感器配置"""
    id: str
    name: str
    sensor_type: SensorType
    protocol: CommunicationProtocol
    protocol_config: Dict[str, Any]
    interval: int = 1  # 数据采集间隔（秒）
    enabled: bool = True
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class Sensor:
    """传感器基类"""
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self.id = config.id
        self.name = config.name
        self.type = config.sensor_type.value
        self.protocol = config.protocol.value
        self.protocol_config = config.protocol_config
        self.status = "online"
        self.last_reading = None
        self.last_updated = None
        self.enabled = config.enabled
    
    async def read_data(self) -> Dict[str, Any]:
        """读取传感器数据
        Returns:
            Dict[str, Any]: 传感器数据
        """
        raise NotImplementedError("子类必须实现read_data方法")
    
    async def connect(self) -> bool:
        """连接传感器
        Returns:
            bool: 连接是否成功
        """
        return True
    
    async def disconnect(self) -> bool:
        """断开传感器连接
        Returns:
            bool: 断开是否成功
        """
        return True


class SimulationSensor(Sensor):
    """模拟传感器基类"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取模拟传感器数据
        Returns:
            Dict[str, Any]: 传感器数据
        """
        raise NotImplementedError("子类必须实现read_data方法")


class TemperatureSensor(SimulationSensor):
    """温度传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.min_temp = self.protocol_config.get("min_temp", -20)
        self.max_temp = self.protocol_config.get("max_temp", 100)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取温度数据
        Returns:
            Dict[str, Any]: 温度数据
        """
        # 模拟温度数据
        temperature = round(random.uniform(self.min_temp, self.max_temp), 2)
        humidity = round(random.uniform(0, 100), 2)
        
        data = self._create_base_data({
            "temperature": temperature,
            "humidity": humidity
        })
        
        self.last_reading = data
        self.last_updated = datetime.now()
        return data


class SoilMoistureSensor(SimulationSensor):
    """土壤湿度传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取土壤湿度数据
        Returns:
            Dict[str, Any]: 土壤湿度数据
        """
        # 模拟土壤湿度数据
        moisture = round(random.uniform(0, 100), 2)
        ph = round(random.uniform(4, 9), 2)
        
        data = self._create_base_data({
            "moisture": moisture,
            "ph": ph
        })
        
        self.last_reading = data
        self.last_updated = datetime.now()
        return data


class LightSensor(SimulationSensor):
    """光照传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取光照数据
        Returns:
            Dict[str, Any]: 光照数据
        """
        # 模拟光照数据
        light_intensity = round(random.uniform(0, 2000), 2)
        spectrum = {
            "red": round(random.uniform(0, 1), 2),
            "green": round(random.uniform(0, 1), 2),
            "blue": round(random.uniform(0, 1), 2),
            "uv": round(random.uniform(0, 0.5), 3)
        }
        
        data = self._create_base_data({
            "light_intensity": light_intensity,
            "spectrum": spectrum
        })
        
        self.last_reading = data
        self.last_updated = datetime.now()
        return data


class AirQualitySensor(SimulationSensor):
    """空气质量传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取空气质量数据
        Returns:
            Dict[str, Any]: 空气质量数据
        """
        # 模拟空气质量数据
        pm25 = round(random.uniform(0, 300), 2)
        pm10 = round(random.uniform(0, 500), 2)
        co2 = round(random.uniform(300, 2000), 2)
        
        data = self._create_base_data({
            "pm25": pm25,
            "pm10": pm10,
            "co2": co2
        })
        
        self.last_reading = data
        self.last_updated = datetime.now()
        return data


class WaterQualitySensor(SimulationSensor):
    """水质传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取水质数据
        Returns:
            Dict[str, Any]: 水质数据
        """
        # 模拟水质数据
        ph = round(random.uniform(6, 9), 2)
        turbidity = round(random.uniform(0, 100), 2)
        dissolved_oxygen = round(random.uniform(0, 15), 2)
        
        data = self._create_base_data({
            "ph": ph,
            "turbidity": turbidity,
            "dissolved_oxygen": dissolved_oxygen
        })
        
        self.last_reading = data
        self.last_updated = datetime.now()
        return data


class WindSensor(SimulationSensor):
    """风速传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
    
    async def read_data(self) -> Dict[str, Any]:
        """读取风速数据
        Returns:
            Dict[str, Any]: 风速数据
        """
        # 模拟风速数据
        speed = round(random.uniform(0, 50), 2)
        direction = round(random.uniform(0, 360), 2)
        
        data = self._create_base_data({
            "speed": speed,
            "direction": direction
        })
        
        self.last_reading = data
        self.last_updated = datetime.now()
        return data


# 传感器工厂类
class SensorFactory:
    """传感器工厂，用于创建不同类型的传感器"""
    
    @staticmethod
    def create_sensor(config: SensorConfig) -> Sensor:
        """创建传感器实例
        Args:
            config: 传感器配置
        Returns:
            Sensor: 传感器实例
        """
        if config.protocol == CommunicationProtocol.SIMULATION:
            # 根据传感器类型创建模拟传感器
            if config.sensor_type == SensorType.TEMPERATURE:
                return TemperatureSensor(config)
            elif config.sensor_type == SensorType.SOIL_MOISTURE:
                return SoilMoistureSensor(config)
            elif config.sensor_type == SensorType.LIGHT:
                return LightSensor(config)
            elif config.sensor_type == SensorType.AIR_QUALITY:
                return AirQualitySensor(config)
            elif config.sensor_type == SensorType.WATER_QUALITY:
                return WaterQualitySensor(config)
            elif config.sensor_type == SensorType.WIND:
                return WindSensor(config)
            # TODO: 添加更多传感器类型的支持
            else:
                raise ValueError(f"不支持的传感器类型: {config.sensor_type}")
        elif config.protocol == CommunicationProtocol.MQTT:
            # 创建MQTT传感器
            return MQTTSensor(config)
        else:
            # TODO: 实现其他通信协议的传感器支持
            raise NotImplementedError(f"通信协议 {config.protocol} 尚未实现")


# MQTT传感器类
class MQTTSensor(Sensor):
    """MQTT传感器"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.client = None
        self.topic = self.protocol_config.get("topic", "sensor/data")
        self.broker = self.protocol_config.get("broker", "localhost")
        self.port = self.protocol_config.get("port", 1883)
        self.username = self.protocol_config.get("username")
        self.password = self.protocol_config.get("password")
        self.connected = False
        self.reconnect_interval = self.protocol_config.get("reconnect_interval", 5)
        self.data_queue = asyncio.Queue()
    
    async def connect(self) -> bool:
        """连接MQTT服务器"""
        import paho.mqtt.client as mqtt
        
        try:
            # 创建MQTT客户端
            self.client = mqtt.Client()
            
            # 设置认证信息
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # 设置回调函数
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            # 连接MQTT服务器
            self.client.connect(self.broker, self.port, 60)
            
            # 启动客户端
            self.client.loop_start()
            
            # 等待连接成功
            for _ in range(5):
                if self.connected:
                    return True
                await asyncio.sleep(1)
            
            return self.connected
        except Exception as e:
            self.logger.error(f"MQTT连接失败: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            self.connected = True
            self.status = "online"
            # 订阅主题
            self.client.subscribe(self.topic)
            self.logger.info(f"MQTT连接成功，已订阅主题: {self.topic}")
        else:
            self.logger.error(f"MQTT连接失败，错误码: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        self.connected = False
        self.status = "offline"
        self.logger.info(f"MQTT断开连接，错误码: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT消息回调"""
        try:
            # 解析消息
            message = json.loads(msg.payload.decode())
            # 将消息放入队列
            asyncio.create_task(self.data_queue.put(message))
        except Exception as e:
            self.logger.error(f"处理MQTT消息失败: {e}")
    
    async def read_data(self) -> Dict[str, Any]:
        """读取MQTT传感器数据"""
        try:
            # 如果未连接，尝试重连
            if not self.connected:
                await self.connect()
            
            # 尝试从队列中获取数据
            try:
                message = await asyncio.wait_for(self.data_queue.get(), timeout=5.0)
                # 处理数据
                data = self._create_base_data(message)
                self.last_reading = data
                self.last_updated = datetime.now()
                return data
            except asyncio.TimeoutError:
                # 如果没有新数据，返回上次读取的数据
                if self.last_reading:
                    return self.last_reading
                else:
                    # 如果还没有任何数据，返回默认值
                    default_data = {
                        "temperature": 25.0,
                        "humidity": 60.0,
                        "light_intensity": 500.0
                    }
                    return self._create_base_data(default_data)
        except Exception as e:
            self.logger.error(f"读取MQTT传感器数据失败: {e}")
            # 失败时返回默认值
            default_data = {
                "temperature": 25.0,
                "humidity": 60.0,
                "light_intensity": 500.0
            }
            return self._create_base_data(default_data)
    
    async def disconnect(self) -> bool:
        """断开MQTT连接"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                self.status = "offline"
            return True
        except Exception as e:
            self.logger.error(f"断开MQTT连接失败: {e}")
            return False


# 创建插件类
class Plugin(BasePlugin):
    """传感器适配器插件"""
    
    def __init__(self):
        super().__init__()
        self.sensors: Dict[str, Sensor] = {}
        self.sensor_configs: Dict[str, SensorConfig] = {}
        self.data_history: Dict[str, List[Dict[str, Any]]] = {}
        self.is_running = False
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.sensor_factory = SensorFactory()
    
    async def initialize(self, app: Any) -> bool:
        """初始化插件
        Args:
            app: FastAPI应用实例
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 注册路由
            if app:
                self.logger.info("传感器适配器插件初始化成功")
            
            # 初始化传感器配置
            self._load_sensor_configs()
            
            # 创建传感器实例
            self._create_sensors()
            
            # 初始化数据历史
            for sensor_id in self.sensors:
                self.data_history[sensor_id] = []
            
            self.status = PluginStatus.LOADED
            return True
            
        except Exception as e:
            self.logger.error(f"传感器适配器插件初始化失败: {e}")
            self.status = PluginStatus.ERROR
            return False
    
    def _load_sensor_configs(self):
        """加载传感器配置
        从配置文件或数据库加载传感器配置
        """
        # 示例配置，实际可以从文件或数据库加载
        default_configs = [
            {
                "id": "temp_001",
                "name": "温室温度传感器",
                "sensor_type": "temperature",
                "protocol": "simulation",
                "protocol_config": {
                    "min_temp": -20,
                    "max_temp": 100
                },
                "interval": 1,
                "enabled": True,
                "tags": ["greenhouse", "temperature"]
            },
            {
                "id": "soil_001",
                "name": "土壤湿度传感器",
                "sensor_type": "soil_moisture",
                "protocol": "simulation",
                "protocol_config": {},
                "interval": 1,
                "enabled": True,
                "tags": ["farmland", "soil"]
            },
            {
                "id": "light_001",
                "name": "光照传感器",
                "sensor_type": "light",
                "protocol": "simulation",
                "protocol_config": {},
                "interval": 1,
                "enabled": True,
                "tags": ["greenhouse", "light"]
            },
            {
                "id": "air_001",
                "name": "空气质量传感器",
                "sensor_type": "air_quality",
                "protocol": "simulation",
                "protocol_config": {},
                "interval": 1,
                "enabled": True,
                "tags": ["greenhouse", "air"]
            },
            {
                "id": "water_001",
                "name": "水质传感器",
                "sensor_type": "water_quality",
                "protocol": "simulation",
                "protocol_config": {},
                "interval": 1,
                "enabled": True,
                "tags": ["aquaculture", "water"]
            },
            {
                "id": "wind_001",
                "name": "风速传感器",
                "sensor_type": "wind",
                "protocol": "simulation",
                "protocol_config": {},
                "interval": 1,
                "enabled": True,
                "tags": ["outdoor", "weather"]
            },
            {
                "id": "mqtt_001",
                "name": "MQTT温度湿度传感器",
                "sensor_type": "temperature",
                "protocol": "mqtt",
                "protocol_config": {
                    "broker": "localhost",
                    "port": 1883,
                    "topic": "sensor/agriculture/data",
                    "reconnect_interval": 5
                },
                "interval": 1,
                "enabled": True,
                "tags": ["greenhouse", "mqtt", "temperature", "humidity"]
            }
        ]
        
        # 解析配置
        for config_data in default_configs:
            config = SensorConfig(
                id=config_data["id"],
                name=config_data["name"],
                sensor_type=SensorType(config_data["sensor_type"]),
                protocol=CommunicationProtocol(config_data["protocol"]),
                protocol_config=config_data["protocol_config"],
                interval=config_data["interval"],
                enabled=config_data["enabled"],
                tags=config_data["tags"]
            )
            self.sensor_configs[config.id] = config
    
    def _create_sensors(self):
        """创建传感器实例"""
        self.sensors = {}
        for config_id, config in self.sensor_configs.items():
            if config.enabled:
                try:
                    sensor = self.sensor_factory.create_sensor(config)
                    self.sensors[config_id] = sensor
                    self.logger.info(f"创建传感器成功: {config.name} (ID: {config.id})")
                except Exception as e:
                    self.logger.error(f"创建传感器失败 {config.name}: {e}")
    
    async def activate(self) -> bool:
        """激活插件
        Returns:
            bool: 激活是否成功
        """
        try:
            # 启动所有传感器监控
            self.is_running = True
            for sensor_id, sensor in self.sensors.items():
                self.monitoring_tasks[sensor_id] = asyncio.create_task(
                    self._monitor_single_sensor(sensor_id, sensor)
                )
            
            self.logger.info("传感器适配器插件激活成功")
            self.status = PluginStatus.ACTIVE
            return True
        except Exception as e:
            self.logger.error(f"传感器适配器插件激活失败: {e}")
            self.status = PluginStatus.ERROR
            return False
    
    async def deactivate(self) -> bool:
        """停用插件
        Returns:
            bool: 停用是否成功
        """
        try:
            # 停止所有传感器监控
            self.is_running = False
            for task in self.monitoring_tasks.values():
                task.cancel()
            self.monitoring_tasks.clear()
            
            self.logger.info("传感器适配器插件停用成功")
            self.status = PluginStatus.LOADED
            return True
        except Exception as e:
            self.logger.error(f"传感器适配器插件停用失败: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理插件资源
        Returns:
            bool: 清理是否成功
        """
        try:
            # 停止所有传感器监控
            self.is_running = False
            for task in self.monitoring_tasks.values():
                task.cancel()
            self.monitoring_tasks.clear()
            
            # 断开所有传感器连接
            for sensor in self.sensors.values():
                await sensor.disconnect()
            
            # 清空传感器列表
            self.sensors.clear()
            self.sensor_configs.clear()
            self.data_history.clear()
            
            self.logger.info("传感器适配器插件清理成功")
            self.status = PluginStatus.UNLOADED
            return True
        except Exception as e:
            self.logger.error(f"传感器适配器插件清理失败: {e}")
            return False
    
    async def _monitor_single_sensor(self, sensor_id: str, sensor: Sensor):
        """监控单个传感器数据
        Args:
            sensor_id: 传感器ID
            sensor: 传感器实例
        """
        while self.is_running and sensor.enabled:
            try:
                # 读取传感器数据
                data = await sensor.read_data()
                
                # 保存数据到历史
                self.data_history[sensor_id].append(data)
                # 只保留最近1000条数据
                if len(self.data_history[sensor_id]) > 1000:
                    self.data_history[sensor_id] = self.data_history[sensor_id][-1000:]
                
                # 根据配置的间隔休眠
                await asyncio.sleep(sensor.config.interval)
                
            except Exception as e:
                self.logger.error(f"监控传感器 {sensor.name} 失败: {e}")
                # 失败时延长休眠时间
                await asyncio.sleep(5)
    
    def get_routes(self):
        """获取插件路由
        Returns:
            Any: FastAPI路由
        """
        return router
    
    def get_services(self):
        """获取插件提供的服务
        Returns:
            Dict[str, Any]: 服务字典
        """
        return {
            "sensor_service": {
                "read_sensor_data": self.read_sensor_data,
                "get_sensors": self.get_sensors,
                "get_sensor_history": self.get_sensor_history,
                "get_sensor_configs": self.get_sensor_configs,
                "update_sensor_config": self.update_sensor_config,
                "create_sensor": self.create_sensor,
                "delete_sensor": self.delete_sensor
            }
        }
    
    def get_brain_navigation_regions(self):
        """获取插件注册的大脑导航区域
        Returns:
            List[BrainNavigationRegion]: 大脑导航区域列表
        """
        return [
            BrainNavigationRegion(
                id="sensor_dashboard",
                name="传感器监控",
                path="/sensors",
                position={"top": "40%", left: "65%", width: "25%", height: "25%"},
                description="传感器数据监控与管理",
                icon="Activity",
                color="primary"
            )
        ]
    
    async def read_sensor_data(self, sensor_id: str) -> Dict[str, Any]:
        """读取传感器数据
        Args:
            sensor_id: 传感器ID
        Returns:
            Dict[str, Any]: 传感器数据
        """
        if sensor_id not in self.sensors:
            raise ValueError(f"传感器ID不存在: {sensor_id}")
        
        return await self.sensors[sensor_id].read_data()
    
    def get_sensors(self) -> List[Dict[str, Any]]:
        """获取传感器列表
        Returns:
            List[Dict[str, Any]]: 传感器列表
        """
        sensors = []
        for sensor in self.sensors.values():
            sensors.append({
                "id": sensor.id,
                "name": sensor.name,
                "type": sensor.type,
                "protocol": sensor.protocol,
                "status": sensor.status,
                "enabled": sensor.enabled,
                "last_updated": sensor.last_updated.isoformat() if sensor.last_updated else None,
                "tags": sensor.config.tags
            })
        return sensors
    
    def get_sensor_history(self, sensor_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取传感器历史数据
        Args:
            sensor_id: 传感器ID
            limit: 返回数据的条数
        Returns:
            List[Dict[str, Any]]: 传感器历史数据
        """
        if sensor_id not in self.data_history:
            return []
        
        return self.data_history[sensor_id][-limit:]
    
    def get_sensor_configs(self) -> List[Dict[str, Any]]:
        """获取传感器配置列表
        Returns:
            List[Dict[str, Any]]: 传感器配置列表
        """
        configs = []
        for config in self.sensor_configs.values():
            config_data = asdict(config)
            # 转换枚举为字符串
            config_data["sensor_type"] = config_data["sensor_type"].value
            config_data["protocol"] = config_data["protocol"].value
            configs.append(config_data)
        return configs
    
    async def update_sensor_config(self, sensor_id: str, config_data: Dict[str, Any]) -> bool:
        """更新传感器配置
        Args:
            sensor_id: 传感器ID
            config_data: 新的配置数据
        Returns:
            bool: 更新是否成功
        """
        try:
            if sensor_id not in self.sensor_configs:
                raise ValueError(f"传感器ID不存在: {sensor_id}")
            
            # 更新配置
            old_config = self.sensor_configs[sensor_id]
            new_config = SensorConfig(
                id=sensor_id,
                name=config_data.get("name", old_config.name),
                sensor_type=SensorType(config_data.get("sensor_type", old_config.sensor_type.value)),
                protocol=CommunicationProtocol(config_data.get("protocol", old_config.protocol.value)),
                protocol_config=config_data.get("protocol_config", old_config.protocol_config),
                interval=config_data.get("interval", old_config.interval),
                enabled=config_data.get("enabled", old_config.enabled),
                tags=config_data.get("tags", old_config.tags)
            )
            
            self.sensor_configs[sensor_id] = new_config
            
            # 重启传感器
            if sensor_id in self.sensors:
                # 停止旧传感器监控
                if sensor_id in self.monitoring_tasks:
                    self.monitoring_tasks[sensor_id].cancel()
                    del self.monitoring_tasks[sensor_id]
                
                # 重新创建传感器
                sensor = self.sensor_factory.create_sensor(new_config)
                self.sensors[sensor_id] = sensor
                
                # 启动新的监控任务
                if self.is_running:
                    self.monitoring_tasks[sensor_id] = asyncio.create_task(
                        self._monitor_single_sensor(sensor_id, sensor)
                    )
            
            self.logger.info(f"更新传感器配置成功: {sensor_id}")
            return True
        except Exception as e:
            self.logger.error(f"更新传感器配置失败 {sensor_id}: {e}")
            return False
    
    async def create_sensor(self, config_data: Dict[str, Any]) -> bool:
        """创建新传感器
        Args:
            config_data: 传感器配置数据
        Returns:
            bool: 创建是否成功
        """
        try:
            sensor_id = config_data["id"]
            if sensor_id in self.sensor_configs:
                raise ValueError(f"传感器ID已存在: {sensor_id}")
            
            # 创建配置
            config = SensorConfig(
                id=sensor_id,
                name=config_data["name"],
                sensor_type=SensorType(config_data["sensor_type"]),
                protocol=CommunicationProtocol(config_data["protocol"]),
                protocol_config=config_data.get("protocol_config", {}),
                interval=config_data.get("interval", 1),
                enabled=config_data.get("enabled", True),
                tags=config_data.get("tags", [])
            )
            
            self.sensor_configs[sensor_id] = config
            
            # 创建传感器实例
            if config.enabled:
                sensor = self.sensor_factory.create_sensor(config)
                self.sensors[sensor_id] = sensor
                
                # 启动监控任务
                if self.is_running:
                    self.monitoring_tasks[sensor_id] = asyncio.create_task(
                        self._monitor_single_sensor(sensor_id, sensor)
                    )
            
            self.logger.info(f"创建传感器成功: {config.name} (ID: {config.id})")
            return True
        except Exception as e:
            self.logger.error(f"创建传感器失败: {e}")
            return False
    
    async def delete_sensor(self, sensor_id: str) -> bool:
        """删除传感器
        Args:
            sensor_id: 传感器ID
        Returns:
            bool: 删除是否成功
        """
        try:
            if sensor_id not in self.sensor_configs:
                raise ValueError(f"传感器ID不存在: {sensor_id}")
            
            # 停止监控任务
            if sensor_id in self.monitoring_tasks:
                self.monitoring_tasks[sensor_id].cancel()
                del self.monitoring_tasks[sensor_id]
            
            # 删除传感器实例
            if sensor_id in self.sensors:
                del self.sensors[sensor_id]
            
            # 删除传感器配置
            if sensor_id in self.sensor_configs:
                del self.sensor_configs[sensor_id]
            
            # 删除数据历史
            if sensor_id in self.data_history:
                del self.data_history[sensor_id]
            
            self.logger.info(f"删除传感器成功: {sensor_id}")
            return True
        except Exception as e:
            self.logger.error(f"删除传感器失败 {sensor_id}: {e}")
            return False
    
    def _create_base_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建基础数据结构
        Args:
            sensor_data: 传感器原始数据
        Returns:
            Dict[str, Any]: 标准化的传感器数据
        """
        return {
            "sensor_id": self.id,
            "sensor_name": self.name,
            "sensor_type": self.type,
            "protocol": self.protocol,
            **sensor_data,
            "timestamp": datetime.now().isoformat(),
            "status": self.status
        }


# API路由实现
@router.get("/list", response_model=Dict[str, Any])
async def list_sensors():
    """获取传感器列表
    Returns:
        Dict[str, Any]: 传感器列表
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "get_sensors"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    sensors = plugin.get_sensors()
    return {
        "success": True,
        "data": sensors
    }


@router.get("/read/{sensor_id}", response_model=Dict[str, Any])
async def read_sensor(sensor_id: str):
    """读取传感器数据
    Args:
        sensor_id: 传感器ID
    Returns:
        Dict[str, Any]: 传感器数据
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "read_sensor_data"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    try:
        data = await plugin.read_sensor_data(sensor_id)
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@router.get("/history/{sensor_id}", response_model=Dict[str, Any])
async def get_sensor_history(sensor_id: str, limit: int = 20):
    """获取传感器历史数据
    Args:
        sensor_id: 传感器ID
        limit: 返回数据的条数
    Returns:
        Dict[str, Any]: 传感器历史数据
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "get_sensor_history"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    history = plugin.get_sensor_history(sensor_id, limit)
    return {
        "success": True,
        "data": history
    }


@router.get("/read-all", response_model=Dict[str, Any])
async def read_all_sensors():
    """读取所有传感器数据
    Returns:
        Dict[str, Any]: 所有传感器数据
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "get_sensors"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    sensors = plugin.get_sensors()
    all_data = {}
    
    for sensor in sensors:
        try:
            data = await plugin.read_sensor_data(sensor["id"])
            all_data[sensor["id"]] = data
        except Exception as e:
            all_data[sensor["id"]] = {"error": str(e)}
    
    return {
        "success": True,
        "data": all_data
    }


@router.get("/configs", response_model=Dict[str, Any])
async def get_sensor_configs():
    """获取传感器配置列表
    Returns:
        Dict[str, Any]: 传感器配置列表
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "get_sensor_configs"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    configs = plugin.get_sensor_configs()
    return {
        "success": True,
        "data": configs
    }


@router.post("/config/{sensor_id}", response_model=Dict[str, Any])
async def update_sensor_config(sensor_id: str, config_data: Dict[str, Any]):
    """更新传感器配置
    Args:
        sensor_id: 传感器ID
        config_data: 新的配置数据
    Returns:
        Dict[str, Any]: 更新结果
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "update_sensor_config"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    try:
        result = await plugin.update_sensor_config(sensor_id, config_data)
        return {
            "success": result,
            "message": "传感器配置更新成功" if result else "传感器配置更新失败"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@router.post("/create", response_model=Dict[str, Any])
async def create_sensor(config_data: Dict[str, Any]):
    """创建新传感器
    Args:
        config_data: 传感器配置数据
    Returns:
        Dict[str, Any]: 创建结果
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "create_sensor"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    try:
        result = await plugin.create_sensor(config_data)
        return {
            "success": result,
            "message": "传感器创建成功" if result else "传感器创建失败"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@router.delete("/delete/{sensor_id}", response_model=Dict[str, Any])
async def delete_sensor(sensor_id: str):
    """删除传感器
    Args:
        sensor_id: 传感器ID
    Returns:
        Dict[str, Any]: 删除结果
    """
    from src.core.plugins.plugin_manager import plugin_manager
    plugin = plugin_manager.get_plugin("SensorAdapter")
    if not plugin or not hasattr(plugin, "delete_sensor"):
        return {
            "success": False,
            "message": "传感器服务未可用"
        }
    
    try:
        result = await plugin.delete_sensor(sensor_id)
        return {
            "success": result,
            "message": "传感器删除成功" if result else "传感器删除失败"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
