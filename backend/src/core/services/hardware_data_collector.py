"""
硬件数据收集器 - 收集硬件数据并提供给AI学习系统
实现硬件数据的实时收集、预处理和集成到AI学习流程
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import jax.numpy as jnp
import logging
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)


class HardwareDataType(Enum):
    """硬件数据类型枚举"""
    SENSORS = "sensors"           # 传感器数据
    CONTROLLERS = "controllers"   # 控制器数据
    STATUS = "status"             # 状态数据
    PERFORMANCE = "performance"   # 性能数据
    ENVIRONMENT = "environment"   # 环境数据


@dataclass
class HardwareDataPoint:
    """硬件数据点"""
    device_id: str
    data_type: HardwareDataType
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float = 1.0
    quality_score: float = 1.0


class HardwareDataCollector:
    """硬件数据收集器"""
    
    def __init__(self):
        self.data_buffer: List[HardwareDataPoint] = []
        self.max_buffer_size = 10000
        self.collection_interval = 1.0  # 秒
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # 数据预处理器
        self.preprocessors: Dict[HardwareDataType, Callable] = {
            HardwareDataType.SENSORS: self._preprocess_sensors,
            HardwareDataType.CONTROLLERS: self._preprocess_controllers,
            HardwareDataType.STATUS: self._preprocess_status,
            HardwareDataType.PERFORMANCE: self._preprocess_performance,
            HardwareDataType.ENVIRONMENT: self._preprocess_environment
        }
        
        # AI学习接口
        self.ai_learning_callback: Optional[Callable] = None
        self.learning_queue = queue.Queue()
        
    def set_ai_learning_callback(self, callback: Callable):
        """设置AI学习回调函数"""
        self.ai_learning_callback = callback
    
    async def start_collection(self):
        """开始数据收集"""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("硬件数据收集器已启动")
    
    async def stop_collection(self):
        """停止数据收集"""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("硬件数据收集器已停止")
    
    async def _collection_loop(self):
        """数据收集循环"""
        while self.is_collecting:
            try:
                # 模拟收集硬件数据
                await self._collect_hardware_data()
                
                # 处理数据并准备发送给AI
                await self._process_and_queue_data()
                
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"数据收集循环发生错误: {e}")
    
    async def _collect_hardware_data(self):
        """收集硬件数据"""
        # 模拟从不同硬件源收集数据
        hardware_sources = [
            self._collect_sensor_data(),
            self._collect_controller_data(),
            self._collect_status_data(),
            self._collect_performance_data(),
            self._collect_environment_data()
        ]
        
        # 并行收集数据
        results = await asyncio.gather(*hardware_sources, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"收集硬件数据时发生错误: {result}")
            elif result:
                for data_point in result:
                    self._add_data_point(data_point)
    
    def _add_data_point(self, data_point: HardwareDataPoint):
        """添加数据点到缓冲区"""
        self.data_buffer.append(data_point)
        
        # 保持缓冲区大小
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
    
    async def _collect_sensor_data(self) -> List[HardwareDataPoint]:
        """收集传感器数据"""
        # 模拟传感器数据收集
        sensor_data = {
            "temperature": np.random.normal(25, 2),  # 温度
            "humidity": np.random.normal(60, 5),     # 湿度
            "light_intensity": np.random.normal(500, 50),  # 光强度
            "co2_level": np.random.normal(400, 20),  # CO2浓度
            "soil_moisture": np.random.normal(30, 5)  # 土壤湿度
        }
        
        data_point = HardwareDataPoint(
            device_id="sensor_001",
            data_type=HardwareDataType.SENSORS,
            timestamp=datetime.now(),
            data=sensor_data,
            confidence=0.95,
            quality_score=0.98
        )
        
        return [data_point]
    
    async def _collect_controller_data(self) -> List[HardwareDataPoint]:
        """收集控制器数据"""
        # 模拟控制器数据收集
        controller_data = {
            "led_uv_intensity": np.random.uniform(0.0, 0.1),      # UV LED强度
            "led_far_red_intensity": np.random.uniform(0.0, 0.15), # 远红LED强度
            "led_white_intensity": np.random.uniform(0.5, 0.9),   # 白光LED强度
            "led_red_intensity": np.random.uniform(0.1, 0.3),     # 红光LED强度
            "controller_status": "active",
            "power_consumption": np.random.uniform(45, 55)        # 功耗
        }
        
        data_point = HardwareDataPoint(
            device_id="controller_001",
            data_type=HardwareDataType.CONTROLLERS,
            timestamp=datetime.now(),
            data=controller_data,
            confidence=0.92,
            quality_score=0.96
        )
        
        return [data_point]
    
    async def _collect_status_data(self) -> List[HardwareDataPoint]:
        """收集状态数据"""
        # 模拟状态数据收集
        status_data = {
            "connection_status": "connected",
            "signal_strength": np.random.uniform(70, 100),  # 信号强度
            "battery_level": np.random.uniform(80, 100),    # 电池电量
            "operational_time": time.time() - 3600,         # 运行时间
            "error_count": np.random.randint(0, 3),         # 错误计数
            "last_update": datetime.now().isoformat()
        }
        
        data_point = HardwareDataPoint(
            device_id="device_001",
            data_type=HardwareDataType.STATUS,
            timestamp=datetime.now(),
            data=status_data,
            confidence=0.98,
            quality_score=0.99
        )
        
        return [data_point]
    
    async def _collect_performance_data(self) -> List[HardwareDataPoint]:
        """收集性能数据"""
        # 模拟性能数据收集
        performance_data = {
            "response_time": np.random.exponential(0.05),    # 响应时间
            "throughput": np.random.uniform(100, 500),       # 吞吐量
            "cpu_usage": np.random.uniform(20, 60),          # CPU使用率
            "memory_usage": np.random.uniform(30, 70),       # 内存使用率
            "network_latency": np.random.exponential(0.02),  # 网络延迟
            "processing_efficiency": np.random.uniform(0.7, 0.95)  # 处理效率
        }
        
        data_point = HardwareDataPoint(
            device_id="system_001",
            data_type=HardwareDataType.PERFORMANCE,
            timestamp=datetime.now(),
            data=performance_data,
            confidence=0.90,
            quality_score=0.94
        )
        
        return [data_point]
    
    async def _collect_environment_data(self) -> List[HardwareDataPoint]:
        """收集环境数据"""
        # 模拟环境数据收集
        environment_data = {
            "ambient_temperature": np.random.normal(22, 3),  # 环境温度
            "ambient_humidity": np.random.normal(55, 8),     # 环境湿度
            "air_quality": np.random.uniform(0.7, 1.0),      # 空气质量
            "barometric_pressure": np.random.normal(1013, 10), # 气压
            "wind_speed": np.random.exponential(2.0),        # 风速
            "uv_index": np.random.uniform(2, 8)              # UV指数
        }
        
        data_point = HardwareDataPoint(
            device_id="environment_001",
            data_type=HardwareDataType.ENVIRONMENT,
            timestamp=datetime.now(),
            data=environment_data,
            confidence=0.88,
            quality_score=0.92
        )
        
        return [data_point]
    
    async def _process_and_queue_data(self):
        """处理并排队数据供AI学习"""
        if not self.data_buffer:
            return
            
        # 获取最新的数据
        latest_data = self.data_buffer[-min(10, len(self.data_buffer)):]
        
        for data_point in latest_data:
            # 预处理数据
            processed_data = await self._preprocess_data(data_point)
            
            # 如果有AI学习回调，则调用
            if self.ai_learning_callback:
                try:
                    await self.ai_learning_callback(processed_data)
                except Exception as e:
                    logger.error(f"AI学习回调执行失败: {e}")
    
    async def _preprocess_data(self, data_point: HardwareDataPoint) -> Dict[str, Any]:
        """预处理数据"""
        if data_point.data_type in self.preprocessors:
            return await self.preprocessors[data_point.data_type](data_point)
        else:
            # 默认预处理
            return {
                "device_id": data_point.device_id,
                "data_type": data_point.data_type.value,
                "timestamp": data_point.timestamp.isoformat(),
                "raw_data": data_point.data,
                "confidence": data_point.confidence,
                "quality_score": data_point.quality_score
            }
    
    async def _preprocess_sensors(self, data_point: HardwareDataPoint) -> Dict[str, Any]:
        """预处理传感器数据"""
        # 标准化传感器数据
        normalized_data = {}
        for key, value in data_point.data.items():
            # 简单的标准化处理（实际应用中可能需要更复杂的标准化）
            if isinstance(value, (int, float)):
                # 假设合理的范围，进行归一化
                if key == "temperature":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, (value - 10) / 40))
                elif key == "humidity":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, value / 100))
                elif key == "light_intensity":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, value / 1000))
                elif key == "co2_level":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, value / 2000))
                elif key == "soil_moisture":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, value / 100))
                else:
                    normalized_data[key] = value
            else:
                normalized_data[key] = value
        
        return {
            "device_id": data_point.device_id,
            "data_type": data_point.data_type.value,
            "timestamp": data_point.timestamp.isoformat(),
            "processed_data": normalized_data,
            "confidence": data_point.confidence,
            "quality_score": data_point.quality_score,
            "data_features": self._extract_features(normalized_data)
        }
    
    async def _preprocess_controllers(self, data_point: HardwareDataPoint) -> Dict[str, Any]:
        """预处理控制器数据"""
        # 提取控制器特征
        features = {}
        for key, value in data_point.data.items():
            if isinstance(value, (int, float)):
                features[f"controller_{key}_feature"] = value
            else:
                features[f"controller_{key}_feature"] = str(value)
        
        return {
            "device_id": data_point.device_id,
            "data_type": data_point.data_type.value,
            "timestamp": data_point.timestamp.isoformat(),
            "processed_data": data_point.data,
            "confidence": data_point.confidence,
            "quality_score": data_point.quality_score,
            "data_features": features
        }
    
    async def _preprocess_status(self, data_point: HardwareDataPoint) -> Dict[str, Any]:
        """预处理状态数据"""
        return {
            "device_id": data_point.device_id,
            "data_type": data_point.data_type.value,
            "timestamp": data_point.timestamp.isoformat(),
            "processed_data": data_point.data,
            "confidence": data_point.confidence,
            "quality_score": data_point.quality_score,
            "data_features": {
                "connection_score": 1.0 if data_point.data.get("connection_status") == "connected" else 0.0,
                "signal_strength_normalized": data_point.data.get("signal_strength", 0) / 100,
                "battery_level_normalized": data_point.data.get("battery_level", 0) / 100
            }
        }
    
    async def _preprocess_performance(self, data_point: HardwareDataPoint) -> Dict[str, Any]:
        """预处理性能数据"""
        # 性能数据特征提取
        features = {}
        for key, value in data_point.data.items():
            if isinstance(value, (int, float)):
                # 归一化性能指标
                if key == "cpu_usage" or key == "memory_usage":
                    features[f"performance_{key}_normalized"] = value / 100
                elif key == "response_time":
                    features[f"performance_{key}_normalized"] = min(1.0, value * 20)  # 响应时间倒数
                else:
                    features[f"performance_{key}_feature"] = value
            else:
                features[f"performance_{key}_feature"] = str(value)
        
        return {
            "device_id": data_point.device_id,
            "data_type": data_point.data_type.value,
            "timestamp": data_point.timestamp.isoformat(),
            "processed_data": data_point.data,
            "confidence": data_point.confidence,
            "quality_score": data_point.quality_score,
            "data_features": features
        }
    
    async def _preprocess_environment(self, data_point: HardwareDataPoint) -> Dict[str, Any]:
        """预处理环境数据"""
        # 环境数据标准化
        normalized_data = {}
        for key, value in data_point.data.items():
            if isinstance(value, (int, float)):
                if key == "ambient_temperature":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, (value - 0) / 50))
                elif key == "ambient_humidity":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, value / 100))
                elif key == "air_quality":
                    normalized_data[f"{key}_normalized"] = value
                elif key == "barometric_pressure":
                    normalized_data[f"{key}_normalized"] = max(0, min(1, (value - 950) / 150))
                elif key == "wind_speed":
                    normalized_data[f"{key}_normalized"] = min(1, value / 20)
                elif key == "uv_index":
                    normalized_data[f"{key}_normalized"] = value / 11
                else:
                    normalized_data[key] = value
            else:
                normalized_data[key] = value
        
        return {
            "device_id": data_point.device_id,
            "data_type": data_point.data_type.value,
            "timestamp": data_point.timestamp.isoformat(),
            "processed_data": normalized_data,
            "confidence": data_point.confidence,
            "quality_score": data_point.quality_score,
            "data_features": self._extract_features(normalized_data)
        }
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """从数据中提取数值特征"""
        features = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
        return features
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.data_buffer:
            return {"total_data_points": 0}
        
        # 按数据类型统计
        type_counts = {}
        total_confidence = 0
        total_quality = 0
        
        for data_point in self.data_buffer:
            type_key = data_point.data_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            total_confidence += data_point.confidence
            total_quality += data_point.quality_score
        
        return {
            "total_data_points": len(self.data_buffer),
            "data_type_distribution": type_counts,
            "average_confidence": total_confidence / len(self.data_buffer),
            "average_quality_score": total_quality / len(self.data_buffer),
            "buffer_size": len(self.data_buffer),
            "max_buffer_size": self.max_buffer_size
        }
    
    async def get_recent_data(self, count: int = 10) -> List[HardwareDataPoint]:
        """获取最近的数据点"""
        return self.data_buffer[-min(count, len(self.data_buffer)):]
    
    async def export_data_for_ai_training(self) -> Dict[str, Any]:
        """导出数据用于AI训练"""
        # 将数据转换为AI模型可以使用的格式
        features_list = []
        targets_list = []
        
        for data_point in self.data_buffer:
            # 使用预处理后的特征，确保所有特征都是数值类型
            processed = await self._preprocess_data(data_point)
            features_dict = processed.get('data_features', {})
            
            # 提取数值特征并确保它们是相同长度的向量
            numeric_features = []
            for value in features_dict.values():
                if isinstance(value, (int, float)):
                    numeric_features.append(float(value))
                elif isinstance(value, str):
                    # 将字符串转换为数值（例如哈希值）
                    numeric_features.append(float(hash(value) % 10000) / 10000.0)
            
            # 确保所有特征向量长度一致，如果长度不一致则跳过或填充
            if numeric_features:
                # 假设期望的特征维度是固定的，这里我们取前N个特征或填充
                expected_dim = 20  # 设定一个期望的维度
                if len(numeric_features) < expected_dim:
                    # 填充零以达到期望维度
                    numeric_features.extend([0.0] * (expected_dim - len(numeric_features)))
                elif len(numeric_features) > expected_dim:
                    # 截断到期望维度
                    numeric_features = numeric_features[:expected_dim]
                
                features_list.append(numeric_features)
                # 目标值可以是质量分数或其他指标
                targets_list.append(float(data_point.quality_score))
        
        if features_list:
            # 转换为JAX数组 - 现在所有特征向量长度一致
            import numpy as np
            features_array = jnp.array(features_list)
            targets_array = jnp.array(targets_list)
            
            return {
                "features": features_array,
                "targets": targets_array,
                "sample_count": len(features_list),
                "feature_dimension": features_array.shape[1] if len(features_list) > 0 else 0
            }
        else:
            return {
                "features": jnp.array([]).reshape(0, 0),
                "targets": jnp.array([]),
                "sample_count": 0,
                "feature_dimension": 0
            }


# 全局硬件数据收集器实例
hardware_data_collector = HardwareDataCollector()