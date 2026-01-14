"""
环境感知模块
增强AI系统的环境感知能力，支持动态环境特征提取和上下文理解
集成多模态输入处理能力
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

logger = logging.getLogger(__name__)

# 导入多模态编码器
from .multimodal_encoder import MultimodalInput, multimodal_encoder


@dataclass
class EnvironmentFeature:
    """环境特征数据结构"""
    name: str
    value: Any
    confidence: float
    timestamp: datetime
    type: str  # sensor, inferred, contextual, multimodal
    source: str
    unit: Optional[str] = None
    modality: Optional[str] = None  # 多模态特征类型: text, image, audio, unified
    encoding: Optional[Any] = None  # 编码后的特征数据


class EnvironmentPerceptionSystem:
    """环境感知系统"""
    
    def __init__(self):
        self.environment_features: Dict[str, EnvironmentFeature] = {}
        self.context_history: List[Dict[str, Any]] = []
        self.dynamic_context: Dict[str, Any] = {}
        self.perception_enabled = True
        self.context_window_size = 100  # 保存最近100个上下文
        
        logger.info("环境感知系统初始化完成")
    
    def perceive_environment(self, raw_sensor_data: Dict[str, Any]) -> Dict[str, EnvironmentFeature]:
        """感知环境，提取环境特征"""
        if not self.perception_enabled:
            return {}
        
        extracted_features = {}
        
        # 提取原始传感器数据
        for key, value in raw_sensor_data.items():
            feature = EnvironmentFeature(
                name=key,
                value=value,
                confidence=0.95,  # 传感器数据置信度较高
                timestamp=datetime.now(),
                type="sensor",
                source="raw_sensor",
                unit=self._get_unit_for_feature(key)
            )
            extracted_features[key] = feature
            self.environment_features[key] = feature
        
        # 推断高级特征
        inferred_features = self._infer_advanced_features(raw_sensor_data)
        for key, value in inferred_features.items():
            extracted_features[key] = value
            self.environment_features[key] = value
        
        # 更新动态上下文
        self._update_dynamic_context(extracted_features)
        
        return extracted_features
    
    def perceive_multimodal(self, 
                           text: Optional[str] = None,
                           images: Optional[List[Union[str, BytesIO]]] = None,
                           audio: Optional[Union[str, BytesIO]] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, EnvironmentFeature]:
        """感知多模态环境，处理文本、图像和音频输入"""
        if not self.perception_enabled:
            return {}
        
        logger.debug(f"开始感知多模态环境: 文本={text is not None}, "
                   f"图像={len(images) if images else 0}, "
                   f"音频={audio is not None}")
        
        # 创建多模态输入
        multimodal_input = MultimodalInput(
            text=text,
            images=images,
            audio=audio,
            context=context
        )
        
        # 使用多模态编码器处理
        encoding_result = multimodal_encoder.encode_multimodal(multimodal_input)
        
        extracted_features = {}
        timestamp = datetime.now()
        
        # 处理文本编码
        if encoding_result.text_encoding:
            text_feature = EnvironmentFeature(
                name="text_input",
                value=encoding_result.text_encoding.encoding,
                confidence=encoding_result.text_encoding.confidence,
                timestamp=timestamp,
                type="multimodal",
                source="multimodal_encoder",
                modality="text",
                encoding=encoding_result.text_encoding.encoding
            )
            extracted_features["text_input"] = text_feature
            self.environment_features["text_input"] = text_feature
        
        # 处理图像编码
        if encoding_result.image_encodings:
            for i, img_encoding in enumerate(encoding_result.image_encodings):
                img_feature = EnvironmentFeature(
                    name=f"image_input_{i+1}",
                    value=img_encoding.encoding,
                    confidence=img_encoding.confidence,
                    timestamp=timestamp,
                    type="multimodal",
                    source="multimodal_encoder",
                    modality="image",
                    encoding=img_encoding.encoding
                )
                extracted_features[f"image_input_{i+1}"] = img_feature
                self.environment_features[f"image_input_{i+1}"] = img_feature
        
        # 处理音频编码
        if encoding_result.audio_encoding:
            audio_feature = EnvironmentFeature(
                name="audio_input",
                value=encoding_result.audio_encoding.encoding,
                confidence=encoding_result.audio_encoding.confidence,
                timestamp=timestamp,
                type="multimodal",
                source="multimodal_encoder",
                modality="audio",
                encoding=encoding_result.audio_encoding.encoding
            )
            extracted_features["audio_input"] = audio_feature
            self.environment_features["audio_input"] = audio_feature
        
        # 添加统一编码表示
        unified_feature = EnvironmentFeature(
            name="unified_multimodal",
            value=encoding_result.unified_representation,
            confidence=encoding_result.confidence,
            timestamp=timestamp,
            type="multimodal",
            source="multimodal_encoder",
            modality="unified"
        )
        extracted_features["unified_multimodal"] = unified_feature
        self.environment_features["unified_multimodal"] = unified_feature
        
        # 更新动态上下文
        self._update_dynamic_context(extracted_features)
        
        logger.debug(f"多模态感知完成，提取了 {len(extracted_features)} 个特征")
        return extracted_features
    
    def process_multimodal_query(self, 
                                query: str,
                                images: Optional[List[Union[str, BytesIO]]] = None,
                                audio: Optional[Union[str, BytesIO]] = None,
                                context: Optional[Dict[str, Any]] = None) -> str:
        """处理多模态查询，返回自然语言响应"""
        logger.debug(f"开始处理多模态查询: {query}")
        
        # 创建多模态输入
        multimodal_input = MultimodalInput(
            text=query,
            images=images,
            audio=audio,
            context=context
        )
        
        # 使用多模态编码器处理
        encoding_result = multimodal_encoder.encode_multimodal(multimodal_input)
        
        # 同时更新环境特征
        self.perceive_multimodal(text=query, images=images, audio=audio, context=context)
        
        # 使用GPT-4o处理查询
        try:
            response = multimodal_encoder.process_with_gpt4o(
                encoding=encoding_result,
                prompt=query
            )
            logger.debug(f"多模态查询处理完成")
            return response
        except Exception as e:
            logger.error(f"处理多模态查询时出错: {str(e)}")
            return f"处理多模态查询失败: {str(e)}"
    
    def _get_unit_for_feature(self, feature_name: str) -> Optional[str]:
        """获取特征的单位"""
        unit_map = {
            "temperature": "°C",
            "humidity": "%",
            "co2_level": "ppm",
            "light_intensity": "lux",
            "energy_consumption": "W",
            "pressure": "Pa",
            "wind_speed": "m/s"
        }
        return unit_map.get(feature_name)
    
    def _infer_advanced_features(self, raw_sensor_data: Dict[str, Any]) -> Dict[str, EnvironmentFeature]:
        """从原始数据中推断高级特征"""
        inferred = {}
        
        # 示例：推断舒适度指数
        if "temperature" in raw_sensor_data and "humidity" in raw_sensor_data:
            comfort_index = self._calculate_comfort_index(
                raw_sensor_data["temperature"],
                raw_sensor_data["humidity"]
            )
            inferred["comfort_index"] = EnvironmentFeature(
                name="comfort_index",
                value=comfort_index,
                confidence=0.85,
                timestamp=datetime.now(),
                type="inferred",
                source="comfort_calculator",
                unit="index"
            )
        
        # 示例：推断天气状况
        if "temperature" in raw_sensor_data and "humidity" in raw_sensor_data:
            weather_condition = self._infer_weather_condition(
                raw_sensor_data["temperature"],
                raw_sensor_data["humidity"]
            )
            inferred["weather_condition"] = EnvironmentFeature(
                name="weather_condition",
                value=weather_condition,
                confidence=0.8,
                timestamp=datetime.now(),
                type="inferred",
                source="weather_inferrer"
            )
        
        # 示例：推断活动状态
        if "energy_consumption" in raw_sensor_data:
            activity_level = self._infer_activity_level(raw_sensor_data["energy_consumption"])
            inferred["activity_level"] = EnvironmentFeature(
                name="activity_level",
                value=activity_level,
                confidence=0.75,
                timestamp=datetime.now(),
                type="inferred",
                source="activity_inferrer"
            )
        
        return inferred
    
    def _calculate_comfort_index(self, temperature: float, humidity: float) -> float:
        """计算舒适度指数"""
        # 简单的舒适度指数计算
        # 基于温度和湿度的综合影响
        temp_factor = 1.0
        if temperature > 30:
            temp_factor = 1.0 - (temperature - 30) / 20
        elif temperature < 10:
            temp_factor = 1.0 - (10 - temperature) / 20
        
        humidity_factor = 1.0
        if humidity > 80:
            humidity_factor = 1.0 - (humidity - 80) / 50
        elif humidity < 30:
            humidity_factor = 1.0 - (30 - humidity) / 50
        
        return round(temp_factor * humidity_factor * 100, 2)
    
    def _infer_weather_condition(self, temperature: float, humidity: float) -> str:
        """推断天气状况"""
        if humidity > 85:
            return "潮湿"
        elif humidity < 30:
            return "干燥"
        elif temperature > 35:
            return "炎热"
        elif temperature < 0:
            return "寒冷"
        elif temperature > 25:
            return "温暖"
        else:
            return "舒适"
    
    def _infer_activity_level(self, energy_consumption: float) -> str:
        """推断活动水平"""
        if energy_consumption > 500:
            return "高"
        elif energy_consumption > 200:
            return "中"
        else:
            return "低"
    
    def _update_dynamic_context(self, features: Dict[str, EnvironmentFeature]):
        """更新动态上下文"""
        current_context = {
            "timestamp": datetime.now(),
            "features": {k: v.value for k, v in features.items()}
        }
        
        # 更新上下文历史
        self.context_history.append(current_context)
        if len(self.context_history) > self.context_window_size:
            self.context_history = self.context_history[-self.context_window_size:]
        
        # 更新动态上下文
        self.dynamic_context = self._analyze_context_trends()
    
    def _analyze_context_trends(self) -> Dict[str, Any]:
        """分析上下文趋势"""
        if len(self.context_history) < 10:
            return {"trend": "stable"}
        
        # 简单的趋势分析示例
        recent_temperatures = []
        recent_humidity = []
        
        for context in self.context_history[-10:]:
            if "temperature" in context["features"]:
                recent_temperatures.append(context["features"]["temperature"])
            if "humidity" in context["features"]:
                recent_humidity.append(context["features"]["humidity"])
        
        trend = "stable"
        
        # 温度趋势分析
        if len(recent_temperatures) >= 5:
            temp_change = recent_temperatures[-1] - recent_temperatures[0]
            if temp_change > 2:
                trend = "warming"
            elif temp_change < -2:
                trend = "cooling"
        
        # 湿度趋势分析
        if len(recent_humidity) >= 5:
            humidity_change = recent_humidity[-1] - recent_humidity[0]
            if humidity_change > 10:
                trend = "humidifying"
            elif humidity_change < -10:
                trend = "drying"
        
        return {
            "trend": trend,
            "context_count": len(self.context_history),
            "recent_changes": {
                "temperature": recent_temperatures[-1] - recent_temperatures[0] if len(recent_temperatures) >= 5 else 0,
                "humidity": recent_humidity[-1] - recent_humidity[0] if len(recent_humidity) >= 5 else 0
            }
        }
    
    def get_environment_context(self) -> Dict[str, Any]:
        """获取当前环境上下文"""
        return {
            "current_features": {k: v.value for k, v in self.environment_features.items()},
            "dynamic_context": self.dynamic_context,
            "context_history": self.context_history[-10:],  # 返回最近10个上下文
            "timestamp": datetime.now()
        }
    
    def detect_environment_changes(self) -> List[Dict[str, Any]]:
        """检测环境变化"""
        changes = []
        
        if len(self.context_history) < 2:
            return changes
        
        # 比较最近两个上下文
        prev_context = self.context_history[-2]
        current_context = self.context_history[-1]
        
        for feature_name, current_value in current_context["features"].items():
            if feature_name in prev_context["features"]:
                prev_value = prev_context["features"][feature_name]
                # 计算变化百分比
                if isinstance(current_value, (int, float)) and isinstance(prev_value, (int, float)) and prev_value != 0:
                    change_percent = ((current_value - prev_value) / prev_value) * 100
                    if abs(change_percent) > 10:  # 变化超过10%时报告
                        changes.append({
                            "feature": feature_name,
                            "previous_value": prev_value,
                            "current_value": current_value,
                            "change_percent": round(change_percent, 2),
                            "timestamp": current_context["timestamp"]
                        })
        
        return changes
    
    def predict_environment_trends(self, steps: int = 5) -> Dict[str, Any]:
        """预测环境趋势"""
        if len(self.context_history) < 10:
            return {"prediction": "insufficient_data"}
        
        # 简单的线性预测示例
        predictions = {}
        
        # 只预测数值类型的特征
        numeric_features = []
        for feature_name, feature in self.environment_features.items():
            if isinstance(feature.value, (int, float)):
                numeric_features.append(feature_name)
        
        for feature_name in numeric_features:
            # 提取历史值
            history_values = []
            for context in self.context_history[-10:]:
                if feature_name in context["features"]:
                    history_values.append(context["features"][feature_name])
            
            if len(history_values) >= 5:
                # 计算趋势（斜率）
                x = list(range(len(history_values)))
                from statistics import mean
                mean_x = mean(x)
                mean_y = mean(history_values)
                
                # 计算斜率
                numerator = sum((x[i] - mean_x) * (history_values[i] - mean_y) for i in range(len(x)))
                denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
                
                if denominator != 0:
                    slope = numerator / denominator
                    # 预测未来值
                    last_value = history_values[-1]
                    predictions[feature_name] = {
                        "current": last_value,
                        "predicted": last_value + slope * steps,
                        "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                        "confidence": 0.6  # 简单预测置信度
                    }
        
        return {
            "prediction_time": datetime.now(),
            "predicted_steps": steps,
            "predictions": predictions
        }


# 创建全局环境感知系统实例
environment_perception_system = EnvironmentPerceptionSystem()
