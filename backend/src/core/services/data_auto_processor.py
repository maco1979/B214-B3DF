"""数据自动处理服务

提供完整的数据自动处理流程，包括：
1. 数据质量验证
2. 数据清洗
3. 数据转换
4. 特征工程
5. 数据标准化/归一化
6. 异常检测和处理
7. 数据质量监控

该服务支持跨行业自适应数据处理，能够根据不同行业的数据特点自动调整处理策略。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# 导入现有的数据验证模块
from src.migration_learning.data_validation import (
    DataQualityReport, DataQualityLevel, AnomalyType,
    DataCredibilityValidator
)


class DataProcessingStage(Enum):
    """数据处理阶段"""
    VALIDATION = "validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    STANDARDIZATION = "standardization"
    ANOMALY_HANDLING = "anomaly_handling"
    OUTPUT = "output"


class IndustryType(Enum):
    """行业类型"""
    AGRICULTURE = "agriculture"
    INDUSTRY = "industry"
    HOME = "home"
    HEALTHCARE = "healthcare"
    COMMERCIAL = "commercial"
    AUTOMOTIVE = "automotive"
    LOGISTICS = "logistics"
    ENERGY = "energy"


@dataclass
class DataProcessingResult:
    """数据处理结果"""
    processed_data: Any
    metadata: Dict[str, Any]
    quality_report: DataQualityReport
    processing_time: float
    processing_stages: List[DataProcessingStage]
    warnings: List[str]
    errors: List[str]


@dataclass
class FeatureConfig:
    """特征配置"""
    name: str
    data_type: str
    is_numeric: bool
    is_categorical: bool
    is_temporal: bool
    missing_value_strategy: str
    normalization_strategy: str
    encoding_strategy: str


class DataAutoProcessor:
    """数据自动处理器
    
    支持跨行业的自适应数据处理，能够根据不同行业的数据特点自动调整处理策略。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.validator = DataCredibilityValidator(config)
        
        # 行业特定配置
        self.industry_configs = self._get_industry_specific_configs()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "processing_stages": [
                DataProcessingStage.VALIDATION,
                DataProcessingStage.CLEANING,
                DataProcessingStage.TRANSFORMATION,
                DataProcessingStage.FEATURE_ENGINEERING,
                DataProcessingStage.STANDARDIZATION,
                DataProcessingStage.ANOMALY_HANDLING,
                DataProcessingStage.OUTPUT
            ],
            "missing_value_strategies": {
                "numeric": "mean",
                "categorical": "mode",
                "temporal": "drop"
            },
            "normalization_strategies": {
                "default": "minmax",
                "skewed": "quantile"
            },
            "encoding_strategies": {
                "low_cardinality": "onehot",
                "high_cardinality": "target"
            },
            "anomaly_detection": {
                "method": "iqr",
                "threshold": 1.5
            },
            "feature_selection": {
                "enabled": True,
                "method": "correlation",
                "threshold": 0.9
            }
        }
    
    def _get_industry_specific_configs(self) -> Dict[IndustryType, Dict[str, Any]]:
        """获取行业特定配置"""
        return {
            IndustryType.AGRICULTURE: {
                "missing_value_strategies": {
                    "numeric": "mean",
                    "categorical": "mode",
                    "temporal": "forward_fill"
                },
                "required_features": ["temperature", "humidity", "soil_moisture", "light_intensity"],
                "outlier_threshold": 2.0
            },
            IndustryType.INDUSTRY: {
                "missing_value_strategies": {
                    "numeric": "median",
                    "categorical": "mode",
                    "temporal": "linear_interpolate"
                },
                "required_features": ["temperature", "pressure", "vibration", "voltage"],
                "outlier_threshold": 3.0
            },
            IndustryType.HOME: {
                "missing_value_strategies": {
                    "numeric": "mean",
                    "categorical": "mode",
                    "temporal": "fill_zero"
                },
                "required_features": ["temperature", "humidity", "energy_consumption", "occupancy"],
                "outlier_threshold": 2.5
            },
            IndustryType.HEALTHCARE: {
                "missing_value_strategies": {
                    "numeric": "median",
                    "categorical": "mode",
                    "temporal": "backward_fill"
                },
                "required_features": ["heart_rate", "blood_pressure", "temperature", "respiratory_rate"],
                "outlier_threshold": 2.0
            },
            IndustryType.COMMERCIAL: {
                "missing_value_strategies": {
                    "numeric": "mean",
                    "categorical": "mode",
                    "temporal": "forward_fill"
                },
                "required_features": ["foot_traffic", "sales", "temperature", "humidity"],
                "outlier_threshold": 2.5
            },
            IndustryType.AUTOMOTIVE: {
                "missing_value_strategies": {
                    "numeric": "median",
                    "categorical": "mode",
                    "temporal": "linear_interpolate"
                },
                "required_features": ["speed", "engine_temperature", "fuel_level", "battery_voltage"],
                "outlier_threshold": 3.0
            },
            IndustryType.LOGISTICS: {
                "missing_value_strategies": {
                    "numeric": "mean",
                    "categorical": "mode",
                    "temporal": "forward_fill"
                },
                "required_features": ["location", "speed", "fuel_consumption", "cargo_weight"],
                "outlier_threshold": 2.5
            },
            IndustryType.ENERGY: {
                "missing_value_strategies": {
                    "numeric": "median",
                    "categorical": "mode",
                    "temporal": "linear_interpolate"
                },
                "required_features": ["power_generation", "demand", "temperature", "humidity"],
                "outlier_threshold": 3.0
            }
        }
    
    def process_data(self, data: Any, industry: str = "agriculture", metadata: Optional[Dict[str, Any]] = None) -> DataProcessingResult:
        """处理数据
        
        Args:
            data: 待处理的数据
            industry: 行业类型
            metadata: 数据元数据
            
        Returns:
            DataProcessingResult: 数据处理结果
        """
        start_time = datetime.now()
        warnings = []
        errors = []
        processing_stages = []
        
        try:
            # 转换行业字符串为枚举
            industry_type = IndustryType(industry)
            industry_config = self.industry_configs[industry_type]
            
            # 1. 数据验证
            if DataProcessingStage.VALIDATION in self.config["processing_stages"]:
                quality_report = self.validator.validate_data_quality(data, metadata)
                processing_stages.append(DataProcessingStage.VALIDATION)
                
                if not quality_report.validation_passed:
                    warnings.extend(quality_report.recommendations)
            
            # 2. 数据清洗
            if DataProcessingStage.CLEANING in self.config["processing_stages"]:
                cleaned_data, cleaning_warnings = self._clean_data(data, industry_config)
                data = cleaned_data
                warnings.extend(cleaning_warnings)
                processing_stages.append(DataProcessingStage.CLEANING)
            
            # 3. 数据转换
            if DataProcessingStage.TRANSFORMATION in self.config["processing_stages"]:
                transformed_data, transformation_warnings = self._transform_data(data, industry_config)
                data = transformed_data
                warnings.extend(transformation_warnings)
                processing_stages.append(DataProcessingStage.TRANSFORMATION)
            
            # 4. 特征工程
            if DataProcessingStage.FEATURE_ENGINEERING in self.config["processing_stages"]:
                engineered_data, engineering_warnings = self._engineer_features(data, industry_config)
                data = engineered_data
                warnings.extend(engineering_warnings)
                processing_stages.append(DataProcessingStage.FEATURE_ENGINEERING)
            
            # 5. 数据标准化
            if DataProcessingStage.STANDARDIZATION in self.config["processing_stages"]:
                standardized_data, standardization_warnings = self._standardize_data(data, industry_config)
                data = standardized_data
                warnings.extend(standardization_warnings)
                processing_stages.append(DataProcessingStage.STANDARDIZATION)
            
            # 6. 异常处理
            if DataProcessingStage.ANOMALY_HANDLING in self.config["processing_stages"]:
                handled_data, anomaly_warnings = self._handle_anomalies(data, industry_config)
                data = handled_data
                warnings.extend(anomaly_warnings)
                processing_stages.append(DataProcessingStage.ANOMALY_HANDLING)
            
            # 7. 输出准备
            if DataProcessingStage.OUTPUT in self.config["processing_stages"]:
                output_data, output_warnings = self._prepare_output(data, industry_config)
                data = output_data
                warnings.extend(output_warnings)
                processing_stages.append(DataProcessingStage.OUTPUT)
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DataProcessingResult(
                processed_data=data,
                metadata={
                    **(metadata or {}),
                    "industry": industry,
                    "processing_time": processing_time,
                    "processing_stages": [stage.value for stage in processing_stages],
                    "processed_at": datetime.now().isoformat()
                },
                quality_report=quality_report,
                processing_time=processing_time,
                processing_stages=processing_stages,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            errors.append(str(e))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DataProcessingResult(
                processed_data=data,
                metadata={**(metadata or {}), "industry": industry},
                quality_report=DataQualityReport(
                    quality_level=DataQualityLevel.UNACCEPTABLE,
                    overall_score=0.0,
                    anomaly_details={},
                    validation_passed=False,
                    recommendations=[f"数据处理过程异常: {str(e)}"]
                ),
                processing_time=processing_time,
                processing_stages=processing_stages,
                warnings=warnings,
                errors=errors
            )
    
    def _clean_data(self, data: Any, industry_config: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """数据清洗
        
        处理缺失值、重复值等
        """
        warnings = []
        
        # 简化实现 - 实际应根据数据类型（如DataFrame、Dict等）实现不同的清洗逻辑
        if isinstance(data, pd.DataFrame):
            # 处理重复值
            initial_shape = data.shape
            data = data.drop_duplicates()
            if data.shape[0] < initial_shape[0]:
                warnings.append(f"移除了 {initial_shape[0] - data.shape[0]} 个重复行")
            
            # 处理缺失值
            for column in data.columns:
                if data[column].isnull().any():
                    if data[column].dtype in ['int64', 'float64']:
                        # 数值型数据
                        strategy = industry_config["missing_value_strategies"]["numeric"]
                        if strategy == "mean":
                            data[column] = data[column].fillna(data[column].mean())
                        elif strategy == "median":
                            data[column] = data[column].fillna(data[column].median())
                        warnings.append(f"使用{strategy}填充了列 {column} 的缺失值")
                    else:
                        # 类别型数据
                        strategy = industry_config["missing_value_strategies"]["categorical"]
                        if strategy == "mode":
                            data[column] = data[column].fillna(data[column].mode()[0])
                        warnings.append(f"使用{strategy}填充了列 {column} 的缺失值")
        elif isinstance(data, dict):
            # 字典类型数据处理
            pass
        
        return data, warnings
    
    def _transform_data(self, data: Any, industry_config: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """数据转换
        
        转换数据格式、单位等
        """
        warnings = []
        
        # 简化实现 - 根据数据类型实现转换逻辑
        return data, warnings
    
    def _engineer_features(self, data: Any, industry_config: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """特征工程
        
        创建新特征、选择重要特征等
        """
        warnings = []
        
        # 简化实现 - 根据数据类型实现特征工程
        if isinstance(data, pd.DataFrame):
            # 特征选择
            if self.config["feature_selection"]["enabled"]:
                # 计算相关性矩阵
                corr_matrix = data.corr().abs()
                
                # 创建上三角矩阵
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # 找出相关性大于阈值的列
                to_drop = [column for column in upper.columns if any(upper[column] > self.config["feature_selection"]["threshold"])]
                
                if to_drop:
                    data = data.drop(to_drop, axis=1)
                    warnings.append(f"移除了高相关性特征: {', '.join(to_drop)}")
        
        return data, warnings
    
    def _standardize_data(self, data: Any, industry_config: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """数据标准化
        
        标准化/归一化数值型特征
        """
        warnings = []
        
        # 简化实现 - 根据数据类型实现标准化
        if isinstance(data, pd.DataFrame):
            # 只处理数值型列
            numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
            
            for column in numeric_columns:
                # 使用Min-Max归一化
                min_val = data[column].min()
                max_val = data[column].max()
                
                if max_val != min_val:
                    data[column] = (data[column] - min_val) / (max_val - min_val)
                else:
                    data[column] = 0.0
        
        return data, warnings
    
    def _handle_anomalies(self, data: Any, industry_config: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """异常处理
        
        检测和处理异常值
        """
        warnings = []
        
        # 简化实现 - 根据数据类型实现异常处理
        if isinstance(data, pd.DataFrame):
            # 只处理数值型列
            numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
            
            for column in numeric_columns:
                # 使用IQR方法检测异常值
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                threshold = industry_config.get("outlier_threshold", 1.5)
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # 计算异常值数量
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                if not outliers.empty:
                    # 替换异常值为上下限
                    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
                    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
                    warnings.append(f"处理了列 {column} 中的 {len(outliers)} 个异常值")
        
        return data, warnings
    
    def _prepare_output(self, data: Any, industry_config: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """准备输出数据
        
        将数据转换为适合模型输入的格式
        """
        warnings = []
        
        # 简化实现 - 根据数据类型准备输出
        return data, warnings
    
    async def process_data_async(self, data: Any, industry: str = "agriculture", metadata: Optional[Dict[str, Any]] = None) -> DataProcessingResult:
        """异步处理数据"""
        # 目前使用同步实现，后续可扩展为异步
        return self.process_data(data, industry, metadata)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """获取处理状态"""
        return {
            "processing_stages": [stage.value for stage in self.config["processing_stages"]],
            "active_stage": None,
            "total_processed": 0,
            "success_count": 0,
            "failure_count": 0
        }
    
    def validate_industry_data(self, industry: str, data: Any) -> bool:
        """验证数据是否适合特定行业"""
        try:
            industry_type = IndustryType(industry)
            industry_config = self.industry_configs[industry_type]
            
            # 检查是否包含所有必需特征
            if isinstance(data, dict) and "features" in data:
                features = set(data["features"])
                required_features = set(industry_config.get("required_features", []))
                return required_features.issubset(features)
            
            return True
        except Exception as e:
            self.logger.error(f"验证行业数据失败: {e}")
            return False
