"""
决策服务数据模型
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

Base = declarative_base()


class DecisionAction(str, Enum):
    """决策动作枚举"""
    ADJUST_SPECTRUM = "adjust_spectrum"
    ADJUST_TEMPERATURE = "adjust_temperature" 
    ADJUST_HUMIDITY = "adjust_humidity"
    ADJUST_NUTRIENTS = "adjust_nutrients"
    NO_ACTION = "no_action"


class DecisionObjective(str, Enum):
    """决策目标枚举"""
    MAXIMIZE_YIELD = "maximize_yield"
    IMPROVE_QUALITY = "improve_quality"
    ENHANCE_RESISTANCE = "enhance_resistance"
    OPTIMIZE_EFFICIENCY = "optimize_efficiency"


class DecisionStatus(str, Enum):
    """决策状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# SQLAlchemy 数据库模型
class DecisionRecord(Base):
    """决策记录表"""
    __tablename__ = "decision_records"
    
    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(String(64), unique=True, index=True, nullable=False)
    batch_id = Column(String(64), index=True)
    crop_type = Column(String(50), nullable=False)
    objective = Column(String(20), nullable=False)
    action = Column(String(20), nullable=False)
    parameters = Column(JSON, nullable=False)
    expected_reward = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=False)
    status = Column(String(20), default=DecisionStatus.PENDING.value)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DecisionFeedbackModel(Base):
    """决策反馈表（ORM模型）"""
    __tablename__ = "decision_feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(String(64), index=True, nullable=False)
    actual_reward = Column(Float, nullable=False)
    next_state = Column(JSON, nullable=False)
    success_indicator = Column(Boolean, nullable=False)
    feedback_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PerformanceMetrics(Base):
    """性能指标表（ORM模型）"""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    average_reward = Column(Float, nullable=False)
    decision_count = Column(Integer, nullable=False)
    success_rate = Column(Float, nullable=False)
    avg_execution_time = Column(Float, nullable=False)
    system_load = Column(Float, nullable=False)


# Pydantic 请求/响应模型
class DecisionRequest(BaseModel):
    """决策请求"""
    temperature: float
    humidity: float
    co2_level: float = 400.0
    light_intensity: float
    spectrum_config: Dict[str, float]
    crop_type: str
    growth_day: int
    growth_rate: float
    health_score: float
    yield_potential: float
    energy_consumption: float
    resource_utilization: float
    objective: DecisionObjective


class BatchDecisionRequest(BaseModel):
    """批量决策请求"""
    requests: List[DecisionRequest]
    batch_id: str
    priority: str = "normal"


class DecisionResponse(BaseModel):
    """决策响应"""
    decision_id: str
    action: str
    parameters: Dict[str, float]
    expected_reward: float
    confidence: float
    execution_time: float
    recommendations: Optional[List[str]] = None


class DecisionFeedback(BaseModel):
    """决策反馈（请求体）"""
    decision_id: str
    actual_reward: float
    next_state: Dict[str, Any]
    success_indicator: bool
    feedback_notes: Optional[str] = None


class PerformanceMetricsResponse(BaseModel):
    """性能指标响应"""
    average_reward: float
    decision_count: int
    success_rate: float
    avg_execution_time: float
    system_load: float
    timestamp: float

# Note: keep ORM models and Pydantic schemas using distinct names where
# necessary to avoid accidental name shadowing.


