"""
决策API路由
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel

router = APIRouter(prefix="/api/decision", tags=["决策"])


class DecisionRequest(BaseModel):
    """决策请求模型"""
    input_data: Dict[str, Any]
    model_type: str = "default"
    confidence_threshold: float = 0.7


class DecisionResponse(BaseModel):
    """决策响应模型"""
    decision: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class ModelInfo(BaseModel):
    """模型信息模型"""
    name: str
    version: str
    status: str
    accuracy: float


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "decision-service"}


@router.get("/models")
async def get_models() -> List[ModelInfo]:
    """获取可用模型列表"""
    return [
        ModelInfo(
            name="agriculture_decision_model",
            version="1.0.0",
            status="active",
            accuracy=0.95
        ),
        ModelInfo(
            name="resource_optimization_model",
            version="1.0.0",
            status="active",
            accuracy=0.92
        ),
        ModelInfo(
            name="risk_assessment_model",
            version="1.0.0",
            status="active",
            accuracy=0.88
        )
    ]


@router.post("/predict", response_model=DecisionResponse)
async def make_decision(request: DecisionRequest) -> DecisionResponse:
    """进行决策预测"""
    try:
        # 模拟决策逻辑
        input_data = request.input_data
        
        # 根据输入数据生成决策
        if "agriculture" in request.model_type:
            decision = "种植建议: 适合种植小麦"
            confidence = 0.95
            reasoning = "基于土壤分析和气候数据，小麦种植条件最优"
        elif "resource" in request.model_type:
            decision = "资源分配: 优先分配水资源"
            confidence = 0.92
            reasoning = "水资源紧张，优先保障农业用水"
        elif "risk" in request.model_type:
            decision = "风险评估: 低风险"
            confidence = 0.88
            reasoning = "当前环境条件稳定，风险较低"
        else:
            decision = "默认决策: 继续监控"
            confidence = 0.8
            reasoning = "数据不足，建议继续收集数据"
        
        # 确保置信度不低于阈值
        if confidence < request.confidence_threshold:
            decision = "决策: 需要更多数据"
            confidence = request.confidence_threshold
            reasoning = "置信度过低，建议收集更多数据"
        
        return DecisionResponse(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "model_used": request.model_type,
                "input_size": len(str(input_data)),
                "processing_time": "0.1s"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"决策处理错误: {str(e)}")


@router.get("/status")
async def get_service_status():
    """获取服务状态"""
    return {
        "service": "decision-service",
        "status": "running",
        "version": "1.0.0",
        "models_available": 3,
        "uptime": "24h"
    }