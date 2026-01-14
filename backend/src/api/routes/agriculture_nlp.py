"""
农业NLP集成服务API路由
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from ...core.services.agriculture_nlp_integration import AgricultureNLPIntegrationService

# 创建路由实例
router = APIRouter(prefix="/agriculture-nlp", tags=["agriculture-nlp"])

# 创建农业NLP集成服务实例（单例模式）
nlp_integration_service = AgricultureNLPIntegrationService()


class ChatInput(BaseModel):
    """聊天输入模型"""
    message: str = Field(..., description="用户输入的自然语言文本")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="系统生成的响应文本")
    intent: str = Field(..., description="识别到的意图")
    confidence: float = Field(..., description="意图置信度")
    action_taken: Optional[str] = Field(None, description="执行的动作")
    decision_result: Optional[Dict[str, Any]] = Field(None, description="决策结果")
    conversation_state: Optional[Dict[str, Any]] = Field(None, description="对话状态")


class IntentStatsResponse(BaseModel):
    """意图统计响应模型"""
    available_intents: list = Field(..., description="可用的意图列表")
    supported_crops: list = Field(..., description="支持的作物列表")


@router.post("/chat", response_model=ChatResponse, summary="农业NLP聊天接口")
async def chat_with_agriculture_nlp(chat_input: ChatInput):
    """
    与农业NLP集成服务进行聊天交互
    
    Args:
        chat_input: 聊天输入，包含用户消息和上下文信息
        
    Returns:
        聊天响应，包含系统生成的响应文本、意图、置信度等
    """
    try:
        result = nlp_integration_service.process_chat_input(
            chat_input.message, 
            chat_input.context
        )
        
        return ChatResponse(
            response=result.response,
            intent=result.intent,
            confidence=result.confidence,
            action_taken=result.action_taken,
            decision_result=result.decision_result,
            conversation_state=result.conversation_state
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")


@router.get("/intents", response_model=IntentStatsResponse, summary="获取可用意图统计")
async def get_available_intents():
    """
    获取农业NLP服务支持的意图和作物统计信息
    
    Returns:
        意图统计响应，包含可用的意图列表和支持的作物列表
    """
    try:
        stats = nlp_integration_service.get_intent_stats()
        
        return IntentStatsResponse(
            available_intents=stats["available_intents"],
            supported_crops=stats["supported_crops"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取意图统计失败: {str(e)}")


@router.post("/clear-context", summary="清除上下文")
async def clear_chat_context():
    """
    清除当前对话的上下文信息
    
    Returns:
        清除结果
    """
    try:
        nlp_integration_service.clear_context()
        return {"message": "上下文已成功清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除上下文失败: {str(e)}")
