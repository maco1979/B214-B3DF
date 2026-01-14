"""
语音处理API路由
提供语音识别、自然语言处理和语音合成相关的API
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.core.response import ApiResponse, success_response, error_response
from src.core.services.nlp_service import NLPService

router = APIRouter(prefix="/voice", tags=["voice"])

# 创建NLP服务实例
nlp_service = NLPService()

# 定义请求模型
class VoiceProcessRequest(BaseModel):
    """语音处理请求模型"""
    text: str
    context: Dict[str, Any] = {}
    entities: Optional[List[Dict[str, Any]]] = None
    intent: Optional[str] = None

class VoiceConfigRequest(BaseModel):
    """语音配置请求模型"""
    language: str = "zh-CN"
    asr_engine: str = "default"
    tts_engine: str = "default"

# 定义响应模型
class VoiceProcessResponse(BaseModel):
    """语音处理响应模型"""
    intent: str
    entities: List[Dict[str, Any]]
    response: str
    action: str

class VoiceConfigResponse(BaseModel):
    """语音配置响应模型"""
    language: str
    asr_engine: str
    tts_engine: str
    available_asr_engines: List[str]
    available_tts_engines: List[str]


@router.post("/process", response_model=ApiResponse)
async def process_voice_request(request: VoiceProcessRequest):
    """
    处理语音请求
    
    Args:
        request: 语音处理请求
        
    Returns:
        ApiResponse: 语音处理结果
    """
    try:
        # 调用NLP服务处理请求
        result = nlp_service.process_text(
            request.text,
            request.context
        )
        
        # 构建响应
        response_data = VoiceProcessResponse(
            intent=result.intent,
            entities=result.entities,
            response=f"已处理请求: {request.text}",
            action=nlp_service.map_intent_to_action(result.intent)
        )
        
        return success_response(
            data=response_data,
            message="语音请求处理成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"语音请求处理失败: {str(e)}"
        )


@router.get("/config", response_model=ApiResponse)
async def get_voice_config():
    """
    获取语音配置
    
    Returns:
        ApiResponse: 语音配置信息
    """
    try:
        # 构建配置响应
        config_data = VoiceConfigResponse(
            language="zh-CN",
            asr_engine="default",
            tts_engine="default",
            available_asr_engines=["default", "cloud", "local"],
            available_tts_engines=["default", "cloud", "local"]
        )
        
        return success_response(
            data=config_data,
            message="获取语音配置成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"获取语音配置失败: {str(e)}"
        )


@router.post("/config", response_model=ApiResponse)
async def update_voice_config(request: VoiceConfigRequest):
    """
    更新语音配置
    
    Args:
        request: 语音配置请求
        
    Returns:
        ApiResponse: 更新结果
    """
    try:
        # 这里可以添加配置更新逻辑
        # 例如保存配置到数据库或配置文件
        
        return success_response(
            data={"message": "语音配置更新成功", "config": request.model_dump()},
            message="语音配置更新成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"更新语音配置失败: {str(e)}"
        )


@router.post("/session/clear", response_model=ApiResponse)
async def clear_voice_session():
    """
    清除语音会话
    
    Returns:
        ApiResponse: 清除结果
    """
    try:
        # 清除NLP服务的上下文
        nlp_service.clear_context()
        
        return success_response(
            data={"message": "语音会话已清除"},
            message="语音会话清除成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"清除语音会话失败: {str(e)}"
        )


@router.get("/health", response_model=ApiResponse)
async def voice_health_check():
    """
    语音服务健康检查
    
    Returns:
        ApiResponse: 健康检查结果
    """
    try:
        # 检查NLP服务是否正常
        result = nlp_service.process_text("健康检查")
        
        return success_response(
            data={"status": "healthy", "nlp_status": "ok"},
            message="语音服务健康检查成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"语音服务健康检查失败: {str(e)}"
        )


@router.post("/intent/add", response_model=ApiResponse)
async def add_intent_rule(request: Dict[str, Any]):
    """
    添加自定义意图规则
    
    Args:
        request: 包含意图规则的请求
        
    Returns:
        ApiResponse: 添加结果
    """
    try:
        # 调用NLP服务添加意图规则
        nlp_service.add_intent_rule(request)
        
        return success_response(
            data={"message": "意图规则添加成功"},
            message="意图规则添加成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_400_BAD_REQUEST,
            message=f"添加意图规则失败: {str(e)}"
        )


@router.get("/intents", response_model=ApiResponse)
async def get_intent_rules():
    """
    获取所有意图规则
    
    Returns:
        ApiResponse: 意图规则列表
    """
    try:
        # 获取意图规则
        rules = nlp_service.get_intent_rules()
        
        return success_response(
            data={"rules": rules},
            message="获取意图规则成功"
        )
    except Exception as e:
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"获取意图规则失败: {str(e)}"
        )
