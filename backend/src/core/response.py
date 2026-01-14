"""
统一响应格式模块
定义所有API的统一响应格式
"""

from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel

T = TypeVar('T')


# 统一响应模型
class ApiResponse(BaseModel, Generic[T]):
    """统一API响应格式"""
    success: bool  # 操作是否成功
    code: int  # 状态码
    data: Optional[T] = None  # 响应数据
    message: str  # 响应消息
    detail: Optional[Any] = None  # 详细信息（用于错误）


# 成功响应快捷方法
def success_response(data: Any = None, message: str = "操作成功") -> ApiResponse:
    """生成成功响应
    
    Args:
        data: 响应数据
        message: 响应消息
        
    Returns:
        ApiResponse: 成功响应对象
    """
    return ApiResponse(
        success=True, 
        code=200, 
        data=data, 
        message=message
    )


# 失败响应快捷方法
def error_response(code: int, message: str, detail: Any = None) -> ApiResponse:
    """生成失败响应
    
    Args:
        code: 状态码
        message: 错误消息
        detail: 详细错误信息
        
    Returns:
        ApiResponse: 失败响应对象
    """
    return ApiResponse(
        success=False, 
        code=code, 
        message=message, 
        detail=detail
    )
