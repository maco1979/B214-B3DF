"""
应用错误定义模块
定义具体的应用错误类
"""

from typing import Any, Dict, Optional, Union
from .error_codes import ErrorCode, ErrorCategory


class AppError(Exception):
    """应用基础错误类"""
    
    def __init__(
        self,
        error_code: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.error_code = error_code
        self.message = message or self._get_default_message(error_code)
        self.details = details or {}
        self.cause = cause
        
        # 获取错误信息
        error_info = ErrorCode.get_error_info(error_code)
        self.error_category = error_info["category"]
        
        super().__init__(self.message)
    
    def _get_default_message(self, error_code: int) -> str:
        """获取默认错误消息"""
        error_info = ErrorCode.get_error_info(error_code)
        return error_info["message"]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.error_category,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        base_str = f"[{self.error_code}] {self.message}"
        if self.details:
            base_str += f" | Details: {self.details}"
        if self.cause:
            base_str += f" | Cause: {self.cause}"
        return base_str


class ValidationError(AppError):
    """数据验证错误"""
    
    def __init__(
        self,
        message: Optional[str] = None,
        field_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.INVALID_REQUEST[0],
            message=message or "数据验证失败",
            details={
                "field_errors": field_errors or {},
                **(details or {})
            }
        )


class AuthorizationError(AppError):
    """授权错误"""
    
    def __init__(
        self,
        message: Optional[str] = None,
        required_permissions: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.FORBIDDEN[0],
            message=message or "权限不足",
            details={
                "required_permissions": required_permissions or [],
                **(details or {})
            }
        )


class NotFoundError(AppError):
    """资源不存在错误"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Union[str, int],
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.RESOURCE_NOT_FOUND[0],
            message=message or f"{resource_type} {resource_id} 不存在",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                **(details or {})
            }
        )


class DatabaseError(AppError):
    """数据库错误"""
    
    def __init__(
        self,
        message: Optional[str] = None,
        operation: Optional[str] = None,
        sql: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.DATABASE_ERROR[0],
            message=message or "数据库操作失败",
            details={
                "operation": operation,
                "sql": sql,
                **(details or {})
            }
        )


class ExternalServiceError(AppError):
    """外部服务错误"""
    
    def __init__(
        self,
        service_name: str,
        endpoint: str,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR[0],
            message=message or f"外部服务 {service_name} 调用失败",
            details={
                "service_name": service_name,
                "endpoint": endpoint,
                "status_code": status_code,
                **(details or {})
            }
        )


class ConfigurationError(AppError):
    """配置错误"""
    
    def __init__(
        self,
        config_key: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.INVALID_REQUEST[0],
            message=message or f"配置项 {config_key} 错误",
            details={
                "config_key": config_key,
                **(details or {})
            }
        )


class BusinessError(AppError):
    """业务逻辑错误"""
    
    def __init__(
        self,
        rule_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            error_code=ErrorCode.BUSINESS_RULE_VIOLATION[0],
            message=message or f"违反业务规则: {rule_name}",
            details={
                "rule_name": rule_name,
                **(details or {})
            }
        )


# 错误工厂函数
def create_error(
    error_code: int,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None
) -> AppError:
    """创建错误实例"""
    return AppError(error_code, message, details, cause)


def create_validation_error(
    field_errors: Dict[str, Any],
    message: Optional[str] = None
) -> ValidationError:
    """创建验证错误实例"""
    return ValidationError(message, field_errors)


def create_not_found_error(
    resource_type: str,
    resource_id: Union[str, int],
    message: Optional[str] = None
) -> NotFoundError:
    """创建资源不存在错误实例"""
    return NotFoundError(resource_type, resource_id, message)