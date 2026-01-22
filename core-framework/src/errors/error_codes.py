"""
错误码定义模块
定义统一的错误码和错误类别
"""

from enum import Enum
from typing import Dict, Any


class ErrorCategory(Enum):
    """错误类别"""
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"
    TIMEOUT = "TIMEOUT"
    RESOURCE = "RESOURCE"


class ErrorCode:
    """错误码定义"""
    
    # 通用错误 (1000-1999)
    UNKNOWN_ERROR = (1000, "未知错误", ErrorCategory.INTERNAL)
    INVALID_REQUEST = (1001, "无效请求", ErrorCategory.VALIDATION)
    MISSING_REQUIRED_FIELD = (1002, "缺少必填字段", ErrorCategory.VALIDATION)
    INVALID_DATA_TYPE = (1003, "无效数据类型", ErrorCategory.VALIDATION)
    
    # 认证授权错误 (2000-2999)
    UNAUTHORIZED = (2000, "未授权访问", ErrorCategory.AUTHENTICATION)
    FORBIDDEN = (2001, "禁止访问", ErrorCategory.AUTHORIZATION)
    INVALID_TOKEN = (2002, "无效令牌", ErrorCategory.AUTHENTICATION)
    TOKEN_EXPIRED = (2003, "令牌已过期", ErrorCategory.AUTHENTICATION)
    INVALID_CREDENTIALS = (2004, "无效凭据", ErrorCategory.AUTHENTICATION)
    
    # 资源错误 (3000-3999)
    RESOURCE_NOT_FOUND = (3000, "资源不存在", ErrorCategory.NOT_FOUND)
    RESOURCE_ALREADY_EXISTS = (3001, "资源已存在", ErrorCategory.CONFLICT)
    RESOURCE_CONFLICT = (3002, "资源冲突", ErrorCategory.CONFLICT)
    RESOURCE_LIMIT_EXCEEDED = (3003, "资源限制超出", ErrorCategory.RESOURCE)
    
    # 业务逻辑错误 (4000-4999)
    INVALID_OPERATION = (4000, "无效操作", ErrorCategory.VALIDATION)
    OPERATION_FAILED = (4001, "操作失败", ErrorCategory.INTERNAL)
    BUSINESS_RULE_VIOLATION = (4002, "违反业务规则", ErrorCategory.VALIDATION)
    
    # 外部服务错误 (5000-5999)
    EXTERNAL_SERVICE_ERROR = (5000, "外部服务错误", ErrorCategory.EXTERNAL)
    EXTERNAL_SERVICE_TIMEOUT = (5001, "外部服务超时", ErrorCategory.TIMEOUT)
    EXTERNAL_SERVICE_UNAVAILABLE = (5002, "外部服务不可用", ErrorCategory.EXTERNAL)
    
    # 数据库错误 (6000-6999)
    DATABASE_ERROR = (6000, "数据库错误", ErrorCategory.INTERNAL)
    DATABASE_CONNECTION_ERROR = (6001, "数据库连接错误", ErrorCategory.INTERNAL)
    DATABASE_TIMEOUT = (6002, "数据库超时", ErrorCategory.TIMEOUT)
    
    # 文件系统错误 (7000-7999)
    FILE_NOT_FOUND = (7000, "文件不存在", ErrorCategory.NOT_FOUND)
    FILE_ACCESS_ERROR = (7001, "文件访问错误", ErrorCategory.INTERNAL)
    FILE_SIZE_EXCEEDED = (7002, "文件大小超出限制", ErrorCategory.RESOURCE)
    
    # 网络错误 (8000-8999)
    NETWORK_ERROR = (8000, "网络错误", ErrorCategory.INTERNAL)
    NETWORK_TIMEOUT = (8001, "网络超时", ErrorCategory.TIMEOUT)
    
    @classmethod
    def get_error_info(cls, error_code: int) -> Dict[str, Any]:
        """根据错误码获取错误信息"""
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
                
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, tuple) and len(attr_value) == 3:
                code, message, category = attr_value
                if code == error_code:
                    return {
                        "code": code,
                        "message": message,
                        "category": category.value
                    }
        
        return {
            "code": cls.UNKNOWN_ERROR[0],
            "message": cls.UNKNOWN_ERROR[1],
            "category": cls.UNKNOWN_ERROR[2].value
        }
    
    @classmethod
    def get_all_errors(cls) -> Dict[int, Dict[str, Any]]:
        """获取所有错误码定义"""
        errors = {}
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
                
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, tuple) and len(attr_value) == 3:
                code, message, category = attr_value
                errors[code] = {
                    "name": attr_name,
                    "message": message,
                    "category": category.value
                }
        
        return errors


# 错误码映射（用于快速查找）
ERROR_CODE_MAP = {
    error_code: {"name": name, "message": message, "category": category.value}
    for name, (error_code, message, category) in vars(ErrorCode).items()
    if not name.startswith("_") and isinstance(error_code, int)
}