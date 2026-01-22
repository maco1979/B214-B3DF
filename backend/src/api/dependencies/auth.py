#!/usr/bin/env python3
"""
认证依赖模块
提供JWT认证相关的FastAPI依赖
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, Dict, Any

from src.core.utils.jwt_utils import verify_token

# OAuth2密码Bearer令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    获取当前用户依赖
    验证JWT令牌并返回用户信息
    
    Args:
        token: JWT令牌
        
    Returns:
        当前用户信息
        
    Raises:
        HTTPException: 如果令牌无效或过期
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    # 从数据库获取用户信息
    # 注意：这里应该从实际数据库中获取用户，而不是使用模拟数据
    from src.api.routes.auth import users_db
    user = users_db.get(user_id)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    获取当前活跃用户依赖
    确保当前用户是活跃的
    
    Args:
        current_user: 当前用户信息
        
    Returns:
        当前活跃用户信息
        
    Raises:
        HTTPException: 如果用户不活跃
    """
    # 这里可以添加用户活跃状态检查
    # if not current_user.get("is_active", True):
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="用户已被禁用"
    #     )
    
    return current_user


async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    获取当前管理员用户依赖
    确保当前用户是管理员
    
    Args:
        current_user: 当前用户信息
        
    Returns:
        当前管理员用户信息
        
    Raises:
        HTTPException: 如果用户不是管理员
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    
    return current_user
