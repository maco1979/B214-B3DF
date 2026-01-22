"""
认证API路由
包含扫码登录相关的端点
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime, timedelta

# 创建路由对象
router = APIRouter(prefix="/auth", tags=["auth"])

# 模拟存储扫码登录会话
qr_login_sessions: Dict[str, Dict[str, Any]] = {}

# 模拟用户数据
users_db: Dict[str, Dict[str, Any]] = {
    "wechat_user_1": {
        "id": "wechat_user_1",
        "username": "微信用户1",
        "avatar": "https://example.com/avatar1.jpg",
        "source": "wechat",
        "created_at": datetime.now().isoformat()
    },
    "alipay_user_1": {
        "id": "alipay_user_1",
        "username": "支付宝用户1",
        "avatar": "https://example.com/avatar2.jpg",
        "source": "alipay",
        "created_at": datetime.now().isoformat()
    },
    "test_user": {
        "id": "test_user",
        "username": "测试用户",
        "email": "test@example.com",
        "password": "test123456",  # 测试密码
        "avatar": "https://example.com/test_avatar.jpg",
        "source": "local",
        "role": "admin",
        "created_at": datetime.now().isoformat()
    }
}

class QRLoginResponse(BaseModel):
    """扫码登录响应模型"""
    qr_id: str
    qr_code_url: str
    expires_in: int
    created_at: str

class QRLoginStatusResponse(BaseModel):
    """扫码登录状态响应模型"""
    qr_id: str
    status: str  # "pending", "scanned", "confirmed", "expired"
    user_info: Optional[Dict[str, Any]] = None
    access_token: Optional[str] = None
    token_type: Optional[str] = "bearer"

class QRCallbackRequest(BaseModel):
    """扫码回调请求模型"""
    qr_id: str
    user_id: str
    source: str  # "wechat", "alipay"
    action: str  # "scan", "confirm", "cancel"

class PasswordLoginRequest(BaseModel):
    """密码登录请求模型"""
    email: str
    password: str

class CodeRegistrationRequest(BaseModel):
    """产品注册码注册请求模型"""
    code: str
    email: str
    password: str

# 模拟产品注册码数据库
product_codes = {
    "AGRI-PRO-2024-0001": {"status": "unused", "type": "pro"},
    "AGRI-PRO-2024-0002": {"status": "unused", "type": "pro"},
    "AGRI-BASIC-2024-0001": {"status": "unused", "type": "basic"},
    "AGRI-BASIC-2024-0002": {"status": "used", "type": "basic"}
}

@router.post("/qr/generate", response_model=QRLoginResponse, status_code=status.HTTP_200_OK)
def generate_qr_code():
    """
    生成扫码登录的二维码
    返回二维码ID和二维码URL（实际应用中应返回真实的二维码图片数据）
    """
    qr_id = str(uuid.uuid4())
    expires_in = 300  # 5分钟过期
    created_at = datetime.now()
    expires_at = created_at + timedelta(seconds=expires_in)
    
    # 保存会话信息
    qr_login_sessions[qr_id] = {
        "status": "pending",  # pending: 等待扫码, scanned: 已扫码, confirmed: 已确认, expired: 已过期
        "created_at": created_at,
        "expires_at": expires_at,
        "user_info": None,
        "last_updated": created_at
    }
    
    # 实际应用中，这里应该生成真实的二维码图片
    # 为了演示，我们返回一个模拟的二维码URL
    qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://ai-agriculture-platform.com/auth/qr/callback/{qr_id}"
    
    return QRLoginResponse(
        qr_id=qr_id,
        qr_code_url=qr_code_url,
        expires_in=expires_in,
        created_at=created_at.isoformat()
    )

@router.get("/qr/status/{qr_id}", response_model=QRLoginStatusResponse, status_code=status.HTTP_200_OK)
def check_qr_status(qr_id: str):
    """
    检查扫码登录状态
    前端轮询此接口以获取登录状态
    """
    # 检查会话是否存在
    if qr_id not in qr_login_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="二维码不存在"
        )
    
    session = qr_login_sessions[qr_id]
    
    # 检查是否过期
    if datetime.now() > session["expires_at"]:
        session["status"] = "expired"
        
    return QRLoginStatusResponse(
        qr_id=qr_id,
        status=session["status"],
        user_info=session["user_info"],
        access_token=session.get("access_token")
    )

@router.post("/qr/callback", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def qr_callback(request: QRCallbackRequest):
    """
    扫码回调接口
    由第三方平台（微信、支付宝）调用，通知扫码状态
    """
    qr_id = request.qr_id
    user_id = request.user_id
    source = request.source
    action = request.action
    
    # 检查会话是否存在
    if qr_id not in qr_login_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="二维码不存在"
        )
    
    session = qr_login_sessions[qr_id]
    
    # 检查是否过期
    if datetime.now() > session["expires_at"]:
        session["status"] = "expired"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="二维码已过期"
        )
    
    # 处理不同的操作
    if action == "scan":
        # 用户已扫码，等待确认
        session["status"] = "scanned"
        session["last_updated"] = datetime.now()
        
    elif action == "confirm":
        # 用户已确认登录
        session["status"] = "confirmed"
        session["last_updated"] = datetime.now()
        
        # 模拟获取用户信息
        # 实际应用中，这里应该从第三方平台获取真实的用户信息
        user_info = users_db.get(user_id)
        if not user_info:
            # 如果是新用户，创建新用户记录
            user_info = {
                "id": user_id,
                "username": f"{source}_{user_id[:8]}",
                "avatar": f"https://example.com/{source}_avatar.jpg",
                "source": source,
                "created_at": datetime.now().isoformat()
            }
            users_db[user_id] = user_info
        
        session["user_info"] = user_info
        
        # 生成访问令牌
        # 实际应用中，这里应该生成JWT令牌
        access_token = str(uuid.uuid4())
        session["access_token"] = access_token
        
    elif action == "cancel":
        # 用户取消登录
        session["status"] = "cancelled"
        session["last_updated"] = datetime.now()
    
    return {
        "message": "操作成功",
        "qr_id": qr_id,
        "status": session["status"]
    }

@router.post("/logout", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
def logout():
    """
    用户登出
    """
    # 实际应用中，这里应该失效用户的访问令牌
    return {"message": "登出成功"}

@router.post("/login", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def password_login(request: PasswordLoginRequest):
    """
    密码登录接口
    """
    # 查找用户
    user = None
    for user_id, user_info in users_db.items():
        if user_info.get("email") == request.email:
            user = user_info
            break
    
    # 验证用户和密码
    if not user or user.get("password") != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误"
        )
    
    # 生成访问令牌
    access_token = str(uuid.uuid4())
    
    # 构建返回的用户信息，不包含密码
    user_response = {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email"),
        "avatar": user.get("avatar"),
        "source": user["source"],
        "role": user.get("role", "user"),
        "created_at": user["created_at"]
    }
    
    return {
        "message": "登录成功",
        "user_info": user_response,
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/register/code", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def register_with_code(request: CodeRegistrationRequest):
    """
    使用产品注册码注册接口
    """
    # 验证注册码
    if request.code not in product_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的产品注册码"
        )
    
    code_info = product_codes[request.code]
    if code_info["status"] != "unused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="产品注册码已被使用"
        )
    
    # 检查邮箱是否已存在
    for user_id, user_info in users_db.items():
        if user_info.get("email") == request.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被注册"
            )
    
    # 创建新用户
    user_id = f"user_{str(uuid.uuid4())[:8]}"
    username = request.email.split("@")[0]  # 使用邮箱前缀作为用户名
    
    new_user = {
        "id": user_id,
        "username": username,
        "email": request.email,
        "password": request.password,  # 注意：实际应用中应该加密存储密码
        "avatar": f"https://example.com/avatar_{user_id[:4]}.jpg",
        "source": "local",
        "role": "user",
        "created_at": datetime.now().isoformat(),
        "product_type": code_info["type"]  # 记录产品类型
    }
    
    # 添加到用户数据库
    users_db[user_id] = new_user
    
    # 标记注册码为已使用
    product_codes[request.code]["status"] = "used"
    product_codes[request.code]["user_id"] = user_id
    
    # 构建返回的用户信息，不包含密码
    user_response = {
        "id": new_user["id"],
        "username": new_user["username"],
        "email": new_user["email"],
        "avatar": new_user["avatar"],
        "source": new_user["source"],
        "role": new_user["role"],
        "created_at": new_user["created_at"],
        "product_type": new_user["product_type"]
    }
    
    return {
        "message": "注册成功",
        "user_info": user_response
    }

@router.get("/me", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_current_user():
    """
    获取当前登录用户信息
    """
    # 实际应用中，这里应该从请求头中获取访问令牌并验证，然后返回用户信息
    return {
        "id": "wechat_user_1",
        "username": "微信用户1",
        "avatar": "https://example.com/avatar1.jpg",
        "source": "wechat",
        "created_at": "2024-01-01T00:00:00Z"
    }
