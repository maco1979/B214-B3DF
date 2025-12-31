"""
完全独立的简化版API应用
不依赖任何AI框架，只提供auth和community接口
"""

from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime, timedelta

# --------------------------
# 认证相关代码
# --------------------------
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

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

@auth_router.post("/qr/generate", response_model=QRLoginResponse, status_code=status.HTTP_200_OK)
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

@auth_router.get("/qr/status/{qr_id}", response_model=QRLoginStatusResponse, status_code=status.HTTP_200_OK)
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

@auth_router.post("/qr/callback", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
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

@auth_router.post("/logout", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
def logout():
    """
    用户登出
    """
    # 实际应用中，这里应该失效用户的访问令牌
    return {"message": "登出成功"}

@auth_router.post("/login", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
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
    user_info = {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email"),
        "avatar": user["avatar"],
        "source": user["source"],
        "role": user.get("role", "user")
    }
    
    return {
        "message": "登录成功",
        "user_info": user_info,
        "access_token": access_token,
        "token_type": "bearer"
    }

@auth_router.get("/me", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_current_user():
    """
    获取当前用户信息
    """
    # 实际应用中，这里应该从请求头的Authorization获取令牌并验证
    # 为了演示，我们返回一个默认用户
    return {
        "id": "test_user",
        "username": "测试用户",
        "email": "test@example.com",
        "avatar": "https://example.com/test_avatar.jpg",
        "source": "local",
        "role": "admin"
    }

# --------------------------
# 社区相关代码
# --------------------------
community_router = APIRouter(prefix="/api/community", tags=["community"])

# 模拟直播流数据
live_streams_db: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "小麦种植技术分享",
        "streamer": "农业专家李明",
        "category": "种植技术",
        "viewers": 1234,
        "status": "live",
        "cover_image": "https://via.placeholder.com/640x360/1f2937/ffffff?text=Live+Stream+1",
        "tags": ["小麦", "种植", "技术"],
        "start_time": datetime.now().isoformat(),
        "streamer_avatar": "https://example.com/avatar_li.jpg",
        "description": "分享小麦种植的最新技术和管理方法"
    },
    {
        "id": 2,
        "title": "果园病虫害防治",
        "streamer": "植保专家王芳",
        "category": "病虫害防治",
        "viewers": 567,
        "status": "upcoming",
        "cover_image": "https://via.placeholder.com/640x360/1f2937/ffffff?text=Live+Stream+2",
        "tags": ["果树", "病虫害", "防治"],
        "start_time": (datetime.now() + timedelta(hours=2)).isoformat(),
        "streamer_avatar": "https://example.com/avatar_wang.jpg",
        "description": "讲解果园常见病虫害的识别和防治技术"
    },
    {
        "id": 3,
        "title": "农产品电商运营",
        "streamer": "电商导师赵强",
        "category": "农产品销售",
        "viewers": 890,
        "status": "live",
        "cover_image": "https://via.placeholder.com/640x360/1f2937/ffffff?text=Live+Stream+3",
        "tags": ["电商", "销售", "农产品"],
        "start_time": datetime.now().isoformat(),
        "streamer_avatar": "https://example.com/avatar_zhao.jpg",
        "description": "分享农产品电商平台的运营策略和技巧"
    }
]

# 模拟社区帖子数据
community_posts_db: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "分享一个提高产量的小技巧",
        "content": "我在种植过程中发现，合理使用有机肥可以显著提高作物产量，同时减少病虫害的发生。",
        "user_id": "user_123",
        "username": "农民张",
        "avatar": "https://example.com/avatar_zhang.jpg",
        "likes": 45,
        "comments": [
            {
                "id": 1,
                "user_id": "user_456",
                "username": "农业爱好者",
                "content": "谢谢分享，我也试试！",
                "time": "2小时前",
                "likes": 5
            },
            {
                "id": 2,
                "user_id": "user_789",
                "username": "农技推广员",
                "content": "这个方法确实有效，我们也在推广。",
                "time": "1小时前",
                "likes": 3
            }
        ],
        "time": "3小时前",
        "tags": ["种植技巧", "有机肥"],
        "category": "种植经验"
    },
    {
        "id": 2,
        "title": "求助：果树叶子发黄怎么办？",
        "content": "我的苹果树最近叶子开始发黄，不知道是什么原因，有谁能帮忙分析一下吗？",
        "user_id": "user_456",
        "username": "果园主小李",
        "avatar": "https://example.com/avatar_li_orchard.jpg",
        "likes": 23,
        "comments": [
            {
                "id": 3,
                "user_id": "user_101",
                "username": "植保专家",
                "content": "可能是缺铁性黄叶病，建议补充铁元素肥料。",
                "time": "45分钟前",
                "likes": 12
            }
        ],
        "time": "1小时前",
        "tags": ["果树", "病虫害", "求助"],
        "category": "病虫害防治"
    },
    {
        "id": 3,
        "title": "新型农业机械使用体验",
        "content": "最近购买了一台新型播种机，效率提高了50%，而且省种子，非常好用！",
        "user_id": "user_789",
        "username": "农机达人",
        "avatar": "https://example.com/avatar_machine.jpg",
        "likes": 67,
        "comments": [],
        "time": "5小时前",
        "tags": ["农业机械", "播种机", "使用体验"],
        "category": "农业机械"
    }
]

class CommentCreateRequest(BaseModel):
    """创建评论请求模型"""
    content: str

class LikePostRequest(BaseModel):
    """点赞帖子请求模型"""
    post_id: int

class LikeCommentRequest(BaseModel):
    """点赞评论请求模型"""
    post_id: int
    comment_id: int

@community_router.get("/live-streams", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def get_live_streams():
    """
    获取直播流列表
    """
    return live_streams_db

@community_router.get("/live-streams/{stream_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_live_stream(stream_id: int):
    """
    获取单个直播流详情
    """
    for stream in live_streams_db:
        if stream["id"] == stream_id:
            return stream
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="直播流不存在"
    )

@community_router.get("/posts", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def get_community_posts(category: Optional[str] = None, search: Optional[str] = None):
    """
    获取社区帖子列表
    """
    posts = community_posts_db
    
    # 按分类过滤
    if category:
        posts = [post for post in posts if post["category"] == category]
    
    # 按关键词搜索
    if search:
        search_lower = search.lower()
        posts = [
            post for post in posts 
            if search_lower in post["title"].lower() 
            or search_lower in post["content"].lower()
            or any(tag.lower().find(search_lower) != -1 for tag in post["tags"])
        ]
    
    return posts

@community_router.get("/posts/{post_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_community_post(post_id: int):
    """
    获取单个社区帖子详情
    """
    for post in community_posts_db:
        if post["id"] == post_id:
            return post
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="帖子不存在"
    )

@community_router.post("/posts/{post_id}/comments", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_comment(post_id: int, request: CommentCreateRequest):
    """
    创建评论
    """
    # 查找帖子
    post = None
    for p in community_posts_db:
        if p["id"] == post_id:
            post = p
            break
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 创建新评论
    new_comment = {
        "id": len(post["comments"]) + 1,
        "user_id": "current_user",  # 实际应用中应该从请求中获取当前用户ID
        "username": "当前用户",  # 实际应用中应该从请求中获取当前用户名
        "content": request.content,
        "time": "刚刚",
        "likes": 0
    }
    
    # 添加到帖子的评论列表
    post["comments"].append(new_comment)
    
    return new_comment

@community_router.post("/posts/{post_id}/like", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def like_post(post_id: int):
    """
    点赞帖子
    """
    # 查找帖子
    post = None
    for p in community_posts_db:
        if p["id"] == post_id:
            post = p
            break
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 增加点赞数
    post["likes"] += 1
    
    return {"likes": post["likes"]}

@community_router.post("/posts/{post_id}/comments/{comment_id}/like", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def like_comment(post_id: int, comment_id: int):
    """
    点赞评论
    """
    # 查找帖子
    post = None
    for p in community_posts_db:
        if p["id"] == post_id:
            post = p
            break
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 查找评论
    comment = None
    for c in post["comments"]:
        if c["id"] == comment_id:
            comment = c
            break
    
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="评论不存在"
        )
    
    # 增加点赞数
    comment["likes"] += 1
    
    return {"likes": comment["likes"]}

@community_router.get("/categories", response_model=List[str], status_code=status.HTTP_200_OK)
def get_categories():
    """
    获取所有帖子分类
    """
    categories = set()
    for post in community_posts_db:
        categories.add(post["category"])
    return list(categories)

# --------------------------
# 创建应用
# --------------------------
def create_app() -> FastAPI:
    """创建FastAPI应用"""
    
    app = FastAPI(
        title="AI项目API服务",
        description="简化版API服务，提供认证和社区功能",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应限制来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(auth_router)
    app.include_router(community_router)
    
    # 根路径
    @app.get("/")
    async def root():
        return {
            "message": "AI项目API服务",
            "version": "1.0.0",
            "docs": "/docs",
            "mode": "simplified"  # 标记为简化模式
        }
    
    # 健康检查接口
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    
    return app

# 创建应用实例
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
