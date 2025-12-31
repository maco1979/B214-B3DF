"""
社区API路由
包含社区相关的端点
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime, timedelta

# 创建路由对象
router = APIRouter(prefix="/community", tags=["community"])

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

@router.get("/live-streams", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def get_live_streams():
    """
    获取直播流列表
    """
    return live_streams_db

@router.get("/live-streams/{stream_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
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

@router.get("/posts", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
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

@router.get("/posts/{post_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
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

@router.post("/posts/{post_id}/comments", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
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

@router.post("/posts/{post_id}/like", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
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

@router.post("/posts/{post_id}/comments/{comment_id}/like", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
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

@router.get("/categories", response_model=List[str], status_code=status.HTTP_200_OK)
def get_categories():
    """
    获取所有帖子分类
    """
    categories = set()
    for post in community_posts_db:
        categories.add(post["category"])
    return list(categories)
