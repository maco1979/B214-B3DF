"""
用户API路由
处理用户相关功能：积分兑换、数据贡献等
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from pydantic import BaseModel

# 创建路由对象
router = APIRouter(prefix="/user", tags=["user"])

# 请求模型
class RedeemRequest(BaseModel):
    """兑换请求模型"""
    points: int
    redemption_type: str  # "hardware_discount", "premium_feature", "cash_reward"

class DataContributionRequest(BaseModel):
    """数据贡献请求模型"""
    contribution_type: str  # 贡献类型
    data_content: Dict[str, Any]  # 贡献的数据内容

class UserStatsResponse(BaseModel):
    """用户统计响应模型"""
    success: bool
    data: Dict[str, Any]

class RedeemResponse(BaseModel):
    """兑换响应模型"""
    success: bool
    message: str
    remaining_points: int = None

class DataContributionResponse(BaseModel):
    """数据贡献响应模型"""
    success: bool
    message: str
    photon_points_earned: int = None

# 简化版兑换功能
@router.post("/redeem", summary="兑换光子积分")
async def redeem_points(request: RedeemRequest, user_id: str) -> RedeemResponse:
    """兑换光子积分"""
    # 这里只是模拟兑换逻辑，实际实现需要连接到用户服务
    if request.points <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="兑换积分必须大于0"
        )
    
    valid_types = ["hardware_discount", "premium_feature", "cash_reward"]
    if request.redemption_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的兑换类型: {request.redemption_type}, 有效类型: {valid_types}"
        )
    
    # 模拟兑换成功
    return RedeemResponse(
        success=True,
        message=f"成功兑换 {request.points} 积分",
        remaining_points=1000  # 模拟剩余积分
    )

@router.post("/contribute-data", summary="贡献数据")
async def contribute_data(request: DataContributionRequest, user_id: str) -> DataContributionResponse:
    """贡献数据获得光子积分"""
    valid_types = ["growth_data", "image_upload", "video_upload", "live_stream", "product_feedback"]
    if request.contribution_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的贡献类型: {request.contribution_type}, 有效类型: {valid_types}"
        )
    
    # 模拟数据贡献成功，返回获得的积分
    points_earned = 10  # 模拟获得的积分
    
    return DataContributionResponse(
        success=True,
        message="数据贡献成功",
        photon_points_earned=points_earned
    )

@router.get("/stats", summary="获取用户统计信息")
async def get_user_stats(user_id: str) -> UserStatsResponse:
    """获取用户统计信息"""
    # 模拟用户统计信息
    stats = {
        "user_id": user_id,
        "photon_points": 1500,
        "total_contributions": 25,
        "tier": "premium",
        "contribution_stats": {
            "total_points_earned": 500,
            "by_type": {
                "growth_data": 10,
                "image_upload": 5,
                "video_upload": 3,
                "live_stream": 2,
                "product_feedback": 5
            }
        }
    }
    
    return UserStatsResponse(success=True, data=stats)

@router.get("/{user_id}", summary="获取用户信息")
async def get_user_info(user_id: str) -> Dict[str, Any]:
    """获取用户信息"""
    user_info = {
        "user_id": user_id,
        "username": f"user_{user_id[:8]}",
        "photon_points": 1500,
        "tier": "premium",
        "total_contributions": 25,
        "created_at": "2024-01-01T00:00:00Z",
        "last_active": "2024-12-30T00:00:00Z"
    }
    
    return {"success": True, "data": user_info}