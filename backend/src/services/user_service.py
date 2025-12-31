"""
用户服务 - 管理C端用户种植规划、数据贡献和光子积分系统
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class UserTier(Enum):
    """用户等级"""
    BASIC = "basic"      # 基础用户
    PREMIUM = "premium"   # 高级用户
    ENTERPRISE = "enterprise"  # 企业用户


class ContributionType(Enum):
    """数据贡献类型"""
    GROWTH_DATA = "growth_data"      # 生长数据
    IMAGE_UPLOAD = "image_upload"    # 图片上传
    VIDEO_UPLOAD = "video_upload"    # 视频上传
    LIVE_STREAM = "live_stream"      # 直播数据
    PRODUCT_FEEDBACK = "product_feedback"  # 产品反馈


@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    username: str
    email: str
    tier: UserTier
    photon_points: int
    total_contributions: int
    created_at: datetime
    last_active: datetime
    planting_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "tier": self.tier.value,
            "photon_points": self.photon_points,
            "total_contributions": self.total_contributions,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "planting_history": self.planting_history
        }


@dataclass
class DataContribution:
    """数据贡献记录"""
    contribution_id: str
    user_id: str
    contribution_type: ContributionType
    data_content: Dict[str, Any]
    photon_points_earned: int
    timestamp: datetime
    verified: bool
    blockchain_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "user_id": self.user_id,
            "contribution_type": self.contribution_type.value,
            "data_content": self.data_content,
            "photon_points_earned": self.photon_points_earned,
            "timestamp": self.timestamp.isoformat(),
            "verified": self.verified,
            "blockchain_hash": self.blockchain_hash
        }


@dataclass
class PlantingPlan:
    """种植计划"""
    plan_id: str
    user_id: str
    crop_type: str
    target_yield: str
    start_date: datetime
    expected_duration: int  # 预计天数
    current_day: int
    status: str  # "planning", "active", "completed", "cancelled"
    light_recipe: Dict[str, float]
    environmental_conditions: Dict[str, float]
    growth_data: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "crop_type": self.crop_type,
            "target_yield": self.target_yield,
            "start_date": self.start_date.isoformat(),
            "expected_duration": self.expected_duration,
            "current_day": self.current_day,
            "status": self.status,
            "light_recipe": self.light_recipe,
            "environmental_conditions": self.environmental_conditions,
            "growth_data": self.growth_data
        }


class UserService:
    """用户服务"""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.contributions: Dict[str, DataContribution] = {}
        self.planting_plans: Dict[str, PlantingPlan] = {}
        
        # 积分奖励规则
        self.point_rules = {
            ContributionType.GROWTH_DATA: 10,
            ContributionType.IMAGE_UPLOAD: 5,
            ContributionType.VIDEO_UPLOAD: 15,
            ContributionType.LIVE_STREAM: 25,
            ContributionType.PRODUCT_FEEDBACK: 8
        }
        
        # 用户等级特权
        self.tier_privileges = {
            UserTier.BASIC: {
                "max_plans": 3,
                "data_retention_days": 90,
                "api_limits": 1000,
                "point_multiplier": 1.0
            },
            UserTier.PREMIUM: {
                "max_plans": 10,
                "data_retention_days": 365,
                "api_limits": 10000,
                "point_multiplier": 1.5
            },
            UserTier.ENTERPRISE: {
                "max_plans": 100,
                "data_retention_days": 9999,
                "api_limits": 100000,
                "point_multiplier": 2.0
            }
        }
    
    def create_user(self, username: str, email: str, tier: UserTier = UserTier.BASIC) -> UserProfile:
        """创建新用户"""
        user_id = str(uuid.uuid4())
        now = datetime.now()
        
        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            tier=tier,
            photon_points=100,  # 新用户赠送100积分
            total_contributions=0,
            created_at=now,
            last_active=now,
            planting_history=[]
        )
        
        self.users[user_id] = user_profile
        logger.info(f"创建新用户: {username} ({user_id})")
        return user_profile
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """获取用户信息"""
        return self.users.get(user_id)
    
    def update_user_activity(self, user_id: str) -> bool:
        """更新用户活跃时间"""
        if user_id in self.users:
            self.users[user_id].last_active = datetime.now()
            return True
        return False
    
    def create_planting_plan(self, 
                           user_id: str, 
                           crop_type: str, 
                           target_yield: str,
                           light_recipe: Dict[str, float],
                           environmental_conditions: Dict[str, float]) -> Optional[PlantingPlan]:
        """创建种植计划"""
        
        user = self.get_user(user_id)
        if not user:
            logger.error(f"用户不存在: {user_id}")
            return None
        
        # 检查用户计划数量限制
        user_plans = [p for p in self.planting_plans.values() if p.user_id == user_id and p.status == "active"]
        max_plans = self.tier_privileges[user.tier]["max_plans"]
        
        if len(user_plans) >= max_plans:
            logger.error(f"用户已达到最大活跃计划数: {max_plans}")
            return None
        
        plan_id = str(uuid.uuid4())
        now = datetime.now()
        
        planting_plan = PlantingPlan(
            plan_id=plan_id,
            user_id=user_id,
            crop_type=crop_type,
            target_yield=target_yield,
            start_date=now,
            expected_duration=90,  # 默认90天
            current_day=1,
            status="active",
            light_recipe=light_recipe,
            environmental_conditions=environmental_conditions,
            growth_data=[]
        )
        
        self.planting_plans[plan_id] = planting_plan
        
        # 添加到用户历史
        user.planting_history.append({
            "plan_id": plan_id,
            "crop_type": crop_type,
            "start_date": now.isoformat(),
            "status": "active"
        })
        
        logger.info(f"创建种植计划: {crop_type} - {target_yield} (用户: {user_id})")
        return planting_plan
    
    def record_data_contribution(self, 
                               user_id: str, 
                               contribution_type: ContributionType,
                               data_content: Dict[str, Any]) -> Optional[DataContribution]:
        """记录数据贡献"""
        
        user = self.get_user(user_id)
        if not user:
            logger.error(f"用户不存在: {user_id}")
            return None
        
        # 计算光子积分
        base_points = self.point_rules.get(contribution_type, 5)
        tier_multiplier = self.tier_privileges[user.tier]["point_multiplier"]
        points_earned = int(base_points * tier_multiplier)
        
        # 数据质量奖励
        quality_bonus = self._calculate_quality_bonus(data_content, contribution_type)
        points_earned += quality_bonus
        
        contribution_id = str(uuid.uuid4())
        now = datetime.now()
        
        contribution = DataContribution(
            contribution_id=contribution_id,
            user_id=user_id,
            contribution_type=contribution_type,
            data_content=data_content,
            photon_points_earned=points_earned,
            timestamp=now,
            verified=False
        )
        
        self.contributions[contribution_id] = contribution
        
        # 更新用户积分和贡献计数
        user.photon_points += points_earned
        user.total_contributions += 1
        user.last_active = now
        
        logger.info(f"记录数据贡献: {contribution_type.value} - {points_earned}积分 (用户: {user_id})")
        return contribution
    
    def update_planting_progress(self, plan_id: str, growth_data: Dict[str, Any]) -> bool:
        """更新种植进度"""
        
        plan = self.planting_plans.get(plan_id)
        if not plan:
            logger.error(f"种植计划不存在: {plan_id}")
            return False
        
        # 添加生长数据
        growth_data["timestamp"] = datetime.now().isoformat()
        plan.growth_data.append(growth_data)
        
        # 更新当前天数
        if "current_day" in growth_data:
            plan.current_day = growth_data["current_day"]
        
        # 检查是否完成
        if plan.current_day >= plan.expected_duration:
            plan.status = "completed"
            logger.info(f"种植计划完成: {plan_id}")
        
        return True
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        
        user = self.get_user(user_id)
        if not user:
            return {}
        
        user_plans = [p for p in self.planting_plans.values() if p.user_id == user_id]
        user_contributions = [c for c in self.contributions.values() if c.user_id == user_id]
        
        # 计算统计信息
        active_plans = len([p for p in user_plans if p.status == "active"])
        completed_plans = len([p for p in user_plans if p.status == "completed"])
        total_points_earned = sum(c.photon_points_earned for c in user_contributions)
        
        # 贡献类型统计
        contribution_stats = {}
        for contrib_type in ContributionType:
            count = len([c for c in user_contributions if c.contribution_type == contrib_type])
            contribution_stats[contrib_type.value] = count
        
        return {
            "user_info": user.to_dict(),
            "planning_stats": {
                "total_plans": len(user_plans),
                "active_plans": active_plans,
                "completed_plans": completed_plans
            },
            "contribution_stats": {
                "total_contributions": user.total_contributions,
                "total_points_earned": total_points_earned,
                "by_type": contribution_stats
            },
            "tier_privileges": self.tier_privileges[user.tier]
        }
    
    def redeem_photon_points(self, user_id: str, points: int, redemption_type: str) -> bool:
        """兑换光子积分"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        if user.photon_points < points:
            logger.error(f"积分不足: 需要{points}, 当前{user.photon_points}")
            return False
        
        # 兑换规则
        redemption_values = {
            "hardware_discount": 100,  # 每100积分兑换1%硬件折扣
            "premium_feature": 500,    # 500积分兑换1个月高级功能
            "cash_reward": 1000        # 1000积分兑换10元现金
        }
        
        if redemption_type not in redemption_values:
            logger.error(f"无效的兑换类型: {redemption_type}")
            return False
        
        # 扣除积分
        user.photon_points -= points
        
        logger.info(f"用户{user_id}兑换{points}积分: {redemption_type}")
        return True
    
    def upgrade_user_tier(self, user_id: str, new_tier: UserTier) -> bool:
        """升级用户等级"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        # 检查升级条件
        if new_tier == UserTier.PREMIUM and user.photon_points < 1000:
            logger.error("升级到高级用户需要至少1000积分")
            return False
        
        if new_tier == UserTier.ENTERPRISE and user.photon_points < 5000:
            logger.error("升级到企业用户需要至少5000积分")
            return False
        
        user.tier = new_tier
        logger.info(f"用户{user_id}升级到{new_tier.value}等级")
        return True
    
    def _calculate_quality_bonus(self, data_content: Dict[str, Any], contrib_type: ContributionType) -> int:
        """计算数据质量奖励"""
        bonus = 0
        
        if contrib_type == ContributionType.GROWTH_DATA:
            # 生长数据质量评估
            if len(data_content.get("growth_metrics", {})) > 5:
                bonus += 3
            if data_content.get("environmental_data"):
                bonus += 2
            if data_content.get("image_references"):
                bonus += 5
        
        elif contrib_type == ContributionType.IMAGE_UPLOAD:
            # 图片质量评估
            if data_content.get("image_quality") == "high":
                bonus += 3
            if data_content.get("annotations"):
                bonus += 2
        
        elif contrib_type == ContributionType.LIVE_STREAM:
            # 直播数据质量评估
            duration = data_content.get("duration_minutes", 0)
            if duration > 60:
                bonus += 10
            elif duration > 30:
                bonus += 5
            if data_content.get("viewer_count", 0) > 100:
                bonus += 5
        
        return bonus


# 全局用户服务实例
user_service = UserService()

# 初始化一些示例用户
if __name__ == "__main__":
    # 创建示例用户
    example_user = user_service.create_user("种植达人", "farmer@example.com", UserTier.PREMIUM)
    
    # 创建种植计划
    light_recipe = {
        "uv_380nm": 0.05,
        "far_red_720nm": 0.1,
        "white_light": 0.7,
        "red_660nm": 0.15
    }
    
    environment = {
        "temperature": 25,
        "humidity": 60,
        "co2": 400
    }
    
    planting_plan = user_service.create_planting_plan(
        example_user.user_id, 
        "番茄", 
        "最大化产量",
        light_recipe,
        environment
    )
    
    # 记录数据贡献
    growth_data = {
        "current_day": 10,
        "growth_metrics": {
            "height": 25.5,
            "leaf_count": 8,
            "health_score": 0.85
        },
        "environmental_data": environment
    }
    
    contribution = user_service.record_data_contribution(
        example_user.user_id,
        ContributionType.GROWTH_DATA,
        growth_data
    )
    
    print(f"示例用户创建成功: {example_user.username}")
    print(f"种植计划ID: {planting_plan.plan_id if planting_plan else 'None'}")
    print(f"获得积分: {contribution.photon_points_earned if contribution else 0}")