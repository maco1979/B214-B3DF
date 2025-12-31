"""
企业服务 - 实现B端企业功能：数据服务、API集成、付费接入
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class BusinessTier(Enum):
    """企业等级"""
    STARTER = "starter"      # 入门版
    PROFESSIONAL = "professional"  # 专业版
    ENTERPRISE = "enterprise"     # 企业版


class ServiceType(Enum):
    """服务类型"""
    DATA_ANALYTICS = "data_analytics"      # 数据分析服务
    API_INTEGRATION = "api_integration"    # API集成服务
    CUSTOM_MODEL = "custom_model"          # 定制模型服务
    PREMIUM_SUPPORT = "premium_support"    # 高级技术支持


@dataclass
class BusinessProfile:
    """企业档案"""
    business_id: str
    company_name: str
    contact_email: str
    tier: BusinessTier
    subscription_status: str  # "active", "trial", "expired", "cancelled"
    subscription_end: datetime
    api_usage: Dict[str, int]  # API使用情况
    data_access_level: int  # 数据访问权限级别
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_id": self.business_id,
            "company_name": self.company_name,
            "contact_email": self.contact_email,
            "tier": self.tier.value,
            "subscription_status": self.subscription_status,
            "subscription_end": self.subscription_end.isoformat(),
            "api_usage": self.api_usage,
            "data_access_level": self.data_access_level,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ServiceSubscription:
    """服务订阅"""
    subscription_id: str
    business_id: str
    service_type: ServiceType
    plan_details: Dict[str, Any]
    monthly_cost: float
    start_date: datetime
    end_date: datetime
    usage_limits: Dict[str, int]
    current_usage: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "business_id": self.business_id,
            "service_type": self.service_type.value,
            "plan_details": self.plan_details,
            "monthly_cost": self.monthly_cost,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "usage_limits": self.usage_limits,
            "current_usage": self.current_usage
        }


@dataclass
class DataReport:
    """数据报告"""
    report_id: str
    business_id: str
    report_type: str
    data_content: Dict[str, Any]
    generation_date: datetime
    access_level: int
    download_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "business_id": self.business_id,
            "report_type": self.report_type,
            "data_content": self.data_content,
            "generation_date": self.generation_date.isoformat(),
            "access_level": self.access_level,
            "download_count": self.download_count
        }


class BusinessService:
    """企业服务"""
    
    def __init__(self):
        self.businesses: Dict[str, BusinessProfile] = {}
        self.subscriptions: Dict[str, ServiceSubscription] = {}
        self.data_reports: Dict[str, DataReport] = {}
        
        # 服务定价
        self.service_pricing = {
            BusinessTier.STARTER: {
                ServiceType.DATA_ANALYTICS: 99.0,
                ServiceType.API_INTEGRATION: 199.0,
                ServiceType.CUSTOM_MODEL: 499.0,
                ServiceType.PREMIUM_SUPPORT: 299.0
            },
            BusinessTier.PROFESSIONAL: {
                ServiceType.DATA_ANALYTICS: 299.0,
                ServiceType.API_INTEGRATION: 599.0,
                ServiceType.CUSTOM_MODEL: 1499.0,
                ServiceType.PREMIUM_SUPPORT: 799.0
            },
            BusinessTier.ENTERPRISE: {
                ServiceType.DATA_ANALYTICS: 999.0,
                ServiceType.API_INTEGRATION: 1999.0,
                ServiceType.CUSTOM_MODEL: 4999.0,
                ServiceType.PREMIUM_SUPPORT: 1999.0
            }
        }
        
        # 服务限制
        self.service_limits = {
            BusinessTier.STARTER: {
                "max_api_calls": 10000,
                "data_retention_days": 90,
                "max_reports": 10,
                "support_level": "basic"
            },
            BusinessTier.PROFESSIONAL: {
                "max_api_calls": 100000,
                "data_retention_days": 365,
                "max_reports": 100,
                "support_level": "priority"
            },
            BusinessTier.ENTERPRISE: {
                "max_api_calls": 1000000,
                "data_retention_days": 9999,
                "max_reports": 1000,
                "support_level": "dedicated"
            }
        }
    
    def register_business(self, company_name: str, contact_email: str, tier: BusinessTier = BusinessTier.STARTER) -> BusinessProfile:
        """注册企业用户"""
        business_id = str(uuid.uuid4())
        now = datetime.now()
        
        business_profile = BusinessProfile(
            business_id=business_id,
            company_name=company_name,
            contact_email=contact_email,
            tier=tier,
            subscription_status="trial",
            subscription_end=now + timedelta(days=30),  # 30天试用期
            api_usage={"total": 0, "this_month": 0},
            data_access_level=1,
            created_at=now
        )
        
        self.businesses[business_id] = business_profile
        logger.info(f"注册新企业用户: {company_name} ({business_id})")
        return business_profile
    
    def subscribe_to_service(self, business_id: str, service_type: ServiceType, plan_details: Dict[str, Any]) -> Optional[ServiceSubscription]:
        """订阅服务"""
        
        business = self.businesses.get(business_id)
        if not business:
            logger.error(f"企业用户不存在: {business_id}")
            return None
        
        # 检查是否已订阅
        existing_subs = [s for s in self.subscriptions.values() 
                        if s.business_id == business_id and s.service_type == service_type]
        
        if existing_subs:
            logger.error(f"企业已订阅该服务: {service_type.value}")
            return None
        
        # 计算费用
        monthly_cost = self.service_pricing[business.tier][service_type]
        
        subscription_id = str(uuid.uuid4())
        now = datetime.now()
        
        subscription = ServiceSubscription(
            subscription_id=subscription_id,
            business_id=business_id,
            service_type=service_type,
            plan_details=plan_details,
            monthly_cost=monthly_cost,
            start_date=now,
            end_date=now + timedelta(days=30),  # 按月订阅
            usage_limits=self._get_usage_limits(business.tier, service_type),
            current_usage={"calls": 0, "data_volume": 0}
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # 更新企业订阅状态
        business.subscription_status = "active"
        business.subscription_end = subscription.end_date
        
        logger.info(f"企业{business_id}订阅服务: {service_type.value} - ${monthly_cost}/月")
        return subscription
    
    def generate_data_report(self, business_id: str, report_type: str, filters: Dict[str, Any]) -> Optional[DataReport]:
        """生成数据报告"""
        
        business = self.businesses.get(business_id)
        if not business:
            logger.error(f"企业用户不存在: {business_id}")
            return None
        
        # 检查数据访问权限
        if not self._check_data_access(business, report_type):
            logger.error(f"企业无权限访问该类型数据: {report_type}")
            return None
        
        # 生成报告数据
        report_data = self._generate_report_data(report_type, filters, business.tier)
        
        report_id = str(uuid.uuid4())
        now = datetime.now()
        
        report = DataReport(
            report_id=report_id,
            business_id=business_id,
            report_type=report_type,
            data_content=report_data,
            generation_date=now,
            access_level=business.data_access_level,
            download_count=0
        )
        
        self.data_reports[report_id] = report
        logger.info(f"为企业{business_id}生成数据报告: {report_type}")
        return report
    
    def get_api_usage_statistics(self, business_id: str) -> Dict[str, Any]:
        """获取API使用统计"""
        
        business = self.businesses.get(business_id)
        if not business:
            return {}
        
        business_subs = [s for s in self.subscriptions.values() if s.business_id == business_id]
        
        # 计算总使用量
        total_calls = sum(sub.current_usage.get("calls", 0) for sub in business_subs)
        total_data = sum(sub.current_usage.get("data_volume", 0) for sub in business_subs)
        
        # 计算限制
        tier_limits = self.service_limits[business.tier]
        
        return {
            "business_info": business.to_dict(),
            "subscriptions": [sub.to_dict() for sub in business_subs],
            "usage_summary": {
                "total_api_calls": total_calls,
                "total_data_volume": total_data,
                "monthly_limit": tier_limits["max_api_calls"],
                "utilization_rate": min(100, (total_calls / tier_limits["max_api_calls"]) * 100)
            }
        }
    
    def record_api_usage(self, business_id: str, service_type: ServiceType, call_data: Dict[str, Any]) -> bool:
        """记录API使用情况"""
        
        business = self.businesses.get(business_id)
        if not business:
            return False
        
        # 查找相关订阅
        business_subs = [s for s in self.subscriptions.values() 
                        if s.business_id == business_id and s.service_type == service_type]
        
        if not business_subs:
            logger.error(f"企业未订阅该服务: {service_type.value}")
            return False
        
        subscription = business_subs[0]
        
        # 检查使用限制
        if subscription.current_usage.get("calls", 0) >= subscription.usage_limits.get("max_calls", 0):
            logger.error(f"API调用次数已达上限")
            return False
        
        # 更新使用统计
        subscription.current_usage["calls"] = subscription.current_usage.get("calls", 0) + 1
        subscription.current_usage["data_volume"] = subscription.current_usage.get("data_volume", 0) + call_data.get("data_size", 0)
        
        # 更新企业总使用量
        business.api_usage["total"] = business.api_usage.get("total", 0) + 1
        business.api_usage["this_month"] = business.api_usage.get("this_month", 0) + 1
        
        return True
    
    def upgrade_business_tier(self, business_id: str, new_tier: BusinessTier) -> bool:
        """升级企业等级"""
        
        business = self.businesses.get(business_id)
        if not business:
            return False
        
        # 检查升级条件
        if new_tier == BusinessTier.PROFESSIONAL and business.tier == BusinessTier.STARTER:
            # 需要支付升级费用
            upgrade_cost = 500.0  # 升级费用
            # 这里应该集成支付系统
            logger.info(f"企业{business_id}升级到专业版，费用: ${upgrade_cost}")
        
        elif new_tier == BusinessTier.ENTERPRISE:
            # 企业版需要定制方案
            logger.info(f"企业{business_id}申请升级到企业版，需要定制方案")
        
        business.tier = new_tier
        business.data_access_level = self._get_access_level(new_tier)
        
        logger.info(f"企业{business_id}升级到{new_tier.value}等级")
        return True
    
    def get_service_catalog(self, business_tier: BusinessTier) -> Dict[str, Any]:
        """获取服务目录"""
        
        catalog = {}
        for service_type in ServiceType:
            catalog[service_type.value] = {
                "description": self._get_service_description(service_type),
                "monthly_cost": self.service_pricing[business_tier][service_type],
                "features": self._get_service_features(service_type, business_tier),
                "limits": self._get_service_limits(service_type, business_tier)
            }
        
        return {
            "tier": business_tier.value,
            "services": catalog,
            "tier_benefits": self.service_limits[business_tier]
        }
    
    def _get_usage_limits(self, tier: BusinessTier, service_type: ServiceType) -> Dict[str, int]:
        """获取使用限制"""
        base_limits = {
            "max_calls": 1000,
            "max_data_volume": 1000000  # 1GB
        }
        
        # 根据等级调整限制
        multiplier = {
            BusinessTier.STARTER: 1,
            BusinessTier.PROFESSIONAL: 10,
            BusinessTier.ENTERPRISE: 100
        }[tier]
        
        return {k: v * multiplier for k, v in base_limits.items()}
    
    def _check_data_access(self, business: BusinessProfile, report_type: str) -> bool:
        """检查数据访问权限"""
        
        access_rules = {
            "basic_analytics": 1,      # 基础分析报告
            "growth_trends": 2,        # 生长趋势报告
            "market_insights": 3,      # 市场洞察报告
            "custom_analysis": 4       # 定制分析报告
        }
        
        required_level = access_rules.get(report_type, 999)
        return business.data_access_level >= required_level
    
    def _generate_report_data(self, report_type: str, filters: Dict[str, Any], tier: BusinessTier) -> Dict[str, Any]:
        """生成报告数据"""
        
        # 模拟报告数据生成
        if report_type == "basic_analytics":
            return {
                "summary": {
                    "total_farms": 1500,
                    "active_growth_cycles": 4500,
                    "average_yield": 0.85
                },
                "crop_distribution": {
                    "番茄": 35,
                    "生菜": 25,
                    "草莓": 15,
                    "其他": 25
                },
                "success_rate": 0.92
            }
        
        elif report_type == "growth_trends":
            return {
                "trend_analysis": {
                    "yield_improvement": 0.15,
                    "growth_efficiency": 0.22,
                    "resource_optimization": 0.18
                },
                "seasonal_patterns": {
                    "spring": 0.95,
                    "summer": 0.88,
                    "autumn": 0.92,
                    "winter": 0.78
                }
            }
        
        elif report_type == "market_insights":
            return {
                "market_trends": {
                    "demand_growth": 0.25,
                    "price_trends": 0.12,
                    "consumer_preferences": {
                        "organic": 0.65,
                        "local": 0.78,
                        "sustainable": 0.82
                    }
                },
                "competitive_analysis": {
                    "market_share": 0.35,
                    "growth_potential": 0.45
                }
            }
        
        else:
            return {"error": "未知报告类型"}
    
    def _get_access_level(self, tier: BusinessTier) -> int:
        """获取访问权限级别"""
        return {
            BusinessTier.STARTER: 1,
            BusinessTier.PROFESSIONAL: 3,
            BusinessTier.ENTERPRISE: 5
        }[tier]
    
    def _get_service_description(self, service_type: ServiceType) -> str:
        """获取服务描述"""
        descriptions = {
            ServiceType.DATA_ANALYTICS: "农业数据分析服务，提供生长趋势、产量预测等深度分析",
            ServiceType.API_INTEGRATION: "完整的API集成服务，支持实时数据访问和控制",
            ServiceType.CUSTOM_MODEL: "定制AI模型开发，针对特定作物和环境的优化方案",
            ServiceType.PREMIUM_SUPPORT: "高级技术支持，包括专家咨询和紧急响应"
        }
        return descriptions[service_type]
    
    def _get_service_features(self, service_type: ServiceType, tier: BusinessTier) -> List[str]:
        """获取服务特性"""
        base_features = {
            ServiceType.DATA_ANALYTICS: ["实时数据分析", "趋势预测", "异常检测"],
            ServiceType.API_INTEGRATION: ["RESTful API", "WebSocket实时数据", "批量处理"],
            ServiceType.CUSTOM_MODEL: ["专业模型定制", "性能优化", "持续更新"],
            ServiceType.PREMIUM_SUPPORT: ["24/7技术支持", "专家咨询", "优先响应"]
        }
        
        tier_features = {
            BusinessTier.PROFESSIONAL: ["高级分析功能", "自定义报告"],
            BusinessTier.ENTERPRISE: ["完全定制化", "专属客户经理"]
        }
        
        features = base_features[service_type]
        if tier in tier_features:
            features.extend(tier_features[tier])
        
        return features
    
    def _get_service_limits(self, service_type: ServiceType, tier: BusinessTier) -> Dict[str, Any]:
        """获取服务限制"""
        base_limits = {
            "api_calls_per_month": 10000,
            "data_storage_gb": 10,
            "concurrent_connections": 10
        }
        
        multiplier = {
            BusinessTier.STARTER: 1,
            BusinessTier.PROFESSIONAL: 5,
            BusinessTier.ENTERPRISE: 20
        }[tier]
        
        return {k: v * multiplier for k, v in base_limits.items()}


# 全局企业服务实例
business_service = BusinessService()

# 初始化一些示例企业
if __name__ == "__main__":
    # 创建示例企业
    example_business = business_service.register_business(
        "先进农业科技", "contact@advancedagri.com", BusinessTier.PROFESSIONAL
    )
    
    # 订阅服务
    analytics_sub = business_service.subscribe_to_service(
        example_business.business_id,
        ServiceType.DATA_ANALYTICS,
        {"report_frequency": "weekly", "data_sources": ["growth", "environment"]}
    )
    
    # 生成报告
    report = business_service.generate_data_report(
        example_business.business_id,
        "growth_trends",
        {"crop_type": "番茄", "time_range": "last_quarter"}
    )
    
    # 获取使用统计
    usage_stats = business_service.get_api_usage_statistics(example_business.business_id)
    
    print(f"示例企业创建成功: {example_business.company_name}")
    print(f"服务订阅: {analytics_sub.service_type.value if analytics_sub else 'None'}")
    print(f"生成报告: {report.report_type if report else 'None'}")