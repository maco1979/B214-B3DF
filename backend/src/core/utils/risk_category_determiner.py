"""
风险类别判定工具模块
提供多种版本的风险类别判定函数，适配不同场景需求
"""

from typing import Dict, Union, Optional, Any
from enum import Enum


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


# ========== 版本3：极致扩展版（推荐用于核心业务） ==========
# 风险配置集中管理（可无限扩展，示例新增critical等级）
# 格式：{风险等级: 阈值}，阈值越大，风险等级越高
RISK_CONFIG = {
    "critical": 0.9,  # 新增：风险值≥0.9 → 极高风险
    "high": 0.7,      # 风险值≥0.7 → 高风险
    "medium": 0.5,    # 风险值≥0.5 → 中风险
    "low": 0.3        # 风险值≥0.3 → 低风险，其他 → 低风险
}

# 按阈值从高到低排序（程序自动适配，新增等级无需手动排序）
SORTED_RISK_RULES = sorted(RISK_CONFIG.items(), key=lambda x: x[1], reverse=True)


def determine_risk_category_extended(
    assessment: Dict[str, Any], 
    risk_value: Optional[Union[int, float]], 
    custom_config: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    确定整体风险类别（极致扩展版）
    特性：新增/删除风险等级，仅需修改RISK_CONFIG，核心逻辑0改动
    
    :param assessment: 风险评估原始字典
    :param risk_value: 风险值（支持int/float，兼容None/负数）
    :param custom_config: 自定义风险配置，如果提供则覆盖默认配置
    :return: 补充风险等级的评估字典
    """
    # 使用自定义配置或默认配置
    config = custom_config or RISK_CONFIG
    sorted_rules = sorted(config.items(), key=lambda x: x[1], reverse=True) if not custom_config else sorted(custom_config.items(), key=lambda x: x[1], reverse=True)
    
    # 1. 输入容错校验
    if risk_value is None or not isinstance(risk_value, (int, float)) or risk_value < 0:
        assessment['risk_category'] = 'low'
        assessment['risk_level_enum'] = RiskLevel.LOW
        return assessment
    
    # 2. 动态匹配风险等级（适配任意数量的风险等级）
    for risk_level, threshold in sorted_rules:
        if risk_value >= threshold:
            assessment['risk_category'] = risk_level
            # 设置对应的枚举值
            assessment['risk_level_enum'] = RiskLevel(risk_level)
            break
    else:
        # 如果没有匹配到任何规则，默认为低风险
        assessment['risk_category'] = 'low'
        assessment['risk_level_enum'] = RiskLevel.LOW
    
    return assessment


# ========== 版本2：生产健壮版（推荐用于一般业务） ==========
def determine_risk_category_robust(assessment: Dict, risk_value: Optional[Union[int, float]]) -> Dict:
    """
    确定整体风险类别（生产健壮版）
    :param assessment: 风险评估原始字典
    :param risk_value: 风险值（支持int/float，兼容None/负数）
    :return: 补充风险等级的评估字典
    """
    # 1. 输入校验与容错处理（核心：防止异常数据导致程序崩溃）
    if risk_value is None or not isinstance(risk_value, (int, float)):
        assessment['risk_category'] = 'low'
        assessment['risk_level_enum'] = RiskLevel.LOW
        return assessment
    
    # 超标值为负数 → 归为低风险
    if risk_value < 0:
        assessment['risk_category'] = 'low'
        assessment['risk_level_enum'] = RiskLevel.LOW
        return assessment
    
    # 2. 风险等级判定（逻辑简洁，易读）
    # 风险配置集中管理（统一维护，无需改动业务逻辑）
    RISK_CONFIG = {
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3
    }
    
    if risk_value >= RISK_CONFIG["high"]:
        assessment['risk_category'] = 'high'
        assessment['risk_level_enum'] = RiskLevel.HIGH
    elif risk_value >= RISK_CONFIG["medium"]:
        assessment['risk_category'] = 'medium'
        assessment['risk_level_enum'] = RiskLevel.MEDIUM
    else:
        assessment['risk_category'] = 'low'
        assessment['risk_level_enum'] = RiskLevel.LOW
    
    return assessment


# ========== 版本1：基础优雅版（最小改动，立竿见影） ==========
def determine_risk_category_basic(assessment: Dict, risk_value: Union[int, float]) -> Dict:
    """
    确定整体风险类别（基础优雅版）
    :param assessment: 风险评估字典（需包含风险等级字段）
    :param risk_value: 风险值
    :return: 补充风险等级后的评估字典
    """
    # ========== 常量定义（所有配置集中管理，一改全改） ==========
    # 风险阈值
    HIGH_RISK_THRESHOLD: float = 0.7
    MEDIUM_RISK_THRESHOLD: float = 0.5
    LOW_RISK_THRESHOLD: float = 0.3
    
    # 风险等级映射
    RISK_LEVEL_HIGH: str = "high"
    RISK_LEVEL_MEDIUM: str = "medium"
    RISK_LEVEL_LOW: str = "low"
    
    if risk_value >= HIGH_RISK_THRESHOLD:
        assessment['risk_category'] = RISK_LEVEL_HIGH
        assessment['risk_level_enum'] = RiskLevel.HIGH
    elif risk_value >= MEDIUM_RISK_THRESHOLD:
        assessment['risk_category'] = RISK_LEVEL_MEDIUM
        assessment['risk_level_enum'] = RiskLevel.MEDIUM
    else:
        assessment['risk_category'] = RISK_LEVEL_LOW
        assessment['risk_level_enum'] = RiskLevel.LOW
    
    return assessment


# ========== 版本4：极简一行版（追求代码精简，适合简单场景） ==========
def determine_risk_category_simple(assessment: dict, risk_value: float) -> dict:
    """
    确定整体风险类别（极简一行版）
    适合阈值/等级固定、无需扩展的简单场景
    """
    # 一行完成风险判定，简洁高效
    assessment['risk_category'] = 'high' if risk_value >= 0.7 else 'medium' if risk_value >= 0.5 else 'low'
    assessment['risk_level_enum'] = RiskLevel(assessment['risk_category'])
    return assessment


# ========== 版本5：企业级增强版（日志+注释+异常捕获，适配大型项目） ==========
import logging

# 配置日志（可接入项目统一日志体系）
logger = logging.getLogger(__name__)

def determine_risk_category_enterprise(
    assessment: Dict[str, Any], 
    risk_value: Optional[Union[int, float]]
) -> Dict[str, Any]:
    """
    【企业级】确定整体风险类别核心方法
    :param assessment: 风险评估原始字典，包含业务基础数据
    :param risk_value: 风险值，核心判定依据
    :return: 新字典（含风险等级，不修改原字典）
    :raises: 无（全异常捕获，保证程序健壮性）
    """
    try:
        # 1. 输入校验与容错
        if not isinstance(risk_value, (int, float)) or risk_value < 0:
            assessment['risk_category'] = 'low'
            assessment['risk_level_enum'] = RiskLevel.LOW
            logger.info(f"风险判定：输入值异常[{risk_value}]，默认低风险")
            return assessment
        
        # 2. 风险等级判定
        # 风险配置集中管理
        RISK_CONFIG = {
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3
        }
        
        if risk_value >= RISK_CONFIG["high"]:
            assessment['risk_category'] = 'high'
            assessment['risk_level_enum'] = RiskLevel.HIGH
        elif risk_value >= RISK_CONFIG["medium"]:
            assessment['risk_category'] = 'medium'
            assessment['risk_level_enum'] = RiskLevel.MEDIUM
        else:
            assessment['risk_category'] = 'low'
            assessment['risk_level_enum'] = RiskLevel.LOW
        
        # 3. 记录判定日志（关键：线上溯源）
        logger.info(f"风险判定成功：风险值[{risk_value}] → 风险等级[{assessment['risk_category']}]")
        return assessment
    
    except Exception as e:
        # 兜底：捕获所有异常，返回低风险，避免程序崩溃
        logger.error(f"风险判定失败：{str(e)}，默认低风险", exc_info=True)
        assessment['risk_category'] = 'low'
        assessment['risk_level_enum'] = RiskLevel.LOW
        return assessment