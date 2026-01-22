"""
AI自主决策风险控制系统配置

提供系统的默认配置和配置管理功能，支持环境变量覆盖和动态配置更新，
确保系统在不同环境下的灵活性和可配置性。
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class Environment(Enum):
    """运行环境"""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class TechnicalRiskConfig:
    """技术风险配置"""
    # 目标对齐检测
    target_alignment_threshold: float = 0.8
    target_drift_detection_enabled: bool = True
    max_target_drift: float = 0.2
    
    # 紧急停止机制
    emergency_stop_enabled: bool = True
    emergency_stop_threshold: float = 0.9
    max_decision_override_count: int = 3
    
    # 黑箱检测
    blackbox_detection_enabled: bool = True
    explainability_threshold: float = 0.7
    max_unexplainable_decisions: int = 5
    
    # 技术稳定性
    system_stability_threshold: float = 0.85
    resource_usage_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_usage_limits is None:
            self.resource_usage_limits = {
                "cpu_usage": 0.9,
                "memory_usage": 0.85,
                "network_bandwidth": 0.8
            }


@dataclass
class DataSecurityConfig:
    """数据安全配置"""
    # 数据加密
    encryption_required: bool = True
    encryption_algorithm: str = "AES-256"
    key_rotation_interval_days: int = 30
    
    # 隐私保护
    privacy_protection_enabled: bool = True
    data_anonymization_enabled: bool = True
    zero_knowledge_proofs_enabled: bool = False
    
    # 访问控制
    access_control_enabled: bool = True
    max_failed_login_attempts: int = 5
    session_timeout_minutes: int = 30
    
    # 数据保留
    data_retention_days: int = 90
    sensitive_data_retention_days: int = 30
    
    # 联邦学习
    federated_learning_enabled: bool = True
    model_aggregation_frequency_hours: int = 24


@dataclass
class AlgorithmBiasConfig:
    """算法偏见配置"""
    # 公平性检测
    fairness_detection_enabled: bool = True
    fairness_threshold: float = 0.8
    protected_attributes: list = None
    
    # 偏见纠正
    bias_correction_enabled: bool = True
    correction_method: str = "reweighting"
    max_bias_correction_iterations: int = 10
    
    # 数据平衡
    data_balancing_enabled: bool = True
    min_group_representation: float = 0.1
    max_group_disparity: float = 0.2
    
    # 可解释性
    explainability_enabled: bool = True
    feature_importance_threshold: float = 0.05
    
    def __post_init__(self):
        if self.protected_attributes is None:
            self.protected_attributes = [
                "gender", "age", "geographic_region", 
                "income_level", "education_level"
            ]


@dataclass
class GovernanceConfig:
    """治理配置"""
    # 人-AI协同
    human_ai_collaboration_enabled: bool = True
    human_override_enabled: bool = True
    max_ai_autonomy_level: float = 0.8
    
    # 社区治理
    community_governance_enabled: bool = True
    voting_threshold: float = 0.6
    min_voting_participation: float = 0.3
    
    # 审计日志
    audit_logging_enabled: bool = True
    log_retention_days: int = 365
    detailed_decision_logging: bool = True
    
    # 透明性
    transparency_enabled: bool = True
    decision_explanation_required: bool = True
    model_parameters_public: bool = False


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 监控间隔
    monitoring_interval_seconds: int = 60
    alert_retention_days: int = 30
    report_generation_interval_hours: int = 24
    
    # 风险阈值
    risk_thresholds: Dict[str, float] = None
    
    # 应急响应
    emergency_response_enabled: bool = True
    auto_mitigation_enabled: bool = False
    
    # 通知设置
    email_notifications_enabled: bool = True
    sms_notifications_enabled: bool = False
    webhook_notifications_enabled: bool = True
    
    def __post_init__(self):
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                "normal": 0.3,
                "warning": 0.5,
                "alert": 0.7,
                "critical": 0.9
            }


@dataclass
class SystemConfig:
    """系统配置"""
    # 基础配置
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # 数据库配置
    database_url: str = "sqlite:///ai_risk_control.db"
    redis_url: Optional[str] = None
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # 模块配置
    technical_risk: TechnicalRiskConfig = None
    data_security: DataSecurityConfig = None
    algorithm_bias: AlgorithmBiasConfig = None
    governance: GovernanceConfig = None
    monitoring: MonitoringConfig = None
    
    def __post_init__(self):
        if self.technical_risk is None:
            self.technical_risk = TechnicalRiskConfig()
        if self.data_security is None:
            self.data_security = DataSecurityConfig()
        if self.algorithm_bias is None:
            self.algorithm_bias = AlgorithmBiasConfig()
        if self.governance is None:
            self.governance = GovernanceConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config = None
        
    def load_config(self) -> SystemConfig:
        """加载配置"""
        # 从环境变量获取基础配置
        environment = Environment(os.getenv("ENVIRONMENT", "development"))
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # 数据库配置
        database_url = os.getenv("DATABASE_URL", "sqlite:///ai_risk_control.db")
        redis_url = os.getenv("REDIS_URL")
        
        # API配置
        api_host = os.getenv("API_HOST", "0.0.0.0")
        api_port = int(os.getenv("API_PORT", "8000"))
        api_debug = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # 从配置文件加载（如果存在）
        file_config = {}
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
        
        # 构建配置对象
        config = SystemConfig(
            environment=environment,
            debug_mode=debug_mode,
            log_level=log_level,
            database_url=database_url,
            redis_url=redis_url,
            api_host=api_host,
            api_port=api_port,
            api_debug=api_debug
        )
        
        # 应用文件配置
        self._apply_file_config(config, file_config)
        
        self._config = config
        return config
    
    def _apply_file_config(self, config: SystemConfig, file_config: Dict[str, Any]):
        """应用文件配置"""
        # 技术风险配置
        if "technical_risk" in file_config:
            tech_config = file_config["technical_risk"]
            config.technical_risk = TechnicalRiskConfig(**tech_config)
        
        # 数据安全配置
        if "data_security" in file_config:
            security_config = file_config["data_security"]
            config.data_security = DataSecurityConfig(**security_config)
        
        # 算法偏见配置
        if "algorithm_bias" in file_config:
            bias_config = file_config["algorithm_bias"]
            config.algorithm_bias = AlgorithmBiasConfig(**bias_config)
        
        # 治理配置
        if "governance" in file_config:
            gov_config = file_config["governance"]
            config.governance = GovernanceConfig(**gov_config)
        
        # 监控配置
        if "monitoring" in file_config:
            monitor_config = file_config["monitoring"]
            config.monitoring = MonitoringConfig(**monitor_config)
    
    def get_config(self) -> SystemConfig:
        """获取配置"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        if self._config is None:
            self.load_config()
        
        # 更新基础配置
        if "environment" in new_config:
            self._config.environment = Environment(new_config["environment"])
        if "debug_mode" in new_config:
            self._config.debug_mode = new_config["debug_mode"]
        if "log_level" in new_config:
            self._config.log_level = new_config["log_level"]
        
        # 保存到文件（如果指定了配置文件）
        if self.config_file:
            self.save_config()
    
    def save_config(self):
        """保存配置到文件"""
        if self.config_file and self._config:
            config_dict = asdict(self._config)
            
            # 处理枚举类型
            config_dict["environment"] = self._config.environment.value
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def get_config_for_environment(self, environment: Environment) -> SystemConfig:
        """获取特定环境的配置"""
        base_config = self.get_config()
        
        # 根据环境调整配置
        if environment == Environment.PRODUCTION:
            base_config.debug_mode = False
            base_config.log_level = "WARNING"
            base_config.api_debug = False
            base_config.monitoring.monitoring_interval_seconds = 30
            
        elif environment == Environment.TESTING:
            base_config.debug_mode = True
            base_config.log_level = "DEBUG"
            base_config.api_debug = True
            base_config.database_url = "sqlite:///test_ai_risk_control.db"
        
        return base_config


# 全局配置管理器实例
_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> SystemConfig:
    """获取当前配置"""
    return get_config_manager().get_config()


def create_default_config_file(config_file: str = "ai_risk_control_config.json"):
    """创建默认配置文件"""
    default_config = SystemConfig()
    config_dict = asdict(default_config)
    
    # 处理枚举类型
    config_dict["environment"] = default_config.environment.value
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"默认配置文件已创建: {config_file}")


# 环境变量配置示例
ENVIRONMENT_VARIABLES = {
    "ENVIRONMENT": "运行环境 (development/testing/staging/production)",
    "DEBUG": "调试模式 (true/false)",
    "LOG_LEVEL": "日志级别 (DEBUG/INFO/WARNING/ERROR)",
    "DATABASE_URL": "数据库连接URL",
    "REDIS_URL": "Redis连接URL (可选)",
    "API_HOST": "API服务主机",
    "API_PORT": "API服务端口",
    "API_DEBUG": "API调试模式 (true/false)"
}


if __name__ == "__main__":
    # 创建默认配置文件
    create_default_config_file("default_config.json")
    
    # 示例：加载和使用配置
    config_manager = get_config_manager("default_config.json")
    config = config_manager.get_config()
    
    print("当前配置:")
    print(f"环境: {config.environment.value}")
    print(f"调试模式: {config.debug_mode}")
    print(f"日志级别: {config.log_level}")
    print(f"数据库URL: {config.database_url}")
    print(f"API端口: {config.api_port}")