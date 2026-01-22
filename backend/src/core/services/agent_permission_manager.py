"""
智能体权限管理器 - 提供细粒度的智能体权限控制和行为审计
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """智能体类型枚举"""
    DECISION_AGENT = "decision_agent"       # 决策智能体
    CONTROL_AGENT = "control_agent"         # 控制智能体
    MONITOR_AGENT = "monitor_agent"         # 监控智能体
    LEARNING_AGENT = "learning_agent"       # 学习智能体
    DATA_AGENT = "data_agent"               # 数据智能体
    SYSTEM_AGENT = "system_agent"           # 系统智能体


class Permission(Enum):
    """权限枚举"""
    # 决策权限
    DECISION_READ = "decision.read"
    DECISION_CREATE = "decision.create"
    DECISION_EXECUTE = "decision.execute"
    
    # 设备控制权限
    DEVICE_READ = "device.read"
    DEVICE_CONTROL = "device.control"
    DEVICE_CONFIGURE = "device.configure"
    
    # 模型权限
    MODEL_READ = "model.read"
    MODEL_TRAIN = "model.train"
    MODEL_DEPLOY = "model.deploy"
    MODEL_DELETE = "model.delete"
    
    # 数据权限
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    
    # 系统权限
    SYSTEM_READ = "system.read"
    SYSTEM_CONFIGURE = "system.configure"
    SYSTEM_ADMIN = "system.admin"
    
    # 监控权限
    METRICS_READ = "metrics.read"
    LOGS_READ = "logs.read"
    ALERTS_MANAGE = "alerts.manage"


@dataclass
class AgentBehaviorLog:
    """智能体行为日志"""
    log_id: str
    agent_id: str
    agent_type: str
    action: str
    resource: str
    permission_used: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class AgentProfile:
    """智能体配置文件"""
    agent_id: str
    agent_type: AgentType
    name: str
    permissions: Set[str]
    created_at: datetime
    last_active: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 100  # 每分钟最大请求数
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentPermissionManager:
    """智能体权限管理器"""
    
    # 默认权限配置（基于智能体类型）
    DEFAULT_PERMISSIONS: Dict[AgentType, List[str]] = {
        AgentType.DECISION_AGENT: [
            Permission.DECISION_READ.value,
            Permission.DECISION_CREATE.value,
            Permission.MODEL_READ.value,
            Permission.METRICS_READ.value,
        ],
        AgentType.CONTROL_AGENT: [
            Permission.DEVICE_READ.value,
            Permission.DEVICE_CONTROL.value,
            Permission.DECISION_READ.value,
            Permission.METRICS_READ.value,
        ],
        AgentType.MONITOR_AGENT: [
            Permission.METRICS_READ.value,
            Permission.LOGS_READ.value,
            Permission.SYSTEM_READ.value,
            Permission.DEVICE_READ.value,
        ],
        AgentType.LEARNING_AGENT: [
            Permission.MODEL_READ.value,
            Permission.MODEL_TRAIN.value,
            Permission.DATA_READ.value,
            Permission.METRICS_READ.value,
        ],
        AgentType.DATA_AGENT: [
            Permission.DATA_READ.value,
            Permission.DATA_WRITE.value,
            Permission.METRICS_READ.value,
        ],
        AgentType.SYSTEM_AGENT: [
            Permission.SYSTEM_READ.value,
            Permission.SYSTEM_CONFIGURE.value,
            Permission.SYSTEM_ADMIN.value,
            Permission.METRICS_READ.value,
            Permission.LOGS_READ.value,
        ],
    }
    
    def __init__(self):
        self._agents: Dict[str, AgentProfile] = {}
        self._behavior_logs: List[AgentBehaviorLog] = []
        self._rate_limiters: Dict[str, List[datetime]] = defaultdict(list)
        self._max_log_size = 10000
        self._lock = asyncio.Lock()
    
    async def register_agent(self, 
                            agent_id: str,
                            agent_type: AgentType,
                            name: str,
                            custom_permissions: Optional[List[str]] = None,
                            rate_limit: int = 100) -> AgentProfile:
        """注册新智能体"""
        async with self._lock:
            if agent_id in self._agents:
                raise ValueError(f"智能体 {agent_id} 已存在")
            
            # 获取默认权限
            default_perms = set(self.DEFAULT_PERMISSIONS.get(agent_type, []))
            
            # 合并自定义权限
            if custom_permissions:
                default_perms.update(custom_permissions)
            
            profile = AgentProfile(
                agent_id=agent_id,
                agent_type=agent_type,
                name=name,
                permissions=default_perms,
                created_at=datetime.now(),
                rate_limit=rate_limit
            )
            
            self._agents[agent_id] = profile
            
            # 记录注册行为
            await self._log_behavior(
                agent_id=agent_id,
                agent_type=agent_type.value,
                action="register",
                resource="agent_registry",
                permission_used="system.register",
                success=True,
                details={"permissions": list(default_perms)}
            )
            
            logger.info(f"智能体 {name} ({agent_id}) 注册成功，类型: {agent_type.value}")
            return profile
    
    async def check_permission(self, 
                              agent_id: str, 
                              permission: str,
                              resource: Optional[str] = None) -> bool:
        """检查智能体权限"""
        async with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"未知智能体尝试访问: {agent_id}")
                return False
            
            profile = self._agents[agent_id]
            
            # 检查智能体是否激活
            if not profile.is_active:
                logger.warning(f"已停用的智能体尝试访问: {agent_id}")
                return False
            
            # 检查速率限制
            if not await self._check_rate_limit(agent_id, profile.rate_limit):
                logger.warning(f"智能体 {agent_id} 超过速率限制")
                return False
            
            # 检查权限
            has_permission = permission in profile.permissions
            
            # 更新最后活动时间
            profile.last_active = datetime.now()
            
            # 记录行为
            await self._log_behavior(
                agent_id=agent_id,
                agent_type=profile.agent_type.value,
                action="check_permission",
                resource=resource or "unknown",
                permission_used=permission,
                success=has_permission,
                details={"requested_permission": permission}
            )
            
            return has_permission
    
    async def grant_permission(self, 
                              agent_id: str, 
                              permission: str,
                              granted_by: str = "system") -> bool:
        """授予权限"""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            profile = self._agents[agent_id]
            profile.permissions.add(permission)
            
            await self._log_behavior(
                agent_id=granted_by,
                agent_type="system",
                action="grant_permission",
                resource=agent_id,
                permission_used="system.admin",
                success=True,
                details={"granted_permission": permission, "target_agent": agent_id}
            )
            
            logger.info(f"已授予智能体 {agent_id} 权限: {permission}")
            return True
    
    async def revoke_permission(self, 
                               agent_id: str, 
                               permission: str,
                               revoked_by: str = "system") -> bool:
        """撤销权限"""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            profile = self._agents[agent_id]
            profile.permissions.discard(permission)
            
            await self._log_behavior(
                agent_id=revoked_by,
                agent_type="system",
                action="revoke_permission",
                resource=agent_id,
                permission_used="system.admin",
                success=True,
                details={"revoked_permission": permission, "target_agent": agent_id}
            )
            
            logger.info(f"已撤销智能体 {agent_id} 权限: {permission}")
            return True
    
    async def deactivate_agent(self, agent_id: str, reason: str = "") -> bool:
        """停用智能体"""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            profile = self._agents[agent_id]
            profile.is_active = False
            
            await self._log_behavior(
                agent_id="system",
                agent_type="system",
                action="deactivate_agent",
                resource=agent_id,
                permission_used="system.admin",
                success=True,
                details={"reason": reason}
            )
            
            logger.info(f"智能体 {agent_id} 已停用，原因: {reason}")
            return True
    
    async def activate_agent(self, agent_id: str) -> bool:
        """激活智能体"""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            profile = self._agents[agent_id]
            profile.is_active = True
            
            await self._log_behavior(
                agent_id="system",
                agent_type="system",
                action="activate_agent",
                resource=agent_id,
                permission_used="system.admin",
                success=True
            )
            
            logger.info(f"智能体 {agent_id} 已激活")
            return True
    
    async def _check_rate_limit(self, agent_id: str, limit: int) -> bool:
        """检查速率限制"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # 清理旧记录
        self._rate_limiters[agent_id] = [
            t for t in self._rate_limiters[agent_id] 
            if t > minute_ago
        ]
        
        # 检查是否超过限制
        if len(self._rate_limiters[agent_id]) >= limit:
            return False
        
        # 记录本次请求
        self._rate_limiters[agent_id].append(now)
        return True
    
    async def _log_behavior(self, 
                           agent_id: str,
                           agent_type: str,
                           action: str,
                           resource: str,
                           permission_used: str,
                           success: bool,
                           details: Optional[Dict] = None,
                           duration_ms: Optional[float] = None):
        """记录智能体行为"""
        log_id = hashlib.md5(
            f"{agent_id}:{action}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        log = AgentBehaviorLog(
            log_id=log_id,
            agent_id=agent_id,
            agent_type=agent_type,
            action=action,
            resource=resource,
            permission_used=permission_used,
            timestamp=datetime.now(),
            success=success,
            details=details or {},
            duration_ms=duration_ms
        )
        
        self._behavior_logs.append(log)
        
        # 限制日志大小
        if len(self._behavior_logs) > self._max_log_size:
            self._behavior_logs = self._behavior_logs[-self._max_log_size // 2:]
    
    async def get_agent_logs(self, 
                            agent_id: Optional[str] = None,
                            action: Optional[str] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """获取智能体行为日志"""
        async with self._lock:
            logs = self._behavior_logs
            
            # 过滤
            if agent_id:
                logs = [l for l in logs if l.agent_id == agent_id]
            if action:
                logs = [l for l in logs if l.action == action]
            if start_time:
                logs = [l for l in logs if l.timestamp >= start_time]
            if end_time:
                logs = [l for l in logs if l.timestamp <= end_time]
            
            # 排序并限制
            logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
            
            return [
                {
                    **asdict(log),
                    "timestamp": log.timestamp.isoformat()
                }
                for log in logs
            ]
    
    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体信息"""
        async with self._lock:
            if agent_id not in self._agents:
                return None
            
            profile = self._agents[agent_id]
            return {
                "agent_id": profile.agent_id,
                "agent_type": profile.agent_type.value,
                "name": profile.name,
                "permissions": list(profile.permissions),
                "is_active": profile.is_active,
                "rate_limit": profile.rate_limit,
                "created_at": profile.created_at.isoformat(),
                "last_active": profile.last_active.isoformat() if profile.last_active else None,
                "metadata": profile.metadata
            }
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """获取所有智能体"""
        async with self._lock:
            return [
                {
                    "agent_id": p.agent_id,
                    "agent_type": p.agent_type.value,
                    "name": p.name,
                    "is_active": p.is_active,
                    "permission_count": len(p.permissions),
                    "created_at": p.created_at.isoformat(),
                    "last_active": p.last_active.isoformat() if p.last_active else None
                }
                for p in self._agents.values()
            ]
    
    async def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取审计摘要"""
        async with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            recent_logs = [l for l in self._behavior_logs if l.timestamp > cutoff]
            
            # 统计
            total_actions = len(recent_logs)
            successful_actions = len([l for l in recent_logs if l.success])
            failed_actions = total_actions - successful_actions
            
            # 按智能体统计
            agent_stats = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0})
            for log in recent_logs:
                agent_stats[log.agent_id]["total"] += 1
                if log.success:
                    agent_stats[log.agent_id]["success"] += 1
                else:
                    agent_stats[log.agent_id]["failed"] += 1
            
            # 按操作类型统计
            action_stats = defaultdict(int)
            for log in recent_logs:
                action_stats[log.action] += 1
            
            return {
                "period_hours": hours,
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "success_rate": round(successful_actions / total_actions * 100, 2) if total_actions > 0 else 0,
                "agent_stats": dict(agent_stats),
                "action_stats": dict(action_stats),
                "active_agents": len([a for a in self._agents.values() if a.is_active]),
                "total_agents": len(self._agents)
            }


# 权限检查装饰器
def require_permission(permission: str):
    """权限检查装饰器"""
    def decorator(func):
        async def wrapper(self, *args, agent_id: str = None, **kwargs):
            if agent_id is None:
                raise ValueError("需要提供agent_id参数")
            
            manager = get_agent_permission_manager()
            has_permission = await manager.check_permission(agent_id, permission)
            
            if not has_permission:
                raise PermissionError(f"智能体 {agent_id} 没有权限: {permission}")
            
            return await func(self, *args, agent_id=agent_id, **kwargs)
        return wrapper
    return decorator


# 全局实例
_agent_permission_manager: Optional[AgentPermissionManager] = None


def get_agent_permission_manager() -> AgentPermissionManager:
    """获取智能体权限管理器单例"""
    global _agent_permission_manager
    if _agent_permission_manager is None:
        _agent_permission_manager = AgentPermissionManager()
    return _agent_permission_manager


async def initialize_default_agents():
    """初始化默认智能体"""
    manager = get_agent_permission_manager()
    
    # 注册系统智能体
    default_agents = [
        ("decision_agent_001", AgentType.DECISION_AGENT, "主决策智能体"),
        ("control_agent_001", AgentType.CONTROL_AGENT, "设备控制智能体"),
        ("monitor_agent_001", AgentType.MONITOR_AGENT, "系统监控智能体"),
        ("learning_agent_001", AgentType.LEARNING_AGENT, "模型学习智能体"),
        ("data_agent_001", AgentType.DATA_AGENT, "数据处理智能体"),
    ]
    
    for agent_id, agent_type, name in default_agents:
        try:
            await manager.register_agent(agent_id, agent_type, name)
        except ValueError:
            pass  # 智能体已存在
    
    logger.info("默认智能体初始化完成")
