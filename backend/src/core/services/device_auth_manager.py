"""
设备认证管理器 - 管理设备认证和授权
支持设备注册、认证、授权和安全连接
"""
import asyncio
import hashlib
import secrets
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """设备状态枚举"""
    PENDING = "pending"      # 待认证
    ACTIVE = "active"        # 活跃
    INACTIVE = "inactive"    # 非活跃
    BLOCKED = "blocked"      # 已封禁
    EXPIRED = "expired"      # 已过期


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_name: str
    device_type: str
    manufacturer: str
    model: str
    firmware_version: str
    serial_number: str
    registration_date: datetime
    last_seen: datetime
    status: DeviceStatus
    public_key: Optional[str] = None
    permissions: Optional[list] = None


class DeviceAuthManager:
    """设备认证管理器"""
    
    def __init__(self, jwt_secret: str = "default_secret_key_for_dev"):
        self.jwt_secret = jwt_secret
        self.devices: Dict[str, DeviceInfo] = {}
        self.device_tokens: Dict[str, str] = {}  # device_id -> token
        self.token_devices: Dict[str, str] = {}  # token -> device_id
        self.registration_codes: Dict[str, str] = {}  # code -> device_id
        self.auth_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5分钟缓存
    
    def generate_registration_code(self, device_id: str) -> str:
        """生成设备注册码"""
        code = secrets.token_urlsafe(16)
        self.registration_codes[code] = device_id
        # 注册码有效期15分钟
        asyncio.get_event_loop().call_later(900, 
                                          lambda: self.registration_codes.pop(code, None))
        return code
    
    def register_device(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """注册新设备"""
        try:
            device_id = device_info.get("device_id")
            if not device_id:
                return {"success": False, "message": "设备ID不能为空"}
            
            if device_id in self.devices:
                return {"success": False, "message": "设备已存在"}
            
            # 创建设备信息
            device = DeviceInfo(
                device_id=device_id,
                device_name=device_info.get("device_name", f"设备_{device_id}"),
                device_type=device_info.get("device_type", "unknown"),
                manufacturer=device_info.get("manufacturer", "unknown"),
                model=device_info.get("model", "unknown"),
                firmware_version=device_info.get("firmware_version", "1.0.0"),
                serial_number=device_info.get("serial_number", ""),
                registration_date=datetime.now(),
                last_seen=datetime.now(),
                status=DeviceStatus.PENDING,
                public_key=device_info.get("public_key"),
                permissions=device_info.get("permissions", ["read"])
            )
            
            self.devices[device_id] = device
            logger.info(f"成功注册设备: {device_id}")
            
            return {
                "success": True,
                "message": "设备注册成功，请使用注册码完成认证",
                "device_id": device_id,
                "registration_code": self.generate_registration_code(device_id)
            }
        except Exception as e:
            logger.error(f"设备注册失败: {str(e)}")
            return {"success": False, "message": f"设备注册失败: {str(e)}"}
    
    def authenticate_device(self, device_id: str, auth_data: Dict[str, Any]) -> Dict[str, Any]:
        """认证设备"""
        try:
            if device_id not in self.devices:
                return {"success": False, "message": "设备不存在"}
            
            device = self.devices[device_id]
            
            # 检查设备状态
            if device.status == DeviceStatus.BLOCKED:
                return {"success": False, "message": "设备已被封禁"}
            
            if device.status == DeviceStatus.EXPIRED:
                return {"success": False, "message": "设备认证已过期"}
            
            # 验证认证数据
            auth_method = auth_data.get("method", "token")
            
            if auth_method == "token":
                token = auth_data.get("token")
                if not token:
                    return {"success": False, "message": "认证令牌不能为空"}
                
                # 验证JWT令牌
                try:
                    payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
                    if payload.get("device_id") != device_id:
                        return {"success": False, "message": "令牌与设备ID不匹配"}
                    
                    # 更新最后活动时间
                    device.last_seen = datetime.now()
                    device.status = DeviceStatus.ACTIVE
                    
                    return {
                        "success": True,
                        "message": "设备认证成功",
                        "device_id": device_id,
                        "permissions": device.permissions,
                        "expires_at": payload.get("exp")
                    }
                except jwt.ExpiredSignatureError:
                    return {"success": False, "message": "认证令牌已过期"}
                except jwt.InvalidTokenError:
                    return {"success": False, "message": "无效的认证令牌"}
            
            elif auth_method == "registration_code":
                code = auth_data.get("code")
                if not code or code not in self.registration_codes:
                    return {"success": False, "message": "无效的注册码"}
                
                registered_device_id = self.registration_codes.pop(code)
                if registered_device_id != device_id:
                    return {"success": False, "message": "注册码与设备ID不匹配"}
                
                # 生成JWT令牌
                expiration = datetime.utcnow() + timedelta(days=30)  # 30天有效期
                token_payload = {
                    "device_id": device_id,
                    "exp": expiration.timestamp(),
                    "iat": datetime.utcnow().timestamp()
                }
                token = jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")
                
                # 更新设备状态
                device.status = DeviceStatus.ACTIVE
                device.last_seen = datetime.now()
                
                # 存储令牌
                self.device_tokens[device_id] = token
                self.token_devices[token] = device_id
                
                return {
                    "success": True,
                    "message": "设备认证成功",
                    "device_id": device_id,
                    "token": token,
                    "permissions": device.permissions,
                    "expires_at": expiration.isoformat()
                }
            
            else:
                return {"success": False, "message": "不支持的认证方式"}
                
        except Exception as e:
            logger.error(f"设备认证失败: {str(e)}")
            return {"success": False, "message": f"设备认证失败: {str(e)}"}
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证认证令牌"""
        # 检查缓存
        if token in self.auth_cache:
            cached = self.auth_cache[token]
            if time.time() < cached.get("expires_at", 0):
                return cached
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            device_id = payload.get("device_id")
            
            if device_id not in self.devices:
                return {"success": False, "message": "设备不存在"}
            
            device = self.devices[device_id]
            if device.status != DeviceStatus.ACTIVE:
                return {"success": False, "message": "设备状态异常"}
            
            # 更新最后活动时间
            device.last_seen = datetime.now()
            
            result = {
                "success": True,
                "device_id": device_id,
                "permissions": device.permissions,
                "expires_at": payload.get("exp")
            }
            
            # 缓存结果
            self.auth_cache[token] = result
            asyncio.get_event_loop().call_later(
                self.cache_ttl, 
                lambda: self.auth_cache.pop(token, None)
            )
            
            return result
        except jwt.ExpiredSignatureError:
            return {"success": False, "message": "认证令牌已过期"}
        except jwt.InvalidTokenError:
            return {"success": False, "message": "无效的认证令牌"}
    
    def revoke_device_token(self, device_id: str) -> Dict[str, Any]:
        """撤销设备令牌"""
        try:
            if device_id in self.device_tokens:
                token = self.device_tokens.pop(device_id)
                self.token_devices.pop(token, None)
                
                if device_id in self.devices:
                    self.devices[device_id].status = DeviceStatus.INACTIVE
                
                logger.info(f"已撤销设备令牌: {device_id}")
                return {
                    "success": True,
                    "message": "设备令牌已撤销",
                    "device_id": device_id
                }
            else:
                return {
                    "success": False,
                    "message": "设备令牌不存在"
                }
        except Exception as e:
            logger.error(f"撤销设备令牌失败: {str(e)}")
            return {"success": False, "message": f"撤销设备令牌失败: {str(e)}"}
    
    def update_device_permissions(self, device_id: str, permissions: list) -> Dict[str, Any]:
        """更新设备权限"""
        if device_id not in self.devices:
            return {"success": False, "message": "设备不存在"}
        
        self.devices[device_id].permissions = permissions
        logger.info(f"已更新设备权限: {device_id}, 权限: {permissions}")
        
        return {
            "success": True,
            "message": "设备权限更新成功",
            "device_id": device_id,
            "permissions": permissions
        }
    
    def block_device(self, device_id: str) -> Dict[str, Any]:
        """封禁设备"""
        if device_id not in self.devices:
            return {"success": False, "message": "设备不存在"}
        
        self.devices[device_id].status = DeviceStatus.BLOCKED
        # 撤销令牌
        self.revoke_device_token(device_id)
        
        logger.info(f"已封禁设备: {device_id}")
        return {
            "success": True,
            "message": "设备已封禁",
            "device_id": device_id
        }
    
    def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """获取设备信息"""
        return self.devices.get(device_id)
    
    def list_devices(self) -> Dict[str, Any]:
        """列出所有设备"""
        devices_info = []
        for device_id, device in self.devices.items():
            devices_info.append({
                "device_id": device.device_id,
                "device_name": device.device_name,
                "device_type": device.device_type,
                "manufacturer": device.manufacturer,
                "model": device.model,
                "firmware_version": device.firmware_version,
                "registration_date": device.registration_date.isoformat(),
                "last_seen": device.last_seen.isoformat(),
                "status": device.status.value,
                "permissions": device.permissions
            })
        
        return {
            "success": True,
            "devices": devices_info,
            "total_count": len(devices_info)
        }


# 全局设备认证管理器实例
device_auth_manager = DeviceAuthManager()