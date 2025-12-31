"""
权限管理智能合约封装类
提供基于区块链的权限管理功能
"""
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime


class AccessControlContract:
    """权限管理智能合约封装"""
    
    def __init__(self, fabric_client):
        self.client = fabric_client
        self.contract_name = "accesscontrol"
    
    async def grant_permission(self, 
                             resource_id: str,
                             permission: str,
                             granted_to: str,
                             granted_by: str,
                             expires_at: Optional[str] = None) -> Dict[str, Any]:
        """授予权限"""
        args = [
            resource_id,
            permission,
            granted_to,
            granted_by,
            expires_at or ""
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "grantPermission",
            args
        )
    
    async def revoke_permission(self,
                              resource_id: str,
                              permission: str,
                              granted_to: str) -> Dict[str, Any]:
        """撤销权限"""
        args = [
            resource_id,
            permission,
            granted_to
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "revokePermission",
            args
        )
    
    async def check_permission(self,
                             resource_id: str,
                             permission: str,
                             user_id: str) -> Dict[str, Any]:
        """检查权限"""
        args = [
            resource_id,
            permission,
            user_id
        ]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "checkPermission",
            args
        )
        
        if result['success'] and result['data']:
            try:
                has_permission = json.loads(result['data'])
                return {
                    'success': True,
                    'has_permission': has_permission,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                return result
        
        return result
    
    async def create_role(self,
                        role_id: str,
                        role_name: str,
                        permissions: List[str],
                        description: str) -> Dict[str, Any]:
        """创建角色"""
        args = [
            role_id,
            role_name,
            json.dumps(permissions),
            description
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "createRole",
            args
        )
    
    async def assign_role(self,
                        user_id: str,
                        role_id: str,
                        assigned_by: str) -> Dict[str, Any]:
        """分配角色给用户"""
        args = [
            user_id,
            role_id,
            assigned_by
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "assignRole",
            args
        )
    
    async def log_access(self,
                       log_id: str,
                       user_id: str,
                       resource_id: str,
                       action: str,
                       status: str,
                       details: Dict[str, str]) -> Dict[str, Any]:
        """记录访问日志"""
        args = [
            log_id,
            user_id,
            resource_id,
            action,
            status,
            json.dumps(details)
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "logAccess",
            args
        )
    
    async def get_access_logs(self,
                            user_id: str,
                            resource_id: str) -> Dict[str, Any]:
        """获取访问日志"""
        args = [
            user_id,
            resource_id
        ]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getAccessLogs",
            args
        )
        
        if result['success'] and result['data']:
            try:
                logs = json.loads(result['data'])
                return {
                    'success': True,
                    'logs': logs,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                return result
        
        return result
    
    async def get_permission(self,
                           resource_id: str,
                           permission: str,
                           granted_to: str) -> Dict[str, Any]:
        """获取权限信息"""
        args = [
            resource_id,
            permission,
            granted_to
        ]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getPermission",
            args
        )
        
        if result['success'] and result['data']:
            try:
                permission_info = json.loads(result['data'])
                return {
                    'success': True,
                    'permission': permission_info,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                return result
        
        return result
    
    async def get_role(self, role_id: str) -> Dict[str, Any]:
        """获取角色信息"""
        args = [role_id]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getRole",
            args
        )
        
        if result['success'] and result['data']:
            try:
                role_info = json.loads(result['data'])
                return {
                    'success': True,
                    'role': role_info,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                return result
        
        return result


# 权限管理工具类
class AccessControlManager:
    """权限管理工具类"""
    
    def __init__(self, fabric_client):
        self.contract = AccessControlContract(fabric_client)
    
    async def has_access(self, user_id: str, resource_id: str, permission: str) -> bool:
        """检查用户是否有访问权限"""
        try:
            result = await self.contract.check_permission(resource_id, permission, user_id)
            return result.get('success', False) and result.get('has_permission', False)
        except Exception:
            return False
    
    async def grant_access(self, 
                          resource_id: str,
                          permission: str,
                          user_id: str,
                          granted_by: str,
                          expires_at: Optional[str] = None) -> Dict[str, Any]:
        """授予权限"""
        return await self.contract.grant_permission(
            resource_id, permission, user_id, granted_by, expires_at
        )
    
    async def revoke_access(self,
                           resource_id: str,
                           permission: str,
                           user_id: str) -> Dict[str, Any]:
        """撤销权限"""
        return await self.contract.revoke_permission(
            resource_id, permission, user_id
        )
    
    async def create_and_assign_role(self,
                                   role_id: str,
                                   role_name: str,
                                   permissions: List[str],
                                   description: str,
                                   user_id: str,
                                   assigned_by: str) -> Dict[str, Any]:
        """创建角色并分配给用户"""
        # 首先创建角色
        role_result = await self.contract.create_role(
            role_id, role_name, permissions, description
        )
        
        if not role_result.get('success'):
            return role_result
        
        # 然后分配角色给用户
        assign_result = await self.contract.assign_role(user_id, role_id, assigned_by)
        
        return {
            'success': assign_result.get('success', False),
            'role_created': role_result,
            'role_assigned': assign_result
        }
    
    async def log_user_access(self,
                            user_id: str,
                            resource_id: str,
                            action: str,
                            status: str,
                            details: Dict[str, str] = None) -> Dict[str, Any]:
        """记录用户访问日志"""
        log_id = f"access_{hashlib.sha256((user_id + resource_id + action + status).encode()).hexdigest()[:16]}"
        details = details or {}
        
        return await self.contract.log_access(
            log_id, user_id, resource_id, action, status, details
        )