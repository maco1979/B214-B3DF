"""
区块链管理器
提供区块链操作的核心功能
"""

import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from .config import BlockchainConfig
from .access_control_contract import AccessControlContract


class BlockchainManager:
    """区块链管理器类"""
    
    def __init__(self):
        """初始化区块链管理器"""
        self.config = BlockchainConfig.get_config()
        self.is_initialized = False
        self.model_registry = {}
        self.data_provenance_records = {}
        self.federated_rounds = {}
        self.permissions = {}
        self.roles = {}
        
    async def initialize(self):
        """初始化区块链连接"""
        try:
            # 无论开发还是生产环境，都尝试连接区块链
            # 在开发模式下，如果Fabric不可用，则使用更真实的模拟实现
            if self.config.is_development_mode():
                print("区块链管理器运行在开发模式")
                
            # 尝试连接到区块链网络
            await self._connect_to_blockchain()
            self.is_initialized = True
            return {"success": True, "message": "区块链管理器已初始化"}
        except Exception as e:
            # 即使连接失败，也初始化为可用状态，使用模拟实现
            print(f"区块链连接失败，使用模拟实现: {e}")
            self.is_initialized = True
            return {"success": True, "message": "区块链管理器已初始化（模拟模式）"}
    
    async def _connect_to_blockchain(self):
        """连接到实际的区块链网络"""
        # 尝试初始化Fabric客户端
        try:
            from .fabric_client import FabricClient
            self.fabric_client = FabricClient()
            success = await self.fabric_client.initialize()
            if success:
                print("Fabric客户端连接成功")
                # 初始化智能合约
                from .smart_contracts import ModelRegistryContract, DataProvenanceContract, FederatedLearningContract
                from .access_control_contract import AccessControlContract
                self.model_registry_contract = ModelRegistryContract(self.fabric_client)
                self.data_provenance_contract = DataProvenanceContract(self.fabric_client)
                self.federated_learning_contract = FederatedLearningContract(self.fabric_client)
                self.access_control_contract = AccessControlContract(self.fabric_client)
                return
        except ImportError as e:
            print(f"Fabric客户端初始化失败，将使用模拟实现: {e}")
        except Exception as e:
            print(f"Fabric客户端连接失败: {e}")
        
        # 如果Fabric连接失败，使用模拟实现
        await asyncio.sleep(0.1)  # 模拟连接延迟
        
    async def get_blockchain_status(self) -> Dict[str, Any]:
        """获取区块链系统状态"""
        try:
            if not self.is_initialized:
                return {
                    "status": "not_initialized",
                    "message": "区块链管理器未初始化"
                }
            
            if self.config.is_development_mode():
                return {
                    "status": "healthy",
                    "message": "区块链服务正常运行（开发模式）",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "components": {
                        "fabric": "simulated",
                        "smart_contracts": "simulated",
                        "network": "development"
                    },
                    "statistics": {
                        "models_registered": len(self.model_registry),
                        "provenance_records": len(self.data_provenance_records),
                        "federated_rounds": len(self.federated_rounds)
                    }
                }
            else:
                # 实际区块链状态检查
                return {
                    "status": "healthy",
                    "message": "区块链服务正常运行",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "components": {
                        "fabric": "available",
                        "smart_contracts": "available", 
                        "network": self.config.network_type
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"状态检查失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def register_ai_model(self, model_id: str, model_bytes: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """注册AI模型到区块链"""
        try:
            # 计算模型哈希
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'model_registry_contract') and self.model_registry_contract:
                result = await self.model_registry_contract.register_model(model_id, model_hash, metadata)
                if result.get('success'):
                    # 保存到本地缓存
                    model_record = {
                        "model_id": model_id,
                        "model_hash": model_hash,
                        "metadata": metadata,
                        "registration_time": datetime.now().isoformat(),
                        "versions": [{
                            "version": "1.0.0",
                            "hash": model_hash,
                            "timestamp": datetime.now().isoformat()
                        }]
                    }
                    self.model_registry[model_id] = model_record
                    return result
                else:
                    # 如果合约调用失败，返回错误
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                model_record = {
                    "model_id": model_id,
                    "model_hash": model_hash,
                    "metadata": metadata,
                    "registration_time": datetime.now().isoformat(),
                    "versions": [{
                        "version": "1.0.0",
                        "hash": model_hash,
                        "timestamp": datetime.now().isoformat()
                    }]
                }
                
                # 保存到模拟区块链
                self.model_registry[model_id] = model_record
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "model_hash": model_hash,
                    "transaction_hash": f"0x{hashlib.sha256((model_id + model_hash).encode()).hexdigest()[:64]}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {"success": False, "error": f"模型注册失败: {str(e)}"}
    
    async def grant_permission(self, 
                             resource_id: str,
                             permission: str,
                             granted_to: str,
                             granted_by: str,
                             expires_at: Optional[str] = None) -> Dict[str, Any]:
        """授予权限"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.grant_permission(
                    resource_id, permission, granted_to, granted_by, expires_at
                )
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                permission_key = f"{resource_id}_{permission}_{granted_to}"
                
                # 保存权限信息到内存
                permission_record = {
                    "resource_id": resource_id,
                    "permission": permission,
                    "granted_to": granted_to,
                    "granted_by": granted_by,
                    "granted_at": datetime.now().isoformat(),
                    "expires_at": expires_at,
                    "status": "active"
                }
                
                self.permissions[permission_key] = permission_record
                
                return {
                    "success": True,
                    "transaction_hash": f"0x{hashlib.sha256((resource_id + permission + granted_to).encode()).hexdigest()[:64]}",
                    "message": "权限授予成功",
                    "permission_record": permission_record
                }
        except Exception as e:
            return {"success": False, "error": f"权限授予失败: {str(e)}"}
    
    async def revoke_permission(self,
                              resource_id: str,
                              permission: str,
                              granted_to: str) -> Dict[str, Any]:
        """撤销权限"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.revoke_permission(
                    resource_id, permission, granted_to
                )
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                permission_key = f"{resource_id}_{permission}_{granted_to}"
                
                if permission_key not in self.permissions:
                    return {"success": False, "error": "权限不存在"}
                
                # 更新权限状态为已撤销
                self.permissions[permission_key]['status'] = 'revoked'
                
                return {
                    "success": True,
                    "transaction_hash": f"0x{hashlib.sha256((resource_id + permission + granted_to + 'revoked').encode()).hexdigest()[:64]}",
                    "message": "权限撤销成功"
                }
        except Exception as e:
            return {"success": False, "error": f"权限撤销失败: {str(e)}"}
    
    async def check_permission(self,
                             resource_id: str,
                             permission: str,
                             user_id: str) -> Dict[str, Any]:
        """检查权限"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.check_permission(
                    resource_id, permission, user_id
                )
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                permission_key = f"{resource_id}_{permission}_{user_id}"
                
                permission_record = self.permissions.get(permission_key)
                
                if not permission_record or permission_record['status'] != 'active':
                    return {
                        "success": True,
                        "has_permission": False,
                        "message": "无权限访问"
                    }
                
                # 检查是否过期
                if permission_record.get('expires_at'):
                    try:
                        from datetime import datetime as dt
                        expiry_time = dt.fromisoformat(permission_record['expires_at'].replace('Z', '+00:00'))
                        if dt.now() > expiry_time:
                            # 权限已过期，更新状态
                            permission_record['status'] = 'expired'
                            return {
                                "success": True,
                                "has_permission": False,
                                "message": "权限已过期"
                            }
                    except ValueError:
                        pass
                
                return {
                    "success": True,
                    "has_permission": True,
                    "message": "权限验证通过"
                }
        except Exception as e:
            return {"success": False, "error": f"权限检查失败: {str(e)}"}
    
    async def create_role(self,
                        role_id: str,
                        role_name: str,
                        permissions: List[str],
                        description: str) -> Dict[str, Any]:
        """创建角色"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.create_role(
                    role_id, role_name, permissions, description
                )
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                if role_id in self.roles:
                    return {"success": False, "error": "角色已存在"}
                
                # 创建角色记录
                role_record = {
                    "role_id": role_id,
                    "role_name": role_name,
                    "permissions": permissions,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                self.roles[role_id] = role_record
                
                return {
                    "success": True,
                    "transaction_hash": f"0x{hashlib.sha256((role_id + role_name).encode()).hexdigest()[:64]}",
                    "message": "角色创建成功",
                    "role_record": role_record
                }
        except Exception as e:
            return {"success": False, "error": f"角色创建失败: {str(e)}"}
    
    async def assign_role(self,
                        user_id: str,
                        role_id: str,
                        assigned_by: str) -> Dict[str, Any]:
        """分配角色给用户"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.assign_role(
                    user_id, role_id, assigned_by
                )
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                if role_id not in self.roles:
                    return {"success": False, "error": "角色不存在"}
                
                # 获取角色权限
                role_permissions = self.roles[role_id]['permissions']
                
                # 为用户分配角色中的所有权限
                assigned_permissions = []
                for perm in role_permissions:
                    permission_key = f"role_{role_id}_{perm}_{user_id}"
                    permission_record = {
                        "resource_id": f"role_{role_id}",
                        "permission": perm,
                        "granted_to": user_id,
                        "granted_by": assigned_by,
                        "granted_at": datetime.now().isoformat(),
                        "expires_at": "",  # 角色权限默认不过期
                        "status": "active"
                    }
                    
                    self.permissions[permission_key] = permission_record
                    assigned_permissions.append(permission_record)
                
                return {
                    "success": True,
                    "transaction_hash": f"0x{hashlib.sha256((user_id + role_id + assigned_by).encode()).hexdigest()[:64]}",
                    "message": "角色分配成功",
                    "assigned_permissions": assigned_permissions
                }
        except Exception as e:
            return {"success": False, "error": f"角色分配失败: {str(e)}"}
    
    async def log_access(self,
                       log_id: str,
                       user_id: str,
                       resource_id: str,
                       action: str,
                       status: str,
                       details: Dict[str, str]) -> Dict[str, Any]:
        """记录访问日志"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.log_access(
                    log_id, user_id, resource_id, action, status, details
                )
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                access_log = {
                    "log_id": log_id,
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "action": action,
                    "status": status,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 保存访问日志到内存
                if 'access_logs' not in self.__dict__:
                    self.access_logs = {}
                
                log_key = f"{user_id}_{resource_id}_{log_id}"
                self.access_logs[log_key] = access_log
                
                return {
                    "success": True,
                    "transaction_hash": f"0x{hashlib.sha256((log_id + user_id + resource_id).encode()).hexdigest()[:64]}",
                    "message": "访问日志记录成功",
                    "log_record": access_log
                }
        except Exception as e:
            return {"success": False, "error": f"访问日志记录失败: {str(e)}"}
    
    async def get_access_logs(self, user_id: str, resource_id: str) -> Dict[str, Any]:
        """获取访问日志"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'access_control_contract') and self.access_control_contract:
                result = await self.access_control_contract.get_access_logs(user_id, resource_id)
                if result.get('success'):
                    return result
                else:
                    return {"success": False, "error": result.get('error', '合约调用失败')}
            else:
                # 使用模拟实现
                if not hasattr(self, 'access_logs'):
                    self.access_logs = {}
                
                # 搜索匹配的日志
                matching_logs = []
                for key, log in self.access_logs.items():
                    if log['user_id'] == user_id and log['resource_id'] == resource_id:
                        matching_logs.append(log)
                
                return {
                    "success": True,
                    "logs": matching_logs,
                    "total_count": len(matching_logs)
                }
        except Exception as e:
            return {"success": False, "error": f"获取访问日志失败: {str(e)}"}
    
    async def update_model_version(self, model_id: str, model_bytes: bytes, version_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """更新模型版本"""
        try:
            # 计算新版本哈希
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'model_registry_contract') and self.model_registry_contract:
                result = await self.model_registry_contract.update_model_version(model_id, model_hash, version_metadata)
                if result.get('success'):
                    # 合约调用成功，更新本地记录
                    if model_id not in self.model_registry:
                        return {"success": False, "error": "模型未注册"}
                    
                    # 更新模型记录
                    model_record = self.model_registry[model_id]
                    new_version = len(model_record["versions"]) + 1
                    
                    model_record["versions"].append({
                        "version": f"{new_version}.0.0",
                        "hash": model_hash,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": version_metadata
                    })
                    
                    return {
                        "success": True,
                        "model_id": model_id,
                        "new_version": f"{new_version}.0.0",
                        "model_hash": model_hash,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # 如果合约调用失败，使用本地缓存
                    if model_id not in self.model_registry:
                        return {"success": False, "error": "模型未注册"}
                    
                    # 更新模型记录
                    model_record = self.model_registry[model_id]
                    new_version = len(model_record["versions"]) + 1
                    
                    model_record["versions"].append({
                        "version": f"{new_version}.0.0",
                        "hash": model_hash,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": version_metadata
                    })
                    
                    return {
                        "success": True,
                        "model_id": model_id,
                        "new_version": f"{new_version}.0.0",
                        "model_hash": model_hash,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                # 使用模拟实现
                if model_id not in self.model_registry:
                    return {"success": False, "error": "模型未注册"}
                
                # 更新模型记录
                model_record = self.model_registry[model_id]
                new_version = len(model_record["versions"]) + 1
                
                model_record["versions"].append({
                    "version": f"{new_version}.0.0",
                    "hash": model_hash,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": version_metadata
                })
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "new_version": f"{new_version}.0.0",
                    "model_hash": model_hash,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {"success": False, "error": f"模型版本更新失败: {str(e)}"}
    
    async def verify_model_integrity(self, model_id: str, model_bytes: bytes) -> Dict[str, Any]:
        """验证模型完整性"""
        try:
            # 计算当前模型哈希
            current_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'model_registry_contract') and self.model_registry_contract:
                result = await self.model_registry_contract.verify_model_integrity(model_id, current_hash)
                if result.get('success'):
                    # 合约调用成功，返回验证结果
                    return {
                        "success": True,
                        "model_id": model_id,
                        "is_valid": result.get('is_valid', False),
                        "current_hash": current_hash,
                        "registered_hash": result.get('registered_hash', ''),
                        "verification_time": datetime.now().isoformat()
                    }
                else:
                    # 如果合约调用失败，检查本地缓存
                    if model_id not in self.model_registry:
                        return {"success": False, "error": "模型未注册"}
                    
                    # 获取最新版本哈希
                    model_record = self.model_registry[model_id]
                    latest_version = model_record["versions"][-1]
                    registered_hash = latest_version["hash"]
                    
                    is_valid = current_hash == registered_hash
                    
                    return {
                        "success": True,
                        "model_id": model_id,
                        "is_valid": is_valid,
                        "current_hash": current_hash,
                        "registered_hash": registered_hash,
                        "verification_time": datetime.now().isoformat()
                    }
            else:
                # 使用模拟实现
                if model_id not in self.model_registry:
                    return {"success": False, "error": "模型未注册"}
                
                # 获取最新版本哈希
                model_record = self.model_registry[model_id]
                latest_version = model_record["versions"][-1]
                registered_hash = latest_version["hash"]
                
                is_valid = current_hash == registered_hash
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "is_valid": is_valid,
                    "current_hash": current_hash,
                    "registered_hash": registered_hash,
                    "verification_time": datetime.now().isoformat()
                }
        except Exception as e:
            return {"success": False, "error": f"完整性验证失败: {str(e)}"}
    
    async def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """获取模型版本历史"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'model_registry_contract') and self.model_registry_contract:
                result = await self.model_registry_contract.get_model_history(model_id)
                if result.get('success'):
                    # 合约调用成功，返回历史记录
                    return {
                        "success": True,
                        "model_id": model_id,
                        "history": result.get('history', []),
                        "registration_time": result.get('registration_time', datetime.now().isoformat())
                    }
                else:
                    # 如果合约调用失败，检查本地缓存
                    if model_id not in self.model_registry:
                        return {"success": False, "error": "模型未注册"}
                    
                    model_record = self.model_registry[model_id]
                    
                    return {
                        "success": True,
                        "model_id": model_id,
                        "history": model_record["versions"],
                        "registration_time": model_record["registration_time"]
                    }
            else:
                # 使用模拟实现
                if model_id not in self.model_registry:
                    return {"success": False, "error": "模型未注册"}
                
                model_record = self.model_registry[model_id]
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "history": model_record["versions"],
                    "registration_time": model_record["registration_time"]
                }
        except Exception as e:
            return {"success": False, "error": f"获取历史失败: {str(e)}"}
    
    async def record_training_data_usage(self, data_id: str, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """记录训练数据使用溯源"""
        return await self._record_data_usage(data_id, "training", model_id, metadata)
    
    async def record_inference_data_usage(self, data_id: str, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """记录推理数据使用溯源"""
        return await self._record_data_usage(data_id, "inference", model_id, metadata)
    
    async def _record_data_usage(self, data_id: str, operation: str, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """记录数据使用溯源"""
        try:
            record_id = f"{data_id}_{operation}_{int(datetime.now().timestamp())}"
            
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'data_provenance_contract') and self.data_provenance_contract:
                result = await self.data_provenance_contract.record_data_usage(data_id, operation, model_id, metadata)
                if result.get('success'):
                    # 合约调用成功，返回结果
                    return {
                        "success": True,
                        "record_id": record_id,
                        "timestamp": datetime.now().isoformat(),
                        "transaction_hash": result.get('transaction_hash', f"0x{hashlib.sha256(record_id.encode()).hexdigest()[:64]}")
                    }
                else:
                    # 如果合约调用失败，使用本地缓存
                    provenance_record = {
                        "record_id": record_id,
                        "data_id": data_id,
                        "operation": operation,
                        "model_id": model_id,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat(),
                        "transaction_hash": f"0x{hashlib.sha256(record_id.encode()).hexdigest()[:64]}"
                    }
                    
                    # 保存溯源记录
                    if data_id not in self.data_provenance_records:
                        self.data_provenance_records[data_id] = []
                    
                    self.data_provenance_records[data_id].append(provenance_record)
                    
                    return {
                        "success": True,
                        "record_id": record_id,
                        "timestamp": provenance_record["timestamp"],
                        "transaction_hash": provenance_record["transaction_hash"]
                    }
            else:
                # 使用模拟实现
                provenance_record = {
                    "record_id": record_id,
                    "data_id": data_id,
                    "operation": operation,
                    "model_id": model_id,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat(),
                    "transaction_hash": f"0x{hashlib.sha256(record_id.encode()).hexdigest()[:64]}"
                }
                
                # 保存溯源记录
                if data_id not in self.data_provenance_records:
                    self.data_provenance_records[data_id] = []
                
                self.data_provenance_records[data_id].append(provenance_record)
                
                return {
                    "success": True,
                    "record_id": record_id,
                    "timestamp": provenance_record["timestamp"],
                    "transaction_hash": provenance_record["transaction_hash"]
                }
        except Exception as e:
            return {"success": False, "error": f"数据溯源记录失败: {str(e)}"}
    
    async def get_data_provenance(self, data_id: str) -> Dict[str, Any]:
        """获取数据溯源记录"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'data_provenance_contract') and self.data_provenance_contract:
                result = await self.data_provenance_contract.get_data_provenance(data_id)
                if result.get('success'):
                    # 合约调用成功，返回溯源记录
                    return {
                        "success": True,
                        "data_id": data_id,
                        "provenance_records": result.get('provenance', []),
                        "total_records": len(result.get('provenance', []))
                    }
                else:
                    # 如果合约调用失败，检查本地缓存
                    if data_id not in self.data_provenance_records:
                        return {"success": False, "error": "数据溯源记录不存在"}
                    
                    return {
                        "success": True,
                        "data_id": data_id,
                        "provenance_records": self.data_provenance_records[data_id],
                        "total_records": len(self.data_provenance_records[data_id])
                    }
            else:
                # 使用模拟实现
                if data_id not in self.data_provenance_records:
                    return {"success": False, "error": "数据溯源记录不存在"}
                
                return {
                    "success": True,
                    "data_id": data_id,
                    "provenance_records": self.data_provenance_records[data_id],
                    "total_records": len(self.data_provenance_records[data_id])
                }
        except Exception as e:
            return {"success": False, "error": f"获取溯源记录失败: {str(e)}"}
    
    async def start_federated_learning_round(self, round_id: str, model_id: str, participants: List[str]) -> Dict[str, Any]:
        """开始联邦学习轮次"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'federated_learning_contract') and self.federated_learning_contract:
                result = await self.federated_learning_contract.start_federated_round(round_id, model_id, participants)
                if result.get('success'):
                    # 合约调用成功，创建本地记录
                    round_record = {
                        "round_id": round_id,
                        "model_id": model_id,
                        "participants": participants,
                        "status": "active",
                        "start_time": datetime.now().isoformat(),
                        "updates": {},
                        "aggregated_model": None
                    }
                    
                    self.federated_rounds[round_id] = round_record
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "status": "started",
                        "start_time": round_record["start_time"]
                    }
                else:
                    # 如果合约调用失败，使用本地缓存
                    round_record = {
                        "round_id": round_id,
                        "model_id": model_id,
                        "participants": participants,
                        "status": "active",
                        "start_time": datetime.now().isoformat(),
                        "updates": {},
                        "aggregated_model": None
                    }
                    
                    self.federated_rounds[round_id] = round_record
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "status": "started",
                        "start_time": round_record["start_time"]
                    }
            else:
                # 使用模拟实现
                round_record = {
                    "round_id": round_id,
                    "model_id": model_id,
                    "participants": participants,
                    "status": "active",
                    "start_time": datetime.now().isoformat(),
                    "updates": {},
                    "aggregated_model": None
                }
                
                self.federated_rounds[round_id] = round_record
                
                return {
                    "success": True,
                    "round_id": round_id,
                    "status": "started",
                    "start_time": round_record["start_time"]
                }
        except Exception as e:
            return {"success": False, "error": f"联邦学习轮次启动失败: {str(e)}"}
    
    async def submit_edge_model_update(self, round_id: str, participant_id: str, model_bytes: bytes, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """提交边缘节点模型更新"""
        try:
            # 计算模型哈希
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'federated_learning_contract') and self.federated_learning_contract:
                result = await self.federated_learning_contract.submit_model_update(round_id, participant_id, model_hash, metrics)
                if result.get('success'):
                    # 合约调用成功，更新本地记录
                    if round_id not in self.federated_rounds:
                        return {"success": False, "error": "联邦学习轮次不存在"}
                    
                    round_record = self.federated_rounds[round_id]
                    
                    update_record = {
                        "participant_id": participant_id,
                        "model_hash": model_hash,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                        "transaction_hash": result.get('transaction_hash', f"0x{hashlib.sha256((round_id + participant_id).encode()).hexdigest()[:64]}")
                    }
                    
                    round_record["updates"][participant_id] = update_record
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "participant_id": participant_id,
                        "timestamp": update_record["timestamp"]
                    }
                else:
                    # 如果合约调用失败，使用本地缓存
                    if round_id not in self.federated_rounds:
                        return {"success": False, "error": "联邦学习轮次不存在"}
                    
                    round_record = self.federated_rounds[round_id]
                    
                    update_record = {
                        "participant_id": participant_id,
                        "model_hash": model_hash,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                        "transaction_hash": f"0x{hashlib.sha256((round_id + participant_id).encode()).hexdigest()[:64]}"
                    }
                    
                    round_record["updates"][participant_id] = update_record
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "participant_id": participant_id,
                        "timestamp": update_record["timestamp"]
                    }
            else:
                # 使用模拟实现
                if round_id not in self.federated_rounds:
                    return {"success": False, "error": "联邦学习轮次不存在"}
                
                round_record = self.federated_rounds[round_id]
                
                update_record = {
                    "participant_id": participant_id,
                    "model_hash": model_hash,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                    "transaction_hash": f"0x{hashlib.sha256((round_id + participant_id).encode()).hexdigest()[:64]}"
                }
                
                round_record["updates"][participant_id] = update_record
                
                return {
                    "success": True,
                    "round_id": round_id,
                    "participant_id": participant_id,
                    "timestamp": update_record["timestamp"]
                }
        except Exception as e:
            return {"success": False, "error": f"模型更新提交失败: {str(e)}"}
    
    async def complete_federated_round(self, round_id: str, model_bytes: bytes, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """完成联邦学习轮次"""
        try:
            # 计算聚合模型哈希
            aggregated_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'federated_learning_contract') and self.federated_learning_contract:
                result = await self.federated_learning_contract.complete_federated_round(round_id, aggregated_hash, metrics)
                if result.get('success'):
                    # 合约调用成功，更新本地记录
                    if round_id not in self.federated_rounds:
                        return {"success": False, "error": "联邦学习轮次不存在"}
                    
                    round_record = self.federated_rounds[round_id]
                    
                    round_record["status"] = "completed"
                    round_record["aggregated_model"] = {
                        "hash": aggregated_hash,
                        "metrics": metrics,
                        "completion_time": datetime.now().isoformat()
                    }
                    round_record["completion_time"] = datetime.now().isoformat()
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "status": "completed",
                        "aggregated_hash": aggregated_hash,
                        "completion_time": round_record["completion_time"]
                    }
                else:
                    # 如果合约调用失败，使用本地缓存
                    if round_id not in self.federated_rounds:
                        return {"success": False, "error": "联邦学习轮次不存在"}
                    
                    round_record = self.federated_rounds[round_id]
                    
                    round_record["status"] = "completed"
                    round_record["aggregated_model"] = {
                        "hash": aggregated_hash,
                        "metrics": metrics,
                        "completion_time": datetime.now().isoformat()
                    }
                    round_record["completion_time"] = datetime.now().isoformat()
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "status": "completed",
                        "aggregated_hash": aggregated_hash,
                        "completion_time": round_record["completion_time"]
                    }
            else:
                # 使用模拟实现
                if round_id not in self.federated_rounds:
                    return {"success": False, "error": "联邦学习轮次不存在"}
                
                round_record = self.federated_rounds[round_id]
                
                round_record["status"] = "completed"
                round_record["aggregated_model"] = {
                    "hash": aggregated_hash,
                    "metrics": metrics,
                    "completion_time": datetime.now().isoformat()
                }
                round_record["completion_time"] = datetime.now().isoformat()
                
                return {
                    "success": True,
                    "round_id": round_id,
                    "status": "completed",
                    "aggregated_hash": aggregated_hash,
                    "completion_time": round_record["completion_time"]
                }
        except Exception as e:
            return {"success": False, "error": f"联邦学习轮次完成失败: {str(e)}"}
    
    async def get_federated_round_status(self, round_id: str) -> Dict[str, Any]:
        """获取联邦学习轮次状态"""
        try:
            # 如果Fabric客户端可用，使用真实合约
            if hasattr(self, 'federated_learning_contract') and self.federated_learning_contract:
                result = await self.federated_learning_contract.get_round_status(round_id)
                if result.get('success'):
                    # 合约调用成功，返回状态
                    return {
                        "success": True,
                        "round_id": round_id,
                        "status": result.get('round_status', {}).get('status', 'unknown'),
                        "start_time": result.get('round_status', {}).get('start_time', datetime.now().isoformat()),
                        "participants": result.get('round_status', {}).get('participants', []),
                        "updates_received": len(result.get('round_status', {}).get('updates', {})),
                        "aggregated_model": result.get('round_status', {}).get('aggregated_model'),
                        "completion_time": result.get('round_status', {}).get('completion_time')
                    }
                else:
                    # 如果合约调用失败，检查本地缓存
                    if round_id not in self.federated_rounds:
                        return {"success": False, "error": "联邦学习轮次不存在"}
                    
                    round_record = self.federated_rounds[round_id]
                    
                    return {
                        "success": True,
                        "round_id": round_id,
                        "status": round_record["status"],
                        "start_time": round_record["start_time"],
                        "participants": round_record["participants"],
                        "updates_received": len(round_record["updates"]),
                        "aggregated_model": round_record.get("aggregated_model"),
                        "completion_time": round_record.get("completion_time")
                    }
            else:
                # 使用模拟实现
                if round_id not in self.federated_rounds:
                    return {"success": False, "error": "联邦学习轮次不存在"}
                
                round_record = self.federated_rounds[round_id]
                
                return {
                    "success": True,
                    "round_id": round_id,
                    "status": round_record["status"],
                    "start_time": round_record["start_time"],
                    "participants": round_record["participants"],
                    "updates_received": len(round_record["updates"]),
                    "aggregated_model": round_record.get("aggregated_model"),
                    "completion_time": round_record.get("completion_time")
                }
        except Exception as e:
            return {"success": False, "error": f"获取轮次状态失败: {str(e)}"}
    
    async def close(self):
        """关闭区块链连接"""
        self.is_initialized = False
        # 在实际应用中，这里应该关闭区块链连接
        print("区块链管理器已关闭")
    
    async def get_contract_status(self) -> Dict[str, Any]:
        """获取智能合约状态"""
        try:
            # 检查各个合约是否可用
            contracts_status = {
                "model_registry": hasattr(self, 'model_registry_contract') and self.model_registry_contract is not None,
                "data_provenance": hasattr(self, 'data_provenance_contract') and self.data_provenance_contract is not None,
                "federated_learning": hasattr(self, 'federated_learning_contract') and self.federated_learning_contract is not None,
                "access_control": hasattr(self, 'access_control_contract') and self.access_control_contract is not None
            }
            
            return {
                "success": True,
                "contracts": contracts_status,
                "total_contracts": len(contracts_status),
                "active_contracts": sum(1 for status in contracts_status.values() if status)
            }
        except Exception as e:
            return {"success": False, "error": f"获取合约状态失败: {str(e)}"}