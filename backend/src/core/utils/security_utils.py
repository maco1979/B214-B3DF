"""安全工具模块
提供敏感数据加密、密码哈希、数据签名等安全功能
"""

import os
import hashlib
import base64
import logging
import secrets
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# 后端
backend = default_backend()

# 安全配置
SECURITY_CONFIG = {
    "password_salt_rounds": 100000,
    "password_hash_algorithm": "sha256",
    "encryption_algorithm": "AES-256-CBC",
    "hmac_algorithm": "HMAC-SHA256",
    "key_length": 32,  # AES-256需要32字节密钥
    "iv_length": 16,   # CBC模式需要16字节IV
    "salt_length": 16, # 盐值长度
    "kdf_iterations": 100000,
    "min_token_length": 32
}


# 敏感数据分类
SENSITIVE_DATA_CATEGORIES = {
    # 高敏感数据 - 必须加密存储
    "high": {
        "password", "passwd", "pwd",
        "token", "access_token", "refresh_token",
        "api_key", "api_secret",
        "secret", "key", "private_key",
        "credit_card", "cc", "card_number",
        "social_security_number", "ssn",
        "id_card", "id_number", "identity",
        "wifi_password", "password_hash"
    },
    
    # 中敏感数据 - 建议加密存储
    "medium": {
        "phone", "phone_number",
        "email", "email_address",
        "address", "location", "coordinates",
        "device_id", "serial_number",
        "mac_address", "ip_address"
    },
    
    # 低敏感数据 - 可以脱敏后存储
    "low": {
        "battery", "signal", "wifi_ssid",
        "device_name", "manufacturer",
        "model", "firmware_version"
    }
}


class SecurityUtils:
    """安全工具类"""
    
    def __init__(self):
        """初始化安全工具"""
        # 从环境变量获取加密密钥，默认使用随机生成的密钥（仅开发环境）
        self.encryption_key = os.environ.get("ENCRYPTION_KEY")
        if not self.encryption_key:
            # 生成随机密钥（仅开发环境使用）
            self.encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            logger.warning("使用随机生成的加密密钥，生产环境请设置ENCRYPTION_KEY环境变量")
        
        # 从环境变量获取盐值，默认使用随机生成的盐值（仅开发环境）
        self.salt = os.environ.get("ENCRYPTION_SALT")
        if not self.salt:
            self.salt = base64.urlsafe_b64encode(os.urandom(16)).decode()
            logger.warning("使用随机生成的盐值，生产环境请设置ENCRYPTION_SALT环境变量")
        
        # 从环境变量获取HMAC密钥，默认使用加密密钥的衍生密钥
        self.hmac_key = os.environ.get("HMAC_KEY")
        if not self.hmac_key:
            # 从加密密钥衍生HMAC密钥
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=SECURITY_CONFIG["key_length"],
                salt=self.salt.encode(),
                iterations=1000,
                backend=backend
            )
            self.hmac_key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode())).decode()
            logger.warning("使用衍生的HMAC密钥，生产环境请设置HMAC_KEY环境变量")
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """哈希密码
        
        Args:
            password: 原始密码
            salt: 盐值，默认生成随机盐值
        
        Returns:
            包含哈希密码、盐值和算法的字典
        """
        if not salt:
            salt = base64.urlsafe_b64encode(os.urandom(SECURITY_CONFIG["salt_length"])).decode()
        
        # 使用PBKDF2进行密码哈希
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=SECURITY_CONFIG["key_length"],
            salt=salt.encode(),
            iterations=SECURITY_CONFIG["kdf_iterations"],
            backend=backend
        )
        
        hashed_password = base64.urlsafe_b64encode(kdf.derive(password.encode())).decode()
        
        return {
            "hashed_password": hashed_password,
            "salt": salt,
            "algorithm": "pbkdf2_sha256",
            "iterations": SECURITY_CONFIG["kdf_iterations"],
            "version": "1.0"
        }
    
    def verify_password(self, password: str, hashed_password: str, salt: str, iterations: int = 100000) -> bool:
        """验证密码
        
        Args:
            password: 原始密码
            hashed_password: 哈希后的密码
            salt: 盐值
            iterations: 迭代次数
        
        Returns:
            密码是否匹配
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=SECURITY_CONFIG["key_length"],
                salt=salt.encode(),
                iterations=iterations,
                backend=backend
            )
            
            # 验证密码
            kdf.verify(password.encode(), base64.urlsafe_b64decode(hashed_password))
            return True
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False
    
    def encrypt_data(self, data: str) -> Dict[str, str]:
        """加密数据
        
        Args:
            data: 要加密的数据
        
        Returns:
            包含加密数据、IV和算法的字典
        """
        # 生成随机IV
        iv = os.urandom(SECURITY_CONFIG["iv_length"])
        
        # 创建加密器
        cipher = Cipher(
            algorithms.AES(base64.urlsafe_b64decode(self.encryption_key)),
            modes.CBC(iv),
            backend=backend
        )
        encryptor = cipher.encryptor()
        
        # 填充数据
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data.encode()) + padder.finalize()
        
        # 加密数据
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # 生成HMAC以验证数据完整性
        hmac_obj = hmac.HMAC(base64.urlsafe_b64decode(self.hmac_key), hashes.SHA256(), backend=backend)
        hmac_obj.update(encrypted_data + iv)
        signature = base64.urlsafe_b64encode(hmac_obj.finalize()).decode()
        
        return {
            "encrypted_data": base64.urlsafe_b64encode(encrypted_data).decode(),
            "iv": base64.urlsafe_b64encode(iv).decode(),
            "algorithm": SECURITY_CONFIG["encryption_algorithm"],
            "signature": signature,
            "version": "1.1",
            "timestamp": str(datetime.utcnow())
        }
    
    def decrypt_data(self, encrypted_data: str, iv: str, signature: Optional[str] = None) -> str:
        """解密数据
        
        Args:
            encrypted_data: 加密的数据
            iv: 初始化向量
            signature: 数据完整性签名
        
        Returns:
            解密后的数据
        """
        try:
            # 解码数据
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data)
            iv_bytes = base64.urlsafe_b64decode(iv)
            
            # 验证数据完整性
            if signature:
                hmac_obj = hmac.HMAC(base64.urlsafe_b64decode(self.hmac_key), hashes.SHA256(), backend=backend)
                hmac_obj.update(encrypted_bytes + iv_bytes)
                hmac_obj.verify(base64.urlsafe_b64decode(signature))
            
            # 创建解密器
            cipher = Cipher(
                algorithms.AES(base64.urlsafe_b64decode(self.encryption_key)),
                modes.CBC(iv_bytes),
                backend=backend
            )
            decryptor = cipher.decryptor()
            
            # 解密数据
            decrypted_padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
            
            # 去除填充
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()
            
            return decrypted_data.decode()
        except hmac.InvalidSignature:
            logger.error("数据完整性验证失败")
            raise ValueError("数据完整性验证失败")
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            raise ValueError("数据解密失败") from e
    
    def generate_token(self, length: int = 32) -> str:
        """生成随机令牌
        
        Args:
            length: 令牌长度
        
        Returns:
            随机令牌
        """
        # 确保令牌长度不小于最小值
        token_length = max(length, SECURITY_CONFIG["min_token_length"])
        return base64.urlsafe_b64encode(secrets.token_bytes(token_length)).decode().rstrip('=')
    
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """哈希数据
        
        Args:
            data: 要哈希的数据
            algorithm: 哈希算法
        
        Returns:
            哈希值
        """
        if algorithm.lower() == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm.lower() == "sha512":
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm.lower() == "md5":
            logger.warning("使用不安全的MD5算法")
            return hashlib.md5(data.encode()).hexdigest()
        else:
            logger.error(f"不支持的哈希算法: {algorithm}")
            raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    def generate_hmac(self, data: bytes) -> str:
        """生成HMAC签名
        
        Args:
            data: 要签名的数据
        
        Returns:
            HMAC签名
        """
        hmac_obj = hmac.HMAC(base64.urlsafe_b64decode(self.hmac_key), hashes.SHA256(), backend=backend)
        hmac_obj.update(data)
        return base64.urlsafe_b64encode(hmac_obj.finalize()).decode()
    
    def verify_hmac(self, data: bytes, signature: str) -> bool:
        """验证HMAC签名
        
        Args:
            data: 原始数据
            signature: 签名
        
        Returns:
            签名是否有效
        """
        try:
            hmac_obj = hmac.HMAC(base64.urlsafe_b64decode(self.hmac_key), hashes.SHA256(), backend=backend)
            hmac_obj.update(data)
            hmac_obj.verify(base64.urlsafe_b64decode(signature))
            return True
        except Exception as e:
            logger.error(f"HMAC验证失败: {e}")
            return False
    
    def get_sensitive_data_category(self, key: str) -> Optional[str]:
        """获取敏感数据类别
        
        Args:
            key: 数据键名
        
        Returns:
            敏感数据类别（high, medium, low）或None
        """
        key_lower = key.lower()
        for category, keys in SENSITIVE_DATA_CATEGORIES.items():
            if key_lower in keys:
                return category
        return None
    
    def is_sensitive_data(self, key: str) -> bool:
        """判断是否为敏感数据
        
        Args:
            key: 数据键名
        
        Returns:
            是否为敏感数据
        """
        return self.get_sensitive_data_category(key) is not None
    
    def is_highly_sensitive_data(self, key: str) -> bool:
        """判断是否为高敏感数据
        
        Args:
            key: 数据键名
        
        Returns:
            是否为高敏感数据
        """
        return self.get_sensitive_data_category(key) == "high"
    
    def sanitize_sensitive_data(self, data: Any, mask: bool = True) -> Any:
        """清理敏感数据，用于日志或调试输出
        
        Args:
            data: 要清理的数据
            mask: 是否掩码数据，否则替换为固定字符串
        
        Returns:
            清理后的数据
        """
        if isinstance(data, dict):
            return {
                k: self._mask_sensitive_value(k, v, mask) if self.is_sensitive_data(k) else self.sanitize_sensitive_data(v, mask)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.sanitize_sensitive_data(item, mask) for item in data]
        else:
            return data
    
    def _mask_sensitive_value(self, key: str, value: Any, mask: bool = True) -> Any:
        """掩码敏感数据
        
        Args:
            key: 数据键名
            value: 数据值
            mask: 是否掩码数据
        
        Returns:
            掩码后的数据
        """
        if not isinstance(value, str):
            return "[SENSITIVE]"
        
        category = self.get_sensitive_data_category(key)
        
        if mask:
            # 根据数据长度进行掩码
            if len(value) <= 4:
                return "****"
            elif len(value) <= 10:
                return f"{value[:2]}****{value[-2:]}"
            elif len(value) <= 20:
                return f"{value[:3]}****{value[-3:]}"
            else:
                return f"{value[:4]}****{value[-4:]}"
        else:
            # 根据敏感级别返回不同的掩码字符串
            if category == "high":
                return "[HIGHLY_SENSITIVE]"
            elif category == "medium":
                return "[MEDIUM_SENSITIVE]"
            else:
                return "[LOW_SENSITIVE]"
    
    def desensitize_data(self, data: Any, level: str = "medium") -> Any:
        """数据脱敏
        
        Args:
            data: 要脱敏的数据
            level: 脱敏级别（low, medium, high）
        
        Returns:
            脱敏后的数据
        """
        if isinstance(data, dict):
            return {
                k: self._desensitize_value(k, v, level) for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.desensitize_data(item, level) for item in data]
        else:
            return data
    
    def _desensitize_value(self, key: str, value: Any, level: str) -> Any:
        """脱敏单个值
        
        Args:
            key: 数据键名
            value: 数据值
            level: 脱敏级别
        
        Returns:
            脱敏后的值
        """
        if not isinstance(value, str):
            return value
        
        category = self.get_sensitive_data_category(key)
        if not category:
            return value
        
        # 根据脱敏级别决定处理方式
        if level == "high":
            # 高脱敏级别 - 完全替换
            return self._mask_sensitive_value(key, value, mask=False)
        elif level == "medium":
            # 中脱敏级别 - 部分掩码
            return self._mask_sensitive_value(key, value, mask=True)
        else:
            # 低脱敏级别 - 仅对高敏感数据处理
            if category == "high":
                return self._mask_sensitive_value(key, value, mask=True)
            return value
    
    def rotate_keys(self) -> Dict[str, str]:
        """轮换加密密钥和HMAC密钥
        
        Returns:
            新生成的密钥（仅用于测试和开发环境）
        """
        logger.warning("密钥轮换功能仅用于测试和开发环境")
        
        # 生成新的加密密钥
        new_encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        new_salt = base64.urlsafe_b64encode(os.urandom(16)).decode()
        new_hmac_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        # 更新当前密钥
        self.encryption_key = new_encryption_key
        self.salt = new_salt
        self.hmac_key = new_hmac_key
        
        return {
            "new_encryption_key": new_encryption_key,
            "new_salt": new_salt,
            "new_hmac_key": new_hmac_key
        }
    
    def validate_encryption_key(self, key: str) -> bool:
        """验证加密密钥格式
        
        Args:
            key: 要验证的密钥
        
        Returns:
            密钥格式是否有效
        """
        try:
            decoded = base64.urlsafe_b64decode(key)
            return len(decoded) == SECURITY_CONFIG["key_length"]
        except Exception:
            return False


# 全局安全工具实例
security_utils = SecurityUtils()


# 装饰器：加密敏感数据
def encrypt_sensitive_data(func):
    """装饰器，用于加密函数返回的敏感数据
    
    示例用法：
    @encrypt_sensitive_data
    def get_user_info(user_id):
        # 返回包含敏感数据的字典
        return {
            "id": user_id,
            "name": "张三",
            "email": "zhangsan@example.com",
            "password": "password123"
        }
    """
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs) if hasattr(func, "__await__") else func(*args, **kwargs)
        
        if isinstance(result, dict):
            # 加密敏感数据
            for key, value in result.items():
                if security_utils.is_sensitive_data(key) and isinstance(value, str):
                    encrypted = security_utils.encrypt_data(value)
                    result[key] = encrypted
        
        return result
    
    return wrapper


# 装饰器：验证密码
def verify_password_decorator(func):
    """装饰器，用于验证密码
    
    示例用法：
    @verify_password_decorator
    def login(username, password, hashed_password, salt):
        # 密码验证通过后执行的逻辑
        return {"success": True, "message": "登录成功"}
    """
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 提取密码相关参数
        password = kwargs.get("password")
        hashed_password = kwargs.get("hashed_password")
        salt = kwargs.get("salt")
        iterations = kwargs.get("iterations", 100000)
        
        if password and hashed_password and salt:
            # 验证密码
            if not security_utils.verify_password(password, hashed_password, salt, iterations):
                return {"success": False, "message": "密码错误"}
        
        # 密码验证通过，执行原函数
        return await func(*args, **kwargs) if hasattr(func, "__await__") else func(*args, **kwargs)
    
    return wrapper


# 装饰器：验证数据完整性
def verify_data_integrity(func):
    """装饰器，用于验证函数返回数据的完整性
    
    示例用法：
    @verify_data_integrity
    def get_sensitive_data(data_id):
        # 返回包含敏感数据的字典
        return {
            "id": data_id,
            "sensitive_field": "sensitive_value",
            "signature": "hmac_signature"
        }
    """
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs) if hasattr(func, "__await__") else func(*args, **kwargs)
        
        if isinstance(result, dict) and "signature" in result:
            # 提取签名
            signature = result.pop("signature")
            
            # 重新生成签名并验证
            import json
            data_bytes = json.dumps(result, sort_keys=True).encode()
            if not security_utils.verify_hmac(data_bytes, signature):
                logger.error("数据完整性验证失败")
                return {"success": False, "message": "数据完整性验证失败"}
        
        return result
    
    return wrapper