"""数据访问层（DAO）
提供统一的数据访问接口，封装数据库操作
"""

from .base_dao import BaseDAO
from .device_dao import DeviceDAO

__all__ = ["BaseDAO", "DeviceDAO"]