"""
插件管理器模块
负责插件的加载、卸载、管理和生命周期控制
实现开放的插件架构，允许动态添加新功能模块
"""

import logging
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Type, Any
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """插件状态"""
    INITIALIZED = "initialized"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    UNLOADED = "unloaded"


class PluginType(Enum):
    """插件类型"""
    AI_MODEL = "ai_model"
    SENSOR_ADAPTER = "sensor_adapter"
    RISK_CONTROLLER = "risk_controller"
    BUSINESS_MODULE = "business_module"
    UI_COMPONENT = "ui_component"
    OTHER = "other"


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    entry_point: str
    status: PluginStatus = PluginStatus.INITIALIZED
    error_message: Optional[str] = None
    loaded_at: Optional[str] = None
    activated_at: Optional[str] = None
    plugin_path: Optional[str] = None


# 大脑区域类型定义
@dataclass
class BrainNavigationRegion:
    """大脑导航区域"""
    id: str
    name: str
    path: str
    position: Dict[str, str]  # { top, left, width, height }
    description: str
    icon: str = "Brain"
    color: str = "primary"

class BasePlugin(ABC):
    """插件基类
    所有插件必须继承此类
    """
    
    def __init__(self):
        self.info = None
        self.status = PluginStatus.INITIALIZED
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self, app: Any) -> bool:
        """初始化插件
        Args:
            app: FastAPI应用实例
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def activate(self) -> bool:
        """激活插件
        Returns:
            bool: 激活是否成功
        """
        pass
    
    @abstractmethod
    async def deactivate(self) -> bool:
        """停用插件
        Returns:
            bool: 停用是否成功
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """清理插件资源
        Returns:
            bool: 清理是否成功
        """
        pass
    
    def get_routes(self) -> Optional[Any]:
        """获取插件的路由
        Returns:
            Optional[Any]: 插件的FastAPI路由
        """
        return None
    
    def get_models(self) -> Optional[List[Any]]:
        """获取插件的模型定义
        Returns:
            Optional[List[Any]]: 插件的模型列表
        """
        return None
    
    def get_services(self) -> Optional[Dict[str, Any]]:
        """获取插件提供的服务
        Returns:
            Optional[Dict[str, Any]]: 插件提供的服务字典
        """
        return None
    
    def get_api_schemas(self) -> Optional[Dict[str, Any]]:
        """获取插件的API模式
        Returns:
            Optional[Dict[str, Any]]: 插件的API模式字典
        """
        return None
    
    def get_brain_navigation_regions(self) -> Optional[List[BrainNavigationRegion]]:
        """获取插件注册的大脑导航区域
        Returns:
            Optional[List[BrainNavigationRegion]]: 大脑导航区域列表
        """
        return None


class PluginManager:
    """插件管理器
    负责插件的加载、卸载、管理和生命周期控制
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.plugin_paths: List[str] = []
        self.app = None
        self.brain_navigation_regions: List[BrainNavigationRegion] = []
    
    def set_app(self, app: Any):
        """设置FastAPI应用实例
        Args:
            app: FastAPI应用实例
        """
        self.app = app
    
    def add_plugin_path(self, path: str):
        """添加插件搜索路径
        Args:
            path: 插件路径
        """
        if os.path.exists(path) and path not in self.plugin_paths:
            self.plugin_paths.append(path)
            logger.info(f"添加插件路径: {path}")
    
    async def discover_plugins(self) -> List[PluginInfo]:
        """发现可用插件
        Returns:
            List[PluginInfo]: 发现的插件信息列表
        """
        discovered_plugins = []
        
        for plugin_path in self.plugin_paths:
            if not os.path.exists(plugin_path):
                continue
            
            for item in os.listdir(plugin_path):
                item_path = os.path.join(plugin_path, item)
                if os.path.isdir(item_path):
                    # 检查是否为插件目录
                    plugin_info = await self._check_plugin_dir(item_path)
                    if plugin_info:
                        discovered_plugins.append(plugin_info)
        
        return discovered_plugins
    
    async def _check_plugin_dir(self, plugin_dir: str) -> Optional[PluginInfo]:
        """检查目录是否为有效插件
        Args:
            plugin_dir: 插件目录
        Returns:
            Optional[PluginInfo]: 插件信息，如果不是有效插件返回None
        """
        try:
            # 检查是否存在plugin.json文件
            plugin_json_path = os.path.join(plugin_dir, "plugin.json")
            if not os.path.exists(plugin_json_path):
                return None
            
            # 读取插件配置
            import json
            with open(plugin_json_path, "r", encoding="utf-8") as f:
                plugin_config = json.load(f)
            
            # 验证插件配置
            required_fields = ["name", "version", "description", "author", "plugin_type", "entry_point"]
            for field in required_fields:
                if field not in plugin_config:
                    logger.warning(f"插件配置缺少必填字段 {field}: {plugin_dir}")
                    return None
            
            # 检查入口点是否存在
            entry_point = plugin_config["entry_point"]
            entry_path = os.path.join(plugin_dir, entry_point.replace(".", "/") + ".py")
            if not os.path.exists(entry_path):
                logger.warning(f"插件入口点不存在: {entry_path}")
                return None
            
            # 创建插件信息
            plugin_info = PluginInfo(
                name=plugin_config["name"],
                version=plugin_config["version"],
                description=plugin_config["description"],
                author=plugin_config["author"],
                plugin_type=PluginType(plugin_config["plugin_type"]),
                dependencies=plugin_config.get("dependencies", []),
                entry_point=plugin_config["entry_point"],
                plugin_path=plugin_dir
            )
            
            return plugin_info
            
        except Exception as e:
            logger.error(f"检查插件目录失败 {plugin_dir}: {e}")
            return None
    
    async def load_plugin(self, plugin_info: PluginInfo) -> bool:
        """加载插件
        Args:
            plugin_info: 插件信息
        Returns:
            bool: 加载是否成功
        """
        try:
            if plugin_info.name in self.plugins:
                logger.warning(f"插件已加载: {plugin_info.name}")
                return True
            
            logger.info(f"开始加载插件: {plugin_info.name} (v{plugin_info.version})")
            
            # 加载插件模块
            plugin_module = await self._load_plugin_module(plugin_info)
            if not plugin_module:
                return False
            
            # 创建插件实例
            plugin_class = getattr(plugin_module, "Plugin", None)
            if not plugin_class or not issubclass(plugin_class, BasePlugin):
                logger.error(f"插件类不存在或未继承BasePlugin: {plugin_info.name}")
                return False
            
            plugin = plugin_class()
            plugin.info = plugin_info
            
            # 初始化插件
            if await plugin.initialize(self.app):
                self.plugins[plugin_info.name] = plugin
                plugin_info.status = PluginStatus.LOADED
                self.plugin_info[plugin_info.name] = plugin_info
                logger.info(f"插件加载成功: {plugin_info.name}")
                return True
            else:
                logger.error(f"插件初始化失败: {plugin_info.name}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "初始化失败"
                return False
                
        except Exception as e:
            logger.error(f"加载插件失败 {plugin_info.name}: {e}")
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            return False
    
    async def _load_plugin_module(self, plugin_info: PluginInfo) -> Optional[Any]:
        """加载插件模块
        Args:
            plugin_info: 插件信息
        Returns:
            Optional[Any]: 加载的模块，失败返回None
        """
        try:
            plugin_dir = plugin_info.plugin_path
            if not plugin_dir:
                return None
            
            # 将插件目录添加到Python路径
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # 动态加载模块
            module_name = f"plugin_{plugin_info.name.lower().replace(' ', '_')}"
            spec = importlib.util.spec_from_file_location(
                module_name,
                os.path.join(plugin_dir, plugin_info.entry_point.replace(".", "/") + ".py")
            )
            
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            logger.error(f"加载插件模块失败 {plugin_info.name}: {e}")
            return None
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """激活插件
        Args:
            plugin_name: 插件名称
        Returns:
            bool: 激活是否成功
        """
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"插件未加载: {plugin_name}")
                return False
            
            plugin = self.plugins[plugin_name]
            if plugin.status == PluginStatus.ACTIVE:
                logger.warning(f"插件已激活: {plugin_name}")
                return True
            
            logger.info(f"激活插件: {plugin_name}")
            
            if await plugin.activate():
                plugin.status = PluginStatus.ACTIVE
                self.plugin_info[plugin_name].status = PluginStatus.ACTIVE
                
                # 注册插件路由
                routes = plugin.get_routes()
                if routes and self.app:
                    self.app.include_router(routes, prefix=f"/api/plugins/{plugin_name}")
                    logger.info(f"注册插件路由: {plugin_name}")
                
                # 更新大脑导航区域
                self._update_brain_navigation_regions()
                logger.info(f"更新大脑导航区域成功")
                
                logger.info(f"插件激活成功: {plugin_name}")
                return True
            else:
                logger.error(f"插件激活失败: {plugin_name}")
                plugin.status = PluginStatus.ERROR
                self.plugin_info[plugin_name].status = PluginStatus.ERROR
                self.plugin_info[plugin_name].error_message = "激活失败"
                return False
                
        except Exception as e:
            logger.error(f"激活插件失败 {plugin_name}: {e}")
            if plugin_name in self.plugins:
                self.plugins[plugin_name].status = PluginStatus.ERROR
                self.plugin_info[plugin_name].status = PluginStatus.ERROR
                self.plugin_info[plugin_name].error_message = str(e)
            return False
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """停用插件
        Args:
            plugin_name: 插件名称
        Returns:
            bool: 停用是否成功
        """
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"插件未加载: {plugin_name}")
                return False
            
            plugin = self.plugins[plugin_name]
            if plugin.status != PluginStatus.ACTIVE:
                logger.warning(f"插件未激活: {plugin_name}")
                return True
            
            logger.info(f"停用插件: {plugin_name}")
            
            if await plugin.deactivate():
                plugin.status = PluginStatus.LOADED
                self.plugin_info[plugin_name].status = PluginStatus.LOADED
                logger.info(f"插件停用成功: {plugin_name}")
                
                # 更新大脑导航区域
                self._update_brain_navigation_regions()
                logger.info(f"更新大脑导航区域成功")
                
                return True
            else:
                logger.error(f"插件停用失败: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"停用插件失败 {plugin_name}: {e}")
            return False
    
    def _update_brain_navigation_regions(self):
        """更新大脑导航区域
        从所有激活的插件中收集大脑导航区域
        """
        try:
            # 清空现有区域
            self.brain_navigation_regions = []
            
            # 从所有激活的插件中收集大脑导航区域
            for plugin in self.plugins.values():
                if plugin.status == PluginStatus.ACTIVE:
                    regions = plugin.get_brain_navigation_regions()
                    if regions:
                        self.brain_navigation_regions.extend(regions)
                        logger.info(f"从插件 {plugin.info.name} 收集到 {len(regions)} 个大脑导航区域")
            
            logger.info(f"总共有 {len(self.brain_navigation_regions)} 个大脑导航区域")
        except Exception as e:
            logger.error(f"更新大脑导航区域失败: {e}")
    
    def get_brain_navigation_regions(self) -> List[BrainNavigationRegion]:
        """获取所有大脑导航区域
        Returns:
            List[BrainNavigationRegion]: 大脑导航区域列表
        """
        return self.brain_navigation_regions
    
    def add_brain_navigation_region(self, region: BrainNavigationRegion) -> bool:
        """添加大脑导航区域
        Args:
            region: 大脑导航区域
        Returns:
            bool: 添加是否成功
        """
        try:
            # 检查区域ID是否已存在
            if any(r.id == region.id for r in self.brain_navigation_regions):
                logger.warning(f"大脑导航区域ID已存在: {region.id}")
                return False
            
            self.brain_navigation_regions.append(region)
            logger.info(f"添加大脑导航区域成功: {region.name}")
            return True
        except Exception as e:
            logger.error(f"添加大脑导航区域失败: {e}")
            return False
    
    def remove_brain_navigation_region(self, region_id: str) -> bool:
        """移除大脑导航区域
        Args:
            region_id: 区域ID
        Returns:
            bool: 移除是否成功
        """
        try:
            initial_count = len(self.brain_navigation_regions)
            self.brain_navigation_regions = [r for r in self.brain_navigation_regions if r.id != region_id]
            
            if len(self.brain_navigation_regions) < initial_count:
                logger.info(f"移除大脑导航区域成功: {region_id}")
                return True
            else:
                logger.warning(f"大脑导航区域不存在: {region_id}")
                return False
        except Exception as e:
            logger.error(f"移除大脑导航区域失败: {e}")
            return False
    

    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件
        Args:
            plugin_name: 插件名称
        Returns:
            bool: 卸载是否成功
        """
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"插件未加载: {plugin_name}")
                return True
            
            plugin = self.plugins[plugin_name]
            
            # 如果插件正在运行，先停用
            if plugin.status == PluginStatus.ACTIVE:
                if not await self.deactivate_plugin(plugin_name):
                    logger.error(f"停用插件失败，无法卸载: {plugin_name}")
                    return False
            
            logger.info(f"卸载插件: {plugin_name}")
            
            # 清理插件资源
            if await plugin.cleanup():
                del self.plugins[plugin_name]
                del self.plugin_info[plugin_name]
                logger.info(f"插件卸载成功: {plugin_name}")
                return True
            else:
                logger.error(f"插件清理失败: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"卸载插件失败 {plugin_name}: {e}")
            return False
    
    async def load_all_plugins(self) -> List[str]:
        """加载所有可用插件
        Returns:
            List[str]: 成功加载的插件名称列表
        """
        discovered_plugins = await self.discover_plugins()
        loaded_plugins = []
        
        for plugin_info in discovered_plugins:
            if await self.load_plugin(plugin_info):
                loaded_plugins.append(plugin_info.name)
        
        return loaded_plugins
    
    async def activate_all_plugins(self) -> List[str]:
        """激活所有已加载插件
        Returns:
            List[str]: 成功激活的插件名称列表
        """
        activated_plugins = []
        
        for plugin_name in self.plugins:
            if await self.activate_plugin(plugin_name):
                activated_plugins.append(plugin_name)
        
        return activated_plugins
    
    async def unload_all_plugins(self) -> List[str]:
        """卸载所有插件
        Returns:
            List[str]: 成功卸载的插件名称列表
        """
        unloaded_plugins = []
        
        # 先复制插件名称列表，避免迭代时修改
        plugin_names = list(self.plugins.keys())
        for plugin_name in plugin_names:
            if await self.unload_plugin(plugin_name):
                unloaded_plugins.append(plugin_name)
        
        return unloaded_plugins
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """获取插件实例
        Args:
            plugin_name: 插件名称
        Returns:
            Optional[BasePlugin]: 插件实例，不存在返回None
        """
        return self.plugins.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息
        Args:
            plugin_name: 插件名称
        Returns:
            Optional[PluginInfo]: 插件信息，不存在返回None
        """
        return self.plugin_info.get(plugin_name)
    
    def list_plugins(self, status: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """列出插件
        Args:
            status: 插件状态，None表示列出所有插件
        Returns:
            List[PluginInfo]: 插件信息列表
        """
        if status:
            return [info for info in self.plugin_info.values() if info.status == status]
        else:
            return list(self.plugin_info.values())
    
    def get_plugin_services(self, plugin_type: Optional[PluginType] = None) -> Dict[str, Any]:
        """获取插件提供的服务
        Args:
            plugin_type: 插件类型，None表示所有类型
        Returns:
            Dict[str, Any]: 插件服务字典
        """
        services = {}
        
        for plugin in self.plugins.values():
            if plugin_type and plugin.info.plugin_type != plugin_type:
                continue
            
            plugin_services = plugin.get_services()
            if plugin_services:
                services.update(plugin_services)
        
        return services


# 全局插件管理器实例
plugin_manager = PluginManager()
