"""
插件管理API路由
提供插件的加载、激活、停用、卸载等管理功能
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from enum import Enum
from src.core.plugins.plugin_manager import plugin_manager, PluginInfo, PluginStatus, PluginType
from src.api.routes.auth import get_current_user


router = APIRouter(prefix="/plugins", tags=["plugins"])


@router.get("/", response_model=List[Dict])
async def list_plugins(
    status: Optional[str] = Query(None, description="插件状态过滤"),
    plugin_type: Optional[str] = Query(None, description="插件类型过滤"),
    _ = Depends(get_current_user)
):
    """列出所有插件"""
    
    # 转换状态参数
    status_enum = None
    if status:
        try:
            status_enum = PluginStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的状态值: {status}")
    
    # 获取插件列表
    plugins = plugin_manager.list_plugins(status_enum)
    
    # 转换类型过滤
    if plugin_type:
        try:
            type_enum = PluginType(plugin_type)
            plugins = [p for p in plugins if p.plugin_type == type_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的插件类型: {plugin_type}")
    
    # 转换为字典返回
    return [
        {
            "name": p.name,
            "version": p.version,
            "description": p.description,
            "author": p.author,
            "plugin_type": p.plugin_type.value,
            "status": p.status.value,
            "error_message": p.error_message,
            "loaded_at": p.loaded_at,
            "activated_at": p.activated_at
        }
        for p in plugins
    ]


@router.get("/discover", response_model=List[Dict])
async def discover_plugins(
    _ = Depends(get_current_user)
):
    """发现可用插件"""
    discovered_plugins = await plugin_manager.discover_plugins()
    
    return [
        {
            "name": p.name,
            "version": p.version,
            "description": p.description,
            "author": p.author,
            "plugin_type": p.plugin_type.value,
            "dependencies": p.dependencies
        }
        for p in discovered_plugins
    ]


@router.post("/load", response_model=Dict)
async def load_plugin(
    plugin_name: str,
    _ = Depends(get_current_user)
):
    """加载插件"""
    # 先发现插件
    discovered_plugins = await plugin_manager.discover_plugins()
    
    # 查找指定插件
    plugin_info = next((p for p in discovered_plugins if p.name == plugin_name), None)
    if not plugin_info:
        raise HTTPException(status_code=404, detail=f"插件未找到: {plugin_name}")
    
    # 加载插件
    result = await plugin_manager.load_plugin(plugin_info)
    if result:
        return {
            "success": True,
            "message": f"插件加载成功: {plugin_name}",
            "plugin": {
                "name": plugin_info.name,
                "status": plugin_info.status.value
            }
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"插件加载失败: {plugin_name}, 错误: {plugin_info.error_message}"
        )


@router.post("/activate", response_model=Dict)
async def activate_plugin(
    plugin_name: str,
    _ = Depends(get_current_user)
):
    """激活插件"""
    result = await plugin_manager.activate_plugin(plugin_name)
    if result:
        return {
            "success": True,
            "message": f"插件激活成功: {plugin_name}"
        }
    else:
        plugin_info = plugin_manager.get_plugin_info(plugin_name)
        error_msg = plugin_info.error_message if plugin_info else "未知错误"
        raise HTTPException(
            status_code=500,
            detail=f"插件激活失败: {plugin_name}, 错误: {error_msg}"
        )


@router.post("/deactivate", response_model=Dict)
async def deactivate_plugin(
    plugin_name: str,
    _ = Depends(get_current_user)
):
    """停用插件"""
    result = await plugin_manager.deactivate_plugin(plugin_name)
    if result:
        return {
            "success": True,
            "message": f"插件停用成功: {plugin_name}"
        }
    else:
        raise HTTPException(status_code=500, detail=f"插件停用失败: {plugin_name}")


@router.post("/unload", response_model=Dict)
async def unload_plugin(
    plugin_name: str,
    _ = Depends(get_current_user)
):
    """卸载插件"""
    result = await plugin_manager.unload_plugin(plugin_name)
    if result:
        return {
            "success": True,
            "message": f"插件卸载成功: {plugin_name}"
        }
    else:
        raise HTTPException(status_code=500, detail=f"插件卸载失败: {plugin_name}")


@router.post("/load-all", response_model=Dict)
async def load_all_plugins(
    _ = Depends(get_current_user)
):
    """加载所有可用插件"""
    loaded_plugins = await plugin_manager.load_all_plugins()
    return {
        "success": True,
        "message": f"成功加载 {len(loaded_plugins)} 个插件",
        "loaded_plugins": loaded_plugins
    }


@router.post("/activate-all", response_model=Dict)
async def activate_all_plugins(
    _ = Depends(get_current_user)
):
    """激活所有已加载插件"""
    activated_plugins = await plugin_manager.activate_all_plugins()
    return {
        "success": True,
        "message": f"成功激活 {len(activated_plugins)} 个插件",
        "activated_plugins": activated_plugins
    }


@router.get("/services", response_model=Dict)
async def get_plugin_services(
    plugin_type: Optional[str] = Query(None, description="插件类型过滤"),
    _ = Depends(get_current_user)
):
    """获取插件提供的服务"""
    
    # 转换类型参数
    type_enum = None
    if plugin_type:
        try:
            type_enum = PluginType(plugin_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的插件类型: {plugin_type}")
    
    services = plugin_manager.get_plugin_services(type_enum)
    
    return {
        "success": True,
        "services": services
    }


@router.get("/{plugin_name}", response_model=Dict)
async def get_plugin_info(
    plugin_name: str,
    _ = Depends(get_current_user)
):
    """获取插件详情"""
    plugin_info = plugin_manager.get_plugin_info(plugin_name)
    if not plugin_info:
        raise HTTPException(status_code=404, detail=f"插件未找到: {plugin_name}")
    
    plugin = plugin_manager.get_plugin(plugin_name)
    
    return {
        "name": plugin_info.name,
        "version": plugin_info.version,
        "description": plugin_info.description,
        "author": plugin_info.author,
        "plugin_type": plugin_info.plugin_type.value,
        "status": plugin_info.status.value,
        "error_message": plugin_info.error_message,
        "loaded_at": plugin_info.loaded_at,
        "activated_at": plugin_info.activated_at,
        "has_routes": bool(plugin.get_routes() if plugin else False),
        "has_services": bool(plugin.get_services() if plugin else False),
        "dependencies": plugin_info.dependencies
    }


@router.get("/brain-regions", response_model=List[Dict])
async def get_brain_navigation_regions(
    _ = Depends(get_current_user)
):
    """获取大脑导航区域"""
    # 获取所有大脑导航区域
    brain_regions = plugin_manager.get_brain_navigation_regions()
    
    # 转换为字典返回
    return [
        {
            "id": region.id,
            "name": region.name,
            "path": region.path,
            "position": region.position,
            "description": region.description,
            "icon": region.icon,
            "color": region.color
        }
        for region in brain_regions
    ]
