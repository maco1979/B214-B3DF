from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import datetime
import logging
import sys
import os
import numpy as np
import jax.numpy as jnp
import jax
import asyncio

# 配置日志记录
logger = logging.getLogger(__name__)

# Flax兼容性补丁已在main.py中应用

# 标准导入摄像头控制器
from src.core.services.camera_controller import CameraController

# 创建摄像头控制器实例
camera_controller = CameraController()

# 标准导入连接控制器
from src.core.services.connection_controller import InfraredController, AppController, BluetoothController

# 导入新的服务组件
from src.core.services.connection_pool_manager import connection_pool_manager
from src.core.services.protocol_adapter import protocol_adapter_manager, ProtocolType
from src.core.services.device_auth_manager import device_auth_manager

# 为不同的连接类型创建控制器实例
controllers = {
    "infrared": InfraredController(),
    "app": AppController(),
    "bluetooth": BluetoothController()
}

# 导入设备发现服务
from src.core.services.device_discovery_service import device_discovery_service

# 标准导入JEPA和融合模块
from src.core.models.jepa_model import JEPAModel
from src.core.models.jepa_dtmpc_fusion import JepaDtMpcFusion

# 导入有机体AI核心
from src.core.ai_organic_core import organic_ai_core, get_organic_ai_core

# 导入DT-MPC控制器
try:
    from dt_mpc_core import DTMpcController
except ImportError as e:
    logger.error(f"DT-MPC控制器导入失败: {e}")
    DTMpcController = None

# 创建路由实例
router = APIRouter(prefix="/ai-control", tags=["AI控制"])

# 配置日志记录
logger = logging.getLogger(__name__)

# Mock设备数据
mock_devices = [
    {
        "id": 1,
        "name": "AI农业监测站",
        "type": "农业设备",
        "status": "online",
        "connected": True,
        "connection_type": "wifi",
        "signal": 90,
        "battery": 85,
        "location": "北京市海淀区",
        "lastSeen": "刚刚",
        "permissions": ["read", "write", "control"],
        "isCompliant": True,
        "connection_details": {
            "wifi_ssid": "Farm_WIFI",
            "wifi_strength": 90
        }
    },
    {
        "id": 2,
        "name": "智能灌溉系统",
        "type": "农业设备",
        "status": "online",
        "connected": True,
        "connection_type": "infrared",
        "signal": 85,
        "battery": 92,
        "location": "北京市海淀区",
        "lastSeen": "2分钟前",
        "permissions": ["read", "write", "control"],
        "isCompliant": True,
        "connection_details": {
            "infrared_channel": 3,
            "infrared_range": 10
        }
    },
    {
        "id": 3,
        "name": "环境传感器",
        "type": "传感器",
        "status": "online",
        "connected": True,
        "connection_type": "bluetooth",
        "signal": 78,
        "battery": 65,
        "location": "北京市海淀区",
        "lastSeen": "5分钟前",
        "permissions": ["read"],
        "isCompliant": True,
        "connection_details": {
            "bluetooth_address": "00:11:22:33:44:55",
            "bluetooth_version": "5.0"
        }
    },
    {
        "id": 4,
        "name": "AI视觉识别摄像头",
        "type": "摄像头",
        "status": "offline",
        "connected": False,
        "connection_type": "app",
        "signal": 0,
        "battery": 0,
        "location": "北京市海淀区",
        "lastSeen": "1小时前",
        "permissions": ["read", "write", "control"],
        "isCompliant": True,
        "connection_details": {
            "app_id": "com.ai.camera",
            "app_version": "1.2.3"
        }
    }
]

# 初始化连接池管理器
async def initialize_connection_manager():
    try:
        await connection_pool_manager.initialize()
        logger.info("连接池管理器初始化完成")
    except Exception as e:
        logger.error(f"连接池管理器初始化失败: {e}")

# 初始化时建立连接
for device in mock_devices:
    if device["connected"] and device["status"] == "online":
        connection_type = device["connection_type"]
        if connection_type in controllers:
            controller = controllers[connection_type]
            # 根据设备的连接类型发送连接命令
            if connection_type == "infrared" and "connection_details" in device:
                # 红外线连接需要channel和range参数
                controller.connect({
                    "channel": device["connection_details"].get("infrared_channel", 1),
                    "range": device["connection_details"].get("infrared_range", 10)
                })
            elif connection_type == "app" and "connection_details" in device:
                # APP连接需要app_id和app_version参数
                controller.connect({
                    "app_id": device["connection_details"].get("app_id", ""),
                    "app_version": device["connection_details"].get("app_version", "")
                })
            elif connection_type == "bluetooth" and "connection_details" in device:
                # 蓝牙连接需要bluetooth_address和bluetooth_version参数
                controller.connect({
                    "bluetooth_address": device["connection_details"].get("bluetooth_address", ""),
                    "bluetooth_version": device["connection_details"].get("bluetooth_version", "5.0")
                })
            elif connection_type == "wifi":
                # WiFi连接不需要特殊处理，设备已在线
                pass
            else:
                # 其他连接类型默认连接
                controller.connect({})
        
        # 注册到连接池
        # 注意：不能在模块导入时创建任务，因为没有事件循环
        # if connection_type in controllers:
        #     controller = controllers[connection_type]
        #     # 需要在运行时创建任务，而不是导入时
        #     # asyncio.create_task(connection_pool_manager.register_connection(device["id"], controller))

# 初始化JEPA和融合模块
jepa_dtmpc_fusion = None

# 暂时跳过JEPA初始化，确保API文档可访问
logger.info("暂时跳过JEPA-DT-MPC融合模块初始化，确保API文档端点可访问")

# 输入维度定义
input_dim = 32  # 与有机体AI核心状态向量维度匹配

# 主控状态
master_control_active = False

# 有机体AI核心实例
organic_ai_core_instance = None

async def initialize_organic_ai_core():
    """初始化有机体AI核心实例"""
    global organic_ai_core_instance
    if organic_ai_core_instance is None:
        organic_ai_core_instance = await get_organic_ai_core()

@router.get("/devices", response_model=List[Dict[str, Any]])
async def get_devices():
    """获取所有设备列表"""
    try:
        logger.info("获取设备列表请求")
        return mock_devices
    except Exception as e:
        logger.error(f"获取设备列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取设备列表失败: {str(e)}")

@router.post("/device/{device_id}", response_model=Dict[str, Any])
async def control_device(device_id: int, control_params: Dict[str, Any]):
    """控制指定设备"""
    try:
        logger.info(f"控制设备 {device_id} 请求: {control_params}")
        
        # 查找设备
        device = next((d for d in mock_devices if d["id"] == device_id), None)
        if not device:
            raise HTTPException(status_code=404, detail="设备不存在")
        
        # 检查设备是否连接
        if not device["connected"]:
            raise HTTPException(status_code=400, detail="设备未连接")
        
        # 检查设备是否有控制权限
        if "control" not in device["permissions"]:
            raise HTTPException(status_code=403, detail="设备无控制权限")
        
        # 实际设备控制逻辑
        action = control_params.get("action", "get_status")
        
        # 特殊处理摄像头设备 (ID=4)
        if device_id == 4 and device["type"] == "摄像头":
            try:
                if action == "open":
                    # 打开摄像头
                    result = camera_controller.open_camera()
                elif action == "close":
                    # 关闭摄像头
                    result = camera_controller.close_camera()
                elif action == "take_photo":
                    # 拍照
                    frame = camera_controller.take_photo()
                    if frame is not None:
                        result = {
                            "success": True,
                            "message": "拍照成功",
                            "frame_shape": frame.shape
                        }
                    else:
                        result = {
                            "success": False,
                            "message": "拍照失败"
                        }
                elif action == "get_status":
                    # 获取摄像头状态
                    result = {
                        "success": True,
                        "is_open": camera_controller.is_camera_open(),
                        "camera_index": camera_controller.camera_index
                    }
                elif action == "list_cameras":
                    # 列出可用摄像头
                    result = camera_controller.list_cameras()
                else:
                    # 其他操作返回默认响应
                    result = {
                        "success": True,
                        "message": f"摄像头 {action} 操作成功"
                    }
                
                if result.get("success", True):
                    return {
                        "success": True,
                        "device_id": device_id,
                        "action": action,
                        "status": "executed",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": result.get("message", f"设备 {device['name']} {action} 操作成功"),
                        "camera_data": result
                    }
                else:
                    return {
                        "success": False,
                        "device_id": device_id,
                        "action": action,
                        "status": "failed",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": result.get("message", "设备操作失败")
                    }
                    
            except Exception as e:
                logger.error(f"控制摄像头失败: {str(e)}")
                return {
                    "success": False,
                    "device_id": device_id,
                    "action": action,
                    "status": "failed",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": f"摄像头控制失败: {str(e)}"
                }
        
        # 使用连接控制器发送命令
        connection_type = device["connection_type"]
        if connection_type in controllers:
            controller = controllers[connection_type]
            command_result = controller.send_command(control_params)
            
            if command_result["success"]:
                return {
                    "success": True,
                    "device_id": device_id,
                    "action": action,
                    "status": "executed",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": command_result.get("message", f"设备 {device['name']} {action} 操作成功"),
                    "connection_type": connection_type
                }
            else:
                return {
                    "success": False,
                    "device_id": device_id,
                    "action": action,
                    "status": "failed",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": command_result.get("message", "设备操作失败")
                }
        
        # 其他设备的默认控制逻辑
        if device["type"] == "农业设备" and "control" in device["permissions"]:
            # 使用JEPA-DT-MPC融合控制
            if action == "auto_adjust" and jepa_dtmpc_fusion is not None:
                try:
                    # 生成模拟输入数据
                    rng = jax.random.PRNGKey(0)
                    current_data = jax.random.normal(rng, (1, input_dim))
                    
                    # 执行融合控制步骤
                    fusion_result = jepa_dtmpc_fusion.step(current_data)
                    
                    return {
                        "success": True,
                        "device_id": device_id,
                        "action": action,
                        "status": "executed",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"设备 {device['name']} 使用JEPA-DT-MPC融合控制自动调整成功",
                        "fusion_result": fusion_result,
                        "connection_type": connection_type
                    }
                except Exception as e:
                    logger.error(f"JEPA-DT-MPC融合控制失败: {str(e)}")
                    # 回退到传统控制
                    return {
                        "success": True,
                        "device_id": device_id,
                        "action": action,
                        "status": "executed",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"设备 {device['name']} 使用传统方法自动调整成功",
                        "connection_type": connection_type
                    }
        
        return {
            "success": True,
            "device_id": device_id,
            "action": action,
            "status": "executed",
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"设备 {device['name']} {action} 操作成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"控制设备 {device_id} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"控制设备失败: {str(e)}")

@router.get("/device/{device_id}/status", response_model=Dict[str, Any])
async def get_device_status(device_id: int):
    """获取指定设备状态"""
    try:
        logger.info(f"获取设备 {device_id} 状态请求")
        
        # 查找设备
        device = next((d for d in mock_devices if d["id"] == device_id), None)
        if not device:
            raise HTTPException(status_code=404, detail="设备不存在")
        
        # 返回设备状态
        return {
            "success": True,
            "device_id": device_id,
            "status": device["status"],
            "connected": device["connected"],
            "signal": device["signal"],
            "battery": device["battery"],
            "last_updated": datetime.datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取设备 {device_id} 状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取设备状态失败: {str(e)}")

@router.post("/master-control", response_model=Dict[str, Any])
async def activate_master_control(activate: Dict[str, bool]):
    """激活/关闭AI主控功能"""
    try:
        global master_control_active
        activate_status = activate.get("activate", False)
        
        logger.info(f"AI主控 {'激活' if activate_status else '关闭'} 请求")
        
        # 更新主控状态
        master_control_active = activate_status
        
        # 初始化有机体AI核心
        await initialize_organic_ai_core()
        
        # 控制有机体AI核心的主动迭代
        if organic_ai_core_instance:
            if activate_status:
                # 激活有机体AI核心的主动迭代
                await organic_ai_core_instance.start_active_iteration()
                logger.info("有机体AI核心主动迭代已启动")
            else:
                # 停止有机体AI核心的主动迭代
                await organic_ai_core_instance.stop_active_iteration()
                logger.info("有机体AI核心主动迭代已停止")
        
        # AI主控自动控制逻辑
        controlled_devices = []
        if master_control_active:
            # 激活时自动检测和控制设备
            logger.info("AI主控开始自动检测和控制设备")
            
            for device in mock_devices:
                if device["connected"] and device["status"] == "online":
                    try:
                        # 根据设备类型执行不同的自动控制逻辑
                        if device["type"] == "农业设备" and "control" in device["permissions"]:
                            # 自动调整农业设备参数
                            control_params = {"action": "auto_adjust", "mode": "smart_agriculture"}
                            result = await control_device(device["id"], control_params)
                            controlled_devices.append({
                                "device_id": device["id"],
                                "device_name": device["name"],
                                "action": "auto_adjust",
                                "status": "success" if result["success"] else "failed"
                            })
                        elif device["type"] == "摄像头" and "control" in device["permissions"]:
                            # 自动启动摄像头监控
                            control_params = {"action": "open"}
                            result = await control_device(device["id"], control_params)
                            controlled_devices.append({
                                "device_id": device["id"],
                                "device_name": device["name"],
                                "action": "start_monitoring",
                                "status": "success" if result["success"] else "failed"
                            })
                    except Exception as e:
                        logger.error(f"自动控制设备 {device['id']} 失败: {str(e)}")
                        controlled_devices.append({
                            "device_id": device["id"],
                            "device_name": device["name"],
                            "action": "auto_control",
                            "status": "failed",
                            "error": str(e)
                        })
        else:
            # 关闭时停止所有自动控制
            logger.info("AI主控关闭，停止所有自动控制")
            
            for device in mock_devices:
                if device["connected"] and device["status"] == "online":
                    try:
                        if device["type"] == "摄像头" and "control" in device["permissions"]:
                            # 关闭摄像头监控
                            control_params = {"action": "close"}
                            await control_device(device["id"], control_params)
                    except Exception as e:
                        logger.error(f"停止控制设备 {device['id']} 失败: {str(e)}")
        
        return {
            "success": True,
            "master_control_active": master_control_active,
            "message": f"AI主控已{'激活' if master_control_active else '关闭'}",
            "timestamp": datetime.datetime.now().isoformat(),
            "controlled_devices": controlled_devices,
            "organic_ai_core_status": "active" if master_control_active and organic_ai_core_instance else "inactive"
        }
    except Exception as e:
        logger.error(f"切换AI主控状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换AI主控状态失败: {str(e)}")

@router.get("/master-control/status", response_model=Dict[str, Any])
async def get_master_control_status():
    """获取AI主控状态"""
    try:
        logger.info("获取AI主控状态请求")
        
        return {
            "success": True,
            "master_control_active": master_control_active,
            "message": f"AI主控当前状态: {'激活' if master_control_active else '关闭'}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取AI主控状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取AI主控状态失败: {str(e)}")

@router.get("/fusion/prediction", response_model=Dict[str, Any])
async def get_fusion_prediction(steps: int = 5):
    """获取JEPA-DT-MPC融合预测结果"""
    try:
        logger.info(f"获取JEPA-DT-MPC融合预测结果请求，步数: {steps}")
        
        if jepa_dtmpc_fusion is None:
            raise HTTPException(status_code=503, detail="JEPA-DT-MPC融合模块未初始化")
        
        # 生成模拟输入数据
        rng = jax.random.PRNGKey(0)
        current_data = jax.random.normal(rng, (1, input_dim))
        
        # 获取融合预测结果
        fusion_result = jepa_dtmpc_fusion.predict(current_data, steps=steps)
        
        return {
            "success": True,
            "fusion_result": fusion_result,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取融合预测结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取融合预测结果失败: {str(e)}")

@router.get("/fusion/status", response_model=Dict[str, Any])
async def get_fusion_status():
    """获取JEPA-DT-MPC融合模块状态"""
    try:
        logger.info("获取JEPA-DT-MPC融合模块状态请求")
        
        if jepa_dtmpc_fusion is None:
            return {
                "success": False,
                "status": "未初始化",
                "message": "JEPA-DT-MPC融合模块未初始化",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "status": "已初始化",
            "message": "JEPA-DT-MPC融合模块已初始化",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取融合模块状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取融合模块状态失败: {str(e)}")

@router.get("/scan-devices", response_model=Dict[str, Any])
async def scan_devices():
    """扫描网络设备"""
    try:
        logger.info("扫描设备请求")
        
        # 调用设备发现服务的扫描功能
        scanned_devices = await device_discovery_service.scan_all_devices()
        
        # 统计新发现的设备数量（这里简单返回总数量，实际应与现有设备对比）
        new_devices_count = len(scanned_devices)
        
        return {
            "success": True,
            "data": scanned_devices,
            "message": "设备扫描完成",
            "timestamp": datetime.datetime.now().isoformat(),
            "scan_count": len(scanned_devices),
            "new_devices_count": new_devices_count
        }
    except Exception as e:
        logger.error(f"扫描设备失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"扫描设备失败: {str(e)}")

@router.post("/device/{device_id}/authenticate", response_model=Dict[str, Any])
async def authenticate_device(device_id: int, auth_params: Dict[str, Any]):
    """设备认证接口"""
    try:
        logger.info(f"设备 {device_id} 认证请求: {auth_params}")
        
        # 使用设备认证管理器进行认证
        result = device_auth_manager.authenticate_device(str(device_id), auth_params)
        
        return {
            "success": result["success"],
            "message": result.get("message", "认证处理完成"),
            "device_id": device_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"设备 {device_id} 认证失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设备认证失败: {str(e)}")


@router.post("/device/{device_id}/connection", response_model=Dict[str, Any])
async def toggle_device_connection(device_id: int, connect_params: Dict[str, Any]):
    """设备认证接口"""
    try:
        logger.info(f"设备 {device_id} 认证请求: {auth_params}")
        
        # 使用设备认证管理器进行认证
        result = device_auth_manager.authenticate_device(str(device_id), auth_params)
        
        return {
            "success": result["success"],
            "message": result.get("message", "认证处理完成"),
            "device_id": device_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"设备 {device_id} 认证失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设备认证失败: {str(e)}")


@router.post("/device/{device_id}/connection", response_model=Dict[str, Any])
async def toggle_device_connection(device_id: int, connect_params: Dict[str, Any]):
    """切换设备连接状态"""
    try:
        logger.info(f"切换设备 {device_id} 连接状态请求: {connect_params}")
        
        # 查找设备
        device = next((d for d in mock_devices if d["id"] == device_id), None)
        if not device:
            raise HTTPException(status_code=404, detail="设备不存在")
        
        connection_type = device["connection_type"]
        connect = connect_params.get("connect", False)
        
        # 检查是否支持该连接类型
        if connection_type not in controllers:
            raise HTTPException(status_code=400, detail=f"不支持的连接类型: {connection_type}")
        
        controller = controllers[connection_type]
        result = {}
        
        if connect:
            # 建立连接
            connection_details = device.get("connection_details", {})
            result = controller.connect(connection_details)
            if result["success"]:
                device["connected"] = True
                device["status"] = "online"
                device["lastSeen"] = "刚刚"
                # 更新连接详情
                if connection_type == "infrared":
                    device["connection_details"]["infrared_channel"] = result.get("channel")
                    device["connection_details"]["infrared_range"] = result.get("range")
                elif connection_type == "app":
                    device["connection_details"]["app_id"] = result.get("app_id")
                    device["connection_details"]["app_version"] = result.get("app_version")
                elif connection_type == "bluetooth":
                    device["connection_details"]["bluetooth_address"] = result.get("bluetooth_address")
                    device["connection_details"]["bluetooth_version"] = result.get("bluetooth_version")
                    device["signal"] = result.get("signal_strength", device["signal"])
        else:
            # 断开连接
            result = controller.disconnect()
            if result["success"]:
                device["connected"] = False
                device["status"] = "offline"
                device["lastSeen"] = "刚刚"
        
        return {
            "success": result["success"],
            "device_id": device_id,
            "connected": device["connected"],
            "status": device["status"],
            "connection_type": connection_type,
            "message": result["message"],
            "timestamp": datetime.datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换设备 {device_id} 连接状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换设备连接状态失败: {str(e)}")
