#!/usr/bin/env python3
"""
独立的摄像头控制服务器
基于FastAPI，提供摄像头操作的RESTful API接口
"""

import sys
import os

# 确保能导入摄像头控制器
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB, imencode
import numpy as np
import base64

# 直接导入CameraController，避免导入不必要的依赖
import importlib.util

# 动态导入camera_controller模块
spec = importlib.util.spec_from_file_location(
    "camera_controller", 
    os.path.join(os.path.dirname(__file__), "src", "core", "services", "camera_controller.py")
)
camera_controller_module = importlib.util.module_from_spec(spec)
sys.modules["camera_controller"] = camera_controller_module
spec.loader.exec_module(camera_controller_module)

# 获取CameraController类
CameraController = camera_controller_module.CameraController

# 创建FastAPI应用
app = FastAPI(
    title="摄像头控制服务",
    description="提供本地摄像头的操作接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建摄像头控制器实例
camera_controller = CameraController()


# Pydantic模型
class CameraOpenRequest(BaseModel):
    """打开摄像头请求"""
    camera_index: int = 0


class ResolutionRequest(BaseModel):
    """分辨率设置请求"""
    width: int
    height: int


class CameraResponse(BaseModel):
    """摄像头操作响应"""
    success: bool
    message: str
    data: Optional[dict] = None


class FrameResponse(BaseModel):
    """帧数据响应"""
    success: bool
    message: str
    frame_base64: Optional[str] = None


# API路由
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "摄像头控制服务",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/camera/open", response_model=CameraResponse)
async def open_camera(request: CameraOpenRequest):
    """
    打开摄像头
    
    Args:
        request: 包含摄像头索引的请求
        
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.open_camera(request.camera_index)
        return CameraResponse(
            success=result["success"],
            message=result["message"],
            data={k: v for k, v in result.items() if k not in ["success", "message"]}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"打开摄像头失败: {str(e)}"
        )


@app.post("/camera/close", response_model=CameraResponse)
async def close_camera():
    """
    关闭摄像头
    
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.close_camera()
        return CameraResponse(
            success=result["success"],
            message=result["message"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关闭摄像头失败: {str(e)}"
        )


@app.get("/camera/status", response_model=CameraResponse)
async def get_camera_status():
    """
    获取摄像头状态
    
    Returns:
        CameraResponse: 包含摄像头状态的响应
    """
    try:
        is_open = camera_controller.is_camera_open()
        return CameraResponse(
            success=True,
            message="摄像头状态查询成功",
            data={
                "is_open": is_open,
                "camera_index": camera_controller.camera_index
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询摄像头状态失败: {str(e)}"
        )


@app.get("/camera/list", response_model=CameraResponse)
async def list_cameras():
    """
    列出可用的摄像头
    
    Returns:
        CameraResponse: 包含可用摄像头列表的响应
    """
    try:
        result = camera_controller.list_cameras()
        return CameraResponse(
            success=result["success"],
            message=result["message"],
            data={"cameras": result["cameras"]}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出摄像头失败: {str(e)}"
        )


@app.post("/camera/resolution", response_model=CameraResponse)
async def set_camera_resolution(request: ResolutionRequest):
    """
    设置摄像头分辨率
    
    Args:
        request: 包含目标分辨率的请求
        
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.set_resolution(request.width, request.height)
        return CameraResponse(
            success=result["success"],
            message=result["message"],
            data={"resolution": result.get("resolution")}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置分辨率失败: {str(e)}"
        )


@app.get("/camera/photo", response_model=FrameResponse)
async def take_photo():
    """
    拍摄照片
    
    Returns:
        FrameResponse: 包含拍摄照片的响应
    """
    try:
        frame = camera_controller.take_photo()
        
        if frame is None:
            return FrameResponse(
                success=False,
                message="拍摄失败，请检查摄像头是否已打开"
            )
        
        # 将BGR格式转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 将图像编码为JPEG格式
        _, buffer = cv2.imencode('.jpg', rgb_frame)
        
        # 转换为base64字符串
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return FrameResponse(
            success=True,
            message="拍摄成功",
            frame_base64=frame_base64
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"拍摄失败: {str(e)}"
        )


@app.get("/camera/frame", response_model=FrameResponse)
async def get_current_frame():
    """
    获取当前帧
    
    Returns:
        FrameResponse: 包含当前帧的响应
    """
    try:
        frame = camera_controller.get_current_frame()
        
        if frame is None:
            return FrameResponse(
                success=False,
                message="获取帧失败，请检查摄像头是否已打开"
            )
        
        # 将BGR格式转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 将图像编码为JPEG格式
        _, buffer = cv2.imencode('.jpg', rgb_frame)
        
        # 转换为base64字符串
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return FrameResponse(
            success=True,
            message="获取帧成功",
            frame_base64=frame_base64
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取帧失败: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
