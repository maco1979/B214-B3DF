"""
摄像头控制API路由
提供摄像头的操作接口
"""

from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB, imencode
import base64

from src.core.services.camera_controller import CameraController

router = APIRouter(prefix="/camera", tags=["camera"])

# 创建摄像头控制器实例
camera_controller = CameraController()


class CameraOpenRequest(BaseModel):
    """打开摄像头请求"""
    camera_index: int = 0


class ResolutionRequest(BaseModel):
    """分辨率设置请求"""
    width: int
    height: int


class TrackingStartRequest(BaseModel):
    """启动视觉跟踪请求"""
    tracker_type: str = 'CSRT'
    initial_bbox: Optional[Tuple[int, int, int, int]] = None


class TrackingUpdateRequest(BaseModel):
    """更新跟踪对象请求"""
    new_bbox: Tuple[int, int, int, int]


class RecognitionStartRequest(BaseModel):
    """启动视觉识别请求"""
    model_type: str = 'haar'
    model_path: Optional[str] = None


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


@router.post("/open", response_model=CameraResponse)
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


@router.post("/close", response_model=CameraResponse)
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


@router.get("/status", response_model=CameraResponse)
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


@router.get("/list", response_model=CameraResponse)
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


@router.post("/resolution", response_model=CameraResponse)
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


@router.get("/photo", response_model=FrameResponse)
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


@router.get("/frame", response_model=CameraResponse)
async def get_current_frame():
    """
    获取当前帧
    
    Returns:
        CameraResponse: 包含当前帧的响应
    """
    try:
        frame = camera_controller.get_current_frame()
        
        if frame is None:
            return CameraResponse(
                success=False,
                message="获取帧失败，请检查摄像头是否已打开"
            )
        
        # 将BGR格式转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像编码为JPEG格式，设置压缩质量为70以减少数据量
        success, buffer = cv2.imencode('.jpg', rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        
        # 转换为base64字符串
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return CameraResponse(
            success=True,
            message="获取帧成功",
            data={"frame_base64": frame_base64}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取帧失败: {str(e)}"
        )


@router.post("/tracking/start", response_model=CameraResponse)
async def start_tracking(request: TrackingStartRequest):
    """
    启动视觉跟踪
    
    Args:
        request: 包含跟踪参数的请求
        
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.start_visual_tracking(
            tracker_type=request.tracker_type,
            initial_bbox=request.initial_bbox
        )
        return CameraResponse(
            success=result["success"],
            message=result["message"],
            data={k: v for k, v in result.items() if k not in ["success", "message"]}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动视觉跟踪失败: {str(e)}"
        )


@router.post("/tracking/stop", response_model=CameraResponse)
async def stop_tracking():
    """
    停止视觉跟踪
    
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.stop_visual_tracking()
        return CameraResponse(
            success=result["success"],
            message=result["message"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止视觉跟踪失败: {str(e)}"
        )


@router.post("/tracking/update", response_model=CameraResponse)
async def update_tracking(request: TrackingUpdateRequest):
    """
    更新跟踪对象
    
    Args:
        request: 包含新边界框的请求
        
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.update_tracked_object(request.new_bbox)
        return CameraResponse(
            success=result["success"],
            message=result["message"],
            data={k: v for k, v in result.items() if k not in ["success", "message"]}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新跟踪对象失败: {str(e)}"
        )


@router.get("/tracking/status", response_model=CameraResponse)
async def get_tracking_status():
    """
    获取视觉跟踪状态
    
    Returns:
        CameraResponse: 包含跟踪状态的响应
    """
    try:
        status = camera_controller.get_tracking_status()
        return CameraResponse(
            success=True,
            message="获取跟踪状态成功",
            data=status
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取跟踪状态失败: {str(e)}"
        )


@router.post("/recognition/start", response_model=CameraResponse)
async def start_recognition(request: RecognitionStartRequest):
    """
    启动视觉识别
    
    Args:
        request: 包含识别参数的请求
        
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.start_visual_recognition(
            model_type=request.model_type,
            model_path=request.model_path
        )
        return CameraResponse(
            success=result["success"],
            message=result["message"],
            data={k: v for k, v in result.items() if k not in ["success", "message"]}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动视觉识别失败: {str(e)}"
        )


@router.post("/recognition/stop", response_model=CameraResponse)
async def stop_recognition():
    """
    停止视觉识别
    
    Returns:
        CameraResponse: 操作结果
    """
    try:
        result = camera_controller.stop_visual_recognition()
        return CameraResponse(
            success=result["success"],
            message=result["message"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止视觉识别失败: {str(e)}"
        )


@router.get("/recognition/status", response_model=CameraResponse)
async def get_recognition_status():
    """
    获取视觉识别状态
    
    Returns:
        CameraResponse: 包含识别状态的响应
    """
    try:
        status = camera_controller.get_recognition_status()
        return CameraResponse(
            success=True,
            message="获取识别状态成功",
            data=status
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取识别状态失败: {str(e)}"
        )
