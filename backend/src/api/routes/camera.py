"""
摄像头控制API路由
提供摄像头的操作接口
"""

from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB, imencode
import base64
import asyncio
import logging

from src.core.services.camera_controller import CameraController
from src.core.services.ptz_camera_controller import (
    PTZCameraController, PTZProtocol, PTZAction, get_ptz_controller
)

router = APIRouter(prefix="/camera", tags=["camera"])

# 配置日志
logger = logging.getLogger(__name__)

# 创建摄像头控制器实例
camera_controller = CameraController()

# PTZ云台控制器（延迟初始化）
ptz_controller: Optional[PTZCameraController] = None


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


# ===== WebSocket 摄像头帧流接口（解决429限流问题）=====
@router.websocket("/ws/frame")
async def websocket_camera_frame(websocket: WebSocket):
    """
    WebSocket 摄像头帧流接口
    后端主动推送帧数据，避免前端高频轮询
    
    使用方法：
    const ws = new WebSocket("ws://127.0.0.1:8005/api/camera/ws/frame");
    ws.onmessage = (event) => {
        const frameData = JSON.parse(event.data);
        if (frameData.success) {
            // 使用 frameData.frame_base64
        }
    };
    """
    await websocket.accept()
    logger.info(f"WebSocket 客户端连接: {websocket.client}")
    
    try:
        # 设置帧率（30FPS = 33.3ms/帧）
        frame_interval = 1.0 / 30  # 30 FPS
        
        while True:
            try:
                # 获取当前帧
                frame = camera_controller.get_current_frame()
                
                if frame is not None:
                    # 转换为RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 压缩编码（质量70，减少带宽）
                    success, buffer = cv2.imencode('.jpg', rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    
                    if success:
                        # base64编码
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # 发送JSON格式数据
                        await websocket.send_json({
                            "success": True,
                            "frame_base64": frame_base64,
                            "timestamp": asyncio.get_event_loop().time()
                        })
                else:
                    # 摄像头未开启或读取失败
                    await websocket.send_json({
                        "success": False,
                        "message": "摄像头未开启或读取失败"
                    })
                
                # 控制帧率
                await asyncio.sleep(frame_interval)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket 客户端断开: {websocket.client}")
                break
            except Exception as e:
                logger.error(f"WebSocket 帧处理错误: {e}")
                await websocket.send_json({
                    "success": False,
                    "message": f"帧处理错误: {str(e)}"
                })
                await asyncio.sleep(frame_interval)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket 连接关闭: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket 连接错误: {e}")


# ==================== PTZ云台摄像头控制API ====================

class PTZConnectRequest(BaseModel):
    """PTZ连接请求"""
    protocol: str = "pelco_d"  # pelco_d, pelco_p, visca, onvif, http
    connection_type: str = "serial"  # serial, network, http
    
    # 串口参数
    port: Optional[str] = None
    baudrate: Optional[int] = 9600
    address: Optional[int] = 1
    
    # 网络参数
    host: Optional[str] = None
    network_port: Optional[int] = None
    
    # HTTP参数
    base_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class PTZActionRequest(BaseModel):
    """PTZ动作请求"""
    action: str  # pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out, stop, etc.
    speed: int = 50  # 0-100
    preset_id: Optional[int] = None  # 预置位编号


class PTZPositionRequest(BaseModel):
    """PTZ位置请求"""
    pan: float  # 水平角度 (-180 to 180)
    tilt: float  # 垂直角度 (-90 to 90)
    zoom: Optional[float] = None  # 变焦倍数
    speed: int = 50


class PTZPresetRequest(BaseModel):
    """PTZ预置位请求"""
    preset_id: int
    name: Optional[str] = None


class PTZPatrolRequest(BaseModel):
    """PTZ巡航请求"""
    presets: list  # [1, 2, 3, ...]
    dwell_time: int = 5  # 每个位置停留时间（秒）


@router.post("/ptz/connect", response_model=CameraResponse)
async def ptz_connect(request: PTZConnectRequest):
    """
    连接PTZ云台摄像头
    
    支持的协议：
    - pelco_d: Pelco-D协议（最常用）
    - pelco_p: Pelco-P协议
    - visca: VISCA协议（Sony等）
    - onvif: ONVIF标准协议
    - http: HTTP API接口
    
    支持的连接类型：
    - serial: 串口连接（RS-485/RS-232）
    - network: 网络连接（TCP/IP）
    - http: HTTP接口
    """
    global ptz_controller
    
    try:
        # 参数校验层：验证必需参数
        if not request.protocol:
            return CameraResponse(
                success=False,
                message="协议类型不能为空。支持的协议: pelco_d, pelco_p, visca, onvif, http",
                data={}
            )
        if not request.connection_type:
            return CameraResponse(
                success=False,
                message="连接类型不能为空。支持的类型: serial, network, http",
                data={}
            )
        
        # 解析协议
        protocol_map = {
            "pelco_d": PTZProtocol.PELCO_D,
            "pelco_p": PTZProtocol.PELCO_P,
            "visca": PTZProtocol.VISCA,
            "onvif": PTZProtocol.ONVIF,
            "http": PTZProtocol.HTTP_API
        }
        
        protocol = protocol_map.get(request.protocol.lower())
        if not protocol:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的协议: {request.protocol}。支持的协议: pelco_d, pelco_p, visca, onvif, http"
            )
        
        # 准备连接参数
        connection_params = {}
        
        if request.connection_type == "serial":
            if not request.port:
                raise HTTPException(status_code=400, detail="串口连接需要提供port参数（如COM3）")
            connection_params = {
                "port": request.port,
                "baudrate": request.baudrate or 9600,
                "address": request.address or 1
            }
            logger.info(f"准备串口连接 - 端口:{request.port}, 波特率:{connection_params['baudrate']}")
            
        elif request.connection_type == "network":
            if not request.host or not request.network_port:
                raise HTTPException(status_code=400, detail="网络连接需要提供host和network_port参数")
            connection_params = {
                "host": request.host,
                "port": request.network_port,
                "address": request.address or 1
            }
            logger.info(f"准备网络连接 - 主机:{request.host}:{request.network_port}")
            
        elif request.connection_type == "http":
            if not request.base_url:
                raise HTTPException(status_code=400, detail="HTTP连接需要提供base_url参数")
            if not request.username or not request.password:
                raise HTTPException(status_code=400, detail="HTTP连接需要提供用户名和密码")
            connection_params = {
                "base_url": request.base_url,
                "username": request.username,
                "password": request.password
            }
            logger.info(f"准备HTTP连接 - URL:{request.base_url}")
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的连接类型: {request.connection_type}。支持的类型: serial, network, http"
            )
        
        # 对象判空层：创建PTZ控制器
        try:
            ptz_controller = PTZCameraController(
                protocol=protocol,
                connection_type=request.connection_type,
                **connection_params
            )
        except Exception as e:
            logger.error(f"PTZ控制器初始化失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"PTZ控制器初始化失败: {str(e)}"
            )
        
        # 判空检查
        if not ptz_controller:
            raise HTTPException(
                status_code=500,
                detail="PTZ控制器创建失败，对象为空"
            )
        
        # 全局异常层：执行连接
        try:
            result = await ptz_controller.connect()
        except Exception as e:
            logger.error(f"PTZ设备连接执行异常: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"PTZ设备连接失败: {str(e)}。请检查设备是否开启、线缆是否连接、端口/IP是否正确"
            )
        
        # 检查连接结果
        if result and result.get("success"):
            logger.info(f"PTZ云台连接成功: {request.protocol}@{request.connection_type}")
            return CameraResponse(
                success=True,
                message="PTZ云台连接成功",
                data=result.get("connection_info", {})
            )
        else:
            error_msg = result.get("message", "未知错误") if result else "连接返回结果为空"
            logger.error(f"PTZ连接失败: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"PTZ连接失败: {error_msg}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PTZ连接未捕获异常: {type(e).__name__} - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PTZ连接错误: {str(e)}"
        )


@router.post("/ptz/disconnect", response_model=CameraResponse)
async def ptz_disconnect():
    """
    断开PTZ云台连接
    """
    global ptz_controller
    
    if not ptz_controller:
        raise HTTPException(status_code=400, detail="PTZ云台未连接")
    
    try:
        result = await ptz_controller.disconnect()
        
        if result["success"]:
            ptz_controller = None
            return CameraResponse(
                success=True,
                message="PTZ云台已断开"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"断开失败: {result['message']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PTZ断开错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PTZ断开错误: {str(e)}"
        )


@router.post("/ptz/action", response_model=CameraResponse)
async def ptz_action(request: PTZActionRequest):
    """
    执行PTZ云台动作
    
    支持的动作：
    - pan_left: 向左转
    - pan_right: 向右转
    - tilt_up: 向上转
    - tilt_down: 向下转
    - zoom_in: 拉近
    - zoom_out: 拉远
    - focus_near: 近焦
    - focus_far: 远焦
    - iris_open: 光圈开
    - iris_close: 光圈关
    - stop: 停止
    - preset_set: 设置预置位（需要preset_id）
    - preset_goto: 转到预置位（需要preset_id）
    - auto_scan: 自动扫描
    - patrol: 巡航
    """
    global ptz_controller
    
    if not ptz_controller or not ptz_controller.is_connected:
        raise HTTPException(status_code=400, detail="PTZ云台未连接")
    
    try:
        # 解析动作
        action_map = {
            "pan_left": PTZAction.PAN_LEFT,
            "pan_right": PTZAction.PAN_RIGHT,
            "tilt_up": PTZAction.TILT_UP,
            "tilt_down": PTZAction.TILT_DOWN,
            "zoom_in": PTZAction.ZOOM_IN,
            "zoom_out": PTZAction.ZOOM_OUT,
            "focus_near": PTZAction.FOCUS_NEAR,
            "focus_far": PTZAction.FOCUS_FAR,
            "iris_open": PTZAction.IRIS_OPEN,
            "iris_close": PTZAction.IRIS_CLOSE,
            "preset_set": PTZAction.PRESET_SET,
            "preset_goto": PTZAction.PRESET_GOTO,
            "auto_scan": PTZAction.AUTO_SCAN,
            "patrol": PTZAction.PATROL,
            "stop": PTZAction.STOP
        }
        
        action = action_map.get(request.action.lower())
        if not action:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的动作: {request.action}"
            )
        
        # 执行动作
        params = {}
        if request.preset_id is not None:
            params["preset_id"] = request.preset_id
        
        result = await ptz_controller.execute_action(
            action=action,
            speed=request.speed,
            **params
        )
        
        if result["success"]:
            return CameraResponse(
                success=True,
                message=result["message"],
                data=result.get("new_state")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"动作执行失败: {result['message']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PTZ动作执行错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PTZ动作执行错误: {str(e)}"
        )


@router.post("/ptz/move", response_model=CameraResponse)
async def ptz_move_to_position(request: PTZPositionRequest):
    """
    移动PTZ云台到指定位置（绝对位置控制）
    
    参数：
    - pan: 水平角度 (-180 to 180)
    - tilt: 垂直角度 (-90 to 90)
    - zoom: 变焦倍数（可选）
    - speed: 移动速度 (0-100)
    """
    global ptz_controller
    
    if not ptz_controller or not ptz_controller.is_connected:
        raise HTTPException(status_code=400, detail="PTZ云台未连接")
    
    try:
        result = await ptz_controller.move_to_position(
            pan=request.pan,
            tilt=request.tilt,
            zoom=request.zoom,
            speed=request.speed
        )
        
        if result["success"]:
            return CameraResponse(
                success=True,
                message=result["message"],
                data=result.get("current_position")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"移动失败: {result['message']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PTZ移动错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PTZ移动错误: {str(e)}"
        )


@router.post("/ptz/preset/set", response_model=CameraResponse)
async def ptz_set_preset(request: PTZPresetRequest):
    """
    设置PTZ预置位
    
    参数：
    - preset_id: 预置位编号 (1-256)
    - name: 预置位名称（可选）
    """
    global ptz_controller
    
    if not ptz_controller or not ptz_controller.is_connected:
        raise HTTPException(status_code=400, detail="PTZ云台未连接")
    
    try:
        result = await ptz_controller.set_preset(
            preset_id=request.preset_id,
            name=request.name
        )
        
        if result["success"]:
            return CameraResponse(
                success=True,
                message=result["message"],
                data=result.get("preset")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"设置预置位失败: {result['message']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置预置位错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"设置预置位错误: {str(e)}"
        )


@router.post("/ptz/preset/goto", response_model=CameraResponse)
async def ptz_goto_preset(request: PTZPresetRequest):
    """
    转到PTZ预置位
    
    参数：
    - preset_id: 预置位编号
    """
    global ptz_controller
    
    if not ptz_controller or not ptz_controller.is_connected:
        raise HTTPException(status_code=400, detail="PTZ云台未连接")
    
    try:
        result = await ptz_controller.goto_preset(preset_id=request.preset_id)
        
        if result["success"]:
            return CameraResponse(
                success=True,
                message=result["message"],
                data={
                    "preset": result.get("preset"),
                    "position": result.get("current_position")
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"转到预置位失败: {result['message']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转到预置位错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"转到预置位错误: {str(e)}"
        )


@router.post("/ptz/patrol", response_model=CameraResponse)
async def ptz_auto_patrol(request: PTZPatrolRequest):
    """
    PTZ自动巡航（依次访问多个预置位）
    
    参数：
    - presets: 预置位列表 [1, 2, 3, ...]
    - dwell_time: 每个位置停留时间（秒）
    """
    global ptz_controller
    
    if not ptz_controller or not ptz_controller.is_connected:
        raise HTTPException(status_code=400, detail="PTZ云台未连接")
    
    try:
        # 异步启动巡航任务
        asyncio.create_task(
            ptz_controller.auto_patrol(
                presets=request.presets,
                dwell_time=request.dwell_time
            )
        )
        
        return CameraResponse(
            success=True,
            message=f"巡航任务已启动，将访问{len(request.presets)}个预置位",
            data={
                "presets": request.presets,
                "dwell_time": request.dwell_time
            }
        )
    
    except Exception as e:
        logger.error(f"启动巡航错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"启动巡航错误: {str(e)}"
        )


@router.get("/ptz/status", response_model=CameraResponse)
async def ptz_get_status():
    """
    获取PTZ云台当前状态
    
    返回：
    - connected: 连接状态
    - protocol: 协议类型
    - connection_type: 连接类型
    - position: 当前位置 {pan, tilt, zoom}
    - presets: 已设置的预置位
    """
    global ptz_controller
    
    if not ptz_controller:
        return CameraResponse(
            success=True,
            message="PTZ云台未连接",
            data={"connected": False}
        )
    
    try:
        status = ptz_controller.get_status()
        
        return CameraResponse(
            success=True,
            message="获取状态成功",
            data=status
        )
    
    except Exception as e:
        logger.error(f"获取PTZ状态错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取状态错误: {str(e)}"
        )
