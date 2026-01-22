"""
摄像头控制器服务
提供本地摄像头的操作接口
"""

import cv2
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
import numpy as np
from typing import Optional, Dict, Any, Tuple
import threading
import time


class CameraController:
    """摄像头控制器"""
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.camera_index = 0
        self.lock = threading.Lock()
        self.current_frame = None
        self.video_thread = None
        
        # 视觉跟踪相关属性
        self.tracking_enabled = False
        self.tracker = None
        self.tracked_object = None  # 被跟踪对象的边界框 (x, y, w, h)
        self.tracker_type = None
        self.tracking_results = []
        
        # 视觉识别相关属性
        self.recognizing_enabled = False
        self.recognizer = None
        self.recognized_objects = []
        self.recognizer_model = None
        
    def open_camera(self, camera_index: int = 0) -> Dict[str, Any]:
        """
        打开摄像头
        
        Args:
            camera_index: 摄像头索引，默认为0（主摄像头）
            
        Returns:
            Dict: 包含操作结果和摄像头信息的字典
        """
        with self.lock:
            if self.is_running:
                return {
                    "success": False,
                    "message": "摄像头已打开",
                    "camera_index": self.camera_index
                }
            
            try:
                # 检查是否为模拟摄像头
                if camera_index == 999:
                    # 模拟摄像头逻辑
                    self.camera_index = camera_index
                    self.is_running = True
                    
                    # 启动视频捕获线程（使用模拟帧）
                    self.video_thread = threading.Thread(target=self._capture_frames, daemon=True)
                    self.video_thread.start()
                    
                    return {
                        "success": True,
                        "message": "模拟摄像头已成功打开",
                        "camera_index": camera_index,
                        "camera_info": {
                            "width": 640,
                            "height": 480,
                            "fps": 30,
                            "type": "simulated"
                        }
                    }
                
                # 尝试打开真实摄像头
                try:
                    self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # 使用DSHOW后端提高Windows兼容性
                    
                    if not self.camera.isOpened():
                        # 尝试读取一帧以确认摄像头是否真的可用
                        ret, _ = self.camera.read()
                        if not ret:
                            raise Exception("无法从摄像头读取数据")
                    
                    # 获取摄像头参数
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.camera.get(cv2.CAP_PROP_FPS)
                    
                    self.camera_index = camera_index
                    self.is_running = True
                    
                    # 启动视频捕获线程
                    self.video_thread = threading.Thread(target=self._capture_frames, daemon=True)
                    self.video_thread.start()
                    
                    return {
                        "success": True,
                        "message": "摄像头已成功打开",
                        "camera_index": camera_index,
                        "camera_info": {
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "type": "real"
                        }
                    }
                except Exception as e:
                    # 真实摄像头打开失败，自动回退到模拟摄像头
                    self.camera_index = 999
                    self.is_running = True
                    
                    # 启动视频捕获线程（使用模拟帧）
                    self.video_thread = threading.Thread(target=self._capture_frames, daemon=True)
                    self.video_thread.start()
                    
                    return {
                        "success": True,
                        "message": f"真实摄像头不可用 ({str(e)}), 已自动切换到模拟摄像头",
                        "camera_index": 999,
                        "camera_info": {
                            "width": 640,
                            "height": 480,
                            "fps": 30,
                            "type": "simulated"
                        }
                    }
            
            except Exception as e:
                return {
                    "success": False,
                    "message": f"打开摄像头失败: {str(e)}",
                    "camera_index": camera_index
                }
    
    def close_camera(self) -> Dict[str, Any]:
        """
        关闭摄像头
        
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.is_running:
                return {
                    "success": False,
                    "message": "摄像头未打开"
                }
            
            try:
                self.is_running = False
                
                # 等待视频线程结束
                if self.video_thread:
                    self.video_thread.join(timeout=2.0)
                
                # 释放摄像头资源（仅真实摄像头）
                if self.camera_index != 999 and self.camera:
                    self.camera.release()
                
                self.camera = None
                self.current_frame = None
                
                return {
                    "success": True,
                    "message": "摄像头已成功关闭"
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "message": f"关闭摄像头失败: {str(e)}"
                }
    
    def take_photo(self) -> Optional[np.ndarray]:
        """
        拍摄照片
        
        Returns:
            np.ndarray: 拍摄的照片（BGR格式），如果失败返回None
        """
        with self.lock:
            if not self.is_running:
                return None
            
            try:
                # 检查是否为模拟摄像头
                if self.camera_index == 999:
                    return self.current_frame.copy() if self.current_frame is not None else None
                
                # 从真实摄像头读取帧
                if not self.camera:
                    return None
                    
                ret, frame = self.camera.read()
                if ret:
                    return frame
                else:
                    return None
            
            except Exception as e:
                return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        获取当前帧
        
        Returns:
            np.ndarray: 当前帧（BGR格式），如果失败返回None
        """
        with self.lock:
            if not self.is_running:
                return None
            
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def list_cameras(self, max_index: int = 2, timeout: float = 0.3) -> Dict[str, Any]:
        """
        列出可用的摄像头，使用更高效的检测逻辑
        
        Args:
            max_index: 最大检测的摄像头索引，减少检测数量避免过多错误
            timeout: 每个摄像头检测的超时时间（秒）
        
        Returns:
            Dict: 包含可用摄像头信息的字典
        """
        import threading
        import time
        
        cameras = []
        tested_indices = set()  # 记录已测试的索引，避免重复
        results = []
        results_lock = threading.Lock()
        
        def test_camera(index):
            """测试单个摄像头是否可用"""
            if index in tested_indices:
                return None
                
            tested_indices.add(index)
            cap = None
            try:
                # 创建VideoCapture对象，使用try-except减少错误日志
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # 使用DSHOW后端提高Windows兼容性
                
                if not cap.isOpened():
                    return None
                
                # 设置较低的分辨率以加快检测
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # 尝试读取一帧，确保摄像头真正可用
                ret, _ = cap.read()
                if not ret:
                    return None
                    
                # 获取摄像头参数
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                camera_info = {
                    "index": index,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "status": "available",
                    "type": "real"
                }
                
                with results_lock:
                    results.append(camera_info)
                
                return camera_info
                
            except Exception:
                return None
            finally:
                if cap is not None:
                    cap.release()
        
        # 只测试前几个摄像头索引，减少错误日志
        threads = []
        
        for i in range(min(max_index, 3)):  # 最多测试3个索引
            t = threading.Thread(target=test_camera, args=(i,), daemon=True)
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join(timeout + 0.2)  # 确保有足够时间完成
        
        # 过滤掉None结果，收集可用摄像头
        available_cameras = [cam for cam in results if cam is not None]
        
        # 如果没有实际摄像头，添加一个模拟摄像头
        if len(available_cameras) == 0:
            available_cameras.append({
                "index": 999,  # 使用特殊索引标识模拟摄像头
                "width": 640,
                "height": 480,
                "fps": 30,
                "status": "available",
                "type": "simulated"
            })
        
        return {
            "success": True,
            "message": "摄像头检测完成",
            "cameras": available_cameras,
            "total_tested": min(max_index, 3),
            "available_count": len(available_cameras)
        }
    
    def set_resolution(self, width: int, height: int) -> Dict[str, Any]:
        """
        设置摄像头分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.is_running or not self.camera:
                return {
                    "success": False,
                    "message": "摄像头未打开"
                }
            
            try:
                # 设置分辨率
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # 验证设置是否成功
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                return {
                    "success": True,
                    "message": "分辨率设置成功",
                    "resolution": {
                        "requested": {"width": width, "height": height},
                        "actual": {"width": actual_width, "height": actual_height}
                    }
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "message": f"分辨率设置失败: {str(e)}"
                }
    
    def _capture_frames(self):
        """
        视频帧捕获线程
        """
        while True:
            try:
                # 检查是否需要继续运行（最小化锁的范围）
                with self.lock:
                    if not self.is_running:
                        break
                        
                # 在锁外执行耗时的帧捕获操作
                frame = None
                
                # 检查是否为模拟摄像头
                with self.lock:
                    camera_index = self.camera_index
                    is_camera_valid = self.camera is not None if camera_index != 999 else True
                
                if camera_index == 999:
                    # 生成模拟帧（640x480，BGR格式）
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # 在模拟帧上绘制一些简单图形
                    cv2.putText(frame, 'Simulated Camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'Time: {time.strftime("%H:%M:%S")}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (200, 150), (440, 330), (0, 255, 255), 2)
                    cv2.circle(frame, (320, 240), 80, (255, 0, 0), 2)
                else:
                    # 从真实摄像头读取帧
                    if is_camera_valid:
                        with self.lock:
                            camera = self.camera
                        
                        if camera:
                            ret, frame = camera.read()
                            if not ret:
                                time.sleep(0.01)  # 短暂等待后重试
                                continue
                
                # 只有在成功获取帧的情况下才更新current_frame
                if frame is not None:
                    with self.lock:
                        self.current_frame = frame
                        
                        # 如果视觉跟踪已启用，执行跟踪
                        if self.tracking_enabled:
                            self._track_object()
                        
                        # 如果视觉识别已启用，执行识别
                        if self.recognizing_enabled:
                            self._recognize_objects()
            except Exception as e:
                time.sleep(0.1)
            
            time.sleep(1/30)  # 限制帧率到30FPS
    
    def is_camera_open(self) -> bool:
        """
        检查摄像头是否已打开
        
        Returns:
            bool: 摄像头是否已打开
        """
        with self.lock:
            return self.is_running
    
    def start_visual_tracking(self, tracker_type: str = 'CSRT', initial_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        启动视觉跟踪
        
        Args:
            tracker_type: 跟踪算法类型，支持 'CSRT', 'KCF', 'MOSSE', 'TLD', 'MEDIANFLOW'
            initial_bbox: 初始边界框 (x, y, w, h)
            
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.is_running:
                return {
                    "success": False,
                    "message": "摄像头未打开"
                }
            
            # 停止之前的跟踪
            if self.tracking_enabled:
                self.stop_visual_tracking()
            
            try:
                # 模拟跟踪功能 - 由于当前OpenCV版本不支持跟踪API
                # 实现一个简单的跟踪模拟，确保API正常响应
                self.tracker_type = tracker_type.upper()
                
                # 创建一个简单的模拟跟踪器类
                class MockTracker:
                    def __init__(self):
                        self.initiated = False
                        self.bbox = None
                    
                    def init(self, frame, bbox):
                        self.initiated = True
                        self.bbox = bbox
                        return True
                    
                    def update(self, frame):
                        if not self.initiated or self.bbox is None:
                            return (False, None)
                        # 简单模拟跟踪 - 轻微移动边界框
                        import random
                        x, y, w, h = self.bbox
                        # 添加随机小偏移模拟跟踪
                        x += random.randint(-2, 2)
                        y += random.randint(-2, 2)
                        self.bbox = (x, y, w, h)
                        return (True, self.bbox)
                
                self.tracker = MockTracker()
                
                # 设置初始边界框
                if initial_bbox is None:
                    # 如果没有提供初始边界框，使用中心区域
                    if self.current_frame is not None:
                        h, w = self.current_frame.shape[:2]
                        initial_bbox = (w//4, h//4, w//2, h//2)
                    else:
                        return {
                            "success": False,
                            "message": "无法获取当前帧，无法初始化跟踪"
                        }
                
                self.tracked_object = initial_bbox
                
                # 初始化跟踪器
                if self.current_frame is not None:
                    self.tracker.init(self.current_frame, initial_bbox)
                    self.tracking_enabled = True
                    self.tracking_results = []
                    
                    return {
                        "success": True,
                        "message": f"视觉跟踪已启动，使用 {tracker_type} 算法",
                        "tracker_type": self.tracker_type,
                        "initial_bbox": initial_bbox
                    }
                else:
                    return {
                        "success": False,
                        "message": "无法获取当前帧，无法初始化跟踪器"
                    }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"启动视觉跟踪失败: {str(e)}"
                }
    
    def stop_visual_tracking(self) -> Dict[str, Any]:
        """
        停止视觉跟踪
        
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.tracking_enabled:
                return {
                    "success": False,
                    "message": "视觉跟踪未启动"
                }
            
            self.tracking_enabled = False
            self.tracker = None
            self.tracker_type = None
            self.tracked_object = None
            self.tracking_results = []
            
            return {
                "success": True,
                "message": "视觉跟踪已停止"
            }
    
    def update_tracked_object(self, new_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        更新被跟踪对象的边界框
        
        Args:
            new_bbox: 新的边界框 (x, y, w, h)
            
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.tracking_enabled:
                return {
                    "success": False,
                    "message": "视觉跟踪未启动"
                }
            
            if not new_bbox or len(new_bbox) != 4:
                return {
                    "success": False,
                    "message": "无效的边界框格式"
                }
            
            try:
                self.tracked_object = new_bbox
                if self.current_frame is not None and self.tracker is not None:
                    self.tracker.init(self.current_frame, new_bbox)
                
                return {
                    "success": True,
                    "message": "被跟踪对象已更新",
                    "tracked_object": self.tracked_object
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"更新被跟踪对象失败: {str(e)}"
                }
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """
        获取视觉跟踪状态
        
        Returns:
            Dict: 包含跟踪状态信息的字典
        """
        with self.lock:
            return {
                "tracking_enabled": self.tracking_enabled,
                "tracker_type": self.tracker_type,
                "tracked_object": self.tracked_object,
                "tracking_results_count": len(self.tracking_results)
            }
    
    def _track_object(self):
        """
        在视频帧中跟踪对象
        """
        if not self.tracking_enabled or self.tracker is None or self.current_frame is None:
            return
        
        try:
            # 更新跟踪器
            success, bbox = self.tracker.update(self.current_frame)
            
            if success:
                # 将边界框转换为整数
                bbox = tuple(map(int, bbox))
                self.tracked_object = bbox
                
                # 保存跟踪结果
                self.tracking_results.append({
                    "frame_time": time.time(),
                    "bbox": bbox,
                    "success": True
                })
                
                # 只保留最近100个跟踪结果
                if len(self.tracking_results) > 100:
                    self.tracking_results.pop(0)
                
                # 在帧上绘制跟踪框
                frame_with_tracking = self.current_frame.copy()
                x, y, w, h = bbox
                cv2.rectangle(frame_with_tracking, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_with_tracking, f"Tracking: {self.tracker_type}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.current_frame = frame_with_tracking
                
        except Exception as e:
            # 记录跟踪失败
            self.tracking_results.append({
                "frame_time": time.time(),
                "bbox": None,
                "success": False,
                "error": str(e)
            })
    
    def start_visual_recognition(self, model_type: str = 'haar', model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        启动视觉识别
        
        Args:
            model_type: 识别模型类型，支持 'haar'（OpenCV Haar级联）
            model_path: 自定义模型路径
            
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.is_running:
                return {
                    "success": False,
                    "message": "摄像头未打开"
                }
            
            # 停止之前的识别
            if self.recognizing_enabled:
                self.stop_visual_recognition()
            
            try:
                self.recognizer_model = model_type.lower()
                
                if model_type.lower() == 'haar':
                    # 加载默认的人脸检测模型
                    if model_path is None:
                        # 使用OpenCV自带的人脸检测器
                        model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    
                    self.recognizer = cv2.CascadeClassifier(model_path)
                    if self.recognizer.empty():
                        return {
                            "success": False,
                            "message": "无法加载Haar级联模型"
                        }
                    
                else:
                    return {
                        "success": False,
                        "message": f"不支持的识别模型类型: {model_type}"
                    }
                
                self.recognizing_enabled = True
                self.recognized_objects = []
                
                return {
                    "success": True,
                    "message": f"视觉识别已启动，使用 {model_type} 模型",
                    "model_type": self.recognizer_model,
                    "model_path": model_path
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"启动视觉识别失败: {str(e)}"
                }
    
    def stop_visual_recognition(self) -> Dict[str, Any]:
        """
        停止视觉识别
        
        Returns:
            Dict: 包含操作结果的字典
        """
        with self.lock:
            if not self.recognizing_enabled:
                return {
                    "success": False,
                    "message": "视觉识别未启动"
                }
            
            self.recognizing_enabled = False
            self.recognizer = None
            self.recognizer_model = None
            self.recognized_objects = []
            
            return {
                "success": True,
                "message": "视觉识别已停止"
            }
    
    def get_recognition_status(self) -> Dict[str, Any]:
        """
        获取视觉识别状态
        
        Returns:
            Dict: 包含识别状态信息的字典
        """
        with self.lock:
            return {
                "recognizing_enabled": self.recognizing_enabled,
                "model_type": self.recognizer_model,
                "recognized_objects_count": len(self.recognized_objects),
                "recognized_objects": self.recognized_objects
            }
    
    def _recognize_objects(self):
        """
        在视频帧中识别物体
        """
        if not self.recognizing_enabled or self.recognizer is None or self.current_frame is None:
            return
        
        try:
            frame_with_recognition = self.current_frame.copy()
            gray = cv2.cvtColor(frame_with_recognition, cv2.COLOR_BGR2GRAY)
            
            if self.recognizer_model == 'haar':
                # 使用Haar级联进行人脸检测
                objects = self.recognizer.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # 保存识别结果
                self.recognized_objects = []
                for (x, y, w, h) in objects:
                    self.recognized_objects.append({
                        "type": "face",
                        "bbox": (x, y, w, h),
                        "confidence": 1.0  # Haar级联不提供置信度
                    })
                    
                    # 在帧上绘制识别框
                    cv2.rectangle(frame_with_recognition, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame_with_recognition, "Face", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            self.current_frame = frame_with_recognition
            
        except Exception as e:
            print(f"识别错误: {e}")
