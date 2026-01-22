#!/usr/bin/env python3
"""AI PTZ云台转动检测脚本"""

import asyncio
import time
import cv2
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class AIPtzDetection:
    """AI PTZ云台转动检测类"""
    
    def __init__(self):
        """初始化检测系统"""
        self.monitor_camera = CameraController()  # 监控摄像头（第二个摄像头）
        self.ptz_camera = CameraController()      # PTZ摄像头（第一个摄像头）
        self.ptz_controller = None                # PTZ控制器
        self.detection_results = []               # 检测结果
        self.is_monitoring = False                # 监控状态
    
    def list_available_cameras(self):
        """列出可用摄像头"""
        print("=== 检测可用摄像头 ===")
        result = self.monitor_camera.list_cameras(max_index=3)
        
        if result["success"]:
            print(f"检测到 {result['available_count']} 个可用摄像头:")
            for i, cam in enumerate(result["cameras"]):
                print(f"  {i+1}. 索引: {cam['index']}, 类型: {cam['type']}, 分辨率: {cam['width']}x{cam['height']}")
        
        return result["cameras"]
    
    def setup_cameras(self, monitor_index=1, ptz_index=0):
        """设置监控摄像头和PTZ摄像头"""
        print(f"\n=== 设置摄像头 ===")
        print(f"1. 打开监控摄像头（索引: {monitor_index}）...")
        
        # 打开监控摄像头
        monitor_result = self.monitor_camera.open_camera(monitor_index)
        if monitor_result["success"]:
            print(f"   ✅ 监控摄像头打开成功: {monitor_result['message']}")
        else:
            print(f"   ❌ 监控摄像头打开失败: {monitor_result['message']}")
            return False
        
        # 打开PTZ摄像头
        print(f"2. 打开PTZ摄像头（索引: {ptz_index}）...")
        ptz_result = self.ptz_camera.open_camera(ptz_index)
        if ptz_result["success"]:
            print(f"   ✅ PTZ摄像头打开成功: {ptz_result['message']}")
        else:
            print(f"   ❌ PTZ摄像头打开失败: {ptz_result['message']}")
            return False
        
        # 初始化PTZ控制器
        print("3. 初始化PTZ控制器...")
        self.ptz_controller = PTZCameraController(
            protocol=PTZProtocol.HTTP_API,
            connection_type="http",
            base_url="http://192.168.1.64",  # 请替换为真实PTZ摄像头IP
            username="admin",
            password="admin"
        )
        
        asyncio.run(self.ptz_controller.connect())
        print(f"   ✅ PTZ控制器初始化完成")
        
        return True
    
    def start_visual_monitoring(self):
        """启动视觉监控"""
        print("\n=== 启动视觉监控 ===")
        
        # 启动视觉识别
        recognition_result = self.monitor_camera.start_visual_recognition(model_type='haar')
        if recognition_result["success"]:
            print(f"   ✅ 视觉识别启动成功: {recognition_result['message']}")
        else:
            print(f"   ❌ 视觉识别启动失败: {recognition_result['message']}")
            return False
        
        # 启动视觉跟踪
        tracking_result = self.monitor_camera.start_visual_tracking(tracker_type='MIL')
        if tracking_result["success"]:
            print(f"   ✅ 视觉跟踪启动成功: {tracking_result['message']}")
        else:
            print(f"   ❌ 视觉跟踪启动失败: {tracking_result['message']}")
            return False
        
        self.is_monitoring = True
        return True
    
    def capture_reference_frame(self):
        """捕获参考帧"""
        print("\n=== 捕获参考帧 ===")
        
        # 等待摄像头稳定
        time.sleep(1)
        
        # 捕获参考帧
        reference_frame = self.monitor_camera.take_photo()
        if reference_frame is not None:
            print(f"   ✅ 参考帧捕获成功，分辨率: {reference_frame.shape[1]}x{reference_frame.shape[0]}")
            cv2.imwrite("reference_frame.jpg", reference_frame)
            print("   📸 参考帧已保存为 reference_frame.jpg")
            return reference_frame
        else:
            print(f"   ❌ 参考帧捕获失败")
            return None
    
    async def execute_ptz_action(self, action, duration=3, speed=100):
        """执行PTZ动作"""
        print(f"\n=== 执行PTZ动作: {action.value} ===")
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行PTZ动作
        if action == PTZAction.PAN_LEFT:
            result = await self.ptz_controller.execute_action(action, speed)
        elif action == PTZAction.PAN_RIGHT:
            result = await self.ptz_controller.execute_action(action, speed)
        elif action == PTZAction.TILT_UP:
            result = await self.ptz_controller.execute_action(action, speed)
        elif action == PTZAction.TILT_DOWN:
            result = await self.ptz_controller.execute_action(action, speed)
        else:
            result = {"success": False, "message": f"不支持的动作: {action.value}"}
        
        if result["success"]:
            print(f"   ✅ {action.value} 动作执行成功")
        else:
            print(f"   ❌ {action.value} 动作执行失败: {result['message']}")
            return False
        
        # 等待动作完成
        print(f"   ⏱️  动作持续 {duration} 秒...")
        await asyncio.sleep(duration)
        
        # 停止PTZ动作
        stop_result = await self.ptz_controller.execute_action(PTZAction.STOP, 0)
        if stop_result["success"]:
            print(f"   ✅ 动作停止成功")
        
        # 记录结束时间
        end_time = time.time()
        
        return True
    
    def detect_movement(self, reference_frame):
        """检测PTZ摄像头的移动"""
        print("\n=== 检测PTZ移动 ===")
        
        # 捕获当前帧
        current_frame = self.monitor_camera.take_photo()
        if current_frame is None:
            print(f"   ❌ 当前帧捕获失败")
            return False
        
        # 保存当前帧
        cv2.imwrite("current_frame.jpg", current_frame)
        print("   📸 当前帧已保存为 current_frame.jpg")
        
        # 使用简单的差异检测
        reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 调整大小以提高性能
        ref_resized = cv2.resize(reference_gray, (320, 240))
        curr_resized = cv2.resize(current_gray, (320, 240))
        
        # 计算差异
        diff = cv2.absdiff(ref_resized, curr_resized)
        
        # 阈值化差异图像
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 计算差异像素数量
        diff_pixels = cv2.countNonZero(thresh)
        total_pixels = ref_resized.shape[0] * ref_resized.shape[1]
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        # 保存差异图像
        cv2.imwrite("diff_image.jpg", thresh)
        
        print(f"   📊 差异检测结果:")
        print(f"      - 差异像素数: {diff_pixels}")
        print(f"      - 总像素数: {total_pixels}")
        print(f"      - 差异百分比: {diff_percentage:.2f}%")
        
        # 判定是否有明显移动（差异超过5%）
        is_moved = diff_percentage > 5.0
        
        if is_moved:
            print(f"   ✅ 检测到明显移动！")
        else:
            print(f"   ⚠️  未检测到明显移动")
        
        # 保存检测结果
        self.detection_results.append({
            "timestamp": time.time(),
            "diff_percentage": diff_percentage,
            "is_moved": is_moved,
            "diff_pixels": diff_pixels
        })
        
        return is_moved
    
    def run_detection_sequence(self):
        """运行完整检测序列"""
        print("\n" + "="*50)
        print("=== AI PTZ云台转动检测系统 ===")
        print("="*50)
        
        # 1. 列出可用摄像头
        available_cameras = self.list_available_cameras()
        
        if len(available_cameras) < 2:
            print("\n❌ 错误：系统需要至少2个摄像头，1个用于监控，1个用于PTZ测试")
            return False
        
        # 2. 设置摄像头
        if not self.setup_cameras(monitor_index=1, ptz_index=0):
            return False
        
        # 3. 启动视觉监控
        if not self.start_visual_monitoring():
            return False
        
        # 4. 捕获参考帧
        reference_frame = self.capture_reference_frame()
        if reference_frame is None:
            return False
        
        # 5. 执行PTZ动作并检测
        asyncio.run(self._execute_detection_async(reference_frame))
        
        # 6. 生成检测报告
        self.generate_report()
        
        # 7. 清理资源
        self.cleanup()
        
        return True
    
    async def _execute_detection_async(self, reference_frame):
        """异步执行检测"""
        # 执行多个PTZ动作并检测
        actions = [
            PTZAction.PAN_LEFT,
            PTZAction.PAN_RIGHT,
            PTZAction.TILT_UP,
            PTZAction.TILT_DOWN
        ]
        
        for action in actions:
            # 执行PTZ动作
            await self.execute_ptz_action(action, duration=2, speed=100)
            
            # 检测移动
            self.detect_movement(reference_frame)
            
            # 等待1秒
            await asyncio.sleep(1)
        
        # 执行大角度移动测试（>100°）
        print("\n=== 执行大角度移动测试（>100°）===")
        
        # 获取初始位置
        initial_state = self.ptz_controller.get_status()
        initial_pan = initial_state["position"]["pan"]
        
        # 执行180°旋转
        result = await self.ptz_controller.move_to_position(pan=initial_pan + 180, tilt=initial_state["position"]["tilt"], speed=100)
        if result["success"]:
            print(f"   ✅ 180°旋转执行成功")
            # 检测大角度移动
            self.detect_movement(reference_frame)
        
        # 复位到初始位置
        await self.ptz_controller.move_to_position(pan=initial_pan, tilt=initial_state["position"]["tilt"], speed=100)
    
    def generate_report(self):
        """生成检测报告"""
        print("\n" + "="*50)
        print("=== AI PTZ云台转动检测报告 ===")
        print("="*50)
        
        # 统计检测结果
        total_tests = len(self.detection_results)
        moved_tests = sum(1 for r in self.detection_results if r["is_moved"])
        accuracy = (moved_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📋 检测结果统计:")
        print(f"   - 总测试次数: {total_tests}")
        print(f"   - 检测到移动次数: {moved_tests}")
        print(f"   - 检测准确率: {accuracy:.2f}%")
        
        print(f"\n📊 详细检测结果:")
        for i, result in enumerate(self.detection_results):
            status = "✅ 移动" if result["is_moved"] else "⚠️  未移动"
            print(f"   {i+1}. 差异: {result['diff_percentage']:.2f}% {status}")
        
        print(f"\n🔍 分析结论:")
        if moved_tests > 0:
            print(f"   ✅ 成功！AI可以检测到PTZ摄像头的转动")
            print(f"   🎯 系统能够通过第二个摄像头监控第一个摄像头的PTZ动作")
        else:
            print(f"   ⚠️  警告：未检测到PTZ摄像头的转动")
            print(f"   💡 建议：检查PTZ摄像头是否支持PTZ控制，或者调整摄像头位置")
        
        # 保存报告
        with open("ptz_detection_report.txt", "w") as f:
            f.write("AI PTZ云台转动检测报告\n")
            f.write("="*50 + "\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总测试次数: {total_tests}\n")
            f.write(f"检测到移动次数: {moved_tests}\n")
            f.write(f"检测准确率: {accuracy:.2f}%\n\n")
            f.write("详细结果:\n")
            for i, result in enumerate(self.detection_results):
                status = "移动" if result["is_moved"] else "未移动"
                f.write(f"{i+1}. 差异: {result['diff_percentage']:.2f}% - {status}\n")
        
        print(f"\n📄 检测报告已保存为 ptz_detection_report.txt")
    
    def cleanup(self):
        """清理资源"""
        print("\n=== 清理资源 ===")
        
        # 关闭监控摄像头
        self.monitor_camera.close_camera()
        print(f"   ✅ 监控摄像头已关闭")
        
        # 关闭PTZ摄像头
        self.ptz_camera.close_camera()
        print(f"   ✅ PTZ摄像头已关闭")
        
        # 断开PTZ控制器
        if self.ptz_controller:
            asyncio.run(self.ptz_controller.disconnect())
            print(f"   ✅ PTZ控制器已断开")
        
        print(f"   ✅ 所有资源已清理")
    
    def get_status(self):
        """获取系统状态"""
        return {
            "is_monitoring": self.is_monitoring,
            "monitor_camera_open": self.monitor_camera.is_camera_open(),
            "ptz_camera_open": self.ptz_camera.is_camera_open(),
            "detection_results_count": len(self.detection_results)
        }

if __name__ == "__main__":
    # 创建检测系统
    detector = AIPtzDetection()
    
    try:
        # 运行检测序列
        detector.run_detection_sequence()
    except KeyboardInterrupt:
        print("\n\n🔴 检测被用户中断")
        detector.cleanup()
    except Exception as e:
        print(f"\n\n❌ 检测过程中发生错误: {e}")
        detector.cleanup()
    finally:
        print("\n🎉 AI PTZ云台转动检测完成")