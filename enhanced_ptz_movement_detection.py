#!/usr/bin/env python3
"""增强版PTZ摄像头移动检测脚本"""

import asyncio
import time
import cv2
import numpy as np
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class EnhancedPTZMovementDetection:
    """增强版PTZ摄像头移动检测"""
    
    def __init__(self):
        """初始化检测系统"""
        self.monitor_camera = CameraController()  # 监控摄像头（联想摄像头）
        self.ptz_controller = None                # PTZ控制器
        self.detection_results = []               # 检测结果
        
        print("=== 增强版PTZ摄像头移动检测 ===")
        print("系统特性:")
        print("1. 优化的移动检测算法，提高敏感度")
        print("2. 多种检测方法结合")
        print("3. 支持手动控制PTZ动作")
        print("4. 实时显示移动检测结果")
        print("5. 记录详细的移动数据")
    
    def setup_ptz_controller(self):
        """设置PTZ控制器"""
        print("\n=== 设置PTZ控制器 ===")
        
        # 用户配置区
        CAMERA_IP = "192.168.1.64"  # 请替换为真实PTZ摄像头IP
        USERNAME = "admin"         # 请替换为真实用户名
        PASSWORD = "admin"         # 请替换为真实密码
        
        print(f"连接到PTZ摄像头: {CAMERA_IP}")
        
        # 初始化PTZ控制器
        self.ptz_controller = PTZCameraController(
            protocol=PTZProtocol.HTTP_API,
            connection_type="http",
            base_url=f"http://{CAMERA_IP}",
            username=USERNAME,
            password=PASSWORD
        )
        
        # 连接PTZ控制器
        result = asyncio.run(self.ptz_controller.connect())
        if result["success"]:
            print(f"✅ PTZ控制器连接成功")
        else:
            print(f"⚠️  PTZ控制器连接失败: {result['message']}")
            print(f"💡 提示: 系统将继续运行，但PTZ动作可能无法执行")
    
    def open_monitor_camera(self):
        """打开监控摄像头"""
        print("\n=== 打开监控摄像头 ===")
        
        # 打开监控摄像头（索引0，通常是默认摄像头）
        result = self.monitor_camera.open_camera(0)
        if result["success"]:
            print(f"✅ 监控摄像头打开成功: {result['message']}")
            return True
        else:
            print(f"❌ 监控摄像头打开失败: {result['message']}")
            return False
    
    def capture_reference_frame(self):
        """捕获参考帧"""
        print("\n=== 捕获参考帧 ===")
        
        # 等待摄像头稳定
        time.sleep(2)
        
        # 捕获参考帧
        reference_frame = self.monitor_camera.take_photo()
        if reference_frame is not None:
            print(f"✅ 参考帧捕获成功，分辨率: {reference_frame.shape[1]}x{reference_frame.shape[0]}")
            cv2.imwrite("enhanced_reference_frame.jpg", reference_frame)
            print("📸 参考帧已保存为 enhanced_reference_frame.jpg")
            return reference_frame
        else:
            print(f"❌ 参考帧捕获失败")
            return None
    
    def detect_movement(self, reference_frame, current_frame, sensitivity=3.0):
        """检测移动，优化版本"""
        print("\n=== 检测移动 ===")
        
        # 保存当前帧
        cv2.imwrite("enhanced_current_frame.jpg", current_frame)
        
        # 1. 调整图像大小以提高性能
        ref_resized = cv2.resize(reference_frame, (320, 240))
        curr_resized = cv2.resize(current_frame, (320, 240))
        
        # 2. 转换为灰度图
        ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)
        
        # 3. 应用高斯模糊，减少噪声
        ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # 4. 计算绝对差异
        diff = cv2.absdiff(ref_blur, curr_blur)
        
        # 5. 应用自适应阈值，提高对不同光照条件的适应能力
        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 6. 膨胀，合并相邻的差异区域
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # 7. 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 8. 计算差异区域的总面积
        total_diff_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # 过滤小面积差异
                total_diff_area += area
                # 在图像上绘制轮廓
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(curr_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 9. 保存差异图像
        cv2.imwrite("enhanced_diff_image.jpg", diff)
        cv2.imwrite("enhanced_threshold_image.jpg", thresh)
        cv2.imwrite("enhanced_contour_image.jpg", curr_resized)
        
        # 10. 计算移动百分比
        total_pixels = ref_resized.shape[0] * ref_resized.shape[1]
        diff_percentage = (total_diff_area / total_pixels) * 100
        
        # 11. 计算差异像素数
        diff_pixels = cv2.countNonZero(diff)
        diff_pixel_percentage = (diff_pixels / total_pixels) * 100
        
        print(f"📊 移动检测结果:")
        print(f"   - 差异像素数: {diff_pixels}")
        print(f"   - 差异像素百分比: {diff_pixel_percentage:.2f}%")
        print(f"   - 差异区域面积: {total_diff_area:.2f} 像素")
        print(f"   - 差异区域百分比: {diff_percentage:.2f}%")
        print(f"   - 检测到的轮廓数: {len(contours)}")
        
        # 12. 使用更敏感的阈值判定是否移动
        is_moved = diff_pixel_percentage > sensitivity  # 敏感度可调，默认3%
        
        if is_moved:
            print(f"✅ 检测到明显移动！")
        else:
            print(f"⚠️  未检测到明显移动")
        
        # 保存检测结果
        self.detection_results.append({
            "timestamp": time.time(),
            "diff_pixels": diff_pixels,
            "diff_pixel_percentage": diff_pixel_percentage,
            "diff_area": total_diff_area,
            "diff_area_percentage": diff_percentage,
            "contours_count": len(contours),
            "is_moved": is_moved,
            "sensitivity": sensitivity
        })
        
        return is_moved
    
    async def execute_ptz_action(self, action, duration=2, speed=100):
        """执行PTZ动作"""
        print(f"\n=== 执行PTZ动作: {action.value} ===")
        
        result = await self.ptz_controller.execute_action(action, speed)
        if result["success"]:
            print(f"✅ {action.value} 动作执行成功")
        else:
            print(f"⚠️  {action.value} 动作执行失败: {result['message']}")
        
        # 保持动作持续时间
        print(f"⏱️  动作持续 {duration} 秒...")
        await asyncio.sleep(duration)
        
        # 停止PTZ动作
        await self.ptz_controller.execute_action(PTZAction.STOP, 0)
    
    def run_detection_sequence(self, sensitivity=3.0):
        """运行检测序列"""
        print("\n" + "="*60)
        print("=== 开始增强版PTZ摄像头移动检测 ===")
        print("="*60)
        
        # 1. 打开监控摄像头
        if not self.open_monitor_camera():
            return
        
        # 2. 设置PTZ控制器
        self.setup_ptz_controller()
        
        # 3. 捕获参考帧
        reference_frame = self.capture_reference_frame()
        if reference_frame is None:
            return
        
        # 4. 执行PTZ动作并检测
        actions = [
            PTZAction.PAN_LEFT,
            PTZAction.PAN_RIGHT,
            PTZAction.TILT_UP,
            PTZAction.TILT_DOWN
        ]
        
        for action in actions:
            # 执行PTZ动作
            asyncio.run(self.execute_ptz_action(action, duration=2, speed=100))
            
            # 捕获当前帧
            current_frame = self.monitor_camera.take_photo()
            if current_frame is not None:
                # 检测移动
                self.detect_movement(reference_frame, current_frame, sensitivity)
            
            # 等待1秒
            time.sleep(1)
        
        # 5. 执行大角度移动测试
        print("\n=== 执行大角度移动测试（180°旋转）===")
        
        # 获取初始位置
        initial_state = self.ptz_controller.get_status()
        initial_pan = initial_state["position"]["pan"]
        
        # 执行180°旋转
        result = asyncio.run(self.ptz_controller.move_to_position(pan=initial_pan + 180, tilt=initial_state["position"]["tilt"], speed=100))
        if result["success"]:
            print(f"✅ 180°旋转执行成功")
            
            # 捕获当前帧
            current_frame = self.monitor_camera.take_photo()
            if current_frame is not None:
                # 检测移动
                self.detect_movement(reference_frame, current_frame, sensitivity)
        
        # 6. 生成检测报告
        self.generate_report()
        
        # 7. 清理资源
        self.cleanup()
    
    def generate_report(self):
        """生成检测报告"""
        print("\n" + "="*60)
        print("=== 增强版PTZ摄像头移动检测报告 ===")
        print("="*60)
        
        # 统计检测结果
        total_tests = len(self.detection_results)
        moved_tests = sum(1 for r in self.detection_results if r["is_moved"])
        accuracy = (moved_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📋 检测结果统计:")
        print(f"   - 总测试次数: {total_tests}")
        print(f"   - 检测到移动次数: {moved_tests}")
        print(f"   - 检测准确率: {accuracy:.2f}%")
        
        # 详细检测结果
        print(f"\n📊 详细检测结果:")
        for i, result in enumerate(self.detection_results):
            status = "✅ 移动" if result["is_moved"] else "⚠️  未移动"
            print(f"   {i+1}. {status} - 差异: {result['diff_pixel_percentage']:.2f}%, 轮廓数: {result['contours_count']}")
        
        # 保存报告
        with open("enhanced_ptz_movement_report.txt", "w") as f:
            f.write("增强版PTZ摄像头移动检测报告\n")
            f.write("="*60 + "\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总测试次数: {total_tests}\n")
            f.write(f"检测到移动次数: {moved_tests}\n")
            f.write(f"检测准确率: {accuracy:.2f}%\n\n")
            f.write("详细结果:\n")
            for i, result in enumerate(self.detection_results):
                status = "移动" if result["is_moved"] else "未移动"
                f.write(f"{i+1}. {status} - 差异: {result['diff_pixel_percentage']:.2f}%, 轮廓数: {result['contours_count']}\n")
        
        print(f"\n📄 检测报告已保存为 enhanced_ptz_movement_report.txt")
        print(f"📸 参考帧、当前帧和差异图像已保存")
    
    def cleanup(self):
        """清理资源"""
        print("\n=== 清理资源 ===")
        
        # 关闭监控摄像头
        self.monitor_camera.close_camera()
        print(f"✅ 监控摄像头已关闭")
        
        # 断开PTZ控制器
        if self.ptz_controller:
            asyncio.run(self.ptz_controller.disconnect())
            print(f"✅ PTZ控制器已断开")
        
        print(f"✅ 所有资源已清理")
    
    def manual_control(self):
        """手动控制模式"""
        print("\n" + "="*60)
        print("=== 手动控制模式 ===")
        print("="*60)
        print("使用键盘控制PTZ摄像头:")
        print("  a: 向左旋转")
        print("  d: 向右旋转")
        print("  w: 向上倾斜")
        print("  s: 向下倾斜")
        print("  q: 退出")
        print("="*60)
        
        # 捕获参考帧
        reference_frame = self.capture_reference_frame()
        if reference_frame is None:
            return
        
        # 打开监控摄像头
        if not self.open_monitor_camera():
            return
        
        # 设置PTZ控制器
        self.setup_ptz_controller()
        
        try:
            import keyboard
            
            while True:
                if keyboard.is_pressed('q'):
                    print("\n❌ 退出手动控制")
                    break
                
                action = None
                if keyboard.is_pressed('a'):
                    action = PTZAction.PAN_LEFT
                elif keyboard.is_pressed('d'):
                    action = PTZAction.PAN_RIGHT
                elif keyboard.is_pressed('w'):
                    action = PTZAction.TILT_UP
                elif keyboard.is_pressed('s'):
                    action = PTZAction.TILT_DOWN
                
                if action:
                    # 执行PTZ动作
                    print(f"\n执行动作: {action.value}")
                    asyncio.run(self.execute_ptz_action(action, duration=1, speed=100))
                    
                    # 捕获当前帧
                    current_frame = self.monitor_camera.take_photo()
                    if current_frame is not None:
                        # 检测移动
                        self.detect_movement(reference_frame, current_frame, sensitivity=3.0)
                    
                time.sleep(0.1)
        except ImportError:
            print("⚠️  keyboard模块未安装，无法使用手动控制")
            print("   请运行: pip install keyboard")

if __name__ == "__main__":
    # 创建检测系统
    detector = EnhancedPTZMovementDetection()
    
    try:
        # 运行检测序列
        detector.run_detection_sequence(sensitivity=3.0)
    except KeyboardInterrupt:
        print("\n\n🔴 检测被用户中断")
        detector.cleanup()
    except Exception as e:
        print(f"\n\n❌ 检测过程中发生错误: {e}")
        detector.cleanup()
    finally:
        print("\n🎉 增强版PTZ摄像头移动检测完成")
        print("\n📋 使用说明:")
        print("1. 确保监控摄像头已连接")
        print("2. 调整CAMERA_IP、USERNAME和PASSWORD为真实值")
        print("3. 运行: python enhanced_ptz_movement_detection.py")
        print("4. 查看生成的报告和图像文件")
        print("5. 可调整sensitivity参数提高/降低检测敏感度")
        print("\n💡 提示: 降低sensitivity值会提高检测敏感度")
        print("   例如: detector.run_detection_sequence(sensitivity=2.0)")
        print("   或: detector.run_detection_sequence(sensitivity=5.0)")
        print("\n💡 手动控制模式:")
        print("   运行: python enhanced_ptz_movement_detection.py manual")
        print("   使用WASD键控制PTZ摄像头，Q键退出")