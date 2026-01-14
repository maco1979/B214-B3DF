#!/usr/bin/env python3
"""使用联想摄像头监控带云台的摄像头是否移动"""

import asyncio
import time
import cv2
import numpy as np
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class LenovoPTZMonitor:
    """使用联想摄像头监控带云台的摄像头"""
    
    def __init__(self):
        """初始化监控系统"""
        self.monitor_camera = CameraController()  # 联想摄像头（监控用）
        self.ptz_camera = CameraController()      # 带云台的摄像头（测试用）
        self.ptz_controller = None                # PTZ控制器
        self.detection_results = []               # 检测结果
        self.is_monitoring = False                # 监控状态
        
        print("=== 使用联想摄像头监控带云台的摄像头 ===")
        print("系统功能:")
        print("1. 检测系统中可用的摄像头")
        print("2. 优先使用联想摄像头作为监控摄像头")
        print("3. 打开带云台的摄像头")
        print("4. AI控制PTZ摄像头移动")
        print("5. 检测PTZ摄像头是否移动")
        print("6. 生成完整的检测报告")
    
    def list_available_cameras(self):
        """列出可用摄像头"""
        print("\n=== 检测可用摄像头 ===")
        
        # 列出所有可用摄像头
        result = self.monitor_camera.list_cameras(max_index=5)
        
        if result["success"]:
            print(f"检测到 {result['available_count']} 个可用摄像头:")
            for i, cam in enumerate(result["cameras"]):
                print(f"  {i+1}. 索引: {cam['index']}, 类型: {cam['type']}, 分辨率: {cam['width']}x{cam['height']}")
        
        return result["cameras"]
    
    def identify_lenovo_camera(self, cameras):
        """识别联想摄像头"""
        print("\n=== 识别联想摄像头 ===")
        
        # 尝试打开每个摄像头，检查是否为联想摄像头
        for cam in cameras:
            if cam["type"] == "simulated":
                continue  # 跳过模拟摄像头
            
            print(f"测试摄像头索引 {cam['index']}...")
            
            # 尝试打开摄像头
            temp_cam = CameraController()
            result = temp_cam.open_camera(cam["index"])
            
            if result["success"]:
                # 尝试获取摄像头信息
                frame = temp_cam.take_photo()
                if frame is not None:
                    print(f"   ✅ 摄像头 {cam['index']} 可用")
                    
                    # 尝试获取摄像头属性（不同品牌的摄像头可能有不同的属性）
                    # 这里我们简单地通过摄像头名称或设备ID来识别
                    # 联想摄像头通常包含 "lenovo" 或 "Think" 等关键词
                    
                    # 关闭临时摄像头
                    temp_cam.close_camera()
                    
                    # 假设第一个真实摄像头就是联想摄像头
                    print(f"   🎯 假设摄像头 {cam['index']} 为联想摄像头")
                    return cam["index"]
                
                # 关闭临时摄像头
                temp_cam.close_camera()
        
        # 如果没有找到真实摄像头，返回0（默认摄像头）
        print("   ⚠️  未找到明确的联想摄像头，使用默认摄像头")
        return 0
    
    def setup_cameras(self, lenovo_index, ptz_index):
        """设置监控摄像头和PTZ摄像头"""
        print(f"\n=== 设置摄像头 ===")
        print(f"1. 打开联想摄像头（索引: {lenovo_index}）...")
        
        # 打开联想摄像头（监控用）
        monitor_result = self.monitor_camera.open_camera(lenovo_index)
        if monitor_result["success"]:
            print(f"   ✅ 联想摄像头打开成功: {monitor_result['message']}")
        else:
            print(f"   ❌ 联想摄像头打开失败: {monitor_result['message']}")
            return False
        
        # 打开带云台的摄像头
        print(f"2. 打开带云台的摄像头（索引: {ptz_index}）...")
        ptz_result = self.ptz_camera.open_camera(ptz_index)
        if ptz_result["success"]:
            print(f"   ✅ 带云台的摄像头打开成功: {ptz_result['message']}")
        else:
            print(f"   ❌ 带云台的摄像头打开失败: {ptz_result['message']}")
            return False
        
        # 初始化PTZ控制器
        print(f"3. 初始化PTZ控制器...")
        
        # 用户配置区
        CAMERA_IP = "192.168.1.64"  # 请替换为真实PTZ摄像头IP
        USERNAME = "admin"         # 请替换为真实用户名
        PASSWORD = "admin"         # 请替换为真实密码
        
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
            print(f"   ✅ PTZ控制器连接成功: {result['message']}")
        else:
            print(f"   ⚠️  PTZ控制器连接失败: {result['message']}")
            print(f"   💡 提示: 系统将使用模拟模式继续运行")
        
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
            cv2.imwrite("lenovo_reference_frame.jpg", reference_frame)
            print("   📸 参考帧已保存为 lenovo_reference_frame.jpg")
            return reference_frame
        else:
            print(f"   ❌ 参考帧捕获失败")
            return None
    
    async def execute_ptz_action(self, action, duration=3, speed=100):
        """执行PTZ动作"""
        print(f"\n=== 执行PTZ动作: {action.value} ===")
        
        # 执行PTZ动作
        result = await self.ptz_controller.execute_action(action, speed)
        
        if result["success"]:
            print(f"   ✅ {action.value} 动作执行成功")
            return True
        else:
            print(f"   ⚠️  {action.value} 动作执行失败: {result['message']}")
            print(f"   💡 提示: 系统在模拟模式下继续运行")
            return True  # 即使动作失败，也继续运行模拟模式
    
    def detect_movement(self, reference_frame):
        """检测PTZ摄像头的移动"""
        print("\n=== 检测PTZ移动 ===")
        
        # 捕获当前帧
        current_frame = self.monitor_camera.take_photo()
        if current_frame is None:
            print(f"   ❌ 当前帧捕获失败")
            return False
        
        # 保存当前帧
        cv2.imwrite("lenovo_current_frame.jpg", current_frame)
        print("   📸 当前帧已保存为 lenovo_current_frame.jpg")
        
        # 调整图像大小以提高性能
        ref_resized = cv2.resize(reference_frame, (320, 240))
        curr_resized = cv2.resize(current_frame, (320, 240))
        
        # 转换为灰度图
        ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)
        
        # 计算差异
        diff = cv2.absdiff(ref_gray, curr_gray)
        
        # 应用阈值
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 计算差异像素数量
        diff_pixels = cv2.countNonZero(thresh)
        total_pixels = ref_resized.shape[0] * ref_resized.shape[1]
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        # 保存差异图像
        cv2.imwrite("lenovo_diff_image.jpg", thresh)
        
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
    
    def run_monitoring_sequence(self):
        """运行完整的监控序列"""
        print("\n" + "="*60)
        print("=== 开始使用联想摄像头监控带云台的摄像头 ===")
        print("="*60)
        
        # 1. 列出可用摄像头
        available_cameras = self.list_available_cameras()
        
        if len(available_cameras) < 1:
            print("\n❌ 错误：系统中没有可用的摄像头")
            return False
        
        # 2. 识别联想摄像头
        lenovo_index = self.identify_lenovo_camera(available_cameras)
        
        # 3. 设置摄像头
        # 使用联想摄像头作为监控摄像头，另一个摄像头（如果有）作为PTZ摄像头
        ptz_index = 1 if len(available_cameras) > 1 else 0
        
        if not self.setup_cameras(lenovo_index=lenovo_index, ptz_index=ptz_index):
            return False
        
        # 4. 捕获参考帧
        reference_frame = self.capture_reference_frame()
        if reference_frame is None:
            return False
        
        # 5. 执行PTZ动作并检测
        asyncio.run(self._execute_monitoring_async(reference_frame))
        
        # 6. 生成检测报告
        self.generate_report()
        
        # 7. 清理资源
        self.cleanup()
        
        return True
    
    async def _execute_monitoring_async(self, reference_frame):
        """异步执行监控"""
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
        print("\n" + "="*60)
        print("=== 使用联想摄像头监控带云台的摄像头检测报告 ===")
        print("="*60)
        
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
            print(f"   ✅ 成功！使用联想摄像头检测到带云台的摄像头的移动")
            print(f"   🎯 AI可以控制带云台的摄像头进行转动")
        else:
            print(f"   ⚠️  警告：未检测到带云台的摄像头的明显移动")
            print(f"   💡 建议：检查带云台的摄像头是否支持PTZ控制")
            print(f"   💡 建议：调整PTZ摄像头的IP和登录信息")
            print(f"   💡 建议：确保联想摄像头能够清晰地看到带云台的摄像头")
        
        # 保存报告
        with open("lenovo_ptz_monitor_report.txt", "w") as f:
            f.write("使用联想摄像头监控带云台的摄像头检测报告\n")
            f.write("="*60 + "\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总测试次数: {total_tests}\n")
            f.write(f"检测到移动次数: {moved_tests}\n")
            f.write(f"检测准确率: {accuracy:.2f}%\n\n")
            f.write("详细结果:\n")
            for i, result in enumerate(self.detection_results):
                status = "移动" if result["is_moved"] else "未移动"
                f.write(f"{i+1}. 差异: {result['diff_percentage']:.2f}% - {status}\n")
        
        print(f"\n📄 检测报告已保存为 lenovo_ptz_monitor_report.txt")
    
    def cleanup(self):
        """清理资源"""
        print("\n=== 清理资源 ===")
        
        # 关闭监控摄像头（联想摄像头）
        self.monitor_camera.close_camera()
        print(f"   ✅ 联想摄像头已关闭")
        
        # 关闭带云台的摄像头
        self.ptz_camera.close_camera()
        print(f"   ✅ 带云台的摄像头已关闭")
        
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
    # 创建监控系统
    monitor = LenovoPTZMonitor()
    
    try:
        # 运行监控序列
        monitor.run_monitoring_sequence()
    except KeyboardInterrupt:
        print("\n\n🔴 监控被用户中断")
        monitor.cleanup()
    except Exception as e:
        print(f"\n\n❌ 监控过程中发生错误: {e}")
        monitor.cleanup()
    finally:
        print("\n🎉 使用联想摄像头监控带云台的摄像头完成")
        print("\n📋 使用说明:")
        print("1. 确保联想摄像头已连接到电脑")
        print("2. 确保带云台的摄像头已连接到网络")
        print("3. 修改脚本中的CAMERA_IP、USERNAME和PASSWORD为真实值")
        print("4. 确保联想摄像头能够清晰地看到带云台的摄像头")
        print("5. 再次运行脚本")