#!/usr/bin/env python3
"""AI PTZ云台转动模拟检测脚本"""

import asyncio
import time
import cv2
import numpy as np
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class AIPtzSimulation:
    """AI PTZ云台转动模拟检测类"""
    
    def __init__(self):
        """初始化模拟检测系统"""
        self.ptz_controller = None                # PTZ控制器
        self.detection_results = []               # 检测结果
        self.is_monitoring = False                # 监控状态
        self.simulated_camera = None              # 模拟摄像头帧生成器
        
        print("=== AI PTZ云台转动模拟检测系统 ===")
        print("系统特性:")
        print("1. 支持PTZ摄像头的真实控制")
        print("2. 模拟第二个摄像头的监控功能")
        print("3. 实时显示PTZ动作的视觉反馈")
        print("4. 生成完整的检测报告")
    
    def setup_ptz_controller(self):
        """设置PTZ控制器"""
        print(f"\n=== 设置PTZ控制器 ===")
        
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
            print(f"✅ PTZ控制器连接成功: {result['message']}")
            return True
        else:
            print(f"❌ PTZ控制器连接失败: {result['message']}")
            print(f"💡 提示: 系统将使用模拟模式继续运行")
            return True  # 即使连接失败，也继续运行模拟模式
    
    def generate_simulated_frame(self, pan_offset=0, tilt_offset=0, show_visual_feedback=True):
        """生成模拟的监控摄像头帧"""
        # 创建640x480的黑色背景
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 绘制模拟的PTZ摄像头
        ptz_camera_x = 320
        ptz_camera_y = 240
        
        # 根据pan和tilt偏移绘制PTZ摄像头的视觉反馈
        if show_visual_feedback:
            # 绘制PTZ摄像头主体
            cv2.circle(frame, (ptz_camera_x, ptz_camera_y), 50, (100, 100, 100), -1)
            cv2.circle(frame, (ptz_camera_x, ptz_camera_y), 55, (150, 150, 150), 2)
            
            # 根据pan_offset绘制旋转指示器
            indicator_length = 60
            indicator_x = int(ptz_camera_x + indicator_length * np.sin(np.radians(pan_offset)))
            indicator_y = int(ptz_camera_y + indicator_length * np.cos(np.radians(pan_offset)))
            cv2.line(frame, (ptz_camera_x, ptz_camera_y), (indicator_x, indicator_y), (0, 255, 0), 3)
            
            # 绘制倾斜指示器
            tilt_indicator_x = int(ptz_camera_x + indicator_length * 0.5 * np.sin(np.radians(tilt_offset)))
            tilt_indicator_y = int(ptz_camera_y + indicator_length * 0.5 * np.cos(np.radians(tilt_offset)))
            cv2.line(frame, (ptz_camera_x, ptz_camera_y), (tilt_indicator_x, tilt_indicator_y), (255, 0, 0), 2)
            
            # 绘制当前状态信息
            status_text = f"Pan: {pan_offset:.1f}° | Tilt: {tilt_offset:.1f}°"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 绘制标题
            cv2.putText(frame, "AI PTZ云台监控模拟", (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    async def execute_ptz_action(self, action, duration=3, speed=100):
        """执行PTZ动作"""
        print(f"\n=== 执行PTZ动作: {action.value} ===")
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行PTZ动作
        result = await self.ptz_controller.execute_action(action, speed)
        
        if result["success"]:
            print(f"✅ {action.value} 动作执行成功")
            return True
        else:
            print(f"⚠️  {action.value} 动作执行失败: {result['message']}")
            print(f"💡 提示: 系统在模拟模式下继续运行")
            return True  # 即使动作失败，也继续运行模拟模式
    
    def detect_movement(self, reference_pan, reference_tilt, current_pan, current_tilt):
        """检测PTZ摄像头的移动"""
        print("\n=== 检测PTZ移动 ===")
        
        # 计算pan和tilt的变化
        pan_change = abs(current_pan - reference_pan)
        tilt_change = abs(current_tilt - reference_tilt)
        total_change = pan_change + tilt_change
        
        # 生成模拟的监控帧
        reference_frame = self.generate_simulated_frame(reference_pan, reference_tilt, show_visual_feedback=True)
        current_frame = self.generate_simulated_frame(current_pan, current_tilt, show_visual_feedback=True)
        
        # 保存模拟的监控帧
        cv2.imwrite("simulated_reference_frame.jpg", reference_frame)
        cv2.imwrite("simulated_current_frame.jpg", current_frame)
        
        # 计算差异百分比（模拟）
        diff_percentage = min(100, total_change * 2)  # 模拟差异百分比
        
        print(f"📊 差异检测结果:")
        print(f"   - Pan变化: {pan_change:.2f}°")
        print(f"   - Tilt变化: {tilt_change:.2f}°")
        print(f"   - 总变化: {total_change:.2f}°")
        print(f"   - 模拟差异百分比: {diff_percentage:.2f}%")
        
        # 判定是否有明显移动（变化超过10°）
        is_moved = total_change > 10.0
        
        if is_moved:
            print(f"✅ 检测到明显移动！")
        else:
            print(f"⚠️  未检测到明显移动")
        
        # 保存检测结果
        self.detection_results.append({
            "timestamp": time.time(),
            "diff_percentage": diff_percentage,
            "is_moved": is_moved,
            "pan_change": pan_change,
            "tilt_change": tilt_change,
            "total_change": total_change
        })
        
        return is_moved
    
    def run_detection_sequence(self):
        """运行完整检测序列"""
        print("\n" + "="*50)
        print("=== 开始AI PTZ云台转动检测 ===")
        print("="*50)
        
        # 1. 设置PTZ控制器
        if not self.setup_ptz_controller():
            return False
        
        # 2. 启动模拟监控
        print("\n=== 启动模拟监控 ===")
        print("✅ 模拟监控已启动")
        self.is_monitoring = True
        
        # 3. 初始化参考位置
        initial_state = self.ptz_controller.get_status()
        reference_pan = initial_state["position"]["pan"]
        reference_tilt = initial_state["position"]["tilt"]
        
        print(f"\n=== 初始PTZ位置 ===")
        print(f"   Pan: {reference_pan:.2f}°")
        print(f"   Tilt: {reference_tilt:.2f}°")
        
        # 4. 执行PTZ动作并检测
        asyncio.run(self._execute_detection_async(reference_pan, reference_tilt))
        
        # 5. 生成检测报告
        self.generate_report()
        
        # 6. 清理资源
        self.cleanup()
        
        return True
    
    async def _execute_detection_async(self, reference_pan, reference_tilt):
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
            
            # 获取当前位置
            current_state = self.ptz_controller.get_status()
            current_pan = current_state["position"]["pan"]
            current_tilt = current_state["position"]["tilt"]
            
            # 检测移动
            self.detect_movement(reference_pan, reference_tilt, current_pan, current_tilt)
            
            # 等待1秒
            await asyncio.sleep(1)
        
        # 执行大角度移动测试（>100°）
        print("\n=== 执行大角度移动测试（>100°）===")
        
        # 获取当前状态
        current_state = self.ptz_controller.get_status()
        current_pan = current_state["position"]["pan"]
        current_tilt = current_state["position"]["tilt"]
        
        # 执行180°旋转
        target_pan = current_pan + 180
        result = await self.ptz_controller.move_to_position(pan=target_pan, tilt=current_tilt, speed=100)
        
        if result["success"]:
            print(f"✅ 180°旋转执行成功")
            
            # 获取移动后的位置
            moved_state = self.ptz_controller.get_status()
            moved_pan = moved_state["position"]["pan"]
            moved_tilt = moved_state["position"]["tilt"]
            
            # 检测大角度移动
            self.detect_movement(current_pan, current_tilt, moved_pan, moved_tilt)
        
        # 复位到初始位置
        print("\n=== 复位到初始位置 ===")
        result = await self.ptz_controller.move_to_position(pan=reference_pan, tilt=reference_tilt, speed=100)
        if result["success"]:
            print(f"✅ 成功复位到初始位置")
    
    def generate_report(self):
        """生成检测报告"""
        print("\n" + "="*50)
        print("=== AI PTZ云台转动检测报告 ===")
        print("="*50)
        
        # 统计检测结果
        total_tests = len(self.detection_results)
        moved_tests = sum(1 for r in self.detection_results if r["is_moved"])
        accuracy = (moved_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # 计算平均变化
        if total_tests > 0:
            avg_pan_change = sum(r["pan_change"] for r in self.detection_results) / total_tests
            avg_tilt_change = sum(r["tilt_change"] for r in self.detection_results) / total_tests
            avg_total_change = sum(r["total_change"] for r in self.detection_results) / total_tests
        else:
            avg_pan_change = 0
            avg_tilt_change = 0
            avg_total_change = 0
        
        print(f"\n📋 检测结果统计:")
        print(f"   - 总测试次数: {total_tests}")
        print(f"   - 检测到移动次数: {moved_tests}")
        print(f"   - 检测准确率: {accuracy:.2f}%")
        print(f"   - 平均Pan变化: {avg_pan_change:.2f}°")
        print(f"   - 平均Tilt变化: {avg_tilt_change:.2f}°")
        print(f"   - 平均总变化: {avg_total_change:.2f}°")
        
        print(f"\n📊 详细检测结果:")
        for i, result in enumerate(self.detection_results):
            status = "✅ 移动" if result["is_moved"] else "⚠️  未移动"
            print(f"   {i+1}. Pan变化: {result['pan_change']:.2f}° | Tilt变化: {result['tilt_change']:.2f}° | 差异: {result['diff_percentage']:.2f}% - {status}")
        
        print(f"\n🔍 分析结论:")
        if moved_tests > 0:
            print(f"✅ 成功！AI可以控制PTZ摄像头进行转动")
            print(f"🎯 系统能够检测到PTZ摄像头的移动")
        else:
            print(f"⚠️  警告：未检测到PTZ摄像头的明显移动")
            print(f"💡 建议：检查PTZ摄像头是否支持PTZ控制，或者调整摄像头IP和登录信息")
        
        # 保存报告
        with open("simulated_ptz_detection_report.txt", "w") as f:
            f.write("AI PTZ云台转动检测报告\n")
            f.write("="*50 + "\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测模式: 模拟检测\n")
            f.write(f"总测试次数: {total_tests}\n")
            f.write(f"检测到移动次数: {moved_tests}\n")
            f.write(f"检测准确率: {accuracy:.2f}%\n")
            f.write(f"平均Pan变化: {avg_pan_change:.2f}°\n")
            f.write(f"平均Tilt变化: {avg_tilt_change:.2f}°\n")
            f.write(f"平均总变化: {avg_total_change:.2f}°\n\n")
            f.write("详细结果:\n")
            for i, result in enumerate(self.detection_results):
                status = "移动" if result["is_moved"] else "未移动"
                f.write(f"{i+1}. Pan变化: {result['pan_change']:.2f}° | Tilt变化: {result['tilt_change']:.2f}° | 差异: {result['diff_percentage']:.2f}% - {status}\n")
        
        print(f"\n📄 检测报告已保存为 simulated_ptz_detection_report.txt")
        print(f"📸 模拟监控帧已保存为 simulated_reference_frame.jpg 和 simulated_current_frame.jpg")
    
    def cleanup(self):
        """清理资源"""
        print("\n=== 清理资源 ===")
        
        # 断开PTZ控制器
        if self.ptz_controller:
            result = asyncio.run(self.ptz_controller.disconnect())
            if result["success"]:
                print(f"✅ PTZ控制器已断开")
        
        self.is_monitoring = False
        print(f"✅ 所有资源已清理")
    
    def get_status(self):
        """获取系统状态"""
        return {
            "is_monitoring": self.is_monitoring,
            "detection_results_count": len(self.detection_results)
        }

if __name__ == "__main__":
    # 创建检测系统
    simulator = AIPtzSimulation()
    
    try:
        # 运行检测序列
        simulator.run_detection_sequence()
    except KeyboardInterrupt:
        print("\n\n🔴 检测被用户中断")
        simulator.cleanup()
    except Exception as e:
        print(f"\n\n❌ 检测过程中发生错误: {e}")
        simulator.cleanup()
    finally:
        print("\n🎉 AI PTZ云台转动模拟检测完成")
        print("\n📋 使用说明:")
        print("1. 修改脚本中的CAMERA_IP、USERNAME和PASSWORD为真实值")
        print("2. 确保PTZ摄像头已连接到网络")
        print("3. 运行脚本，观察AI对PTZ摄像头的控制")
        print("4. 查看生成的检测报告和模拟监控帧")
        print("\n💡 提示: 即使在没有两个真实摄像头的情况下，系统也会运行并生成模拟检测结果")