#!/usr/bin/env python3
"""PTZ摄像头视觉对比验证脚本"""

import asyncio
import time
import cv2
import numpy as np
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class PTZVisualComparison:
    """PTZ摄像头视觉对比验证"""
    
    def __init__(self):
        """初始化对比验证系统"""
        self.monitor_camera = CameraController()  # 监控摄像头
        self.ptz_controller = None                # PTZ控制器
        self.comparison_results = []              # 对比结果
        
        print("=== PTZ摄像头视觉对比验证 ===")
        print("系统功能:")
        print("1. 拍摄初始位置照片")
        print("2. 执行PTZ动作")
        print("3. 拍摄动作后照片")
        print("4. 生成对比照片（并排显示）")
        print("5. 高亮显示差异区域")
        print("6. 计算差异百分比")
        print("7. 生成详细的对比报告")
    
    def setup_systems(self):
        """设置系统"""
        print(f"\n=== 设置系统 ===")
        
        # 1. 打开监控摄像头
        print(f"1. 打开监控摄像头...")
        result = self.monitor_camera.open_camera(0)
        if result["success"]:
            print(f"   ✅ 监控摄像头打开成功: {result['message']}")
        else:
            print(f"   ❌ 监控摄像头打开失败: {result['message']}")
            return False
        
        # 2. 设置PTZ控制器
        print(f"\n2. 设置PTZ控制器...")
        
        # 用户配置区
        CAMERA_IP = "192.168.1.1"  # 扫描结果显示为海康威视设备
        USERNAME = "admin"         # 默认用户名
        PASSWORD = "admin"         # 默认密码
        
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
            print(f"   💡 提示: 系统将继续运行，但PTZ动作可能无法执行")
        
        return True
    
    def take_photo(self, description):
        """拍摄照片"""
        print(f"\n📸 拍摄{description}照片...")
        
        # 等待摄像头稳定
        time.sleep(1)
        
        # 拍摄照片
        frame = self.monitor_camera.take_photo()
        if frame is not None:
            # 保存照片
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ptz_comparison_{description}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   ✅ 照片保存成功: {filename}")
            return frame, filename
        else:
            print(f"   ❌ 照片拍摄失败")
            return None, None
    
    def create_comparison_image(self, before_frame, after_frame, action):
        """创建对比图像"""
        print(f"\n🔍 生成对比图像...")
        
        # 确保两张图像大小相同
        if before_frame.shape != after_frame.shape:
            after_frame = cv2.resize(after_frame, (before_frame.shape[1], before_frame.shape[0]))
        
        # 1. 并排显示
        comparison_img = np.hstack((before_frame, after_frame))
        
        # 2. 添加文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison_img, "BEFORE", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison_img, "AFTER", (before_frame.shape[1] + 50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison_img, f"ACTION: {action}", (50, before_frame.shape[0] - 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 3. 计算差异图像
        gray_before = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_before, gray_after)
        
        # 4. 高亮差异区域
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # 5. 在原图上绘制差异区域
        after_with_diff = after_frame.copy()
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(after_with_diff, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # 6. 创建包含差异高亮的对比图
        comparison_with_diff = np.hstack((before_frame, after_with_diff))
        
        # 7. 保存对比图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 并排对比
        comparison_filename = f"ptz_comparison_side_by_side_{action}_{timestamp}.jpg"
        cv2.imwrite(comparison_filename, comparison_img)
        print(f"   ✅ 并排对比图保存成功: {comparison_filename}")
        
        # 差异高亮对比
        diff_filename = f"ptz_comparison_with_diff_{action}_{timestamp}.jpg"
        cv2.imwrite(diff_filename, comparison_with_diff)
        print(f"   ✅ 差异高亮图保存成功: {diff_filename}")
        
        # 差异热图
        heatmap_filename = f"ptz_diff_heatmap_{action}_{timestamp}.jpg"
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_filename, heatmap)
        print(f"   ✅ 差异热图保存成功: {heatmap_filename}")
        
        # 计算差异百分比
        total_pixels = diff.size
        diff_pixels = cv2.countNonZero(diff)
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        print(f"   📊 差异统计:")
        print(f"      - 总像素数: {total_pixels}")
        print(f"      - 差异像素数: {diff_pixels}")
        print(f"      - 差异百分比: {diff_percentage:.2f}%")
        print(f"      - 检测到的轮廓数: {len(contours)}")
        
        return {
            "before_frame": before_frame,
            "after_frame": after_frame,
            "diff": diff,
            "diff_percentage": diff_percentage,
            "diff_pixels": diff_pixels,
            "contours_count": len(contours),
            "comparison_filename": comparison_filename,
            "diff_filename": diff_filename,
            "heatmap_filename": heatmap_filename
        }
    
    def run_comparison_sequence(self):
        """运行对比序列"""
        print("\n" + "="*60)
        print("=== 开始PTZ摄像头视觉对比验证 ===")
        print("=== 本脚本将拍摄对比照片，直观显示摄像头移动 ===")
        print("="*60)
        
        # 1. 设置系统
        if not self.setup_systems():
            print(f"\n❌ 无法设置系统，验证失败")
            return
        
        # 2. 提供使用说明
        print(f"\n📋 视觉对比验证说明:")
        print(f"   - 系统将拍摄PTZ动作前后的对比照片")
        print(f"   - 对比照片将并排显示，方便直观观察变化")
        print(f"   - 差异区域将用红色框高亮显示")
        print(f"   - 生成差异热图，显示变化强度")
        print(f"   - 计算差异百分比，量化显示变化程度")
        
        # 3. 执行多种PTZ动作对比
        actions = [
            (PTZAction.PAN_LEFT, "pan_left"),
            (PTZAction.PAN_RIGHT, "pan_right"),
            (PTZAction.TILT_UP, "tilt_up"),
            (PTZAction.TILT_DOWN, "tilt_down")
        ]
        
        for action, action_name in actions:
            print(f"\n" + "="*50)
            print(f"执行 {action_name} 对比验证")
            print("="*50)
            
            # 拍摄动作前照片
            before_frame, before_filename = self.take_photo(f"动作前_{action_name}")
            if before_frame is None:
                continue
            
            # 执行PTZ动作
            print(f"\n🔄 执行 {action_name} 动作...")
            result = asyncio.run(self.ptz_controller.execute_action(action, speed=100))
            if result["success"]:
                print(f"   ✅ {action_name} 动作执行成功")
            else:
                print(f"   ⚠️  {action_name} 动作执行失败: {result['message']}")
            
            # 保持动作1秒
            time.sleep(1)
            
            # 拍摄动作后照片
            after_frame, after_filename = self.take_photo(f"动作后_{action_name}")
            if after_frame is None:
                continue
            
            # 创建对比图像
            comparison_result = self.create_comparison_image(before_frame, after_frame, action_name)
            
            # 记录对比结果
            self.comparison_results.append({
                "action": action_name,
                "before_filename": before_filename,
                "after_filename": after_filename,
                **comparison_result
            })
            
            # 停止PTZ动作
            asyncio.run(self.ptz_controller.execute_action(PTZAction.STOP, 0))
        
        # 4. 执行大角度移动对比
        print(f"\n" + "="*50)
        print(f"执行大角度移动对比验证")
        print("="*50)
        
        # 拍摄初始位置照片
        before_frame, before_filename = self.take_photo("大角度动作前")
        if before_frame is not None:
            # 执行180°旋转
            print(f"\n🔄 执行180°旋转...")
            initial_state = self.ptz_controller.get_status()
            initial_pan = initial_state["position"]["pan"]
            
            result = asyncio.run(self.ptz_controller.move_to_position(pan=initial_pan + 180, tilt=initial_state["position"]["tilt"], speed=100))
            if result["success"]:
                print(f"   ✅ 180°旋转执行成功")
            else:
                print(f"   ⚠️  180°旋转执行失败: {result['message']}")
            
            # 等待动作完成
            time.sleep(2)
            
            # 拍摄动作后照片
            after_frame, after_filename = self.take_photo("大角度动作后")
            if after_frame is not None:
                # 创建对比图像
                comparison_result = self.create_comparison_image(before_frame, after_frame, "pan_180_degrees")
                
                # 记录对比结果
                self.comparison_results.append({
                    "action": "pan_180_degrees",
                    "before_filename": before_filename,
                    "after_filename": after_filename,
                    **comparison_result
                })
        
        # 5. 生成对比报告
        self.generate_comparison_report()
        
        # 6. 清理资源
        self.cleanup()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print(f"\n" + "="*60)
        print("=== PTZ摄像头视觉对比验证报告 ===")
        print("="*60)
        
        # 统计对比结果
        total_comparisons = len(self.comparison_results)
        
        if total_comparisons == 0:
            print(f"\n❌ 没有生成对比结果")
            return
        
        print(f"\n📋 对比结果统计:")
        print(f"   - 总对比次数: {total_comparisons}")
        
        print(f"\n📊 详细对比结果:")
        for i, result in enumerate(self.comparison_results):
            print(f"   {i+1}. {result['action']}")
            print(f"      差异像素数: {result['diff_pixels']}")
            print(f"      差异百分比: {result['diff_percentage']:.2f}%")
            print(f"      检测到的轮廓数: {result['contours_count']}")
            print(f"      动作前照片: {result['before_filename']}")
            print(f"      动作后照片: {result['after_filename']}")
            print(f"      并排对比图: {result['comparison_filename']}")
            print(f"      差异高亮图: {result['diff_filename']}")
            print(f"      差异热图: {result['heatmap_filename']}")
        
        # 保存对比报告
        with open("ptz_visual_comparison_report.txt", "w") as f:
            f.write("PTZ摄像头视觉对比验证报告\n")
            f.write("="*60 + "\n")
            f.write(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证类型: 视觉对比验证\n")
            f.write(f"总对比次数: {total_comparisons}\n\n")
            f.write("详细结果:\n")
            for i, result in enumerate(self.comparison_results):
                f.write(f"{i+1}. {result['action']}\n")
                f.write(f"   差异像素数: {result['diff_pixels']}\n")
                f.write(f"   差异百分比: {result['diff_percentage']:.2f}%\n")
                f.write(f"   检测到的轮廓数: {result['contours_count']}\n")
                f.write(f"   动作前照片: {result['before_filename']}\n")
                f.write(f"   动作后照片: {result['after_filename']}\n")
                f.write(f"   并排对比图: {result['comparison_filename']}\n")
                f.write(f"   差异高亮图: {result['diff_filename']}\n")
                f.write(f"   差异热图: {result['heatmap_filename']}\n\n")
        
        print(f"\n📄 对比报告已保存为 ptz_visual_comparison_report.txt")
    
    def cleanup(self):
        """清理资源"""
        print(f"\n=== 清理资源 ===")
        
        # 关闭监控摄像头
        self.monitor_camera.close_camera()
        print(f"   ✅ 监控摄像头已关闭")
        
        # 断开PTZ控制器
        if self.ptz_controller:
            asyncio.run(self.ptz_controller.disconnect())
            print(f"   ✅ PTZ控制器已断开")

if __name__ == "__main__":
    # 创建对比验证系统
    comparator = PTZVisualComparison()
    
    try:
        # 运行对比序列
        comparator.run_comparison_sequence()
    except KeyboardInterrupt:
        print("\n\n🔴 验证被用户中断")
        comparator.cleanup()
    except Exception as e:
        print(f"\n\n❌ 验证过程中发生错误: {e}")
        comparator.cleanup()
    finally:
        print("\n🎉 PTZ摄像头视觉对比验证完成")
        print("\n📋 最终建议:")
        print("1. 查看生成的对比照片，直观观察摄像头是否移动")
        print("2. 查看差异高亮图，了解具体移动区域")
        print("3. 查看差异热图，了解变化强度")
        print("4. 对比差异百分比，量化移动程度")
        print("5. 照片文件已保存在当前目录")