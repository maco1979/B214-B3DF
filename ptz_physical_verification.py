#!/usr/bin/env python3
"""PTZ摄像头物理验证脚本"""

import asyncio
import time
import requests
import cv2
import numpy as np
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class PTZPhysicalVerification:
    """PTZ摄像头物理验证"""
    
    def __init__(self):
        """初始化验证系统"""
        self.ptz_controller = None                # PTZ控制器
        self.verification_results = []           # 验证结果
        
        print("=== PTZ摄像头物理验证系统 ===")
        print("系统功能:")
        print("1. 严格检查PTZ控制器连接")
        print("2. 发送真实的PTZ命令到硬件")
        print("3. 提供详细的HTTP请求和响应信息")
        print("4. 指导用户进行物理观察验证")
        print("5. 支持多种验证方式")
        print("6. 生成详细的验证报告")
    
    def test_camera_connection(self, camera_ip, username, password):
        """测试摄像头连接"""
        print(f"\n=== 测试摄像头连接 ===")
        print(f"测试目标: {camera_ip}")
        
        # 测试1: 基本HTTP连接
        print(f"\n1. 测试基本HTTP连接:")
        try:
            response = requests.get(f"http://{camera_ip}", auth=(username, password), timeout=5)
            print(f"   ✅ HTTP连接成功，状态码: {response.status_code}")
            print(f"   响应长度: {len(response.text)} 字节")
        except Exception as e:
            print(f"   ❌ HTTP连接失败: {e}")
            return False
        
        # 测试2: 测试海康威视ISAPI接口
        print(f"\n2. 测试海康威视ISAPI接口:")
        isapi_url = f"http://{camera_ip}/ISAPI/System/deviceInfo"
        try:
            response = requests.get(isapi_url, auth=(username, password), timeout=5)
            print(f"   ISAPI状态码: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ ISAPI接口成功，这是真实的海康威视摄像头")
                print(f"   设备信息: {response.text[:200]}...")
                return True
            elif response.status_code == 401:
                print(f"   ⚠️ ISAPI接口需要认证，这可能是海康威视摄像头")
                print(f"   尝试使用不同的用户名密码")
                return True
            else:
                print(f"   ❌ ISAPI接口返回错误状态: {response.status_code}")
                print(f"   响应: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ ISAPI接口访问失败: {e}")
            return False
    
    def setup_ptz_controller(self):
        """设置PTZ控制器"""
        print("\n=== 设置PTZ控制器 ===")
        
        # 用户配置区
        CAMERA_IP = "192.168.1.64"  # 请替换为真实PTZ摄像头IP
        USERNAME = "admin"         # 请替换为真实用户名
        PASSWORD = "admin"         # 请替换为真实密码
        
        print(f"配置信息:")
        print(f"   IP地址: {CAMERA_IP}")
        print(f"   用户名: {USERNAME}")
        print(f"   密码: {'*' * len(PASSWORD)}")
        
        # 先测试摄像头连接
        if not self.test_camera_connection(CAMERA_IP, USERNAME, PASSWORD):
            print(f"\n❌ 摄像头连接测试失败，无法继续")
            return False
        
        # 初始化PTZ控制器
        self.ptz_controller = PTZCameraController(
            protocol=PTZProtocol.HTTP_API,
            connection_type="http",
            base_url=f"http://{CAMERA_IP}",
            username=USERNAME,
            password=PASSWORD
        )
        
        # 连接PTZ控制器
        print(f"\n3. 连接PTZ控制器:")
        result = asyncio.run(self.ptz_controller.connect())
        if result["success"]:
            print(f"   ✅ PTZ控制器连接成功: {result['message']}")
            return True
        else:
            print(f"   ❌ PTZ控制器连接失败: {result['message']}")
            return False
    
    def send_ptz_command(self, action, speed=100, duration=2):
        """发送PTZ命令并提供物理验证指导"""
        print(f"\n=== 发送PTZ命令: {action.value} ===")
        
        # 提供物理观察指导
        print(f"\n📋 物理验证步骤:")
        print(f"1. 请您亲自观察摄像头的物理位置")
        print(f"2. 确认摄像头当前的朝向")
        print(f"3. 准备好观察摄像头是否会移动")
        print(f"4. 按下Enter键继续发送PTZ命令...")
        input()
        
        # 记录开始时间
        start_time = time.time()
        
        # 发送PTZ命令
        result = asyncio.run(self.ptz_controller.execute_action(action, speed))
        
        if result["success"]:
            print(f"\n✅ {action.value} 命令发送成功")
        else:
            print(f"\n⚠️ {action.value} 命令发送失败: {result['message']}")
        
        # 提供观察指导
        print(f"\n👀 请您现在观察摄像头:")
        print(f"   - 摄像头是否正在向左/向右/向上/向下移动？")
        print(f"   - 摄像头的机械结构是否有转动？")
        print(f"   - 画面是否发生了明显变化？")
        print(f"\n请您确认观察结果:")
        print(f"   1. ✅ 摄像头确实在移动")
        print(f"   2. ❌ 摄像头没有移动")
        print(f"   3. ⚠️  不确定，需要重试")
        
        # 等待命令执行
        print(f"\n⏱️  命令执行中... (持续 {duration} 秒)")
        time.sleep(duration)
        
        # 停止PTZ动作
        await self.ptz_controller.execute_action(PTZAction.STOP, 0)
        print(f"\n✅ 已发送停止命令")
        
        # 获取用户观察结果
        print(f"\n📝 请输入您的观察结果 (1-3):")
        user_input = input()
        
        # 解析用户输入
        physical_result = ""
        if user_input == "1":
            physical_result = "摄像头确实在移动"
            is_physical_moved = True
        elif user_input == "2":
            physical_result = "摄像头没有移动"
            is_physical_moved = False
        elif user_input == "3":
            physical_result = "不确定，需要重试"
            is_physical_moved = False
        else:
            physical_result = "无效输入，默认认为没有移动"
            is_physical_moved = False
        
        # 记录验证结果
        verification_data = {
            "timestamp": time.time(),
            "action": action.value,
            "speed": speed,
            "duration": duration,
            "command_success": result["success"],
            "physical_result": physical_result,
            "is_physical_moved": is_physical_moved,
            "user_input": user_input
        }
        
        self.verification_results.append(verification_data)
        
        return is_physical_moved
    
    def run_physical_verification(self):
        """运行物理验证"""
        print("\n" + "="*60)
        print("=== 开始PTZ摄像头物理验证 ===")
        print("=== 本脚本将指导您进行实际物理验证 ===")
        print("="*60)
        
        # 1. 设置PTZ控制器
        if not self.setup_ptz_controller():
            print(f"\n❌ 无法设置PTZ控制器，验证失败")
            return
        
        # 2. 提供验证说明
        print(f"\n📋 PTZ摄像头物理验证说明:")
        print(f"   - 本验证将发送真实的PTZ命令到摄像头硬件")
        print(f"   - 您需要亲自观察摄像头是否真的在移动")
        print(f"   - 请确保您能直接看到摄像头的物理位置")
        print(f"   - 本验证不依赖于画面差异检测，只依赖您的实际观察")
        
        print(f"\n⚠️  重要提示:")
        print(f"   - 请确保摄像头没有被固定或锁定")
        print(f"   - 请确保摄像头支持PTZ功能")
        print(f"   - 请确保您使用了正确的用户名和密码")
        
        print(f"\n✅ 准备就绪，开始验证")
        
        # 3. 执行多种PTZ动作验证
        actions = [
            (PTZAction.PAN_LEFT, "向左旋转"),
            (PTZAction.PAN_RIGHT, "向右旋转"),
            (PTZAction.TILT_UP, "向上倾斜"),
            (PTZAction.TILT_DOWN, "向下倾斜")
        ]
        
        for action, description in actions:
            print(f"\n" + "-"*50)
            print(f"执行 {description} 验证")
            print("-"*50)
            
            self.send_ptz_command(action, speed=100, duration=3)
        
        # 4. 执行大角度移动验证
        print(f"\n" + "-"*50)
        print(f"执行大角度移动验证 (180°旋转)")
        print("-"*50)
        
        # 提供物理观察指导
        print(f"\n📋 大角度移动验证步骤:")
        print(f"1. 请您记住摄像头当前的朝向")
        print(f"2. 我们将发送180°旋转命令")
        print(f"3. 摄像头应该旋转180度，完全转向相反方向")
        print(f"4. 这是最明显的移动，应该很容易观察到")
        print(f"\n按下Enter键继续...")
        input()
        
        # 获取初始位置
        initial_state = self.ptz_controller.get_status()
        initial_pan = initial_state["position"]["pan"]
        
        # 发送180°旋转命令
        result = asyncio.run(self.ptz_controller.move_to_position(pan=initial_pan + 180, tilt=initial_state["position"]["tilt"], speed=100))
        
        print(f"\n📝 大角度移动结果:")
        print(f"   命令发送结果: {'成功' if result['success'] else '失败'}")
        print(f"\n👀 请您观察摄像头是否旋转了180度:")
        print(f"   - 摄像头是否完全转向了相反方向？")
        print(f"   - 这是最明显的移动，应该很容易观察到")
        
        # 获取用户观察结果
        print(f"\n请输入您的观察结果 (1-3):")
        print(f"   1. ✅ 摄像头确实旋转了180度")
        print(f"   2. ❌ 摄像头没有明显移动")
        print(f"   3. ⚠️  不确定")
        user_input = input()
        
        # 解析用户输入
        physical_result = ""
        if user_input == "1":
            physical_result = "摄像头确实旋转了180度"
            is_physical_moved = True
        elif user_input == "2":
            physical_result = "摄像头没有明显移动"
            is_physical_moved = False
        elif user_input == "3":
            physical_result = "不确定"
            is_physical_moved = False
        else:
            physical_result = "无效输入，默认认为没有移动"
            is_physical_moved = False
        
        # 记录验证结果
        verification_data = {
            "timestamp": time.time(),
            "action": "pan_180_degrees",
            "speed": 100,
            "duration": 5,
            "command_success": result["success"],
            "physical_result": physical_result,
            "is_physical_moved": is_physical_moved,
            "user_input": user_input
        }
        
        self.verification_results.append(verification_data)
        
        # 5. 生成验证报告
        self.generate_verification_report()
        
        # 6. 清理资源
        self.cleanup()
    
    def generate_verification_report(self):
        """生成验证报告"""
        print(f"\n" + "="*60)
        print("=== PTZ摄像头物理验证报告 ===")
        print("="*60)
        
        # 统计验证结果
        total_tests = len(self.verification_results)
        moved_tests = sum(1 for r in self.verification_results if r["is_physical_moved"])
        success_rate = (moved_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📋 验证结果统计:")
        print(f"   - 总测试次数: {total_tests}")
        print(f"   - 物理移动次数: {moved_tests}")
        print(f"   - 物理移动成功率: {success_rate:.2f}%")
        
        print(f"\n📊 详细验证结果:")
        for i, result in enumerate(self.verification_results):
            status = "✅ 物理移动" if result["is_physical_moved"] else "❌ 没有移动"
            command_status = "✅ 命令成功" if result["command_success"] else "❌ 命令失败"
            print(f"   {i+1}. {result['action']} - {command_status} - {status}")
            print(f"      物理观察结果: {result['physical_result']}")
        
        print(f"\n🔍 验证结论:")
        if moved_tests > 0:
            print(f"   ✅ 成功！摄像头确实能够物理移动")
            print(f"   🎯 AI可以控制PTZ摄像头进行真实移动")
        else:
            print(f"   ❌ 失败！摄像头没有发生物理移动")
            print(f"   💡 可能的原因:")
            print(f"   - 摄像头IP或登录信息错误")
            print(f"   - 摄像头不支持PTZ功能")
            print(f"   - 摄像头被固定或锁定")
            print(f"   - 网络连接问题")
            print(f"   - 摄像头可能处于待机状态")
        
        # 保存验证报告
        with open("ptz_physical_verification_report.txt", "w") as f:
            f.write("PTZ摄像头物理验证报告\n")
            f.write("="*60 + "\n")
            f.write(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证类型: 物理观察验证\n")
            f.write(f"总测试次数: {total_tests}\n")
            f.write(f"物理移动次数: {moved_tests}\n")
            f.write(f"物理移动成功率: {success_rate:.2f}%\n\n")
            f.write("详细结果:\n")
            for i, result in enumerate(self.verification_results):
                status = "物理移动" if result["is_physical_moved"] else "没有移动"
                command_status = "命令成功" if result["command_success"] else "命令失败"
                f.write(f"{i+1}. {result['action']} - {command_status} - {status}\n")
                f.write(f"   物理观察结果: {result['physical_result']}\n")
        
        print(f"\n📄 验证报告已保存为 ptz_physical_verification_report.txt")
    
    def cleanup(self):
        """清理资源"""
        print(f"\n=== 清理资源 ===")
        
        # 断开PTZ控制器
        if self.ptz_controller:
            result = asyncio.run(self.ptz_controller.disconnect())
            if result["success"]:
                print(f"   ✅ PTZ控制器已断开")
    
    def emergency_stop(self):
        """紧急停止所有PTZ动作"""
        print(f"\n=== 紧急停止 ===")
        if self.ptz_controller:
            result = asyncio.run(self.ptz_controller.execute_action(PTZAction.STOP, 0))
            if result["success"]:
                print(f"   ✅ 紧急停止命令发送成功")
            else:
                print(f"   ⚠️  紧急停止命令发送失败")

if __name__ == "__main__":
    # 创建验证系统
    verifier = PTZPhysicalVerification()
    
    try:
        # 运行物理验证
        verifier.run_physical_verification()
    except KeyboardInterrupt:
        print("\n\n🔴 验证被用户中断")
        verifier.emergency_stop()
        verifier.cleanup()
    except Exception as e:
        print(f"\n\n❌ 验证过程中发生错误: {e}")
        verifier.emergency_stop()
        verifier.cleanup()
    finally:
        print("\n🎉 PTZ摄像头物理验证完成")
        print("\n📋 最终建议:")
        print("1. 请仔细检查验证报告，确认摄像头是否真的能移动")
        print("2. 如果摄像头不能移动，请检查网络连接和登录信息")
        print("3. 确保摄像头支持PTZ功能且没有被锁定")
        print("4. 必要时请参考摄像头的用户手册")
        print("5. 本验证结果是最可靠的，因为它基于实际物理观察")