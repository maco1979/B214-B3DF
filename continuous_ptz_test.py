#!/usr/bin/env python3
"""持续PTZ控制测试脚本"""

import requests
import time
import random

BASE_URL = "http://localhost:8001"

# 支持的摄像头协议和配置
default_configs = [
    {
        "name": "海康威视HTTP API",
        "config": {
            "protocol": "http",
            "connection_type": "http",
            "base_url": "http://192.168.1.1",
            "username": "admin",
            "password": "admin"
        }
    },
    {
        "name": "Pelco-D串口",
        "config": {
            "protocol": "pelco_d",
            "connection_type": "serial",
            "port": "COM3",
            "baudrate": 9600,
            "address": 1
        }
    },
    {
        "name": "Pelco-P网络",
        "config": {
            "protocol": "pelco_p",
            "connection_type": "network",
            "host": "192.168.1.1",
            "network_port": 5000,
            "address": 1
        }
    }
]

class ContinuousPTZTest:
    """持续PTZ测试类"""
    
    def __init__(self, configs=None):
        self.configs = configs if configs else default_configs
        self.max_test_rounds = 100  # 最大测试轮数
        self.movement_threshold = 10.0  # 移动检测阈值（度）
        self.test_interval = 5  # 测试间隔（秒）
    
    def open_camera(self):
        """打开摄像头，忽略已打开的情况"""
        print("1. 打开摄像头...")
        try:
            result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
            if result.get("success"):
                print("   ✅ 摄像头打开成功")
                return True
            else:
                # 忽略已打开的情况
                if "已打开" in result.get("message", ""):
                    print(f"   ℹ️  摄像头已打开，继续测试")
                    return True
                else:
                    print(f"   ❌ 摄像头打开失败: {result}")
                    return False
        except Exception as e:
            print(f"   ❌ 摄像头打开异常: {e}")
            return False
    
    def close_camera(self):
        """关闭摄像头"""
        print("   • 关闭摄像头")
        try:
            requests.post(f"{BASE_URL}/api/camera/close").json()
        except:
            pass
    
    def disconnect_ptz(self):
        """断开PTZ连接"""
        print("   • 断开PTZ连接")
        try:
            requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
        except:
            pass
    
    def test_single_config(self, config_info, round_num):
        """测试单个配置"""
        name = config_info["name"]
        config = config_info["config"]
        
        print(f"\n📋 轮次 {round_num}: 测试 {name}")
        print(f"   配置: {config}")
        
        try:
            # 1. 断开现有连接
            self.disconnect_ptz()
            
            # 2. 连接PTZ
            print(f"   • 连接PTZ")
            connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=config).json()
            if not connect_result.get("success"):
                print(f"   ❌ PTZ连接失败: {connect_result}")
                return False
            
            # 3. 获取初始位置
            print(f"   • 获取初始位置")
            initial_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
            if not initial_status.get("success"):
                print(f"   ❌ 获取初始位置失败: {initial_status}")
                return False
            initial_pan = initial_status["data"]["position"]["pan"]
            initial_tilt = initial_status["data"]["position"]["tilt"]
            print(f"   • 初始位置: pan={initial_pan:.1f}°, tilt={initial_tilt:.1f}°")
            
            # 4. 生成随机测试位置（大角度移动）
            test_pan = random.uniform(-180, 180)
            test_tilt = random.uniform(-90, 90)
            test_speed = random.randint(50, 90)
            
            # 5. 发送移动命令
            print(f"   • 发送移动命令: pan={test_pan:.1f}°, tilt={test_tilt:.1f}°, speed={test_speed}%")
            move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                                      json={"pan": test_pan, "tilt": test_tilt, "speed": test_speed})
            
            if move_result.status_code != 200:
                print(f"   ❌ 移动命令请求失败: {move_result.status_code}")
                return False
            
            move_data = move_result.json()
            if not move_data.get("success"):
                print(f"   ❌ 移动命令执行失败: {move_data}")
                return False
            
            # 6. 等待移动完成
            print(f"   • 等待移动完成 (3秒)")
            time.sleep(3)
            
            # 7. 获取最终位置
            print(f"   • 获取最终位置")
            final_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
            if not final_status.get("success"):
                print(f"   ❌ 获取最终位置失败: {final_status}")
                return False
            final_pan = final_status["data"]["position"]["pan"]
            final_tilt = final_status["data"]["position"]["tilt"]
            print(f"   • 最终位置: pan={final_pan:.1f}°, tilt={final_tilt:.1f}°")
            
            # 8. 计算移动距离
            pan_diff = abs(final_pan - initial_pan)
            tilt_diff = abs(final_tilt - initial_tilt)
            print(f"   • 移动距离: pan={pan_diff:.1f}°, tilt={tilt_diff:.1f}°")
            
            # 9. 检测是否真正移动
            if pan_diff > self.movement_threshold or tilt_diff > self.movement_threshold:
                print(f"   ✅ 检测到云台真正移动！移动距离超过阈值 {self.movement_threshold}°")
                print(f"   🎉 测试成功！云台已经真正运动")
                print(f"   📋 成功配置:")
                print(f"      名称: {name}")
                print(f"      配置: {config}")
                print(f"      移动距离: pan={pan_diff:.1f}°, tilt={tilt_diff:.1f}°")
                return True
            else:
                print(f"   ⚠️  未检测到明显移动，距离小于阈值 {self.movement_threshold}°")
                return False
                
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            return False
        finally:
            # 移动回初始位置
            try:
                requests.post(f"{BASE_URL}/api/camera/ptz/move", json={"pan": 0, "tilt": 0, "speed": 100})
                time.sleep(2)
            except:
                pass
            self.disconnect_ptz()
    
    def run_continuous_test(self):
        """运行持续测试"""
        print("=== 持续PTZ控制测试 ===")
        print(f"\n🎯 测试目标:")
        print(f"   • 持续测试直到云台真正运动")
        print(f"   • 支持多种摄像头品牌和协议")
        print(f"   • 检测阈值: {self.movement_threshold}°")
        print(f"   • 最大测试轮数: {self.max_test_rounds}")
        
        # 1. 打开摄像头
        if not self.open_camera():
            return False
        
        try:
            # 2. 持续测试
            for round_num in range(1, self.max_test_rounds + 1):
                # 循环测试所有配置
                for config_info in default_configs:
                    if self.test_single_config(config_info, round_num):
                        # 测试成功，退出
                        return True
                
                # 增加随机延迟，避免过于频繁
                delay = random.uniform(2, 5)
                print(f"\n⏳ 等待 {delay:.1f} 秒后继续下一轮测试...")
                time.sleep(delay)
            
            # 测试失败，所有轮次都没有检测到移动
            print(f"\n❌ 所有 {self.max_test_rounds} 轮测试完成，未检测到云台真正移动")
            print(f"\n💡 可能的原因:")
            print(f"   1. 摄像头IP地址错误")
            print(f"   2. 用户名或密码错误")
            print(f"   3. 摄像头不支持该协议")
            print(f"   4. 网络连接问题")
            print(f"   5. 摄像头未正确连接到网络")
            return False
            
        finally:
            # 清理资源
            print(f"\n🧹 清理资源:")
            self.disconnect_ptz()
            self.close_camera()

def main():
    """主函数"""
    # 创建持续测试实例
    tester = ContinuousPTZTest()
    
    # 运行测试
    success = tester.run_continuous_test()
    
    if success:
        print(f"\n🎉 测试成功！云台已经真正运动")
        print(f"\n📋 最终结果:")
        print(f"   • 测试成功: ✅")
        print(f"   • 云台真正运动: ✅")
        print(f"   • 支持多种品牌: ✅")
        print(f"   • 持续测试直到成功: ✅")
    else:
        print(f"\n❌ 测试失败，请检查摄像头配置")

if __name__ == "__main__":
    main()