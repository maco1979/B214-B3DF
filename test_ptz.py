#!/usr/bin/env python3
"""
PTZ云台测试脚本
测试云台的连接、控制、预置位等功能
"""

import requests
import time

# 后端API地址
BASE_URL = "http://localhost:8001/api"

# 测试配置
TEST_CONFIG = {
    "protocol": "pelco_d",
    "connection_type": "http",
    "speed": 50,
    "test_preset_id": 1
}

class PTZTest:
    """PTZ云台测试类"""
    
    def __init__(self, base_url, config):
        self.base_url = base_url
        self.config = config
        self.session = requests.Session()
        self.connected = False
        self.test_results = []
    
    def log_result(self, test_name, success, message):
        """记录测试结果"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message
        })
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
    
    def test_ptz_status(self):
        """测试PTZ状态查询"""
        test_name = "PTZ状态查询"
        try:
            response = self.session.get(f"{self.base_url}/camera/ptz/status", timeout=5)
            if response.status_code == 200:
                result = response.json()
                self.log_result(test_name, True, f"状态：{result}")
                return True
            else:
                self.log_result(test_name, False, f"HTTP错误: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"连接失败: {str(e)}")
            return False
    
    def test_ptz_connect(self):
        """测试PTZ连接"""
        test_name = "PTZ连接"
        try:
            # 准备连接参数
            params = {
                "protocol": self.config["protocol"],
                "connection_type": self.config["connection_type"]
            }
            
            # 根据连接类型添加额外参数
            if self.config["connection_type"] == "serial":
                params.update({
                    "port": self.config["serial_port"],
                    "baudrate": self.config["baudrate"],
                    "address": 1
                })
            elif self.config["connection_type"] == "network":
                params.update({
                    "host": "192.168.1.100",
                    "network_port": 5000,
                    "address": 1
                })
            
            # 发送连接请求
            response = self.session.post(f"{self.base_url}/camera/ptz/connect", json=params, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.connected = True
                    self.log_result(test_name, True, "连接成功")
                    return True
                else:
                    self.log_result(test_name, False, f"连接失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"HTTP错误: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"连接异常: {str(e)}")
            return False
    
    def test_ptz_disconnect(self):
        """测试PTZ断开连接"""
        test_name = "PTZ断开连接"
        try:
            response = self.session.post(f"{self.base_url}/camera/ptz/disconnect", timeout=5)
            if response.status_code == 200:
                result = response.json()
                self.connected = False
                self.log_result(test_name, True, "断开成功")
                return True
            else:
                self.log_result(test_name, False, f"HTTP错误: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"断开异常: {str(e)}")
            return False
    
    def test_direction_control(self, action):
        """测试方向控制"""
        test_name = f"方向控制 - {action}"
        try:
            params = {
                "action": action,
                "speed": self.config["speed"]
            }
            response = self.session.post(f"{self.base_url}/camera/ptz/action", json=params, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, "控制成功")
                    return True
                else:
                    self.log_result(test_name, False, f"控制失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"HTTP错误: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"控制异常: {str(e)}")
            return False
    
    def test_all_directions(self):
        """测试所有方向"""
        directions = ["pan_left", "pan_right", "tilt_up", "tilt_down"]
        success_count = 0
        for direction in directions:
            if self.test_direction_control(direction):
                success_count += 1
            time.sleep(0.5)  # 等待转动完成
        
        # 测试停止命令
        if self.test_direction_control("stop"):
            success_count += 1
        
        return success_count == len(directions) + 1
    
    def test_preset_set(self):
        """测试设置预置位"""
        test_name = "设置预置位"
        try:
            params = {
                "preset_id": self.config["test_preset_id"],
                "name": "测试预置位"
            }
            response = self.session.post(f"{self.base_url}/camera/ptz/preset/set", json=params, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, f"预置位{self.config['test_preset_id']}设置成功")
                    return True
                else:
                    self.log_result(test_name, False, f"设置失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"HTTP错误: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"设置异常: {str(e)}")
            return False
    
    def test_preset_goto(self):
        """测试调用预置位"""
        test_name = "调用预置位"
        try:
            params = {
                "preset_id": self.config["test_preset_id"]
            }
            response = self.session.post(f"{self.base_url}/camera/ptz/preset/goto", json=params, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, f"转到预置位{self.config['test_preset_id']}成功")
                    return True
                else:
                    self.log_result(test_name, False, f"调用失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"HTTP错误: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"调用异常: {str(e)}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== PTZ云台测试开始 ===")
        print(f"API地址: {self.base_url}")
        print(f"配置: {self.config}")
        print("="*40)
        
        # 1. 测试状态查询
        status_result = self.test_ptz_status()
        
        # 2. 获取PTZ状态，根据状态决定是否继续测试
        try:
            response = self.session.get(f"{self.base_url}/camera/ptz/status", timeout=5)
            if response.status_code == 200:
                ptz_status = response.json()
                is_connected = ptz_status.get("data", {}).get("connected", False)
                
                if not is_connected:
                    print("\n⚠️  PTZ云台未连接，跳过控制测试")
                    print("⚠️  请先确保PTZ设备已连接并配置正确")
                    print("⚠️  当前状态:")
                    print(f"   - 连接状态: {ptz_status.get('data', {}).get('connected', False)}")
                    print(f"   - 协议: {ptz_status.get('data', {}).get('protocol', '未知')}")
                    print(f"   - 连接类型: {ptz_status.get('data', {}).get('connection_type', '未知')}")
                    print(f"   - 位置: {ptz_status.get('data', {}).get('position', {})}")
                else:
                    # 3. 测试方向控制
                    print("\n🔄 PTZ云台已连接，开始测试控制功能...")
                    self.test_all_directions()
                    
                    # 4. 测试预置位
                    self.test_preset_set()
                    self.test_preset_goto()
        except Exception as e:
            print(f"\n⚠️  获取PTZ状态失败: {str(e)}")
            print("⚠️  跳过控制测试")
        
        print("="*40)
        print("=== 测试结果总结 ===")
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        
        print(f"总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        print("\n详细结果:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['message']}")
        
        print("\n=== 测试完成 ===")
        return passed == total

if __name__ == "__main__":
    # 创建测试实例
    ptz_test = PTZTest(BASE_URL, TEST_CONFIG)
    
    # 运行所有测试
    ptz_test.run_all_tests()
