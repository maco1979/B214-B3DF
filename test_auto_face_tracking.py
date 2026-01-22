#!/usr/bin/env python3
"""
自动人脸跟踪功能测试脚本
测试打开摄像头后是否自动寻找脸部并进行跟踪
"""

import requests
import time
import json

# 后端API地址
BASE_URL = "http://localhost:8001/api"

class AutoFaceTrackingTest:
    """自动人脸跟踪功能测试类"""
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_result(self, test_name, success, message):
        """记录测试结果"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.time()
        })
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
    
    def test_api_connection(self):
        """测试API连接"""
        test_name = "API连接测试"
        try:
            response = self.session.get(f"{self.base_url}/camera/status", timeout=5)
            if response.status_code == 200:
                self.log_result(test_name, True, "API连接成功")
                return True
            else:
                self.log_result(test_name, False, f"API连接失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"API连接失败: {str(e)}")
            return False
    
    def open_camera(self):
        """打开摄像头"""
        test_name = "打开摄像头"
        try:
            response = self.session.post(f"{self.base_url}/camera/open", json={"camera_index": 999}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, "摄像头打开成功")
                    return True
                else:
                    self.log_result(test_name, False, f"摄像头打开失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"摄像头打开失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"摄像头打开失败: {str(e)}")
            return False
    
    def close_camera(self):
        """关闭摄像头"""
        test_name = "关闭摄像头"
        try:
            response = self.session.post(f"{self.base_url}/camera/close", timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, "摄像头关闭成功")
                    return True
                else:
                    self.log_result(test_name, False, f"摄像头关闭失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"摄像头关闭失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"摄像头关闭失败: {str(e)}")
            return False
    
    def start_face_recognition(self):
        """启动人脸识别"""
        test_name = "启动人脸识别"
        try:
            response = self.session.post(f"{self.base_url}/camera/recognition/start", json={"model_type": "haar"}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, "人脸识别启动成功")
                    return True
                else:
                    self.log_result(test_name, False, f"人脸识别启动失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"人脸识别启动失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"人脸识别启动失败: {str(e)}")
            return False
    
    def stop_face_recognition(self):
        """停止人脸识别"""
        test_name = "停止人脸识别"
        try:
            response = self.session.post(f"{self.base_url}/camera/recognition/stop", timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, "人脸识别停止成功")
                    return True
                else:
                    self.log_result(test_name, False, f"人脸识别停止失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"人脸识别停止失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"人脸识别停止失败: {str(e)}")
            return False
    
    def stop_tracking(self):
        """停止跟踪"""
        test_name = "停止跟踪"
        try:
            response = self.session.post(f"{self.base_url}/camera/tracking/stop", timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    self.log_result(test_name, True, "跟踪停止成功")
                    return True
                else:
                    self.log_result(test_name, False, f"跟踪停止失败: {result['message']}")
                    return False
            else:
                self.log_result(test_name, False, f"跟踪停止失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, f"跟踪停止失败: {str(e)}")
            return False
    
    def check_tracking_status(self):
        """检查跟踪状态"""
        try:
            response = self.session.get(f"{self.base_url}/camera/tracking/status", timeout=5)
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                return None
        except Exception as e:
            return None
    
    def check_recognition_status(self):
        """检查识别状态"""
        try:
            response = self.session.get(f"{self.base_url}/camera/recognition/status", timeout=5)
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                return None
        except Exception as e:
            return None
    
    def test_auto_face_tracking(self, duration=20):
        """测试自动人脸跟踪功能"""
        print("=== 测试自动人脸跟踪功能 ===")
        print(f"测试持续时间: {duration}秒")
        
        # 1. 测试API连接
        if not self.test_api_connection():
            print("API连接失败，测试终止")
            return False
        
        try:
            # 2. 打开摄像头
            if not self.open_camera():
                print("无法打开摄像头，测试终止")
                return False
            
            # 3. 启动人脸识别
            if not self.start_face_recognition():
                print("无法启动人脸识别，测试终止")
                self.close_camera()
                return False
            
            # 4. 等待并观察自动跟踪启动
            print("\n等待自动跟踪启动...")
            start_time = time.time()
            tracking_started = False
            tracking_start_time = 0
            
            while time.time() - start_time < 10 and not tracking_started:
                tracking_status = self.check_tracking_status()
                if tracking_status and tracking_status["data"]["tracking_enabled"]:
                    tracking_started = True
                    tracking_start_time = time.time()
                    print("✅ 自动跟踪已启动")
                    self.log_result("自动跟踪启动", True, "成功检测到人脸并自动启动跟踪")
                else:
                    print("\r等待自动跟踪启动...", end="")
                time.sleep(1)
            
            if not tracking_started:
                self.log_result("自动跟踪启动", False, "10秒内未检测到人脸或未启动跟踪")
                self.stop_face_recognition()
                self.close_camera()
                return False
            
            # 5. 测试跟踪持续时间
            print("\n测试跟踪持续时间...")
            tracking_duration = 0
            while time.time() - tracking_start_time < duration:
                tracking_status = self.check_tracking_status()
                if tracking_status and tracking_status["data"]["tracking_enabled"]:
                    tracking_duration = time.time() - tracking_start_time
                    tracked_object = tracking_status["data"]["tracked_object"]
                    print(f"\r跟踪持续时间: {int(tracking_duration)}秒, 跟踪对象: {tracked_object}", end="")
                else:
                    print("\r跟踪停止，尝试重新检测...", end="")
                time.sleep(1)
            
            print()
            self.log_result("跟踪持续时间", True, f"成功跟踪 {int(tracking_duration)} 秒")
            
            # 6. 测试结果
            final_tracking_status = self.check_tracking_status()
            final_recognition_status = self.check_recognition_status()
            
            if final_tracking_status and final_tracking_status["data"]["tracking_enabled"]:
                self.log_result("最终跟踪状态", True, "跟踪仍在持续")
            else:
                self.log_result("最终跟踪状态", False, "跟踪已停止")
            
            return True
            
        except Exception as e:
            self.log_result("自动人脸跟踪测试", False, f"测试异常: {str(e)}")
            return False
        finally:
            # 清理资源
            self.stop_tracking()
            self.stop_face_recognition()
            self.close_camera()
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 自动人脸跟踪功能测试 ===")
        print(f"API地址: {self.base_url}")
        print("="*50)
        
        # 运行自动人脸跟踪测试
        self.test_auto_face_tracking(duration=15)
        
        print("="*50)
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

def main():
    """主函数"""
    auto_face_test = AutoFaceTrackingTest(BASE_URL)
    auto_face_test.run_all_tests()

if __name__ == "__main__":
    main()
