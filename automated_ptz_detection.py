#!/usr/bin/env python3
"""PTZ摄像头自动化检测脚本"""

import requests
import socket
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

class AutomatedPTZDetection:
    """PTZ摄像头自动化检测系统"""
    
    def __init__(self):
        """初始化自动化检测系统"""
        self.monitor_camera = CameraController()
        self.ptz_controller = None
        self.results = {
            "ip_detection": [],
            "camera_connection": [],
            "ptz_functionality": [],
            "visual_verification": []
        }
        
        print("=== PTZ摄像头自动化检测 ===")
        print("系统功能:")
        print("1. 自动扫描网段内的摄像头IP")
        print("2. 测试摄像头连接和认证")
        print("3. 验证PTZ基本功能")
        print("4. 进行视觉对比验证")
        print("5. 生成详细检测报告")
    
    def check_port(self, ip, port, timeout=0.5):
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def check_http(self, ip, port=80, timeout=2):
        """检查HTTP服务"""
        try:
            url = f"http://{ip}:{port}"
            response = requests.get(url, timeout=timeout)
            return {
                "status": True,
                "status_code": response.status_code,
                "url": url,
                "content_type": response.headers.get("Content-Type", "")
            }
        except:
            return {
                "status": False
            }
    
    def check_hikvision(self, ip):
        """检查海康威视设备"""
        try:
            # 检查海康威视设备信息接口
            url = f"http://{ip}/ISAPI/System/deviceInfo"
            response = requests.get(url, timeout=2)
            return response.status_code in [200, 401]  # 200成功，401需要认证
        except:
            return False
    
    def scan_network(self, subnet="192.168.1."):
        """扫描网络中的摄像头"""
        print(f"\n=== 1. 网络扫描 ===")
        print(f"扫描网段: {subnet}0/24")
        print("扫描中，请稍候...")
        
        potential_cameras = []
        
        def scan_ip(ip):
            """扫描单个IP"""
            result = {
                "ip": ip,
                "ports": [],
                "is_http": False,
                "is_hikvision": False,
                "is_router": False,
                "status": "unknown"
            }
            
            # 检查常用端口
            ports_to_check = [554, 80, 8080, 8000, 37777]
            for port in ports_to_check:
                if self.check_port(ip, port):
                    result["ports"].append(port)
            
            # 检查HTTP服务
            http_result = self.check_http(ip)
            if http_result["status"]:
                result["is_http"] = True
                
                # 检测是否是路由器
                if "text/html" in http_result["content_type"]:
                    try:
                        response = requests.get(http_result["url"], timeout=2)
                        router_keywords = ["路由器", "router", "login", "登录", "admin"]
                        for keyword in router_keywords:
                            if keyword.lower() in response.text.lower():
                                result["is_router"] = True
                                result["status"] = "router"
                                break
                    except:
                        pass
            
            # 检查海康威视设备
            if self.check_hikvision(ip):
                result["is_hikvision"] = True
                result["status"] = "hikvision_camera"
            
            # 如果有任何匹配，添加到结果列表
            if result["ports"] or result["is_http"] or result["is_hikvision"]:
                potential_cameras.append(result)
        
        # 使用多线程加速扫描
        ip_list = [f"{subnet}{i}" for i in range(1, 255)]
        with ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(scan_ip, ip_list)
        
        # 筛选出可能的摄像头
        camera_candidates = [cam for cam in potential_cameras if not cam["is_router"]]
        
        print(f"扫描完成！")
        print(f"发现 {len(potential_cameras)} 个设备")
        print(f"筛选出 {len(camera_candidates)} 个潜在摄像头")
        
        if potential_cameras:
            print(f"\n详细结果:")
            print("-" * 70)
            for idx, device in enumerate(potential_cameras, 1):
                device_type = "路由器" if device["is_router"] else "海康威视摄像头" if device["is_hikvision"] else "未知设备"
                print(f"{idx:2d}. IP: {device['ip']}")
                print(f"    端口: {', '.join(map(str, device['ports'])) if device['ports'] else '无'}")
                print(f"    类型: {device_type}")
                print("-" * 70)
        
        self.results["ip_detection"] = camera_candidates
        return camera_candidates
    
    def test_camera_connection(self, camera_ips, username="admin", password="admin"):
        """测试摄像头连接"""
        print(f"\n=== 2. 摄像头连接测试 ===")
        print(f"测试 {len(camera_ips)} 个摄像头")
        
        connected_cameras = []
        
        for ip in camera_ips:
            print(f"\n测试设备: {ip}")
            
            try:
                # 创建PTZ控制器
                ptz_controller = PTZCameraController(
                    protocol=PTZProtocol.HTTP_API,
                    connection_type="http",
                    base_url=f"http://{ip}",
                    username=username,
                    password=password
                )
                
                # 尝试连接
                result = asyncio.run(ptz_controller.connect())
                if result["success"]:
                    print(f"✅ 连接成功: {result['message']}")
                    connected_cameras.append({
                        "ip": ip,
                        "username": username,
                        "password": password,
                        "controller": ptz_controller,
                        "status": "connected"
                    })
                else:
                    print(f"❌ 连接失败: {result['message']}")
            except Exception as e:
                print(f"❌ 连接异常: {e}")
        
        print(f"\n连接测试完成！")
        print(f"成功连接 {len(connected_cameras)} 个摄像头")
        
        self.results["camera_connection"] = connected_cameras
        return connected_cameras
    
    def test_ptz_functionality(self, connected_cameras):
        """测试PTZ功能"""
        print(f"\n=== 3. PTZ功能测试 ===")
        print(f"测试 {len(connected_cameras)} 个摄像头的PTZ功能")
        
        functional_cameras = []
        
        for camera in connected_cameras:
            print(f"\n测试摄像头: {camera['ip']}")
            ptz_controller = camera["controller"]
            
            # 测试PTZ动作
            actions_to_test = [
                (PTZAction.PAN_LEFT, "向左旋转"),
                (PTZAction.PAN_RIGHT, "向右旋转"),
                (PTZAction.TILT_UP, "向上倾斜"),
                (PTZAction.TILT_DOWN, "向下倾斜"),
                (PTZAction.STOP, "停止")
            ]
            
            camera_result = {
                "ip": camera["ip"],
                "actions": [],
                "status": "functional"
            }
            
            for action, action_name in actions_to_test:
                try:
                    result = asyncio.run(ptz_controller.execute_action(action, speed=50))
                    if result["success"]:
                        print(f"✅ {action_name}: 成功")
                        camera_result["actions"].append({
                            "action": action_name,
                            "success": True
                        })
                    else:
                        print(f"❌ {action_name}: 失败 - {result['message']}")
                        camera_result["actions"].append({
                            "action": action_name,
                            "success": False
                        })
                        camera_result["status"] = "partial_functional"
                
                    # 等待动作执行
                    time.sleep(0.5)
                except Exception as e:
                    print(f"❌ {action_name}: 异常 - {e}")
                    camera_result["actions"].append({
                        "action": action_name,
                        "success": False
                    })
                    camera_result["status"] = "non_functional"
            
            functional_cameras.append(camera_result)
        
        print(f"\nPTZ功能测试完成！")
        functional_count = len([cam for cam in functional_cameras if cam["status"] == "functional"])
        partial_count = len([cam for cam in functional_cameras if cam["status"] == "partial_functional"])
        non_count = len([cam for cam in functional_cameras if cam["status"] == "non_functional"])
        
        print(f"完全功能: {functional_count} 个")
        print(f"部分功能: {partial_count} 个")
        print(f"无功能: {non_count} 个")
        
        self.results["ptz_functionality"] = functional_cameras
        return functional_cameras
    
    def visual_verification(self, connected_cameras):
        """视觉验证"""
        print(f"\n=== 4. 视觉验证 ===")
        
        # 打开监控摄像头
        print(f"1. 打开监控摄像头...")
        result = self.monitor_camera.open_camera(0)
        if not result["success"]:
            print(f"❌ 无法打开监控摄像头: {result['message']}")
            return []
        print(f"✅ 监控摄像头打开成功")
        
        verification_results = []
        
        for camera in connected_cameras:
            print(f"\n测试摄像头: {camera['ip']}")
            ptz_controller = camera["controller"]
            
            # 拍摄初始照片
            print(f"2. 拍摄初始位置照片...")
            time.sleep(1)
            before_frame = self.monitor_camera.take_photo()
            if before_frame is None:
                print(f"❌ 无法拍摄初始照片")
                continue
            
            # 执行PTZ动作
            print(f"3. 执行PTZ动作...")
            asyncio.run(ptz_controller.execute_action(PTZAction.PAN_RIGHT, speed=80))
            time.sleep(2)  # 等待动作完成
            asyncio.run(ptz_controller.execute_action(PTZAction.STOP, 0))
            time.sleep(1)  # 稳定画面
            
            # 拍摄动作后照片
            print(f"4. 拍摄动作后照片...")
            after_frame = self.monitor_camera.take_photo()
            if after_frame is None:
                print(f"❌ 无法拍摄动作后照片")
                continue
            
            # 计算差异
            print(f"5. 分析视觉差异...")
            # 确保两张图像大小相同
            if before_frame.shape != after_frame.shape:
                after_frame = cv2.resize(after_frame, (before_frame.shape[1], before_frame.shape[0]))
            
            # 转换为灰度图
            gray_before = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)
            
            # 计算差异
            diff = cv2.absdiff(gray_before, gray_after)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # 计算差异百分比
            total_pixels = diff.size
            diff_pixels = cv2.countNonZero(thresh)
            diff_percentage = (diff_pixels / total_pixels) * 100
            
            # 保存对比照片
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            before_filename = f"auto_verification_before_{camera['ip']}_{timestamp}.jpg"
            after_filename = f"auto_verification_after_{camera['ip']}_{timestamp}.jpg"
            cv2.imwrite(before_filename, before_frame)
            cv2.imwrite(after_filename, after_frame)
            
            print(f"✅ 视觉验证完成")
            print(f"   差异百分比: {diff_percentage:.2f}%")
            print(f"   差异像素数: {diff_pixels}/{total_pixels}")
            
            verification_results.append({
                "ip": camera["ip"],
                "diff_percentage": diff_percentage,
                "before_filename": before_filename,
                "after_filename": after_filename,
                "status": "verified" if diff_percentage > 5 else "no_movement"
            })
        
        # 关闭监控摄像头
        self.monitor_camera.close_camera()
        print(f"\n视觉验证完成！")
        
        self.results["visual_verification"] = verification_results
        return verification_results
    
    def generate_report(self):
        """生成检测报告"""
        print(f"\n=== 5. 生成检测报告 ===")
        
        report_filename = f"ptz_automated_detection_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("PTZ摄像头自动化检测报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测类型: 全面自动化检测\n\n")
            
            # 网络扫描结果
            f.write("1. 网络扫描结果\n")
            f.write("-" * 40 + "\n")
            f.write(f"发现设备数: {len(self.results['ip_detection'])}\n")
            for device in self.results['ip_detection']:
                f.write(f"IP: {device['ip']}, 端口: {', '.join(map(str, device['ports']))}, 类型: {'海康威视摄像头' if device['is_hikvision'] else '未知'}\n")
            f.write("\n")
            
            # 连接测试结果
            f.write("2. 连接测试结果\n")
            f.write("-" * 40 + "\n")
            f.write(f"测试设备数: {len(self.results['camera_connection'])}\n")
            for device in self.results['camera_connection']:
                f.write(f"IP: {device['ip']}, 状态: {'成功' if device['status'] == 'connected' else '失败'}\n")
            f.write("\n")
            
            # PTZ功能测试结果
            f.write("3. PTZ功能测试结果\n")
            f.write("-" * 40 + "\n")
            for device in self.results['ptz_functionality']:
                f.write(f"IP: {device['ip']}, 状态: {device['status']}\n")
                for action in device['actions']:
                    f.write(f"  {action['action']}: {'成功' if action['success'] else '失败'}\n")
            f.write("\n")
            
            # 视觉验证结果
            f.write("4. 视觉验证结果\n")
            f.write("-" * 40 + "\n")
            for result in self.results['visual_verification']:
                f.write(f"IP: {result['ip']}, 差异百分比: {result['diff_percentage']:.2f}%, 状态: {'有移动' if result['status'] == 'verified' else '无移动'}\n")
                f.write(f"  初始照片: {result['before_filename']}\n")
                f.write(f"  动作后照片: {result['after_filename']}\n")
            f.write("\n")
            
            # 总结
            f.write("5. 检测总结\n")
            f.write("-" * 40 + "\n")
            f.write(f"总检测设备数: {len(self.results['ip_detection'])}\n")
            f.write(f"成功连接设备数: {len(self.results['camera_connection'])}\n")
            f.write(f"功能正常设备数: {len([d for d in self.results['ptz_functionality'] if d['status'] == 'functional'])}\n")
            f.write(f"视觉验证通过设备数: {len([r for r in self.results['visual_verification'] if r['status'] == 'verified'])}\n")
            f.write("\n")
            
            # 建议
            f.write("6. 建议\n")
            f.write("-" * 40 + "\n")
            if len(self.results['visual_verification']) > 0:
                for result in self.results['visual_verification']:
                    if result['status'] == 'verified':
                        f.write(f"✅ 设备 {result['ip']} 功能正常，可以正常使用\n")
                    else:
                        f.write(f"⚠️  设备 {result['ip']} 视觉验证未通过，建议检查PTZ机械结构或控制设置\n")
            else:
                f.write("⚠️  未发现可用的PTZ摄像头，建议检查网络连接或设备配置\n")
        
        print(f"✅ 检测报告已生成: {report_filename}")
        print(f"\n=== 检测报告摘要 ===")
        print(f"总检测设备数: {len(self.results['ip_detection'])}")
        print(f"成功连接设备数: {len(self.results['camera_connection'])}")
        print(f"功能正常设备数: {len([d for d in self.results['ptz_functionality'] if d['status'] == 'functional'])}")
        print(f"视觉验证通过设备数: {len([r for r in self.results['visual_verification'] if r['status'] == 'verified'])}")
        print(f"\n完整报告已保存为: {report_filename}")
    
    def run(self):
        """运行完整检测流程"""
        print(f"\n" + "=" * 60)
        print("开始PTZ摄像头自动化检测流程")
        print("=" * 60)
        
        try:
            # 1. 网络扫描
            camera_candidates = self.scan_network()
            
            if not camera_candidates:
                print(f"\n❌ 未发现任何潜在摄像头")
                self.generate_report()
                return
            
            # 2. 连接测试
            camera_ips = [cam['ip'] for cam in camera_candidates]
            connected_cameras = self.test_camera_connection(camera_ips)
            
            if not connected_cameras:
                print(f"\n❌ 无法连接到任何摄像头")
                self.generate_report()
                return
            
            # 3. PTZ功能测试
            functional_cameras = self.test_ptz_functionality(connected_cameras)
            
            # 4. 视觉验证
            verification_results = self.visual_verification(connected_cameras)
            
            # 5. 生成报告
            self.generate_report()
            
            print(f"\n🎉 PTZ摄像头自动化检测完成！")
            print(f"\n📋 最终建议:")
            print("1. 查看生成的检测报告，了解详细结果")
            print("2. 对功能正常的摄像头，可以开始使用")
            print("3. 对异常摄像头，根据报告中的建议进行排查")
            print("4. 定期运行本脚本，监控摄像头状态")
            
        except KeyboardInterrupt:
            print(f"\n\n🔴 检测被用户中断")
            self.generate_report()
        except Exception as e:
            print(f"\n\n❌ 检测过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.generate_report()
        finally:
            # 清理资源
            if hasattr(self, 'monitor_camera'):
                self.monitor_camera.close_camera()

if __name__ == "__main__":
    # 创建自动化检测系统
    detector = AutomatedPTZDetection()
    
    # 运行检测流程
    detector.run()