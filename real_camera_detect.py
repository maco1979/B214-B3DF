#!/usr/bin/env python3
"""真实摄像头检测脚本"""

import requests
import time

def detect_real_camera():
    """检测真实摄像头"""
    print("=== 真实摄像头检测 ===")
    
    # 扫描网络中的可能摄像头IP
    potential_ips = [
        "192.168.1.1",    # 路由器
        "192.168.1.33",   # 网络设备
        "192.168.1.64",   # 海康威视默认IP
        "192.168.1.100"   # 常见摄像头IP
    ]
    
    for ip in potential_ips:
        print(f"\n🔍 检测 IP: http://{ip}")
        
        # 检测ISAPI接口（海康威视特征）
        try:
            isapi_url = f"http://{ip}/ISAPI/System/deviceInfo"
            response = requests.get(isapi_url, auth=("admin", "admin"), timeout=3)
            print(f"   ISAPI状态: {response.status_code}")
            
            if response.status_code in [200, 401]:  # 200=成功，401=需要认证
                print(f"   ✅ 可能是海康威视摄像头！")
                print(f"   响应: {response.text[:100]}...")
            elif response.status_code == 302:
                print(f"   ⚠️  重定向（可能是路由器）")
            else:
                print(f"   ❌ 不是海康威视摄像头")
                print(f"   响应: {response.text[:50]}...")
                
        except requests.exceptions.ConnectTimeout:
            print(f"   ⏱️  连接超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ 连接失败")
        except Exception as e:
            print(f"   ⚠️  异常: {e}")

if __name__ == "__main__":
    detect_real_camera()