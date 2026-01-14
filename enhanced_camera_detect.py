#!/usr/bin/env python3
"""增强版真实摄像头检测脚本"""

import requests
import time
import re
import socket
from concurrent.futures import ThreadPoolExecutor

def check_port(ip, port, timeout=0.5):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def check_http(ip, port=80, timeout=2):
    """检查HTTP服务"""
    try:
        url = f"http://{ip}:{port}"
        response = requests.get(url, timeout=timeout)
        return {
            "status": True,
            "status_code": response.status_code,
            "url": url
        }
    except:
        return {
            "status": False
        }

def check_hikvision(ip):
    """检查海康威视设备"""
    try:
        # 检查海康威视设备信息接口
        url = f"http://{ip}/ISAPI/System/deviceInfo"
        response = requests.get(url, timeout=2)
        return response.status_code in [200, 401]  # 200成功，401需要认证
    except:
        return False

def detect_enhanced_camera():
    """增强版摄像头检测"""
    print("=== 增强版真实摄像头检测 ===\n")
    
    # 1. 检测192.168.1.1的真实身份
    print("1. 确认192.168.1.1的真实身份...")
    try:
        response = requests.get("http://192.168.1.1", timeout=3)
        print(f"   状态码: {response.status_code}")
        
        # 检测是否是路由器
        if "<title>" in response.text:
            title = re.search(r"<title>(.*?)</title>", response.text, re.IGNORECASE)
            if title:
                print(f"   页面标题: {title.group(1)}")
            
            # 检测是否包含路由器特征词
            router_keywords = ["路由器", "router", "login", "登录", "admin"]
            for keyword in router_keywords:
                if keyword.lower() in response.text.lower():
                    print(f"   ✅ 包含关键词: {keyword}")
                    print(f"   🎯 结论: 这是**路由器**，不是摄像头！")
                    break
    except Exception as e:
        print(f"   ❌ 检测失败: {e}")
    
    # 2. 扫描整个网段的IP
    print("\n2. 扫描192.168.1.0/24网段的摄像头IP...")
    print("   扫描中，请稍候...")
    
    potential_cameras = []
    
    def scan_ip(ip):
        """扫描单个IP"""
        result = {
            "ip": ip,
            "ports": [],
            "is_hikvision": False,
            "http_status": None
        }
        
        # 检查常用端口
        ports_to_check = [554, 80, 8080, 8000]
        for port in ports_to_check:
            if check_port(ip, port):
                result["ports"].append(port)
        
        # 检查HTTP服务
        http_result = check_http(ip)
        if http_result["status"]:
            result["http_status"] = http_result["status_code"]
        
        # 检查海康威视设备
        if check_hikvision(ip):
            result["is_hikvision"] = True
        
        # 如果有任何匹配，添加到结果列表
        if result["ports"] or result["http_status"] or result["is_hikvision"]:
            potential_cameras.append(result)
    
    # 使用多线程加速扫描
    ip_list = [f"192.168.1.{i}" for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(scan_ip, ip_list)
    
    # 3. 显示扫描结果
    print(f"\n3. 扫描完成！发现 {len(potential_cameras)} 个潜在设备：")
    print("   " + "-" * 60)
    
    if potential_cameras:
        for idx, camera in enumerate(potential_cameras, 1):
            print(f"   {idx}. IP: {camera['ip']}")
            print(f"      开放端口: {', '.join(map(str, camera['ports'])) if camera['ports'] else '无'}")
            if camera['http_status']:
                print(f"      HTTP状态: {camera['http_status']}")
            if camera['is_hikvision']:
                print(f"      设备类型: 海康威视设备")
            print(f"      访问地址: http://{camera['ip']}")
            print(f"   " + "-" * 60)
    else:
        print("   未发现任何潜在摄像头设备")
    
    # 4. 海康威视设备详细检测
    print("\n4. 海康威视设备详细检测...")
    hikvision_ips = [cam['ip'] for cam in potential_cameras if cam['is_hikvision']]
    
    if hikvision_ips:
        for ip in hikvision_ips:
            print(f"   检测海康威视设备: {ip}")
            # 尝试访问海康威视登录页面
            try:
                response = requests.get(f"http://{ip}", timeout=3)
                if response.status_code == 200:
                    print(f"      ✅ 可以访问登录页面")
                    print(f"      📌 建议: 使用此IP作为PTZ摄像头IP")
                elif response.status_code == 401:
                    print(f"      ✅ 设备需要认证，确认为海康威视设备")
                    print(f"      📌 建议: 使用此IP作为PTZ摄像头IP")
            except Exception as e:
                print(f"      ❌ 检测失败: {e}")
    else:
        print("   未检测到海康威视设备")
    
    # 5. 手动配置指南
    print("\n5. 手动配置指南:")
    print("   📌 请按照以下步骤操作：")
    print("   1. 查看摄像头标签，获取默认IP和登录信息")
    print("   2. 使用海康威视SADP工具扫描网络")
    print("   3. 登录路由器，查看已连接设备列表")
    print("   4. 将真实IP填入PTZ配置中")
    print("   \n   🔧 配置示例:")
    print("   - 如果发现设备IP: 192.168.1.X")
    print("   - 请修改 ptz_visual_comparison.py 中的 CAMERA_IP 变量")

if __name__ == "__main__":
    detect_enhanced_camera()