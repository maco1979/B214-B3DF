#!/usr/bin/env python3
"""ONVIF协议测试脚本"""

from onvif import ONVIFCamera
import time

def test_onvif_control():
    """测试ONVIF协议控制"""
    print("=== ONVIF协议测试 ===")
    
    try:
        # 初始化ONVIF连接
        print("1. 初始化ONVIF连接...")
        cam = ONVIFCamera("192.168.1.1", 80, "admin", "admin")
        cam.update_xaddrs()
        
        # 获取服务
        ptz = cam.create_ptz_service()
        media = cam.create_media_service()
        
        print("2. 获取摄像头配置...")
        profiles = media.GetProfiles()
        profile_token = profiles[0].token
        print(f"   使用配置文件: {profile_token}")
        
        # 获取PTZ状态
        print("3. 获取PTZ状态...")
        status = ptz.GetStatus({'ProfileToken': profile_token})
        print(f"   当前状态: {status}")
        
        # 测试连续移动
        print("4. 测试向右转动...")
        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = profile_token
        request.Velocity = {
            'PanTilt': {'x': 0.1, 'y': 0},
            'Zoom': {'x': 0}
        }
        ptz.ContinuousMove(request)
        time.sleep(2)
        
        # 停止移动
        print("5. 停止移动...")
        ptz.Stop({'ProfileToken': profile_token})
        
        # 检查最终状态
        print("6. 检查最终状态...")
        final_status = ptz.GetStatus({'ProfileToken': profile_token})
        print(f"   最终状态: {final_status}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_onvif_control()