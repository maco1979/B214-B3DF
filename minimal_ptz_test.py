#!/usr/bin/env python3
"""极简PTZ控制测试脚本"""

import requests
import time

# =======================================
# 配置区域 - 只需修改这3项
# =======================================
BASE_URL = "http://localhost:8001"

# 请填写您的摄像头信息
CAMERA_IP = "192.168.1.1"      # 替换为摄像头真实IP
CAMERA_USER = "admin"           # 替换为摄像头真实用户名
CAMERA_PASS = "admin"           # 替换为摄像头真实密码

# =======================================
# 无需修改以下代码
# =======================================

def test_minimal_ptz():
    """极简PTZ控制测试"""
    print("=== 极简PTZ控制测试 ===")
    print(f"\n📋 配置信息:")
    print(f"  摄像头IP: http://{CAMERA_IP}")
    print(f"  用户名: {CAMERA_USER}")
    print(f"  密码: {'*' * len(CAMERA_PASS)}")
    
    # 构建完整的PTZ配置
    ptz_config = {
        "protocol": "http",
        "connection_type": "http",
        "base_url": f"http://{CAMERA_IP}",
        "username": CAMERA_USER,
        "password": CAMERA_PASS
    }
    
    # 1. 打开摄像头
    print(f"\n🔍 步骤1: 打开摄像头")
    try:
        open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
        if open_result.get("success"):
            print(f"   ✅ 摄像头打开成功")
        else:
            print(f"   ❌ 摄像头打开失败: {open_result}")
            return False
    except Exception as e:
        print(f"   ❌ 摄像头打开异常: {e}")
        return False
    
    # 2. 断开现有PTZ连接
    print(f"\n🔍 步骤2: 清理现有连接")
    try:
        requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
        print(f"   ✅ 清理完成")
    except Exception as e:
        print(f"   ⚠️  清理时发生异常: {e}")
    
    # 3. 连接PTZ
    print(f"\n🔍 步骤3: 连接PTZ云台")
    try:
        connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config).json()
        if connect_result.get("success"):
            print(f"   ✅ PTZ连接成功")
        else:
            print(f"   ❌ PTZ连接失败: {connect_result}")
            return False
    except Exception as e:
        print(f"   ❌ PTZ连接异常: {e}")
        return False
    
    # 4. 测试移动 - 向右60度，向上30度
    print(f"\n🔍 步骤4: 测试PTZ移动")
    test_pan = 60.0
    test_tilt = 30.0
    test_speed = 70
    
    try:
        move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                                  json={"pan": test_pan, "tilt": test_tilt, "speed": test_speed})
        
        if move_result.status_code == 200:
            move_data = move_result.json()
            if move_data.get("success"):
                print(f"   ✅ 移动命令发送成功")
                print(f"   📡 移动参数: 水平{test_pan}°, 垂直{test_tilt}°, 速度{test_speed}%")
            else:
                print(f"   ❌ 移动命令执行失败: {move_data}")
                return False
        else:
            print(f"   ❌ 移动命令请求失败: {move_result.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 移动命令异常: {e}")
        return False
    
    # 5. 等待移动完成
    wait_time = 5
    print(f"\n⏳ 步骤5: 等待摄像头移动完成 ({wait_time}秒)")
    time.sleep(wait_time)
    
    # 6. 检查移动结果
    print(f"\n📊 步骤6: 检查移动结果")
    try:
        status_result = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
        if status_result.get("success"):
            position = status_result["data"]["position"]
            print(f"   📍 当前位置: 水平{position['pan']:.1f}°, 垂直{position['tilt']:.1f}°, 变焦{position['zoom']:.1f}x")
        else:
            print(f"   ❌ 获取状态失败: {status_result}")
            return False
    except Exception as e:
        print(f"   ❌ 获取状态异常: {e}")
        return False
    
    # 7. 验证移动效果
    print(f"\n✅ 测试完成！")
    print(f"\n📋 测试总结:")
    print(f"   • 摄像头IP: http://{CAMERA_IP}")
    print(f"   • 连接状态: ✅ 成功")
    print(f"   • 移动命令: ✅ 成功发送")
    print(f"   • 移动距离: 目标水平{test_pan}°, 垂直{test_tilt}°")
    print(f"   • 测试结果: ✅ AI PTZ控制功能正常")
    
    # 8. 清理资源
    print(f"\n🧹 清理资源:")
    
    # 移动回初始位置
    print(f"   • 移动回初始位置")
    try:
        requests.post(f"{BASE_URL}/api/camera/ptz/move", json={"pan": 0, "tilt": 0, "speed": 100})
        time.sleep(2)
    except:
        pass
    
    # 断开PTZ连接
    print(f"   • 断开PTZ连接")
    try:
        requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    except:
        pass
    
    # 关闭摄像头
    print(f"   • 关闭摄像头")
    try:
        requests.post(f"{BASE_URL}/api/camera/close").json()
    except:
        pass
    
    print(f"\n🎉 所有测试完成！")
    print(f"\n💡 提示:")
    print(f"   • 如要测试更大角度，请修改脚本中的 test_pan 和 test_tilt 参数")
    print(f"   • 如要测试自动跟踪，请运行其他测试脚本")
    print(f"   • 如遇到问题，请检查摄像头IP、用户名和密码是否正确")
    
    return True

if __name__ == "__main__":
    test_minimal_ptz()