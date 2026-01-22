#!/usr/bin/env python3
"""
手动配置并测试PTZ硬件控制
"""

import asyncio
import logging
from backend.src.core.services.ptz_camera_controller import (
    PTZCameraController, PTZProtocol, PTZAction, get_ptz_controller
)

# 配置日志
logging.basicConfig(level=logging.DEBUG)  # 使用DEBUG级别，查看详细日志
logger = logging.getLogger(__name__)

async def test_ptz_hardware_manual():
    """
    手动配置并测试PTZ硬件控制
    """
    print("=== 手动配置并测试PTZ硬件控制 ===")
    
    print("请根据您的实际PTZ硬件情况选择以下配置：")
    print("1. 协议选择：")
    print("   a. Pelco-D（最常用）")
    print("   b. Pelco-P")
    print("   c. VISCA")
    print("   d. HTTP API")
    
    print("\n2. 连接类型：")
    print("   a. 串口连接（COMx或/dev/ttyUSBx）")
    print("   b. 网络连接（IP地址）")
    print("   c. HTTP连接")
    
    print("\n当前测试使用的默认配置：")
    print("- 协议：Pelco-D")
    print("- 连接类型：串口连接")
    print("- 串口：COM3")
    print("- 波特率：9600")
    print("- 设备地址：1")
    
    # 根据实际硬件情况修改以下参数
    ptz_config = {
        # 选择合适的协议
        # 常用协议：PTZProtocol.PELCO_D, PTZProtocol.VISCA, PTZProtocol.HTTP_API
        "protocol": PTZProtocol.PELCO_D,
        
        # 选择合适的连接类型
        # 常用连接类型：serial, network, http
        "connection_type": "serial",
        
        # 串口连接参数
        "port": "COM3",  # 串口端口，Windows系统通常是COMx，Linux系统通常是/dev/ttyUSBx
        "baudrate": 9600,  # 波特率，通常是9600
        "address": 1,  # 设备地址，通常是1
        
        # 网络连接参数（如果使用网络连接，取消下面的注释并修改）
        # "host": "192.168.1.100",  # 设备IP地址
        # "port": 5000,  # 设备端口
        # "address": 1,  # 设备地址
        
        # HTTP连接参数（如果使用HTTP连接，取消下面的注释并修改）
        # "base_url": "http://192.168.1.100",  # 设备HTTP地址
        # "username": "admin",  # 用户名
        # "password": "admin"  # 密码
    }
    
    print(f"\n当前配置: {ptz_config}")
    
    # 创建PTZ控制器
    ptz_controller = PTZCameraController(**ptz_config)
    
    # 连接PTZ硬件
    print("\n1. 连接PTZ硬件...")
    connect_result = await ptz_controller.connect()
    print(f"   连接结果: {connect_result}")
    
    if not connect_result["success"]:
        print("   ❌ PTZ硬件连接失败")
        print("   请检查以下几点：")
        print("   - 设备是否已通电")
        print("   - 连接线缆是否正确连接")
        print("   - 串口端口是否正确")
        print("   - 波特率、地址等参数是否与设备匹配")
        print("   - 设备是否支持所选协议")
        return
    
    print("   ✅ PTZ硬件连接成功")
    
    # 测试转动功能
    print("\n2. 测试PTZ转动功能...")
    print("   按顺序执行：向左转 → 向右转 → 向上转 → 向下转 → 停止")
    
    test_actions = [
        {"action": PTZAction.PAN_LEFT, "name": "向左转", "speed": 30, "duration": 2.0},
        {"action": PTZAction.PAN_RIGHT, "name": "向右转", "speed": 30, "duration": 2.0},
        {"action": PTZAction.TILT_UP, "name": "向上转", "speed": 30, "duration": 2.0},
        {"action": PTZAction.TILT_DOWN, "name": "向下转", "speed": 30, "duration": 2.0},
        {"action": PTZAction.STOP, "name": "停止", "speed": 0, "duration": 1.0},
    ]
    
    for test_action in test_actions:
        action = test_action["action"]
        name = test_action["name"]
        speed = test_action["speed"]
        duration = test_action["duration"]
        
        print(f"   执行 {name} (速度: {speed})...")
        result = await ptz_controller.execute_action(
            action=action,
            speed=speed
        )
        print(f"   执行结果: {result}")
        await asyncio.sleep(duration)
    
    # 获取PTZ状态
    print("\n3. 获取PTZ状态...")
    status = ptz_controller.get_status()
    print(f"   状态: {status}")
    
    # 断开连接
    print("\n4. 断开PTZ连接...")
    disconnect_result = await ptz_controller.disconnect()
    print(f"   断开结果: {disconnect_result}")
    
    print("\n=== PTZ硬件控制测试完成 ===")
    print("\n如果云台没有转动，请尝试以下解决方法：")
    print("1. 检查协议是否正确")
    print("2. 检查连接类型是否正确")
    print("3. 检查连接参数是否正确（串口、波特率、地址等）")
    print("4. 检查设备是否支持所选协议")
    print("5. 检查设备是否已通电并正常工作")
    print("6. 检查设备地址是否正确")
    print("7. 尝试调整速度参数")

if __name__ == "__main__":
    asyncio.run(test_ptz_hardware_manual())
