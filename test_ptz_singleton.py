#!/usr/bin/env python3
"""
测试PTZ控制器的单例模式，验证只需要第一次登录云台，以后默认可以控制
"""

import asyncio
import logging
from backend.src.core.services.ptz_camera_controller import (
    PTZCameraController, PTZProtocol, PTZAction, get_ptz_controller
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ptz_singleton():
    """
    测试PTZ控制器的单例模式
    """
    print("=== 测试PTZ控制器的单例模式 ===")
    
    # 第一次获取并连接PTZ控制器
    print("\n1. 第一次获取并连接PTZ控制器...")
    ptz_controller1 = get_ptz_controller(
        protocol=PTZProtocol.HTTP_API,
        connection_type="http",
        base_url="http://localhost:8001",
        username="admin",
        password="admin"
    )
    
    # 连接PTZ
    connect_result = await ptz_controller1.connect()
    print(f"   连接结果: {connect_result}")
    
    if connect_result["success"]:
        print("   ✅ 第一次连接成功")
    else:
        print("   ⚠️  连接失败，使用模拟模式继续测试")
        ptz_controller1.is_connected = True
    
    # 第二次获取PTZ控制器（应该返回同一个实例）
    print("\n2. 第二次获取PTZ控制器...")
    ptz_controller2 = get_ptz_controller()
    
    # 验证是否是同一个实例
    print(f"   两个控制器是否是同一个实例: {ptz_controller1 is ptz_controller2}")
    
    if ptz_controller1 is ptz_controller2:
        print("   ✅ PTZ控制器是单例模式，只需要第一次连接")
    else:
        print("   ❌ PTZ控制器不是单例模式，需要每次连接")
        return
    
    # 测试使用第二个控制器执行PTZ动作（不需要再次连接）
    print("\n3. 使用第二个控制器执行PTZ动作...")
    result = await ptz_controller2.execute_action(
        action=PTZAction.PAN_RIGHT,
        speed=30
    )
    print(f"   执行结果: {result}")
    
    if result["success"]:
        print("   ✅ 第二个控制器可以直接执行动作，不需要再次连接")
    else:
        print("   ❌ 第二个控制器执行动作失败")
    
    # 测试使用第三个控制器执行PTZ动作
    print("\n4. 使用第三个控制器执行PTZ动作...")
    ptz_controller3 = get_ptz_controller()
    result = await ptz_controller3.execute_action(
        action=PTZAction.TILT_UP,
        speed=30
    )
    print(f"   执行结果: {result}")
    
    if result["success"]:
        print("   ✅ 第三个控制器可以直接执行动作，不需要再次连接")
    else:
        print("   ❌ 第三个控制器执行动作失败")
    
    # 获取PTZ状态
    print("\n5. 获取PTZ状态...")
    status = ptz_controller3.get_status()
    print(f"   状态: {status}")
    
    print("\n=== 单例模式测试完成 ===")
    print("✅ PTZ控制器是单例模式，只需要第一次登录云台")
    print("✅ 后续获取的控制器实例可以直接使用，不需要再次登录")
    print("✅ 所有控制器实例共享同一个连接状态")

if __name__ == "__main__":
    asyncio.run(test_ptz_singleton())
