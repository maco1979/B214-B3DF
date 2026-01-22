#!/usr/bin/env python3
"""
测试真实的转动云台功能
"""

import asyncio
from backend.src.core.services.ptz_camera_controller import (
    PTZCameraController, PTZProtocol, PTZAction, get_ptz_controller
)

async def test_ptz_rotation():
    """
    测试PTZ云台转动功能
    """
    print("=== 测试真实的转动云台功能 ===")
    
    # 1. 获取或创建PTZ控制器实例
    print("\n1. 创建PTZ控制器实例...")
    ptz_controller = get_ptz_controller(
        protocol=PTZProtocol.HTTP_API,  # 使用HTTP API协议
        connection_type="http",
        base_url="http://localhost:8001",  # 使用本地服务器作为代理
        username="admin",
        password="admin"
    )
    print(f"   PTZ控制器已创建: {ptz_controller.protocol.value}")
    
    # 2. 连接到云台
    print("\n2. 连接到云台...")
    connect_result = await ptz_controller.connect()
    print(f"   连接结果: {connect_result}")
    if not connect_result["success"]:
        print("   ❌ 连接失败，退出测试")
        return
    print("   ✅ 连接成功")
    
    # 3. 测试基本转动功能
    print("\n3. 测试基本转动功能...")
    
    # 测试向左转动
    print("   测试向左转动...")
    result = await ptz_controller.execute_action(
        action=PTZAction.PAN_LEFT,
        speed=30
    )
    print(f"     结果: {result}")
    await asyncio.sleep(1)  # 等待1秒
    
    # 停止
    await ptz_controller.execute_action(PTZAction.STOP, 0)
    await asyncio.sleep(0.5)
    
    # 测试向右转动
    print("   测试向右转动...")
    result = await ptz_controller.execute_action(
        action=PTZAction.PAN_RIGHT,
        speed=30
    )
    print(f"     结果: {result}")
    await asyncio.sleep(1)
    
    # 停止
    await ptz_controller.execute_action(PTZAction.STOP, 0)
    await asyncio.sleep(0.5)
    
    # 测试向上转动
    print("   测试向上转动...")
    result = await ptz_controller.execute_action(
        action=PTZAction.TILT_UP,
        speed=30
    )
    print(f"     结果: {result}")
    await asyncio.sleep(1)
    
    # 停止
    await ptz_controller.execute_action(PTZAction.STOP, 0)
    await asyncio.sleep(0.5)
    
    # 测试向下转动
    print("   测试向下转动...")
    result = await ptz_controller.execute_action(
        action=PTZAction.TILT_DOWN,
        speed=30
    )
    print(f"     结果: {result}")
    await asyncio.sleep(1)
    
    # 停止
    await ptz_controller.execute_action(PTZAction.STOP, 0)
    await asyncio.sleep(0.5)
    
    # 4. 测试变焦功能
    print("\n4. 测试变焦功能...")
    
    # 测试拉近
    print("   测试拉近...")
    result = await ptz_controller.execute_action(
        action=PTZAction.ZOOM_IN,
        speed=20
    )
    print(f"     结果: {result}")
    await asyncio.sleep(1)
    
    # 停止
    await ptz_controller.execute_action(PTZAction.STOP, 0)
    await asyncio.sleep(0.5)
    
    # 测试拉远
    print("   测试拉远...")
    result = await ptz_controller.execute_action(
        action=PTZAction.ZOOM_OUT,
        speed=20
    )
    print(f"     结果: {result}")
    await asyncio.sleep(1)
    
    # 停止
    await ptz_controller.execute_action(PTZAction.STOP, 0)
    await asyncio.sleep(0.5)
    
    # 5. 测试移动到指定位置
    print("\n5. 测试移动到指定位置...")
    result = await ptz_controller.move_to_position(
        pan=0.0,  # 回到中心位置
        tilt=0.0,
        speed=20
    )
    print(f"   结果: {result}")
    
    # 6. 获取当前状态
    print("\n6. 获取当前状态...")
    status = ptz_controller.get_status()
    print(f"   状态: {status}")
    
    # 7. 断开连接
    print("\n7. 断开连接...")
    disconnect_result = await ptz_controller.disconnect()
    print(f"   断开结果: {disconnect_result}")
    
    print("\n=== 测试完成 ===")
    print("✅ 云台转动测试已完成")

if __name__ == "__main__":
    asyncio.run(test_ptz_rotation())
