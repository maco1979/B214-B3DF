#!/usr/bin/env python3
"""简化版AI PTZ测试脚本"""

import asyncio
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol

async def simple_ai_ptz_test():
    """简化版AI PTZ测试"""
    print("=== 简化版AI PTZ控制测试 ===\n")
    
    # 用户配置区
    CAMERA_IP = "192.168.1.64"  # 请替换为真实摄像头IP
    USERNAME = "admin"         # 请替换为真实用户名
    PASSWORD = "admin"         # 请替换为真实密码
    
    # 创建PTZ控制器
    ptz = PTZCameraController(
        protocol=PTZProtocol.HTTP_API,
        connection_type="http",
        base_url=f"http://{CAMERA_IP}",
        username=USERNAME,
        password=PASSWORD
    )
    
    # 连接摄像头
    print(f"连接到摄像头: {CAMERA_IP}...")
    result = await ptz.connect()
    print(f"连接结果: {result['success']}\n")
    
    if not result["success"]:
        print("❌ 连接失败，请检查配置")
        return
    
    # 大角度移动测试
    print("执行大角度移动测试...")
    
    # 获取初始位置
    initial = ptz.get_status()
    print(f"初始位置: pan={initial['position']['pan']}, tilt={initial['position']['tilt']}")
    
    # 执行180°旋转测试
    print("\n1. 执行180°向右旋转...")
    await ptz.move_to_position(pan=initial['position']['pan'] + 180, tilt=initial['position']['tilt'], speed=100)
    after_pan = ptz.get_status()['position']['pan']
    print(f"旋转后位置: pan={after_pan}")
    
    # 执行90°向上倾斜测试
    print("\n2. 执行90°向上倾斜...")
    await ptz.move_to_position(pan=after_pan, tilt=initial['position']['tilt'] + 90, speed=100)
    after_tilt = ptz.get_status()['position']['tilt']
    print(f"倾斜后位置: tilt={after_tilt}")
    
    # 执行复位
    print("\n3. 复位到初始位置...")
    await ptz.move_to_position(pan=initial['position']['pan'], tilt=initial['position']['tilt'], speed=100)
    final = ptz.get_status()
    print(f"复位后位置: pan={final['position']['pan']}, tilt={final['position']['tilt']}")
    
    print("\n=== 测试完成 ===")
    print("🎉 AI PTZ控制功能已实现，可通过以下步骤进一步验证:")
    print("1. 确认摄像头物理位置是否随命令移动")
    print("2. 检查监控画面是否随PTZ动作变化")
    print("3. 调整CAMERA_IP、USERNAME、PASSWORD为真实值")
    print("4. 查看系统日志获取详细执行信息")

if __name__ == "__main__":
    asyncio.run(simple_ai_ptz_test())