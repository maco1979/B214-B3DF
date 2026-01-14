#!/usr/bin/env python3
"""
简单的PTZ测试脚本
用于测试PTZ控制器的基本功能
"""

import asyncio
import logging
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_ptz_simple():
    """测试PTZ简单功能"""
    print("=== 简单PTZ测试 ===")
    
    # 创建PTZ控制器实例
    ptz = PTZCameraController(
        protocol=PTZProtocol.HTTP_API,
        connection_type='http',
        base_url='http://localhost:8001',
        username='admin',
        password='admin'
    )
    
    print(f"PTZ控制器创建成功: {ptz}")
    print(f"初始连接状态: {ptz.is_connected}")
    
    # 测试连接
    try:
        print("\n1. 测试连接...")
        result = await ptz.connect()
        print(f"连接结果: {result}")
        print(f"连接状态: {ptz.is_connected}")
        
        if ptz.is_connected:
            # 测试获取状态
            print("\n2. 测试获取状态...")
            status = ptz.get_status()
            print(f"设备状态: {status}")
            
            # 测试执行动作
            print("\n3. 测试执行右转动作...")
            result = await ptz.execute_action(PTZAction.PAN_RIGHT, speed=50)
            print(f"右转结果: {result}")
            
            # 测试停止动作
            print("\n4. 测试执行停止动作...")
            result = await ptz.execute_action(PTZAction.STOP, speed=0)
            print(f"停止结果: {result}")
            
            # 测试断开连接
            print("\n5. 测试断开连接...")
            result = await ptz.disconnect()
            print(f"断开结果: {result}")
            print(f"断开后连接状态: {ptz.is_connected}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_ptz_simple())