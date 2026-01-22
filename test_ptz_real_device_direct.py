#!/usr/bin/env python3
"""
直接测试真实PTZ设备
"""

import asyncio
import logging
import aiohttp

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_ptz_device():
    """
    直接测试真实PTZ设备
    """
    print("=== 直接测试真实PTZ设备 ===")
    
    # 替换为实际的PTZ设备信息
    ptz_config = {
        "base_url": "http://localhost:8001",  # 实际PTZ设备的IP地址
        "username": "admin",  # 实际PTZ设备的用户名
        "password": "admin"   # 实际PTZ设备的密码
    }
    
    print(f"\n1. 配置信息: {ptz_config}")
    
    # 测试HTTP连接
    print("\n2. 测试HTTP连接...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ptz_config['base_url']}",
                timeout=5
            ) as resp:
                print(f"   状态码: {resp.status}")
                print(f"   响应: {await resp.text()}")
                print("   ✅ HTTP连接成功")
    except Exception as e:
        print(f"   ❌ HTTP连接失败: {e}")
        return
    
    # 测试PTZ控制命令
    print("\n3. 测试PTZ控制命令...")
    
    test_actions = [
        {"action": "pan_left", "name": "向左转"},
        {"action": "pan_right", "name": "向右转"},
        {"action": "tilt_up", "name": "向上转"},
        {"action": "tilt_down", "name": "向下转"}
    ]
    
    for test_action in test_actions:
        action = test_action["action"]
        name = test_action["name"]
        
        print(f"   测试 {name}...")
        try:
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(ptz_config["username"], ptz_config["password"])
                async with session.post(
                    f"{ptz_config['base_url']}/api/ptz/control",  # 替换为实际PTZ设备的API路径
                    json={
                        "action": action,
                        "speed": 50
                    },
                    auth=auth,
                    timeout=5
                ) as resp:
                    print(f"     状态码: {resp.status}")
                    print(f"     响应: {await resp.text()}")
                    print(f"     ✅ {name}命令发送成功")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"     ❌ {name}命令发送失败: {e}")
            
    # 发送停止命令
    print("\n4. 发送停止命令...")
    try:
        async with aiohttp.ClientSession() as session:
            auth = aiohttp.BasicAuth(ptz_config["username"], ptz_config["password"])
            async with session.post(
                f"{ptz_config['base_url']}/api/ptz/control",
                json={
                    "action": "stop",
                    "speed": 0
                },
                auth=auth,
                timeout=5
            ) as resp:
                print(f"   状态码: {resp.status}")
                print(f"   响应: {await resp.text()}")
                print("   ✅ 停止命令发送成功")
    except Exception as e:
        print(f"   ❌ 停止命令发送失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_real_ptz_device())
