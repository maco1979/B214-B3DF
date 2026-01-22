"""
测试控制机制改进功能
验证连接池管理、协议适配器和设备认证功能
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.core.services.connection_pool_manager import connection_pool_manager
from backend.src.core.services.protocol_adapter import protocol_adapter_manager, ProtocolType
from backend.src.core.services.device_auth_manager import device_auth_manager
from backend.src.core.services.connection_controller import InfraredController


async def test_connection_pool():
    """测试连接池管理功能"""
    print("\n=== 测试连接池管理功能 ===")
    
    # 初始化连接池
    await connection_pool_manager.initialize()
    print("✅ 连接池管理器初始化完成")
    
    # 创建一个控制器实例
    controller = InfraredController()
    connection_params = {"channel": 1, "range": 5}
    controller.connect(connection_params)
    
    # 注册连接
    result = await connection_pool_manager.register_connection("test_device_001", controller)
    print(f"注册连接结果: {result}")
    
    # 获取连接
    retrieved_controller = await connection_pool_manager.get_connection("test_device_001")
    print(f"获取连接结果: {retrieved_controller is not None}")
    
    # 获取统计信息
    stats = connection_pool_manager.get_pool_stats()
    print(f"连接池统计: {stats}")
    
    print("✅ 连接池管理功能测试完成")


async def test_protocol_adapter():
    """测试协议适配器功能"""
    print("\n=== 测试协议适配器功能 ===")
    
    # 测试WiFi适配器
    print("测试WiFi适配器...")
    wifi_params = {
        "ssid": "test_network",
        "password": "test_password",
        "ip_address": "192.168.1.100",
        "port": 8080
    }
    
    result = await protocol_adapter_manager.create_connection(ProtocolType.WIFI, wifi_params)
    print(f"Wifi连接结果: {result}")
    
    # 测试Zigbee适配器
    print("测试Zigbee适配器...")
    zigbee_params = {
        "network_id": "0x1234",
        "channel": 15,
        "pan_id": "0xFFFF"
    }
    
    result = await protocol_adapter_manager.create_connection(ProtocolType.ZIGBEE, zigbee_params)
    print(f"Zigbee连接结果: {result}")
    
    # 测试LoRa适配器
    print("测试LoRa适配器...")
    lora_params = {
        "frequency": 868.0,
        "spreading_factor": 7,
        "bandwidth": 125
    }
    
    result = await protocol_adapter_manager.create_connection(ProtocolType.LORA, lora_params)
    print(f"LoRa连接结果: {result}")
    
    print("✅ 协议适配器功能测试完成")


def test_device_auth():
    """测试设备认证功能"""
    print("\n=== 测试设备认证功能 ===")
    
    # 注册设备
    device_info = {
        "device_id": "sensor_001",
        "device_name": "温湿度传感器",
        "device_type": "sensor",
        "manufacturer": "TestCorp",
        "model": "TH-S100",
        "firmware_version": "1.2.0",
        "serial_number": "SN123456789"
    }
    
    result = device_auth_manager.register_device(device_info)
    print(f"设备注册结果: {result}")
    
    if result["success"]:
        reg_code = result["registration_code"]
        print(f"获取注册码: {reg_code}")
        
        # 使用注册码认证设备
        auth_result = device_auth_manager.authenticate_device(
            "sensor_001", 
            {"method": "registration_code", "code": reg_code}
        )
        print(f"设备认证结果: {auth_result}")
        
        # 验证令牌
        if "token" in auth_result:
            token = auth_result["token"]
            verify_result = device_auth_manager.verify_token(token)
            print(f"令牌验证结果: {verify_result}")
    
    print("✅ 设备认证功能测试完成")


async def main():
    """主测试函数"""
    print("开始测试控制机制改进功能...")
    
    try:
        # 测试连接池管理
        await test_connection_pool()
        
        # 测试协议适配器
        await test_protocol_adapter()
        
        # 测试设备认证
        test_device_auth()
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        await connection_pool_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())