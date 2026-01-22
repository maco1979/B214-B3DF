#!/usr/bin/env python3
"""
硬件连接功能测试脚本
测试红外线、APP、蓝牙连接的建立、命令发送和断开功能
"""

import os
import sys
import importlib.util
import time

# 动态导入连接控制器
spec = importlib.util.spec_from_file_location(
    "connection_controller", 
    os.path.join(os.path.dirname(__file__), "backend", "src", "core", "services", "connection_controller.py")
)
connection_controller_module = importlib.util.module_from_spec(spec)
sys.modules["connection_controller"] = connection_controller_module
spec.loader.exec_module(connection_controller_module)

# 创建连接控制器实例
InfraredController = connection_controller_module.InfraredController
AppController = connection_controller_module.AppController
BluetoothController = connection_controller_module.BluetoothController

def test_infrared_connection():
    """测试红外线连接功能"""
    print("\n=== 测试红外线连接 ===")
    
    # 创建控制器实例
    infrared_controller = InfraredController()
    
    # 测试连接建立
    print("1. 测试红外线连接建立...")
    connect_params = {"channel": 3, "range": 10}
    connect_result = infrared_controller.connect(connect_params)
    print(f"   连接结果: {connect_result}")
    assert connect_result["success"] == True, "红外线连接建立失败"
    
    # 测试获取连接状态
    print("2. 测试获取红外线连接状态...")
    status_result = infrared_controller.get_connection_status()
    print(f"   连接状态: {status_result}")
    assert status_result["connected"] == True, "红外线连接状态获取失败"
    
    # 测试发送命令
    print("3. 测试发送红外线命令...")
    command_params = {"action": "turn_on", "device": "电视"}
    command_result = infrared_controller.send_command(command_params)
    print(f"   命令结果: {command_result}")
    assert command_result["success"] == True, "红外线命令发送失败"
    
    # 测试断开连接
    print("4. 测试红外线连接断开...")
    disconnect_result = infrared_controller.disconnect()
    print(f"   断开结果: {disconnect_result}")
    assert disconnect_result["success"] == True, "红外线连接断开失败"
    
    # 测试断开后的状态
    status_result = infrared_controller.get_connection_status()
    assert status_result["connected"] == False, "红外线连接断开状态不正确"
    
    print("✅ 红外线连接功能测试通过！")

def test_app_connection():
    """测试APP连接功能"""
    print("\n=== 测试APP连接 ===")
    
    # 创建控制器实例
    app_controller = AppController()
    
    # 测试连接建立
    print("1. 测试APP连接建立...")
    connect_params = {"app_id": "com.ai.camera", "app_version": "1.2.3", "device_token": "test_token_123"}
    connect_result = app_controller.connect(connect_params)
    print(f"   连接结果: {connect_result}")
    assert connect_result["success"] == True, "APP连接建立失败"
    
    # 测试获取连接状态
    print("2. 测试获取APP连接状态...")
    status_result = app_controller.get_connection_status()
    print(f"   连接状态: {status_result}")
    assert status_result["connected"] == True, "APP连接状态获取失败"
    
    # 测试发送命令
    print("3. 测试发送APP命令...")
    command_params = {"action": "start_recording", "duration": 60}
    command_result = app_controller.send_command(command_params)
    print(f"   命令结果: {command_result}")
    assert command_result["success"] == True, "APP命令发送失败"
    
    # 测试断开连接
    print("4. 测试APP连接断开...")
    disconnect_result = app_controller.disconnect()
    print(f"   断开结果: {disconnect_result}")
    assert disconnect_result["success"] == True, "APP连接断开失败"
    
    # 测试断开后的状态
    status_result = app_controller.get_connection_status()
    assert status_result["connected"] == False, "APP连接断开状态不正确"
    
    print("✅ APP连接功能测试通过！")

def test_bluetooth_connection():
    """测试蓝牙连接功能"""
    print("\n=== 测试蓝牙连接 ===")
    
    # 创建控制器实例
    bluetooth_controller = BluetoothController()
    
    # 测试连接建立
    print("1. 测试蓝牙连接建立...")
    connect_params = {"bluetooth_address": "00:11:22:33:44:55", "bluetooth_version": "5.0"}
    connect_result = bluetooth_controller.connect(connect_params)
    print(f"   连接结果: {connect_result}")
    assert connect_result["success"] == True, "蓝牙连接建立失败"
    
    # 测试获取连接状态
    print("2. 测试获取蓝牙连接状态...")
    status_result = bluetooth_controller.get_connection_status()
    print(f"   连接状态: {status_result}")
    assert status_result["connected"] == True, "蓝牙连接状态获取失败"
    assert status_result["signal_strength"] > 0, "蓝牙信号强度不正确"
    
    # 测试发送命令
    print("3. 测试发送蓝牙命令...")
    command_params = {"action": "play_music", "track": "1"}
    command_result = bluetooth_controller.send_command(command_params)
    print(f"   命令结果: {command_result}")
    assert command_result["success"] == True, "蓝牙命令发送失败"
    
    # 测试断开连接
    print("4. 测试蓝牙连接断开...")
    disconnect_result = bluetooth_controller.disconnect()
    print(f"   断开结果: {disconnect_result}")
    assert disconnect_result["success"] == True, "蓝牙连接断开失败"
    
    # 测试断开后的状态
    status_result = bluetooth_controller.get_connection_status()
    assert status_result["connected"] == False, "蓝牙连接断开状态不正确"
    assert status_result["signal_strength"] == 0, "蓝牙断开后信号强度不正确"
    
    print("✅ 蓝牙连接功能测试通过！")

def test_controller_factory():
    """测试控制器工厂模式（模拟ai_control.py中的使用方式）"""
    print("\n=== 测试控制器工厂模式 ===")
    
    # 创建控制器实例字典
    controllers = {
        "infrared": InfraredController(),
        "app": AppController(),
        "bluetooth": BluetoothController()
    }
    
    # 测试批量连接
    test_devices = [
        {
            "name": "智能电视",
            "connection_type": "infrared",
            "connection_params": {"channel": 1, "range": 8}
        },
        {
            "name": "手机APP",
            "connection_type": "app",
            "connection_params": {"app_id": "com.ai.device", "app_version": "2.0.0"}
        },
        {
            "name": "蓝牙耳机",
            "connection_type": "bluetooth",
            "connection_params": {"bluetooth_address": "AA:BB:CC:DD:EE:FF"}
        }
    ]
    
    for device in test_devices:
        print(f"连接设备: {device['name']} ({device['connection_type']})")
        controller = controllers[device['connection_type']]
        result = controller.connect(device['connection_params'])
        print(f"   结果: {result['message']}")
        assert result["success"] == True, f"设备 {device['name']} 连接失败"
    
    # 测试批量断开
    print("\n断开所有设备连接...")
    for connection_type, controller in controllers.items():
        result = controller.disconnect()
        print(f"   {connection_type}: {result['message']}")
        assert result["success"] == True, f"{connection_type} 断开失败"
    
    print("✅ 控制器工厂模式测试通过！")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试未连接状态下发送命令
    print("1. 测试未连接状态下发送命令...")
    infrared_controller = InfraredController()
    command_result = infrared_controller.send_command({"action": "test"})
    print(f"   结果: {command_result}")
    assert command_result["success"] == False, "未连接状态下命令发送应该失败"
    
    # 测试空参数连接
    print("2. 测试空参数连接...")
    bluetooth_controller = BluetoothController()
    connect_result = bluetooth_controller.connect({})
    print(f"   结果: {connect_result}")
    assert connect_result["success"] == True, "空参数连接应该成功"
    bluetooth_controller.disconnect()
    
    print("✅ 边界情况测试通过！")

def main():
    """主测试函数"""
    print("硬件连接功能测试脚本")
    print("=" * 50)
    
    try:
        # 运行所有测试
        test_infrared_connection()
        test_app_connection()
        test_bluetooth_connection()
        test_controller_factory()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 所有硬件连接功能测试通过！")
        print("✅ 红外线连接功能正常")
        print("✅ APP连接功能正常")
        print("✅ 蓝牙连接功能正常")
        print("=" * 50)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
