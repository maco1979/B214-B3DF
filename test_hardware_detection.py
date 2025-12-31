#!/usr/bin/env python3
"""
硬件检测和控制测试脚本
测试项目中的硬件检测和控制功能
"""

import sys
import os
import asyncio

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.core.services.device_discovery_service import device_discovery_service
from backend.src.core.services.connection_controller import InfraredController, AppController, BluetoothController


async def test_hardware_detection():
    """测试硬件检测功能"""
    print('🔍 开始检测本地硬件...')
    
    # 扫描所有设备
    devices = await device_discovery_service.scan_all_devices()
    
    print(f'✅ 发现 {len(devices)} 个设备:')
    for device in devices:
        print(f'  - ID: {device["id"]}, 名称: {device["name"]}, 类型: {device["type"]}, 连接类型: {device["connection_type"]}, 状态: {device["status"]}')
    
    # 测试连接控制器
    print('\n🔗 测试连接控制器...')
    
    # 测试红外控制器
    infrared_ctrl = InfraredController()
    infrared_result = infrared_ctrl.connect({'channel': 1, 'range': 10})
    print(f'  红外连接: {infrared_result["message"]}')
    
    # 测试APP控制器
    app_ctrl = AppController()
    app_result = app_ctrl.connect({'app_id': 'com.test.app', 'app_version': '1.0.0'})
    print(f'  APP连接: {app_result["message"]}')
    
    # 测试蓝牙控制器
    bt_ctrl = BluetoothController()
    bt_result = bt_ctrl.connect({'bluetooth_address': 'AA:BB:CC:DD:EE:FF'})
    print(f'  蓝牙连接: {bt_result["message"]}')
    
    # 测试摄像头控制器
    try:
        from backend.src.core.services.camera_controller import CameraController
        camera_ctrl = CameraController()
        camera_result = camera_ctrl.open_camera(999)  # 使用模拟摄像头
        print(f'  摄像头连接: {camera_result["message"]}')
        
        # 关闭摄像头
        close_result = camera_ctrl.close_camera()
        print(f'  摄像头关闭: {close_result["message"]}')
    except Exception as e:
        print(f'  摄像头控制: 失败 - {str(e)}')
    
    print('\n🎯 硬件检测和控制测试完成!')
    
    # 检查是否所有连接控制器都正常工作
    success_count = 0
    total_tests = 4  # 红外、APP、蓝牙、摄像头
    
    if infrared_result["success"]:
        success_count += 1
    if app_result["success"]:
        success_count += 1
    if bt_result["success"]:
        success_count += 1
    try:
        if camera_result["success"]:
            success_count += 1
    except:
        pass  # 摄像头可能失败
    
    print(f'\n📊 测试结果: {success_count}/{total_tests} 项功能正常')
    
    if success_count >= 3:  # 至少3项功能正常
        print('✅ 硬件检测和控制功能基本正常')
        return True
    else:
        print('❌ 硬件检测和控制功能存在问题')
        return False


def main():
    """主函数"""
    print("🧪 硬件检测和控制功能测试")
    print("="*50)
    
    # 运行异步测试
    success = asyncio.run(test_hardware_detection())
    
    print("\n" + "="*50)
    if success:
        print("🎉 硬件检测和控制功能测试通过！")
    else:
        print("⚠️  硬件检测和控制功能测试未完全通过")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())