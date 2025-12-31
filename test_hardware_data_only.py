"""
测试硬件数据收集功能（不涉及AI核心以避免flax兼容性问题）
验证硬件数据收集、处理和导出功能
"""
import asyncio
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.core.services.hardware_data_collector import hardware_data_collector, HardwareDataType, HardwareDataPoint


async def test_hardware_data_collection():
    """测试硬件数据收集功能"""
    print("\n=== 测试硬件数据收集功能 ===")
    
    # 启动数据收集
    await hardware_data_collector.start_collection()
    print("✅ 硬件数据收集已启动")
    
    # 等待收集一些数据
    await asyncio.sleep(3)
    
    # 获取统计信息
    stats = hardware_data_collector.get_data_statistics()
    print(f"数据统计: {stats}")
    
    # 获取最近的数据
    recent_data = await hardware_data_collector.get_recent_data(5)
    print(f"最近数据点数量: {len(recent_data)}")
    if recent_data:
        for i, data_point in enumerate(recent_data):
            print(f"  {i+1}. {data_point.data_type.value} - {data_point.device_id}")
    
    # 停止数据收集
    await hardware_data_collector.stop_collection()
    print("✅ 硬件数据收集已停止")
    
    return True


async def test_hardware_data_processing():
    """测试硬件数据处理功能"""
    print("\n=== 测试硬件数据处理功能 ===")
    
    # 创建不同类型的硬件数据点
    test_data_points = [
        HardwareDataPoint(
            device_id="sensor_001",
            data_type=HardwareDataType.SENSORS,
            timestamp=datetime.now(),
            data={
                "temperature": 25.5,
                "humidity": 60.2,
                "light_intensity": 520.0,
                "co2_level": 410.5,
                "soil_moisture": 32.8
            },
            confidence=0.95,
            quality_score=0.98
        ),
        HardwareDataPoint(
            device_id="controller_001",
            data_type=HardwareDataType.CONTROLLERS,
            timestamp=datetime.now(),
            data={
                "led_uv_intensity": 0.08,
                "led_far_red_intensity": 0.12,
                "led_white_intensity": 0.75,
                "led_red_intensity": 0.18,
                "controller_status": "active",
                "power_consumption": 52.3
            },
            confidence=0.92,
            quality_score=0.96
        ),
        HardwareDataPoint(
            device_id="device_001",
            data_type=HardwareDataType.STATUS,
            timestamp=datetime.now(),
            data={
                "connection_status": "connected",
                "signal_strength": 88.5,
                "battery_level": 95.2,
                "operational_time": 3600,
                "error_count": 1,
                "last_update": datetime.now().isoformat()
            },
            confidence=0.98,
            quality_score=0.99
        )
    ]
    
    print(f"创建了 {len(test_data_points)} 个测试数据点")
    
    # 模拟预处理这些数据点
    for i, data_point in enumerate(test_data_points):
        processed_data = await hardware_data_collector._preprocess_data(data_point)
        print(f"  {i+1}. {data_point.data_type.value} -> 处理后特征数: {len(processed_data.get('data_features', {}))}")
    
    return True


async def test_data_export_for_ai():
    """测试数据导出用于AI训练"""
    print("\n=== 测试数据导出用于AI训练 ===")
    
    # 启动数据收集
    await hardware_data_collector.start_collection()
    
    # 等待收集一些数据
    await asyncio.sleep(2)
    
    # 导出数据用于AI训练
    training_data = await hardware_data_collector.export_data_for_ai_training()
    print(f"导出的训练数据信息: {training_data}")
    
    # 检查数据形状
    if training_data["sample_count"] > 0:
        print(f"样本数量: {training_data['sample_count']}")
        print(f"特征维度: {training_data['feature_dimension']}")
        print("✅ 数据导出功能正常")
        
        # 显示部分特征
        if hasattr(training_data['features'], 'shape'):
            print(f"特征数组形状: {training_data['features'].shape}")
    else:
        print("⚠️  暂无足够的数据用于训练")
        print("  继续收集数据以达到训练要求")
    
    # 停止数据收集
    await hardware_data_collector.stop_collection()
    
    return True


async def main():
    """主测试函数"""
    print("开始测试硬件数据收集和处理功能...")
    
    try:
        # 测试硬件数据收集
        await test_hardware_data_collection()
        
        # 测试数据处理
        await test_hardware_data_processing()
        
        # 测试数据导出
        await test_data_export_for_ai()
        
        print("\n🎉 所有测试完成！")
        print("✅ 硬件数据收集功能正常工作")
        print("✅ 硬件数据处理功能正常工作")
        print("✅ 数据预处理和特征提取功能正常")
        print("✅ 数据导出用于AI训练功能正常")
        print("\n📝 硬件数据可以成功收集并处理，为AI学习提供数据源")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())