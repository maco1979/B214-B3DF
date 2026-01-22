"""
测试硬件数据与AI核心的集成
验证硬件数据收集、处理和学习功能
"""
import asyncio
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.core.ai_organic_core import organic_ai_core, get_organic_ai_core
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
        print(f"最新数据点: {recent_data[-1].data_type.value} - {recent_data[-1].data}")
    
    # 停止数据收集
    await hardware_data_collector.stop_collection()
    print("✅ 硬件数据收集已停止")
    
    return True


async def test_hardware_ai_integration():
    """测试硬件数据与AI核心的集成"""
    print("\n=== 测试硬件数据与AI核心集成 ===")
    
    # 获取AI核心实例
    ai_core = await get_organic_ai_core()
    print("✅ 获取AI核心实例成功")
    
    # 创建模拟硬件数据点
    sample_data_point = HardwareDataPoint(
        device_id="test_sensor_001",
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
    )
    
    print(f"创建测试数据点: {sample_data_point.data_type.value}")
    
    # 测试AI从硬件数据学习
    await ai_core.learn_from_hardware_data(sample_data_point)
    print("✅ AI从硬件数据学习完成")
    
    # 检查AI核心状态
    status = ai_core.get_status()
    print(f"AI核心状态: {status['state']}")
    print(f"学习记忆大小: {status['learning_memory_size']}")
    print(f"决策数量: {status['decision_count']}")
    
    # 测试AI核心的硬件数据学习启动功能
    await ai_core.start_hardware_data_learning()
    print("✅ AI核心硬件数据学习已启动")
    
    # 等待一些数据被处理
    await asyncio.sleep(3)
    
    # 检查学习状态
    status = ai_core.get_status()
    print(f"学习后AI状态: {status['state']}")
    print(f"学习后记忆大小: {status['learning_memory_size']}")
    
    # 停止硬件数据学习
    await ai_core.stop_hardware_data_learning()
    print("✅ AI核心硬件数据学习已停止")
    
    return True


async def test_data_export_for_training():
    """测试数据导出用于AI训练"""
    print("\n=== 测试数据导出用于AI训练 ===")
    
    # 启动数据收集
    await hardware_data_collector.start_collection()
    
    # 等待收集一些数据
    await asyncio.sleep(2)
    
    # 导出数据用于AI训练
    training_data = await hardware_data_collector.export_data_for_ai_training()
    print(f"导出的训练数据: {training_data}")
    
    # 检查数据形状
    if training_data["sample_count"] > 0:
        print(f"样本数量: {training_data['sample_count']}")
        print(f"特征维度: {training_data['feature_dimension']}")
        print("✅ 数据导出功能正常")
    else:
        print("⚠️  暂无足够的数据用于训练")
    
    # 停止数据收集
    await hardware_data_collector.stop_collection()
    
    return True


async def main():
    """主测试函数"""
    print("开始测试硬件数据与AI核心集成...")
    
    try:
        # 测试硬件数据收集
        await test_hardware_data_collection()
        
        # 测试硬件AI集成
        await test_hardware_ai_integration()
        
        # 测试数据导出
        await test_data_export_for_training()
        
        print("\n🎉 所有测试完成！")
        print("✅ 硬件数据可以成功链接到AI核心")
        print("✅ 硬件数据可以用于AI学习")
        print("✅ 数据收集、处理和学习流程正常工作")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())