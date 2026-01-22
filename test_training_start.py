#!/usr/bin/env python3
"""
测试脚本：验证训练启动功能

该脚本用于测试模型训练启动功能，确保修复后能够成功启动训练
"""

import asyncio
import time
from backend.src.core.services.model_manager import ModelManager

async def test_training_start():
    """测试训练启动功能"""
    print("=== 测试训练启动功能 ===")
    
    # 创建模型管理器实例
    model_manager = ModelManager()
    
    # 初始化模型管理器
    init_result = await model_manager.initialize()
    if not init_result["success"]:
        print(f"初始化模型管理器失败: {init_result['error']}")
        return False
    
    print("模型管理器初始化成功")
    
    # 创建测试模型
    print("\n1. 创建测试模型...")
    
    model_id = "test_training_model"
    model_data = {
        "name": "测试训练模型",
        "type": "classification",
        "framework": "pytorch",
        "version": "1.0.0",
        "status": "registered"
    }
    
    register_result = await model_manager.register_model(model_id, model_data)
    print(f"注册模型结果: {register_result}")
    
    if not register_result["success"]:
        print("注册模型失败，测试终止")
        return False
    
    # 尝试启动训练
    print("\n2. 尝试启动训练...")
    
    training_data = {
        "data_path": "/path/to/training_data",
        "parameters": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    train_result = await model_manager.start_training(model_id, training_data)
    print(f"启动训练结果: {train_result}")
    
    if not train_result["success"]:
        print(f"训练启动失败: {train_result['error']}")
        return False
    
    print("✅ 训练启动成功！")
    
    # 验证训练任务已添加到任务列表
    print("\n3. 验证训练任务已添加到任务列表...")
    
    all_tasks = await model_manager.get_all_training_tasks()
    print(f"当前训练任务数: {len(all_tasks['tasks'])}")
    
    task_ids = [task['task_id'] for task in all_tasks['tasks']]
    if train_result['task_id'] in task_ids:
        print("✅ 训练任务已成功添加到任务列表")
    else:
        print("❌ 训练任务未添加到任务列表")
        return False
    
    # 检查训练任务状态
    print("\n4. 检查训练任务状态...")
    
    task_status = await model_manager.get_training_status(train_result['task_id'])
    print(f"训练任务状态: {task_status['status']}")
    print(f"训练进度: {task_status['progress']}%")
    print(f"训练阶段: {task_status['stage']}")
    
    if task_status['status'] == 'training':
        print("✅ 训练任务正在运行中")
    else:
        print(f"❌ 训练任务状态异常: {task_status['status']}")
        return False
    
    # 等待一段时间，检查训练进度是否更新
    print("\n5. 等待训练进度更新...")
    time.sleep(1.0)
    
    updated_status = await model_manager.get_training_status(train_result['task_id'])
    print(f"更新后的训练进度: {updated_status['progress']}%")
    print(f"更新后的训练阶段: {updated_status['stage']}")
    
    if updated_status['progress'] > 0:
        print("✅ 训练进度正在更新")
    else:
        print("⚠️  训练进度未更新，可能是正常的（训练刚开始）")
    
    print("\n=== 测试完成 ===")
    return True

async def main():
    """主函数"""
    success = await test_training_start()
    if success:
        print("\n🎉 所有测试通过！训练启动功能正常")
        return 0
    else:
        print("\n💥 测试失败！训练启动功能异常")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
