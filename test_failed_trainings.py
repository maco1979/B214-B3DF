#!/usr/bin/env python3
"""
测试脚本：显示所有模型训练失败

该脚本用于测试模型训练失败后，系统是否能正确记录和显示失败信息
"""

import asyncio
import requests
import json
import time
from backend.src.core.services.model_manager import ModelManager

async def test_get_all_failed_trainings():
    """测试获取所有失败的训练任务"""
    print("=== 测试获取所有失败的训练任务 ===")
    
    # 创建模型管理器实例
    model_manager = ModelManager()
    
    # 初始化模型管理器
    init_result = await model_manager.initialize()
    if not init_result["success"]:
        print(f"初始化模型管理器失败: {init_result['error']}")
        return False
    
    print("模型管理器初始化成功")
    
    # 先创建一些模型
    print("\n1. 创建测试模型...")
    
    # 模型1：正常训练
    model1_id = "test_model_1"
    model1_data = {
        "name": "测试模型1",
        "type": "classification",
        "framework": "pytorch",
        "version": "1.0.0",
        "status": "registered"
    }
    register_result1 = await model_manager.register_model(model1_id, model1_data)
    print(f"注册模型1结果: {register_result1}")
    
    # 模型2：用于触发失败
    model2_id = "test_model_2"
    model2_data = {
        "name": "测试模型2",
        "type": "regression",
        "framework": "pytorch",
        "version": "1.0.0",
        "status": "registered"
    }
    register_result2 = await model_manager.register_model(model2_id, model2_data)
    print(f"注册模型2结果: {register_result2}")
    
    # 模型3：用于触发失败
    model3_id = "test_model_3"
    model3_data = {
        "name": "测试模型3",
        "type": "optimization",
        "framework": "pytorch",
        "version": "1.0.0",
        "status": "registered"
    }
    register_result3 = await model_manager.register_model(model3_id, model3_data)
    print(f"注册模型3结果: {register_result3}")
    
    # 2. 开始训练任务，有些会失败
    print("\n2. 开始训练任务...")
    
    # 训练任务1：正常训练
    training_data1 = {
        "data_path": "/path/to/data1",
        "parameters": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    train_result1 = await model_manager.start_training(model1_id, training_data1)
    print(f"开始训练任务1结果: {train_result1}")
    
    # 模拟一个失败的训练任务（通过直接修改状态）
    print("\n3. 模拟失败的训练任务...")
    
    # 为模型2创建一个失败的训练任务
    failed_task_id1 = f"{model2_id}_{int(time.time())}"
    model_manager.training_tasks[failed_task_id1] = {
        "model_id": model2_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "failed",
        "progress": 50,
        "stage": "模型训练",
        "current_step": 5,
        "total_steps": 10,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "error": "模拟训练失败：数据格式错误",
        "training_data": {
            "data_path": "/path/to/invalid_data",
            "parameters": {
                "epochs": 10,
                "batch_size": 32
            }
        }
    }
    print(f"创建失败训练任务1: {failed_task_id1}")
    
    # 为模型3创建一个失败的训练任务
    failed_task_id2 = f"{model3_id}_{int(time.time()) + 1}"
    model_manager.training_tasks[failed_task_id2] = {
        "model_id": model3_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "failed",
        "progress": 70,
        "stage": "模型评估",
        "current_step": 7,
        "total_steps": 10,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "error": "模拟训练失败：内存不足",
        "training_data": {
            "data_path": "/path/to/large_data",
            "parameters": {
                "epochs": 20,
                "batch_size": 64
            }
        }
    }
    print(f"创建失败训练任务2: {failed_task_id2}")
    
    # 4. 测试获取所有训练任务
    print("\n4. 测试获取所有训练任务...")
    all_tasks = await model_manager.get_all_training_tasks()
    print(f"所有训练任务: {json.dumps(all_tasks, indent=2, ensure_ascii=False)}")
    
    # 5. 测试获取失败的训练任务
    print("\n5. 测试获取失败的训练任务...")
    failed_tasks = await model_manager.get_all_training_tasks(status_filter="failed")
    print(f"失败的训练任务: {json.dumps(failed_tasks, indent=2, ensure_ascii=False)}")
    
    # 6. 验证结果
    print("\n6. 验证结果...")
    
    # 检查是否返回了正确数量的失败任务
    expected_failed_count = 2
    actual_failed_count = len(failed_tasks["tasks"])
    
    print(f"预期失败任务数: {expected_failed_count}")
    print(f"实际失败任务数: {actual_failed_count}")
    
    if actual_failed_count == expected_failed_count:
        print("✅ 测试通过：获取失败训练任务数量正确")
    else:
        print("❌ 测试失败：获取失败训练任务数量不正确")
        return False
    
    # 检查每个失败任务是否有错误信息
    for task in failed_tasks["tasks"]:
        if "error" in task and task["error"]:
            print(f"✅ 任务 {task['task_id']} 有错误信息: {task['error']}")
        else:
            print(f"❌ 任务 {task['task_id']} 缺少错误信息")
            return False
    
    # 7. 通过API测试获取失败的训练任务
    print("\n7. 通过API测试获取失败的训练任务...")
    
    try:
        # 使用API获取失败的训练任务
        response = requests.get("http://localhost:8001/api/models/training/tasks?status=failed")
        api_result = response.json()
        
        print(f"API返回状态码: {response.status_code}")
        print(f"API返回结果: {json.dumps(api_result, indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200 and api_result["success"]:
            print("✅ API测试通过：成功获取失败的训练任务")
        else:
            print("❌ API测试失败：无法获取失败的训练任务")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  警告：无法连接到API服务，跳过API测试")
    except Exception as e:
        print(f"❌ API测试失败：{str(e)}")
    
    print("\n=== 测试完成 ===")
    return True

async def main():
    """主函数"""
    success = await test_get_all_failed_trainings()
    if success:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n💥 测试失败！")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
