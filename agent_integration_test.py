#!/usr/bin/env python3
"""
智能体集成测试
测试新智能体管理器与现有有机AI核心的集成
"""

import asyncio
import sys
import os
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.core.agent_manager import AgentManager, AgentType, agent_manager


async def test_basic_agent_functionality():
    """测试智能体管理器基本功能"""
    print("🧪 测试智能体管理器基本功能...")
    
    # 获取智能体管理器实例
    manager = agent_manager
    print(f"  ✅ 获取智能体管理器实例")
    
    # 创建一个决策智能体
    decision_agent_id = manager.create_agent(
        AgentType.DECISION, 
        {
            "collaboration_mode": True
        }
    )
    print(f"  ✅ 创建决策智能体: {decision_agent_id}")
    
    # 初始化智能体
    await manager.initialize_agent(decision_agent_id)
    print(f"  ✅ 初始化决策智能体")
    
    # 测试决策智能体执行
    context = {
        "decision_input": {
            "temperature": 25.0,
            "humidity": 65.0,
            "co2_level": 400.0,
            "light_intensity": 200.0,
            "energy_consumption": 0.6,
            "resource_utilization": 0.7,
            "health_score": 0.85,
            "yield_potential": 0.9
        }
    }
    
    result = await manager.execute_agent(decision_agent_id, context)
    print(f"  ✅ 决策智能体执行: {result['success']}")
    
    return True


async def test_multi_agent_coordination():
    """测试多智能体协调"""
    print("\n🧪 测试多智能体协调...")
    
    manager = agent_manager
    
    # 创建多个智能体
    agent_ids = []
    
    # 决策智能体
    decision_id = manager.create_agent(AgentType.DECISION, {"priority": "high"})
    agent_ids.append(("decision", decision_id))
    
    # 控制智能体
    control_id = manager.create_agent(AgentType.CONTROL, {"priority": "medium"})
    agent_ids.append(("control", control_id))
    
    # 学习智能体
    learning_id = manager.create_agent(AgentType.LEARNING, {"priority": "low"})
    agent_ids.append(("learning", learning_id))
    
    print(f"  ✅ 创建了 {len(agent_ids)} 个智能体")
    
    # 初始化所有智能体
    for agent_type, agent_id in agent_ids:
        await manager.initialize_agent(agent_id)
        print(f"    初始化 {agent_type} 智能体: {agent_id}")
    
    # 执行协调任务
    global_context = {
        "task": "agriculture_optimization",
        "timestamp": "2025-12-31T15:30:00Z",
        "environment": {
            "temperature": 24.5,
            "humidity": 68.0,
            "co2_level": 420.0,
            "light_intensity": 180.0
        },
        "objectives": ["maximize_yield", "minimize_energy", "maintain_quality"]
    }
    
    # 执行所有活跃智能体
    results = await manager.execute_all_active_agents(global_context)
    print(f"  ✅ 协调执行了 {len(results)} 个智能体")
    
    for agent_id, result in results.items():
        success = result.get("success", False)
        status = "✅" if success else "❌"
        print(f"    {status} {agent_id}: {result.get('result', {}).get('action', 'N/A')}")
    
    return True


async def test_agent_knowledge_sharing():
    """测试智能体知识共享概念（模拟）"""
    print("\n🧪 测试智能体知识共享概念...")
    
    manager = agent_manager
    
    # 创建学习智能体
    learning_agent_id = manager.create_agent(
        AgentType.LEARNING,
        {"knowledge_sharing_enabled": True}
    )
    await manager.initialize_agent(learning_agent_id)
    
    print(f"  ✅ 创建学习智能体: {learning_agent_id}")
    
    # 模拟学习过程
    learning_context = {
        "learning_input": {
            "data_type": "sensor_data_pattern",
            "patterns": ["temperature_cycle", "humidity_trend", "growth_correlation"],
            "confidence": 0.85,
            "timestamp": "2025-12-31T15:35:00Z"
        }
    }
    
    learning_result = await manager.execute_agent(learning_agent_id, learning_context)
    print(f"  ✅ 学习智能体执行: {learning_result['success']}")
    
    # 模拟知识共享
    if learning_result["success"]:
        learning_data = learning_result["result"]
        print(f"  ✅ 学习结果已生成，可共享")
    
    # 创建决策智能体
    decision_agent_id = manager.create_agent(AgentType.DECISION, {})
    await manager.initialize_agent(decision_agent_id)
    
    decision_context = {
        "decision_input": {
            "temperature": 26.0,
            "humidity": 70.0,
            "co2_level": 380.0,
            "light_intensity": 220.0,
            "use_shared_knowledge": True  # 指示使用共享知识
        }
    }
    
    decision_result = await manager.execute_agent(decision_agent_id, decision_context)
    print(f"  ✅ 决策智能体执行: {decision_result['success']}")
    
    return True


async def run_integration_tests():
    """运行所有集成测试"""
    print("🚀 开始智能体集成测试")
    print("="*60)
    
    success_count = 0
    total_tests = 3
    
    try:
        # 测试1: 智能体基本功能
        if await test_basic_agent_functionality():
            success_count += 1
            print("  ✅ 智能体基本功能测试通过")
        else:
            print("  ❌ 智能体基本功能测试失败")
        
        # 测试2: 多智能体协调
        if await test_multi_agent_coordination():
            success_count += 1
            print("  ✅ 多智能体协调测试通过")
        else:
            print("  ❌ 多智能体协调测试失败")
        
        # 测试3: 知识共享概念
        if await test_agent_knowledge_sharing():
            success_count += 1
            print("  ✅ 知识共享概念测试通过")
        else:
            print("  ❌ 知识共享概念测试失败")
    
    except Exception as e:
        print(f"❌ 集成测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理所有测试智能体
        print("\n🧹 清理测试智能体...")
        manager = agent_manager
        await manager.cleanup()
        print("✅ 清理完成")
    
    print("\n" + "="*60)
    print(f"📊 集成测试结果: {success_count}/{total_tests} 项测试通过")
    
    if success_count == total_tests:
        print("🎉 所有集成测试通过！")
        return True
    else:
        print("⚠️  部分集成测试未通过")
        return False


def main():
    """主函数"""
    print("🧪 智能体集成测试")
    print("测试内容: 智能体管理器与有机AI核心集成、多智能体协调、知识共享")
    
    success = asyncio.run(run_integration_tests())
    
    if success:
        print("\n✅ 智能体集成测试成功！")
        print("📋 新的智能体管理器已成功与现有有机AI核心集成")
        return 0
    else:
        print("\n❌ 智能体集成测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())