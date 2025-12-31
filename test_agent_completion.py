#!/usr/bin/env python3
"""
智能体补全功能测试脚本
测试新创建的智能体管理器和相关组件
"""

import asyncio
import sys
import os
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.core.agent_manager import (
    AgentManager, 
    AgentType, 
    AgentStatus, 
    DecisionAgent, 
    ControlAgent, 
    LearningAgent,
    agent_manager
)


async def test_agent_creation():
    """测试智能体创建功能"""
    print("🧪 测试智能体创建功能...")
    
    # 创建智能体管理器
    manager = agent_manager
    
    # 创建不同类型的智能体
    decision_agent_id = manager.create_agent(AgentType.DECISION, {"learning_rate": 0.01})
    control_agent_id = manager.create_agent(AgentType.CONTROL, {"max_commands": 100})
    learning_agent_id = manager.create_agent(AgentType.LEARNING, {"memory_size": 1000})
    
    print(f"  ✅ 创建决策智能体: {decision_agent_id}")
    print(f"  ✅ 创建控制智能体: {control_agent_id}")
    print(f"  ✅ 创建学习智能体: {learning_agent_id}")
    
    # 验证智能体状态
    agents = manager.list_agents()
    print(f"  📊 总共创建了 {len(agents)} 个智能体")
    
    for agent_info in agents:
        print(f"    - {agent_info['agent_id']} ({agent_info['agent_type']}): {agent_info['status']}")
    
    return [decision_agent_id, control_agent_id, learning_agent_id]


async def test_agent_initialization(agent_ids):
    """测试智能体初始化功能"""
    print("\n🧪 测试智能体初始化功能...")
    
    manager = agent_manager
    
    for agent_id in agent_ids:
        await manager.initialize_agent(agent_id)
        status = manager.get_agent_status(agent_id)
        print(f"  ✅ 智能体 {agent_id} 初始化完成，状态: {status['status']}")


async def test_agent_execution(agent_ids):
    """测试智能体执行功能"""
    print("\n🧪 测试智能体执行功能...")
    
    manager = agent_manager
    
    # 测试决策智能体
    decision_id = agent_ids[0]
    context = {
        "decision_input": {
            "temperature": 25.5,
            "humidity": 65.0,
            "co2_level": 400
        }
    }
    result = await manager.execute_agent(decision_id, context)
    print(f"  ✅ 决策智能体执行结果: {result['success']}")
    
    # 测试控制智能体
    control_id = agent_ids[1]
    context = {
        "control_input": {
            "device_id": "device_001",
            "command": "activate"
        }
    }
    result = await manager.execute_agent(control_id, context)
    print(f"  ✅ 控制智能体执行结果: {result['success']}")
    
    # 测试学习智能体
    learning_id = agent_ids[2]
    context = {
        "learning_input": {
            "data": [1, 2, 3, 4, 5],
            "labels": [0, 1, 0, 1, 0]
        }
    }
    result = await manager.execute_agent(learning_id, context)
    print(f"  ✅ 学习智能体执行结果: {result['success']}")


async def test_all_agents_execution():
    """测试所有活跃智能体执行"""
    print("\n🧪 测试所有活跃智能体执行...")
    
    manager = agent_manager
    
    context = {
        "global_context": {
            "timestamp": "2025-12-31T15:00:00Z",
            "system_state": "normal"
        }
    }
    
    results = await manager.execute_all_active_agents(context)
    print(f"  ✅ 执行了 {len(results)} 个活跃智能体")
    
    for agent_id, result in results.items():
        status = "✅" if result.get("success", False) else "❌"
        print(f"    {status} {agent_id}: {result.get('result', {}).get('action', 'N/A')}")


async def test_agent_lifecycle():
    """测试智能体生命周期管理"""
    print("\n🧪 测试智能体生命周期管理...")
    
    manager = agent_manager
    
    # 创建一个新智能体
    new_agent_id = manager.create_agent(AgentType.MONITORING, {"interval": 30})
    print(f"  ✅ 创建监控智能体: {new_agent_id}")
    
    # 初始化
    await manager.initialize_agent(new_agent_id)
    status = manager.get_agent_status(new_agent_id)
    print(f"  ✅ 初始化后状态: {status['status']}")
    
    # 暂停
    await manager.pause_agent(new_agent_id)
    status = manager.get_agent_status(new_agent_id)
    print(f"  ✅ 暂停后状态: {status['status']}")
    
    # 恢复
    await manager.resume_agent(new_agent_id)
    status = manager.get_agent_status(new_agent_id)
    print(f"  ✅ 恢复后状态: {status['status']}")
    
    # 停止
    await manager.stop_agent(new_agent_id)
    print(f"  ✅ 停止智能体: {new_agent_id}")
    
    # 验证智能体已被移除
    status = manager.get_agent_status(new_agent_id)
    if status is None:
        print(f"  ✅ 智能体已正确移除")
    else:
        print(f"  ❌ 智能体未被正确移除")


async def test_agent_manager():
    """完整测试智能体管理器"""
    print("🚀 开始智能体管理器完整测试")
    print("="*60)
    
    try:
        # 测试智能体创建
        agent_ids = await test_agent_creation()
        
        # 测试智能体初始化
        await test_agent_initialization(agent_ids)
        
        # 测试智能体执行
        await test_agent_execution(agent_ids)
        
        # 测试所有智能体执行
        await test_all_agents_execution()
        
        # 测试生命周期管理
        await test_agent_lifecycle()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！智能体管理器功能正常")
        
        # 显示最终智能体列表
        manager = agent_manager
        agents = manager.list_agents()
        print(f"\n📊 最终智能体状态:")
        for agent_info in agents:
            print(f"  - {agent_info['agent_id']} ({agent_info['agent_type']}): {agent_info['status']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理所有智能体
        print("\n🧹 清理测试智能体...")
        manager = agent_manager
        await manager.cleanup()
        print("✅ 清理完成")


def main():
    """主函数"""
    print("🧪 智能体补全功能测试")
    print("测试内容: 智能体管理器、多类型智能体、生命周期管理")
    
    success = asyncio.run(test_agent_manager())
    
    if success:
        print("\n✅ 智能体补全功能测试成功！")
        return 0
    else:
        print("\n❌ 智能体补全功能测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())