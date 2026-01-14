#!/usr/bin/env python3
"""
测试按用户使用习惯自动筛选最优智能体的功能
"""

import sys
import os
import requests
import json
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 后端服务URL
BASE_URL = "http://localhost:8000/api"


def test_agent_selection():
    """测试基于用户习惯的智能体选择"""
    print("=== 测试基于用户习惯的智能体选择 ===")
    
    # 测试用户ID
    user_id = "test_user_001"
    
    # 1. 首先记录一些智能体交互数据，模拟用户使用习惯
    print("\n1. 记录智能体交互数据...")
    
    # 模拟用户多次使用不同智能体的情况
    agent_interactions = [
        # 模拟用户更喜欢使用code类型的智能体
        {"agent_id": "auto_test_agent", "success": True, "satisfaction": 0.9},  # 成功，高满意度
        {"agent_id": "auto_test_agent", "success": True, "satisfaction": 0.85},  # 成功，高满意度
        {"agent_id": "auto_test_agent", "success": True, "satisfaction": 0.95},  # 成功，非常高满意度
        {"agent_id": "static_analysis_agent", "success": True, "satisfaction": 0.8},  # 成功，中等满意度
        {"agent_id": "static_analysis_agent", "success": False, "satisfaction": 0.5},  # 失败，低满意度
        {"agent_id": "error_monitor_agent", "success": True, "satisfaction": 0.75},  # 成功，中等满意度
    ]
    
    # 记录这些交互数据
    for i, interaction in enumerate(agent_interactions):
        print(f"   记录交互 {i+1}/{len(agent_interactions)}: {interaction}")
        try:
            response = requests.post(f"{BASE_URL}/ai-assistant/habits/record", json={
                "user_id": user_id,
                "behavior_type": "agent_interaction",
                "params": interaction
            })
            if response.status_code == 200:
                print(f"      ✓ 成功")
            else:
                print(f"      ✗ 失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"      ✗ 错误: {str(e)}")
        time.sleep(0.5)  # 等待一小段时间，模拟真实交互
    
    # 2. 分析用户智能体偏好
    print("\n2. 分析用户智能体偏好...")
    try:
        response = requests.get(f"{BASE_URL}/ai-assistant/habits/analyze/{user_id}")
        if response.status_code == 200:
            analysis_result = response.json()
            print(f"   ✓ 成功获取偏好分析")
            print(f"   用户总交互次数: {analysis_result.get('total_interactions', 0)}")
            
            # 获取智能体偏好排序
            if 'top_agents' in analysis_result:
                print("   智能体偏好排序:")
                for i, agent in enumerate(analysis_result['top_agents'][:3]):
                    print(f"      {i+1}. {agent['agent_id']} - 评分: {agent['score']:.3f} - 成功率: {agent['success_rate']:.2%} - 满意度: {agent['satisfaction']:.2f}")
        else:
            print(f"   ✗ 失败: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ✗ 错误: {str(e)}")
    
    # 3. 获取用户完整画像，查看智能体交互详情
    print("\n3. 获取用户完整画像...")
    try:
        response = requests.get(f"{BASE_URL}/ai-assistant/habits/profile/{user_id}")
        if response.status_code == 200:
            profile = response.json()
            print(f"   ✓ 成功获取用户画像")
            print(f"   智能体交互记录:")
            if 'agent_interactions' in profile:
                for agent_id, interaction in profile['agent_interactions'].items():
                    print(f"      {agent_id}: 交互次数={interaction['count']}, 成功次数={interaction['success_count']}, 平均满意度={interaction['satisfaction']:.2f}")
        else:
            print(f"   ✗ 失败: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ✗ 错误: {str(e)}")
    
    # 4. 测试智能体预测功能
    print("\n4. 测试智能体偏好预测...")
    try:
        # 使用Python的requests库调用服务
        from backend.src.core.services.user_habit_service import user_habit_service
        
        # 预测代码类型任务的智能体偏好
        predictions = user_habit_service.predict_agent_preference(user_id, "code")
        print(f"   ✓ 成功获取预测结果")
        print(f"   代码类型任务的智能体推荐:")
        for i, pred in enumerate(predictions):
            print(f"      {i+1}. {pred['agent_id']} - 置信度: {pred['confidence']:.3f} - 推荐理由: {pred['reason']}")
    except Exception as e:
        print(f"   ✗ 错误: {str(e)}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 检查后端服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/ai-assistant/agents")
        if response.status_code != 200:
            print(f"错误: 后端服务未运行或不可访问 (状态码: {response.status_code})")
            print("请确保后端服务正在运行: python -m uvicorn backend.src.api:create_app --reload --host 0.0.0.0 --port 8000")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到后端服务")
        print("请确保后端服务正在运行: python -m uvicorn backend.src.api:create_app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # 运行测试
    test_agent_selection()
