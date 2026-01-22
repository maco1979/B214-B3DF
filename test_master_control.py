#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试AI主控功能与有机体AI核心的集成
"""

import requests
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_master_control_integration():
    """测试AI主控与有机体AI核心的集成"""
    print("=== 测试AI主控与有机体AI核心集成 ===")
    
    # 后端服务器URL
    backend_url = "http://localhost:8000"
    
    try:
        print("\n1. 获取初始AI主控状态...")
        status_response = requests.get(f"{backend_url}/api/ai-control/master-control/status")
        print(f"   状态码: {status_response.status_code}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   初始状态: {status_data}")
        else:
            print(f"   获取状态失败: {status_response.text}")
            return False
        
        print("\n2. 激活AI主控...")
        activate_response = requests.post(
            f"{backend_url}/api/ai-control/master-control", 
            json={"activate": True}
        )
        print(f"   状态码: {activate_response.status_code}")
        if activate_response.status_code == 200:
            activate_data = activate_response.json()
            print(f"   激活响应: {activate_data}")
            
            if activate_data.get("success"):
                print("   ✅ AI主控激活成功")
            else:
                print(f"   ❌ AI主控激活失败: {activate_data}")
                return False
        else:
            print(f"   激活请求失败: {activate_response.text}")
            return False
        
        # 等待片刻以确保有机体AI核心有时间启动
        time.sleep(1)
        
        print("\n3. 获取激活后的状态...")
        status_response = requests.get(f"{backend_url}/api/ai-control/master-control/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   激活后状态: {status_data}")
            
            if status_data.get("master_control_active"):
                print("   ✅ AI主控状态已正确设置为激活")
            else:
                print("   ❌ AI主控状态未正确更新")
                return False
        else:
            print(f"   获取状态失败: {status_response.text}")
            return False
        
        print("\n4. 关闭AI主控...")
        deactivate_response = requests.post(
            f"{backend_url}/api/ai-control/master-control", 
            json={"activate": False}
        )
        print(f"   状态码: {deactivate_response.status_code}")
        if deactivate_response.status_code == 200:
            deactivate_data = deactivate_response.json()
            print(f"   关闭响应: {deactivate_data}")
            
            if deactivate_data.get("success"):
                print("   ✅ AI主控关闭成功")
            else:
                print(f"   ❌ AI主控关闭失败: {deactivate_data}")
                return False
        else:
            print(f"   关闭请求失败: {deactivate_response.text}")
            return False
        
        print("\n5. 获取关闭后的状态...")
        status_response = requests.get(f"{backend_url}/api/ai-control/master-control/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   关闭后状态: {status_data}")
            
            if not status_data.get("master_control_active"):
                print("   ✅ AI主控状态已正确设置为关闭")
            else:
                print("   ❌ AI主控状态未正确更新")
                return False
        else:
            print(f"   获取状态失败: {status_response.text}")
            return False
        
        print("\n✅ 所有测试通过！AI主控与有机体AI核心集成正常工作")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到后端服务器 {backend_url}")
        print("   请确保后端服务正在运行")
        return False
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_master_control_integration()
    if not success:
        print("\n❌ 测试失败")
        sys.exit(1)
    else:
        print("\n✅ 测试成功")
        sys.exit(0)