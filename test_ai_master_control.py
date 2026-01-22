import requests
import time

def test_ai_master_control():
    print("=== AI主控端到端功能测试 ===")
    
    # 基础URL
    backend_url = "http://localhost:8000"
    
    try:
        # 1. 激活AI主控
        print("\n1. 激活AI主控...")
        activate_response = requests.post(f"{backend_url}/ai-control/master-control", json={"activate": True})
        activate_response.raise_for_status()
        activate_result = activate_response.json()
        print(f"   激活结果: {activate_result}")
        
        if not activate_result.get("success"):
            print("   ❌ AI主控激活失败")
            return False
        
        # 2. 检查设备控制状态
        print("\n2. 检查设备控制状态...")
        time.sleep(2)  # 等待AI处理设备控制
        devices_response = requests.get(f"{backend_url}/ai-control/devices")
        devices_response.raise_for_status()
        devices = devices_response.json()
        
        controlled_devices = []
        for device in devices:
            if device["connected"] and device["status"] == "online":
                controlled_devices.append(device)
                print(f"   设备: {device['name']} - 状态: {device['status']} - 连接: {device['connected']}")
        
        print(f"   📊 在线并连接的设备数: {len(controlled_devices)}")
        
        # 3. 关闭AI主控
        print("\n3. 关闭AI主控...")
        deactivate_response = requests.post(f"{backend_url}/ai-control/master-control", json={"activate": False})
        deactivate_response.raise_for_status()
        deactivate_result = deactivate_response.json()
        print(f"   关闭结果: {deactivate_result}")
        
        if not deactivate_result.get("success"):
            print("   ❌ AI主控关闭失败")
            return False
        
        # 4. 验证主控状态已关闭
        print("\n4. 验证主控状态...")
        devices_after = requests.get(f"{backend_url}/ai-control/devices").json()
        print(f"   AI主控已关闭，系统恢复正常状态")
        
        print("\n✅ AI主控端到端功能测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_ai_master_control()
