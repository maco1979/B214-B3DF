import requests
import time

def test_ai_auto_device_control():
    print("=== AI自动设备控制功能测试 ===")
    
    # 基础URL
    backend_url = "http://localhost:8000"
    
    try:
        # 1. 检查初始设备状态
        print("\n1. 检查初始设备状态...")
        devices_initial = requests.get(f"{backend_url}/ai-control/devices").json()
        print(f"   设备总数: {len(devices_initial)}")
        
        online_devices_initial = [d for d in devices_initial if d["connected"] and d["status"] == "online"]
        print(f"   在线并连接的设备: {[d['name'] for d in online_devices_initial]}")
        
        # 2. 激活AI主控
        print("\n2. 激活AI主控...")
        activate_response = requests.post(f"{backend_url}/ai-control/master-control", json={"activate": True})
        activate_response.raise_for_status()
        activate_result = activate_response.json()
        print(f"   激活结果: {activate_result['message']}")
        
        if not activate_result.get("success"):
            print("   ❌ AI主控激活失败")
            return False
        
        # 检查受控设备
        controlled_devices = activate_result.get("controlled_devices", [])
        print(f"   📊 AI自动控制的设备数: {len(controlled_devices)}")
        
        for device in controlled_devices:
            status = "✅ 成功" if device["status"] == "success" else "❌ 失败"
            print(f"   - {device['device_name']}: {status}")
        
        # 3. 验证设备控制结果
        print("\n3. 验证设备控制结果...")
        time.sleep(2)  # 等待AI完成设备控制
        
        devices_after = requests.get(f"{backend_url}/ai-control/devices").json()
        online_devices_after = [d for d in devices_after if d["connected"] and d["status"] == "online"]
        
        print(f"   激活后在线设备: {[d['name'] for d in online_devices_after]}")
        print(f"   设备状态保持: {'正常' if len(online_devices_after) == len(online_devices_initial) else '异常'}")
        
        # 4. 关闭AI主控
        print("\n4. 关闭AI主控...")
        deactivate_response = requests.post(f"{backend_url}/ai-control/master-control", json={"activate": False})
        deactivate_response.raise_for_status()
        deactivate_result = deactivate_response.json()
        print(f"   关闭结果: {deactivate_result['message']}")
        
        if not deactivate_result.get("success"):
            print("   ❌ AI主控关闭失败")
            return False
        
        # 5. 验证主控关闭后的设备状态
        print("\n5. 验证主控关闭后的设备状态...")
        devices_final = requests.get(f"{backend_url}/ai-control/devices").json()
        online_devices_final = [d for d in devices_final if d["connected"] and d["status"] == "online"]
        
        print(f"   关闭后在线设备: {[d['name'] for d in online_devices_final]}")
        print(f"   设备状态恢复: {'正常' if len(online_devices_final) == len(online_devices_initial) else '异常'}")
        
        # 6. 检查主控状态
        print("\n6. 检查最终主控状态...")
        status_response = requests.get(f"{backend_url}/ai-control/master-control/status")
        status_response.raise_for_status()
        status_result = status_response.json()
        print(f"   主控状态: {'激活' if status_result['master_control_active'] else '关闭'}")
        
        if status_result['master_control_active']:
            print("   ❌ AI主控状态未正确关闭")
            return False
        
        print("\n✅ AI自动设备控制功能测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_ai_auto_device_control()
