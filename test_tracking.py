import requests
import json

BASE_URL = "http://localhost:8001"

def test_camera_list():
    """测试摄像头列表API"""
    try:
        response = requests.get(f"{BASE_URL}/camera/list")
        response.raise_for_status()
        result = response.json()
        print("摄像头列表API响应:")
        print(f"状态码: {response.status_code}")
        print(f"内容: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"摄像头列表API测试失败: {str(e)}")
        return None

def test_open_camera(camera_index):
    """测试打开摄像头API"""
    try:
        response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": camera_index})
        response.raise_for_status()
        result = response.json()
        print("\n打开摄像头API响应:")
        print(f"状态码: {response.status_code}")
        print(f"内容: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"打开摄像头API测试失败: {str(e)}")
        return None

def test_tracking_start():
    """测试视觉跟踪API"""
    try:
        response = requests.post(f"{BASE_URL}/camera/tracking/start", json={"tracker_type": "CSRT"})
        response.raise_for_status()
        result = response.json()
        print("\n视觉跟踪API响应:")
        print(f"状态码: {response.status_code}")
        print(f"内容: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"视觉跟踪API测试失败: {str(e)}")
        return None

def test_close_camera():
    """测试关闭摄像头API"""
    try:
        response = requests.post(f"{BASE_URL}/camera/close")
        response.raise_for_status()
        result = response.json()
        print("\n关闭摄像头API响应:")
        print(f"状态码: {response.status_code}")
        print(f"内容: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"关闭摄像头API测试失败: {str(e)}")
        return None

if __name__ == "__main__":
    print("开始测试视觉跟踪API...")
    print("=" * 50)
    
    # 1. 获取摄像头列表
    camera_list = test_camera_list()
    
    if camera_list and camera_list.get('success') and camera_list.get('data', {}).get('cameras'):
        camera_index = camera_list['data']['cameras'][0]['index']
        
        # 2. 打开摄像头
        open_response = test_open_camera(camera_index)
        
        if open_response and open_response.get('success'):
            # 3. 测试视觉跟踪
            tracking_response = test_tracking_start()
            
            if tracking_response and tracking_response.get('success'):
                print("\n✅ 视觉跟踪API测试成功!")
            else:
                print("\n❌ 视觉跟踪API测试失败!")
            
            # 4. 关闭摄像头
            test_close_camera()
    
    print("\n测试完成!")
    print("=" * 50)