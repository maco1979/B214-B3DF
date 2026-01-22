import requests
import json
import time

# 测试API端点
base_url = "http://localhost:8003/api"

def test_model_management():
    print("测试模型管理API...")
    
    # 等待服务器响应
    print("等待后端服务响应...")
    time.sleep(2)
    
    # 1. 获取所有模型
    print("\n1. 获取模型列表...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"响应内容: {models}")
            print(f"模型数量: {len(models) if isinstance(models, list) else 'N/A'}")
            if isinstance(models, list) and len(models) > 0:
                print(f"第一个模型: {models[0].get('name', 'N/A') if len(models) > 0 else 'N/A'}")
        else:
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"获取模型列表失败: {e}")
    
    # 2. 创建一个测试模型
    print("\n2. 创建测试模型...")
    try:
        test_model_data = {
            "name": "农业分类模型",
            "description": "用于识别农业图像中的作物和病虫害",
            "status": "ready",
            "version": "1.0.0",
            "model_type": "ai"
        }
        response = requests.post(f"{base_url}/models/", json=test_model_data)
        print(f"创建模型状态码: {response.status_code}")
        print(f"创建模型响应: {response.text}")
        if response.status_code == 201:
            created_model = response.json()
            model_id = created_model.get('id', created_model.get('model_id'))
            print(f"创建的模型ID: {model_id}")
            
            # 3. 测试启动模型
            if model_id:
                print(f"\n3. 启动模型 (ID: {model_id})...")
                response = requests.post(f"{base_url}/models/{model_id}/start")
                print(f"启动模型状态码: {response.status_code}")
                print(f"启动模型响应: {response.text}")
                if response.status_code == 200:
                    start_result = response.json()
                    print(f"启动结果: {start_result}")
                else:
                    print(f"启动失败")
                
                # 4. 测试暂停模型
                print(f"\n4. 暂停模型 (ID: {model_id})...")
                response = requests.post(f"{base_url}/models/{model_id}/pause")
                print(f"暂停模型状态码: {response.status_code}")
                print(f"暂停模型响应: {response.text}")
                if response.status_code == 200:
                    pause_result = response.json()
                    print(f"暂停结果: {pause_result}")
                else:
                    print(f"暂停失败")
        else:
            print(f"创建模型失败")
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_model_management()