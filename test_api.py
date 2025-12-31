import requests
import json

# API基础URL
API_BASE_URL = 'http://localhost:8000/api'

def test_auth_login():
    """测试登录API"""
    print("测试登录API...")
    url = f'{API_BASE_URL}/auth/login'
    data = {
        'email': 'test@example.com',
        'password': 'test123456'
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"登录状态码: {response.status_code}")
        if response.status_code == 200:
            print("登录成功！")
            return response.json().get('data', {}).get('access_token')
        else:
            print(f"登录失败: {response.text}")
            return None
    except Exception as e:
        print(f"登录请求错误: {e}")
        return None

def test_models_list(token):
    """测试模型列表API"""
    print("\n测试模型列表API...")
    url = f'{API_BASE_URL}/models'
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    
    try:
        response = requests.get(url, headers=headers)
        print(f"模型列表状态码: {response.status_code}")
        if response.status_code == 200:
            print("获取模型列表成功！")
            print(f"返回数据: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"获取模型列表失败: {response.text}")
            return None
    except Exception as e:
        print(f"模型列表请求错误: {e}")
        return None

def test_create_model(token):
    """测试创建模型API"""
    print("\n测试创建模型API...")
    url = f'{API_BASE_URL}/models'
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    data = {
        'name': '测试模型',
        'description': '这是一个测试模型',
        'version': 'v1.0.0',
        'model_type': 'ai'
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"创建模型状态码: {response.status_code}")
        if response.status_code == 201:
            print("创建模型成功！")
            print(f"返回数据: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"创建模型失败: {response.text}")
            return None
    except Exception as e:
        print(f"创建模型请求错误: {e}")
        return None

if __name__ == "__main__":
    print("=== AI项目API测试 ===")
    
    # 测试登录
    token = test_auth_login()
    
    # 测试模型列表
    test_models_list(token)
    
    # 测试创建模型
    test_create_model(token)
    
    print("\n=== 测试完成 ===")
