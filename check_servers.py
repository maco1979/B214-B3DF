import requests

def check_frontend():
    try:
        response = requests.get('http://localhost:3000')
        print('Frontend status:', response.status_code)
        return True
    except Exception as e:
        print('Frontend error:', e)
        return False

def check_backend_health():
    try:
        response = requests.get('http://localhost:8000/api/system/health')
        print('Backend health status:', response.status_code)
        print('Backend response:', response.json())
        return True
    except Exception as e:
        print('Backend health error:', e)
        return False

def check_ai_control_devices():
    try:
        response = requests.get('http://localhost:8000/api/ai-control/devices')
        print('AI control devices status:', response.status_code)
        print('Devices:', response.json())
        return True
    except Exception as e:
        print('AI control devices error:', e)
        return False

def check_decision_service():
    try:
        response = requests.get('http://localhost:8001/api/decision/models')
        print('Decision service status:', response.status_code)
        print('Models:', response.json())
        return True
    except Exception as e:
        print('Decision service error:', e)
        return False

def check_api_gateway():
    try:
        response = requests.get('http://localhost:8080/health')
        print('API Gateway status:', response.status_code)
        return True
    except Exception as e:
        print('API Gateway error:', e)
        return False

def check_jepa_status():
    try:
        response = requests.get('http://localhost:8000/api/jepa-dtmpc/status')
        print('JEPA status:', response.status_code)
        print('JEPA data:', response.json())
        return True
    except Exception as e:
        print('JEPA error:', e)
        return False

def check_performance_summary():
    try:
        response = requests.get('http://localhost:8000/api/performance/summary')
        print('Performance summary status:', response.status_code)
        print('Performance data:', response.json())
        return True
    except Exception as e:
        print('Performance summary error:', e)
        return False

def check_inference():
    try:
        payload = {"input": [1, 2, 3], "model_name": "default"}
        response = requests.post('http://localhost:8000/api/inference', json=payload)
        print('Inference status:', response.status_code)
        print('Inference result:', response.json())
        return True
    except Exception as e:
        print('Inference error:', e)
        return False

def check_agriculture_light_recipe():
    try:
        payload = {"crop": "lettuce", "growth_stage": "seedling", "environment": {"temperature": 24, "humidity": 65}}
        response = requests.post('http://localhost:8000/api/agriculture/light-recipe', json=payload)
        print('Light recipe status:', response.status_code)
        print('Light recipe result:', response.json())
        return True
    except Exception as e:
        print('Light recipe error:', e)
        return False

def check_decision_predict():
    try:
        # 模拟一个农业决策请求
        payload = {
            "model_name": "agriculture_decision_model",
            "data": {"crop": "lettuce", "humidity": 65, "temperature": 24}
        }
        headers = {
            "X-API-KEY": "default-api-key",
            "Content-Type": "application/json"
        }
        response = requests.post('http://localhost:8080/api/decision/predict', json=payload, headers=headers)
        print('Decision prediction status (via Gateway):', response.status_code)
        print('Prediction result:', response.json())
        return True
    except Exception as e:
        print('Decision prediction error:', e)
        return False

if __name__ == '__main__':
    print('Checking server status...')
    check_frontend()
    check_backend_health()
    check_ai_control_devices()
    check_decision_service()
    check_api_gateway()
    check_jepa_status()
    check_decision_predict()
    check_performance_summary()
    check_inference()
    check_agriculture_light_recipe()
