import requests
import time
import json

BASE_URL = 'http://localhost:8005'

def test_ptz_advanced_features():
    """测试PTZ云台高级功能"""
    print('开始测试PTZ云台高级功能...')
    
    # 1. 获取当前状态
    print('\n1. 获取当前PTZ状态:')
    status_resp = requests.get(f'{BASE_URL}/api/camera/ptz/status')
    status_data = status_resp.json()
    print(f'当前状态: {json.dumps(status_data, indent=2, ensure_ascii=False)}')
    
    # 2. 设置几个预置位
    print('\n2. 设置预置位...')
    presets = [
        {'preset_id': 1, 'name': 'Home Position', 'pan': 0, 'tilt': 0},
        {'preset_id': 2, 'name': 'Right Position', 'pan': 60, 'tilt': 10},
        {'preset_id': 3, 'name': 'Left Position', 'pan': -60, 'tilt': 10},
        {'preset_id': 4, 'name': 'Down Position', 'pan': 0, 'tilt': -30}
    ]
    
    for preset in presets:
        # 移动到预置位位置
        print(f'  移动到预置位 {preset["preset_id"]} ({preset["pan"]}°, {preset["tilt"]}°)...')
        move_resp = requests.post(f'{BASE_URL}/api/camera/ptz/move', json={
            'pan': preset['pan'],
            'tilt': preset['tilt'],
            'speed': 50
        })
        response_data = move_resp.json()
        message = response_data.get('message', str(response_data))
        print(f'  移动结果: {message}')
        
        # 设置预置位
        set_resp = requests.post(f'{BASE_URL}/api/camera/ptz/preset/set', json={
            'preset_id': preset['preset_id'],
            'name': preset['name']
        })
        response_data = set_resp.json()
        message = response_data.get('message', str(response_data))
        print(f'  预置位{preset["preset_id"]}设置: {message}')
        
        time.sleep(1)  # 等待1秒
    
    # 3. 测试转到预置位
    print('\n3. 测试转到预置位...')
    for preset_id in [2, 3, 4, 1]:  # 按顺序访问各个预置位
        print(f'  转到预置位 {preset_id}...')
        goto_resp = requests.post(f'{BASE_URL}/api/camera/ptz/preset/goto', json={
            'preset_id': preset_id
        })
        response_data = goto_resp.json()
        message = response_data.get('message', str(response_data))
        print(f'  转到预置位{preset_id}结果: {message}')
        time.sleep(2)  # 等待2秒
    
    # 4. 测试连续动作
    print('\n4. 测试连续动作...')
    
    # 向右转
    print('  向右转...')
    resp = requests.post(f'{BASE_URL}/api/camera/ptz/action', json={
        'action': 'pan_right',
        'speed': 30
    })
    response_data = resp.json()
    message = response_data.get('message', str(response_data))
    print(f'  向右转结果: {message}')
    time.sleep(2)  # 持续2秒
    
    # 停止
    print('  停止...')
    resp = requests.post(f'{BASE_URL}/api/camera/ptz/action', json={
        'action': 'stop'
    })
    response_data = resp.json()
    message = response_data.get('message', str(response_data))
    print(f'  停止结果: {message}')
    
    # 向上转
    print('  向上转...')
    resp = requests.post(f'{BASE_URL}/api/camera/ptz/action', json={
        'action': 'tilt_up',
        'speed': 30
    })
    response_data = resp.json()
    message = response_data.get('message', str(response_data))
    print(f'  向上转结果: {message}')
    time.sleep(2)  # 持续2秒
    
    # 停止
    print('  停止...')
    resp = requests.post(f'{BASE_URL}/api/camera/ptz/action', json={
        'action': 'stop'
    })
    response_data = resp.json()
    message = response_data.get('message', str(response_data))
    print(f'  停止结果: {message}')
    
    # 5. 最后回到位置1
    print('\n5. 回到初始位置...')
    goto_resp = requests.post(f'{BASE_URL}/api/camera/ptz/preset/goto', json={
        'preset_id': 1
    })
    response_data = goto_resp.json()
    message = response_data.get('message', str(response_data))
    print(f'  回到初始位置结果: {message}')
    
    # 6. 最终状态
    print('\n6. 最终PTZ状态:')
    final_status_resp = requests.get(f'{BASE_URL}/api/camera/ptz/status')
    final_status_data = final_status_resp.json()
    print(f'最终状态: {json.dumps(final_status_data, indent=2, ensure_ascii=False)}')
    
    print('\nPTZ云台高级功能测试完成!')

if __name__ == "__main__":
    test_ptz_advanced_features()