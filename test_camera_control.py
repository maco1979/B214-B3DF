#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
摄像头控制功能测试脚本
用于测试AI主控页面的摄像头控制功能
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"
AI_CONTROL_PREFIX = "/ai-control"

print("=== 摄像头控制功能测试 ===")

# 1. 获取设备列表
print("\n1. 获取设备列表...")
response = requests.get(f"{BASE_URL}{AI_CONTROL_PREFIX}/devices")
if response.status_code != 200:
    print(f"   [错误] 获取设备列表失败: {response.status_code}")
    exit(1)

devices = response.json()
camera_device = None

print("   [成功] 设备列表获取成功")
print("   设备列表:")
for device in devices:
    print(f"     ID: {device['id']}, 名称: {device['name']}, 类型: {device['type']}, 状态: {device['status']}, 连接: {device['connected']}")
    if device['type'] == '摄像头':
        camera_device = device

if not camera_device:
    print("   [错误] 未找到摄像头设备")
    exit(1)

camera_id = camera_device['id']
print(f"\n   找到摄像头设备: ID={camera_id}, 名称={camera_device['name']}")

# 2. 激活AI主控
print("\n2. 激活AI主控...")
response = requests.post(
    f"{BASE_URL}{AI_CONTROL_PREFIX}/master-control",
    json={"activate": True}
)

if response.status_code != 200:
    print(f"   [错误] 激活AI主控失败: {response.status_code}")
    exit(1)

master_control_status = response.json()
print(f"   [成功] AI主控已{'' if master_control_status['master_control_active'] else '未'}激活")

# 3. 连接摄像头设备
if not camera_device['connected']:
    print(f"\n3. 连接摄像头设备 {camera_id}...")
    response = requests.post(
        f"{BASE_URL}{AI_CONTROL_PREFIX}/device/{camera_id}/connection",
        json={"connect": True}
    )

    if response.status_code != 200:
        print(f"   [错误] 连接摄像头设备失败: {response.status_code}")
        exit(1)

    connection_status = response.json()
    print(f"   [成功] 摄像头设备已{'' if connection_status['connected'] else '未'}连接")
else:
    print(f"\n3. 摄像头设备 {camera_id} 已连接")

# 4. 测试摄像头控制功能
print(f"\n4. 测试摄像头控制功能...")

# 4.1 获取摄像头状态
print("\n   4.1 获取摄像头状态...")
response = requests.post(
    f"{BASE_URL}{AI_CONTROL_PREFIX}/device/{camera_id}",
    json={"action": "get_status"}
)

if response.status_code != 200:
    print(f"   [错误] 获取摄像头状态失败: {response.status_code}")
else:
    status = response.json()
    print(f"   [成功] 摄像头状态: {status}")

# 4.2 打开摄像头
print("\n   4.2 打开摄像头...")
response = requests.post(
    f"{BASE_URL}{AI_CONTROL_PREFIX}/device/{camera_id}",
    json={"action": "open"}
)

if response.status_code != 200:
    print(f"   [错误] 打开摄像头失败: {response.status_code}")
    print(f"   响应: {response.json()}")
else:
    open_result = response.json()
    print(f"   [成功] 打开摄像头: {open_result}")

# 4.3 拍照
print("\n   4.3 拍照...")
time.sleep(1)  # 等待摄像头初始化
response = requests.post(
    f"{BASE_URL}{AI_CONTROL_PREFIX}/device/{camera_id}",
    json={"action": "take_photo"}
)

if response.status_code != 200:
    print(f"   [错误] 拍照失败: {response.status_code}")
    print(f"   响应: {response.json()}")
else:
    photo_result = response.json()
    print(f"   [成功] 拍照: {photo_result}")

# 4.4 列出可用摄像头
print("\n   4.4 列出可用摄像头...")
response = requests.post(
    f"{BASE_URL}{AI_CONTROL_PREFIX}/device/{camera_id}",
    json={"action": "list_cameras"}
)

if response.status_code != 200:
    print(f"   [错误] 列出可用摄像头失败: {response.status_code}")
else:
    list_result = response.json()
    print(f"   [成功] 可用摄像头列表: {list_result}")

# 4.5 关闭摄像头
print("\n   4.5 关闭摄像头...")
response = requests.post(
    f"{BASE_URL}{AI_CONTROL_PREFIX}/device/{camera_id}",
    json={"action": "close"}
)

if response.status_code != 200:
    print(f"   [错误] 关闭摄像头失败: {response.status_code}")
else:
    close_result = response.json()
    print(f"   [成功] 关闭摄像头: {close_result}")

print("\n=== 摄像头控制功能测试完成 ===")
