#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
摄像头测试脚本
用于测试本地摄像头的连接和基本功能
"""

import cv2
import os
import sys

def test_camera():
    """
    测试本地摄像头连接
    """
    print("=== 开始摄像头测试 ===")
    
    # 尝试检测可用摄像头
    print("1. 检测可用摄像头...")
    available_cameras = []
    
    # 尝试检测前5个摄像头索引
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 获取摄像头基本信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available_cameras.append({
                "index": i,
                "width": width,
                "height": height,
                "fps": fps
            })
            cap.release()
    
    if not available_cameras:
        print("   [错误] 未检测到可用摄像头")
        return False
    
    print(f"   [成功] 检测到 {len(available_cameras)} 个可用摄像头:")
    for cam in available_cameras:
        print(f"     摄像头 {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps")
    
    # 尝试打开第一个可用摄像头
    print("\n2. 测试摄像头连接...")
    cam_index = available_cameras[0]['index']
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"   [错误] 无法打开摄像头 {cam_index}")
        return False
    
    print(f"   [成功] 摄像头 {cam_index} 已成功打开")
    
    # 测试获取视频帧
    print("\n3. 测试视频帧获取...")
    ret, frame = cap.read()
    
    if not ret:
        print("   [错误] 无法获取视频帧")
        cap.release()
        return False
    
    print("   [成功] 视频帧获取成功")
    print(f"   帧大小: {frame.shape[1]}x{frame.shape[0]}")
    
    # 保存测试照片
    output_dir = "camera_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_photo_path = os.path.join(output_dir, "test_photo.jpg")
    cv2.imwrite(test_photo_path, frame)
    print(f"   [成功] 测试照片已保存至: {test_photo_path}")
    
    # 释放摄像头
    cap.release()
    
    print("\n=== 摄像头测试完成 ===")
    print("   摄像头连接正常，可以使用！")
    
    return True

def main():
    """
    主函数
    """
    try:
        success = test_camera()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[错误] 测试过程中发生异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
