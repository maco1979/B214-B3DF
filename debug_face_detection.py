#!/usr/bin/env python3
"""
人脸检测调试脚本
测试Haar分类器是否能检测到我们绘制的模拟人脸
"""

import cv2
import numpy as np
import time

# 加载Haar人脸检测器
def test_face_detection():
    """测试人脸检测"""
    print("=== 人脸检测调试 ===")
    
    # 1. 加载Haar分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ 无法加载Haar级联模型")
        return
    
    print("✅ 成功加载Haar级联模型")
    
    # 2. 生成模拟帧，与摄像头控制器中的代码完全相同
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 在模拟帧上绘制一些简单图形
    cv2.putText(frame, 'Simulated Camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Time: {time.strftime("%H:%M:%S")}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (200, 150), (440, 330), (0, 255, 255), 2)
    
    # 添加模拟人脸，让人脸识别能检测到（绘制一个肤色椭圆作为人脸）
    # 肤色范围（BGR）
    face_color = (64, 144, 190)  # 肤色的BGR值
    # 绘制椭圆作为人脸
    cv2.ellipse(frame, (320, 200), (60, 80), 0, 0, 360, face_color, -1)
    # 绘制眼睛
    cv2.circle(frame, (300, 180), 10, (0, 0, 0), -1)  # 左眼
    cv2.circle(frame, (340, 180), 10, (0, 0, 0), -1)  # 右眼
    # 绘制嘴巴
    cv2.ellipse(frame, (320, 220), (30, 20), 0, 0, 180, (0, 0, 0), 3)  # 嘴巴
    
    # 3. 测试人脸检测
    print("\n=== 测试人脸检测 ===")
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 保存灰度图用于调试
    cv2.imwrite('debug_gray.png', gray)
    print("✅ 保存灰度图到 debug_gray.png")
    
    # 测试不同的参数组合
    for scale_factor in [1.05, 1.1, 1.2]:
        for min_neighbors in [3, 5, 7]:
            print(f"\n测试参数: scale_factor={scale_factor}, min_neighbors={min_neighbors}")
            
            # 检测人脸
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors, 
                minSize=(30, 30)
            )
            
            print(f"检测到 {len(faces)} 个人脸")
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    print(f"  人脸位置: x={x}, y={y}, w={w}, h={h}")
                    # 在帧上绘制检测框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Face ({scale_factor}, {min_neighbors})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 保存结果图像
    cv2.imwrite('debug_face_detection.png', frame)
    print("\n✅ 保存检测结果到 debug_face_detection.png")
    
    # 4. 测试一个更简单的人脸模型
    print("\n=== 测试更简单的人脸模型 ===")
    simple_face_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 绘制一个简单的矩形作为人脸
    cv2.rectangle(simple_face_frame, (290, 160), (350, 240), (255, 255, 255), -1)
    
    simple_gray = cv2.cvtColor(simple_face_frame, cv2.COLOR_BGR2GRAY)
    simple_faces = face_cascade.detectMultiScale(
        simple_gray, 
        scaleFactor=1.1, 
        minNeighbors=3, 
        minSize=(30, 30)
    )
    
    print(f"简单人脸模型检测到 {len(simple_faces)} 个人脸")
    for (x, y, w, h) in simple_faces:
        print(f"  人脸位置: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(simple_face_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imwrite('debug_simple_face.png', simple_face_frame)
    print("✅ 保存简单人脸检测结果到 debug_simple_face.png")
    
    # 5. 结论
    print("\n=== 结论 ===")
    print("Haar分类器可能无法识别我们绘制的简单人脸图形")
    print("建议：")
    print("1. 使用真实人脸图像进行测试")
    print("2. 或者修改摄像头控制器，直接在代码中模拟人脸检测结果")

if __name__ == "__main__":
    test_face_detection()
