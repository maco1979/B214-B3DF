#!/usr/bin/env python3
"""
测试基本的HTTP GET请求
"""

import http.client
import time

def test_http_get():
    """
    测试基本的HTTP GET请求
    """
    print("=== 测试基本的HTTP GET请求 ===")
    
    # 使用http.client模块进行简单的HTTP请求
    print("\n1. 测试HTTP GET请求...")
    try:
        conn = http.client.HTTPConnection("localhost", 8001, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        print(f"   状态码: {response.status}")
        print(f"   原因: {response.reason}")
        data = response.read()
        print(f"   响应数据: {data.decode('utf-8')}")
        conn.close()
        print("   ✅ HTTP GET请求成功")
        return True
    except Exception as e:
        print(f"   HTTP GET请求失败: {e}")
        return False

if __name__ == "__main__":
    test_http_get()
