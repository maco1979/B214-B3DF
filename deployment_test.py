#!/usr/bin/env python3
"""
AI平台部署测试脚本
验证系统部署状态和功能
"""

import requests
import time
import sys

def test_backend_service():
    """测试后端服务状态"""
    print("=== AI平台部署测试 ===")
    print("正在测试后端服务连接...")
    
    try:
        # 等待服务完全启动
        time.sleep(3)
        
        # 测试健康检查接口
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            print("✅ 后端服务健康检查正常")
            print(f"响应内容: {response.text}")
            return True
        else:
            print(f"⚠️  后端服务响应状态: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 后端服务未启动或端口被占用")
        print("请检查后端服务是否正在运行")
        return False
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

def test_api_docs():
    """测试API文档接口"""
    try:
        response = requests.get('http://localhost:8000/docs', timeout=10)
        if response.status_code == 200:
            print("✅ API文档接口可访问")
            return True
        else:
            print(f"⚠️  API文档响应状态: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API文档测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n🔧 开始系统部署测试...")
    
    # 测试后端服务
    backend_ok = test_backend_service()
    
    # 测试API文档
    docs_ok = test_api_docs()
    
    print("\n=== 部署状态总结 ===")
    if backend_ok and docs_ok:
        print("🎉 AI平台部署成功！")
        print("✅ 后端API服务正常运行")
        print("✅ API文档可访问")
        print("✅ 系统核心功能就绪")
    else:
        print("⚠️  部分服务需要进一步配置")
    
    print("\n🌐 可访问的服务链接:")
    print("1. 后端API: http://localhost:8000")
    print("2. API文档: http://localhost:8000/docs")
    print("3. 健康检查: http://localhost:8000/health")
    
    print("\n🚀 下一步操作:")
    print("1. 保持后端服务运行（当前终端窗口）")
    print("2. 新终端启动前端: cd frontend && npm run dev")
    print("3. 访问 http://localhost:8000/docs 查看完整API")
    print("4. 前端启动后访问 http://localhost:3000")
    
    return backend_ok and docs_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)