#!/usr/bin/env python3
"""
系统集成测试脚本
测试AI平台各组件是否能够正常集成和运行
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试后端模块导入
        from backend.src.api import create_app
        print("  ✅ 后端API模块导入成功")
        
        # 测试模型管理器导入
        from backend.src.core.services.model_manager import ModelManager
        print("  ✅ 模型管理器导入成功")
        
        # 测试推理引擎导入
        from backend.src.core.services.inference_engine import InferenceEngine
        print("  ✅ 推理引擎导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def test_model_manager():
    """测试模型管理器功能"""
    print("\n🔧 测试模型管理器功能...")
    
    try:
        from backend.src.core.services.model_manager import ModelManager
        
        # 创建模型管理器实例
        model_manager = ModelManager()
        print("  ✅ 模型管理器实例创建成功")
        
        # 测试模型统计
        async def test_statistics():
            result = await model_manager.get_model_statistics()
            if result["success"]:
                print("  ✅ 模型统计功能正常")
                return True
            else:
                print(f"  ❌ 模型统计失败: {result['error']}")
                return False
        
        # 运行异步测试
        result = asyncio.run(test_statistics())
        return result
        
    except Exception as e:
        print(f"  ❌ 模型管理器测试失败: {e}")
        return False

def test_api_routes():
    """测试API路由配置"""
    print("\n🌐 测试API路由配置...")
    
    try:
        from backend.src.api import create_app
        
        # 创建应用实例
        app = create_app()
        print("  ✅ FastAPI应用创建成功")
        
        # 检查路由注册
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        print(f"  ✅ 已注册 {len(routes)} 个路由")
        
        # 检查关键路由是否存在
        key_routes = [
            "/api/models",
            "/api/model-manager",
            "/api/ai-control",
            "/api/blockchain"
        ]
        
        missing_routes = []
        for route in key_routes:
            if any(route in r for r in routes):
                print(f"  ✅ 路由 {route} 存在")
            else:
                missing_routes.append(route)
                print(f"  ⚠️  路由 {route} 缺失")
        
        return len(missing_routes) == 0
        
    except Exception as e:
        print(f"  ❌ API路由测试失败: {e}")
        return False

def test_frontend_config():
    """测试前端配置"""
    print("\n🎨 测试前端配置...")
    
    try:
        # 检查前端配置文件
        frontend_config_path = project_root / "frontend" / "src" / "config" / "index.ts"
        
        if frontend_config_path.exists():
            with open(frontend_config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查API基础URL配置
            if 'http://localhost:8000/api' in content:
                print("  ✅ 前端API配置正确 (端口8000)")
            else:
                print("  ⚠️  前端API配置可能需要检查")
                
            print("  ✅ 前端配置文件存在")
            return True
        else:
            print("  ❌ 前端配置文件不存在")
            return False
            
    except Exception as e:
        print(f"  ❌ 前端配置测试失败: {e}")
        return False

def test_project_structure():
    """测试项目结构完整性"""
    print("\n📁 测试项目结构完整性...")
    
    required_dirs = [
        "backend/src",
        "frontend/src", 
        "api-gateway/src",
        "infrastructure"
    ]
    
    required_files = [
        "backend/requirements.txt",
        "frontend/package.json",
        "docker-compose.yml",
        "README.md"
    ]
    
    all_exists = True
    
    # 检查目录
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✅ 目录 {dir_path} 存在")
        else:
            print(f"  ❌ 目录 {dir_path} 缺失")
            all_exists = False
    
    # 检查文件
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ 文件 {file_path} 存在")
        else:
            print(f"  ❌ 文件 {file_path} 缺失")
            all_exists = False
    
    return all_exists

def main():
    """主测试函数"""
    print("🚀 AI平台系统集成测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("项目结构", test_project_structure),
        ("模块导入", test_imports),
        ("模型管理器", test_model_manager),
        ("API路由", test_api_routes),
        ("前端配置", test_frontend_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("📊 测试结果摘要:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统集成正常")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())