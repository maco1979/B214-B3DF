#!/usr/bin/env python3
"""
模型管理器功能测试脚本
测试模型管理器的各项功能是否正常工作
"""

import asyncio
import sys
import os

# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_model_manager():
    """测试模型管理器功能"""
    try:
        # 导入模型管理器
        from src.core.services.model_manager import ModelManager
        
        async def run_tests():
            print("🚀 开始测试模型管理器功能...")
            
            # 创建模型管理器实例
            manager = ModelManager()
            
            # 1. 测试初始化
            print("\n1. 测试模型管理器初始化...")
            init_result = await manager.initialize()
            if init_result["success"]:
                print("✅ 模型管理器初始化成功")
            else:
                print(f"❌ 模型管理器初始化失败: {init_result['error']}")
                return False
            
            # 2. 测试注册模型
            print("\n2. 测试模型注册功能...")
            model_data = {
                "name": "测试模型",
                "type": "classification",
                "framework": "pytorch",
                "version": "1.0.0",
                "metadata": {"description": "这是一个测试模型"}
            }
            register_result = await manager.register_model("test_model_v1", model_data)
            if register_result["success"]:
                print("✅ 模型注册成功")
                print(f"   模型ID: {register_result['model_id']}")
            else:
                print(f"❌ 模型注册失败: {register_result['error']}")
                return False
            
            # 3. 测试加载模型
            print("\n3. 测试模型加载功能...")
            load_result = await manager.load_model("test_model_v1")
            if load_result["success"]:
                print("✅ 模型加载成功")
                print(f"   是否从缓存加载: {load_result['from_cache']}")
            else:
                print(f"❌ 模型加载失败: {load_result['error']}")
                return False
            
            # 4. 测试模型预测
            print("\n4. 测试模型预测功能...")
            input_data = {"features": {"temperature": 26, "humidity": 65}}
            predict_result = await manager.predict("test_model_v1", input_data)
            if predict_result["success"]:
                print("✅ 模型预测成功")
                print(f"   预测结果: {predict_result['prediction']}")
            else:
                print(f"❌ 模型预测失败: {predict_result['error']}")
                return False
            
            # 5. 测试获取模型列表
            print("\n5. 测试获取模型列表...")
            list_result = await manager.list_models()
            if list_result["success"]:
                print("✅ 获取模型列表成功")
                print(f"   模型总数: {list_result['total_count']}")
                print(f"   已加载模型数: {list_result['loaded_count']}")
            else:
                print(f"❌ 获取模型列表失败: {list_result['error']}")
                return False
            
            # 6. 测试获取统计信息
            print("\n6. 测试获取统计信息...")
            stats_result = await manager.get_model_statistics()
            if stats_result["success"]:
                print("✅ 获取统计信息成功")
                stats = stats_result["statistics"]
                print(f"   总模型数: {stats['total_models']}")
                print(f"   按类型统计: {stats['by_type']}")
                print(f"   按状态统计: {stats['by_status']}")
            else:
                print(f"❌ 获取统计信息失败: {stats_result['error']}")
                return False
            
            # 7. 测试搜索模型
            print("\n7. 测试搜索模型功能...")
            search_result = await manager.search_models("测试", {"type": "classification"})
            if search_result["success"]:
                print("✅ 搜索模型成功")
                print(f"   搜索结果数量: {search_result['total_count']}")
            else:
                print(f"❌ 搜索模型失败: {search_result['error']}")
                return False
            
            # 8. 测试模型导出
            print("\n8. 测试模型导出功能...")
            export_result = await manager.export_model("test_model_v1", "onnx")
            if export_result["success"]:
                print("✅ 模型导出成功")
                print(f"   导出格式: {export_result['export_info']['format']}")
                print(f"   导出路径: {export_result['export_info']['export_path']}")
            else:
                print(f"❌ 模型导出失败: {export_result['error']}")
                return False
            
            # 9. 测试备份功能
            print("\n9. 测试元数据备份功能...")
            backup_result = await manager.backup_model_metadata()
            if backup_result["success"]:
                print("✅ 元数据备份成功")
                print(f"   备份路径: {backup_result['backup_info']['backup_path']}")
                print(f"   备份模型数: {backup_result['backup_info']['total_models']}")
            else:
                print(f"❌ 元数据备份失败: {backup_result['error']}")
                return False
            
            # 10. 测试清理资源
            print("\n10. 测试资源清理...")
            await manager.close()
            print("✅ 资源清理成功")
            
            print("\n🎉 所有测试通过！模型管理器功能正常。")
            return True
        
        # 运行异步测试
        return asyncio.run(run_tests())
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_integration():
    """测试API集成"""
    try:
        print("\n🔗 测试API集成...")
        
        # 检查API路由文件是否存在
        api_routes_path = os.path.join("backend", "src", "api", "routes", "model_manager.py")
        if os.path.exists(api_routes_path):
            print("✅ 模型管理器API路由文件存在")
        else:
            print("❌ 模型管理器API路由文件不存在")
            return False
        
        # 检查API文件是否包含模型管理器路由
        api_file_path = os.path.join("backend", "src", "api.py")
        if os.path.exists(api_file_path):
            with open(api_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "model_manager" in content:
                    print("✅ API文件已集成模型管理器路由")
                else:
                    print("❌ API文件未集成模型管理器路由")
                    return False
        else:
            print("❌ API文件不存在")
            return False
        
        print("✅ API集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ API集成测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🧪 AI平台模型管理器功能测试")
    print("=" * 50)
    
    # 运行模型管理器功能测试
    manager_test_passed = test_model_manager()
    
    # 运行API集成测试
    api_test_passed = test_api_integration()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"   模型管理器功能测试: {'✅ 通过' if manager_test_passed else '❌ 失败'}")
    print(f"   API集成测试: {'✅ 通过' if api_test_passed else '❌ 失败'}")
    
    if manager_test_passed and api_test_passed:
        print("\n🎉 所有测试通过！模型管理器服务已成功创建并集成。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查相关代码。")
        return 1

if __name__ == "__main__":
    sys.exit(main())