"""
部署验证测试脚本
验证服务是否正确部署和运行
"""

import requests
import time
import sys
from pathlib import Path


def test_deployment():
    """测试部署"""
    print("🔍 测试部署状态...")
    
    # 等待服务启动
    print("⏳ 等待服务启动...")
    time.sleep(5)
    
    # 测试端点
    endpoints = [
        {
            "name": "健康检查",
            "url": "http://127.0.0.1:8000/health",
            "expected_status": 200
        },
        {
            "name": "根路径",
            "url": "http://127.0.0.1:8000/",
            "expected_status": 200
        },
        {
            "name": "API文档",
            "url": "http://127.0.0.1:8000/docs",
            "expected_status": 200
        },
        {
            "name": "模型列表",
            "url": "http://127.0.0.1:8000/api/models",
            "expected_status": 200
        },
        {
            "name": "用户服务",
            "url": "http://127.0.0.1:8000/api/user/test",
            "expected_status": 404  # 预期返回404，因为用户不存在
        },
        {
            "name": "企业服务",
            "url": "http://127.0.0.1:8000/api/enterprise/test",
            "expected_status": 404  # 预期返回404，因为企业不存在
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        try:
            print(f"  📡 测试 {endpoint['name']}...")
            response = requests.get(endpoint['url'], timeout=10)
            
            expected = endpoint['expected_status']
            actual = response.status_code
            status = "✅" if actual == expected else "❌"
            
            result = {
                "name": endpoint['name'],
                "url": endpoint['url'],
                "expected": expected,
                "actual": actual,
                "status": status
            }
            
            results.append(result)
            print(f"    {status} {endpoint['name']}: {actual} (期望: {expected})")
            
        except requests.exceptions.ConnectionError:
            result = {
                "name": endpoint['name'],
                "url": endpoint['url'],
                "expected": endpoint['expected_status'],
                "actual": "CONNECTION_ERROR",
                "status": "❌"
            }
            results.append(result)
            print(f"    ❌ {endpoint['name']}: 连接错误")
        except requests.exceptions.Timeout:
            result = {
                "name": endpoint['name'],
                "url": endpoint['url'],
                "expected": endpoint['expected_status'],
                "actual": "TIMEOUT",
                "status": "❌"
            }
            results.append(result)
            print(f"    ❌ {endpoint['name']}: 请求超时")
        except Exception as e:
            result = {
                "name": endpoint['name'],
                "url": endpoint['url'],
                "expected": endpoint['expected_status'],
                "actual": f"ERROR: {str(e)}",
                "status": "❌"
            }
            results.append(result)
            print(f"    ❌ {endpoint['name']}: {str(e)}")
    
    # 输出测试摘要
    print(f"\n📊 测试摘要:")
    total_tests = len(results)
    passed_tests = len([r for r in results if r['status'] == '✅'])
    failed_tests = len([r for r in results if r['status'] == '❌'])
    
    print(f"  总测试数: {total_tests}")
    print(f"  通过: {passed_tests}")
    print(f"  失败: {failed_tests}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 部署测试通过! 所有端点正常工作")
        success_rate = 100.0
    elif failed_tests == total_tests:
        print(f"\n❌ 部署测试失败! 所有端点都无法访问")
        success_rate = 0.0
    else:
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n⚠️  部署测试部分通过! 成功率: {success_rate:.1f}%")
    
    # 输出详细结果
    print(f"\n📋 详细结果:")
    for result in results:
        print(f"  {result['status']} {result['name']}")
        print(f"    URL: {result['url']}")
        print(f"    状态: {result['actual']} (期望: {result['expected']})")
        print()
    
    return success_rate, results


def main():
    """主函数"""
    print("🚀 部署验证测试")
    print("="*50)
    
    try:
        success_rate, results = test_deployment()
        
        print("="*50)
        print(f"✅ 部署验证完成")
        print(f"📈 成功率: {success_rate:.1f}%")
        
        # 根据结果返回适当的退出码
        if success_rate == 100:
            print("🎯 部署完全成功!")
            return 0
        elif success_rate >= 80:
            print("✅ 部署基本成功，大部分功能正常")
            return 0
        else:
            print("⚠️  部署存在问题，需要检查")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())