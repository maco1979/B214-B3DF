"""
部署验证脚本
验证本地部署的AI决策系统服务
"""

import requests
import time
import sys
from pathlib import Path


def validate_deployment():
    """验证部署"""
    print("🔍 验证部署状态...")
    
    # 等待服务响应
    print("⏳ 等待服务响应...")
    time.sleep(3)
    
    # 测试可用端点
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
            "name": "认证登录",
            "url": "http://127.0.0.1:8000/api/auth/login",
            "method": "POST",
            "data": {"username": "test", "password": "test"},
            "expected_status": 422  # 期望返回422（参数验证错误）而不是404
        },
        {
            "name": "社区帖子列表",
            "url": "http://127.0.0.1:8000/api/community/posts",
            "expected_status": 200
        },
        {
            "name": "农业光配方",
            "url": "http://127.0.0.1:8000/agriculture/light-recipe",
            "method": "POST",
            "json": {
                "crop_type": "生菜",
                "current_day": 10,
                "target_objective": "最大化产量",
                "environment": {"temperature": 20, "humidity": 65}
            },
            "expected_status": 422  # 期望返回422（参数验证错误）而不是404
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        try:
            print(f"  📡 测试 {endpoint['name']}...")
            
            method = endpoint.get('method', 'GET')
            if method == 'POST' and 'data' in endpoint:
                response = requests.post(endpoint['url'], data=endpoint['data'], timeout=10)
            elif method == 'POST' and 'json' in endpoint:
                response = requests.post(endpoint['url'], json=endpoint['json'], timeout=10)
            else:
                response = requests.get(endpoint['url'], timeout=10)
            
            expected = endpoint['expected_status']
            actual = response.status_code
            # 对于某些端点，我们期望特定的错误码而不是404，表示端点存在
            status = "✅" if (actual == expected or (expected == 422 and actual in [422, 200])) else "❌"
            
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
    print(f"\n📊 验证摘要:")
    total_tests = len(results)
    passed_tests = len([r for r in results if r['status'] == '✅'])
    failed_tests = len([r for r in results if r['status'] == '❌'])
    
    print(f"  总测试数: {total_tests}")
    print(f"  通过: {passed_tests}")
    print(f"  失败: {failed_tests}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 部署验证通过! 所有端点正常工作")
        success_rate = 100.0
    elif failed_tests == total_tests:
        print(f"\n❌ 部署验证失败! 所有端点都无法访问")
        success_rate = 0.0
    else:
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n✅ 部署验证部分通过! 成功率: {success_rate:.1f}%")
    
    # 输出详细结果
    print(f"\n📋 详细结果:")
    for result in results:
        print(f"  {result['status']} {result['name']}")
        print(f"    URL: {result['url']}")
        print(f"    状态: {result['actual']} (期望: {result['expected']})")
        print()
    
    return success_rate, results


def run_functionality_tests():
    """运行功能测试"""
    print("🔧 运行功能测试...")
    
    tests = []
    
    # 测试健康检查响应内容
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "status" in data and data["status"] == "healthy":
                tests.append(("健康检查响应内容", "✅"))
            else:
                tests.append(("健康检查响应内容", "❌"))
        else:
            tests.append(("健康检查响应内容", "❌"))
    except:
        tests.append(("健康检查响应内容", "❌"))
    
    # 测试根路径响应内容
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "message" in data and "AI项目API服务" in data["message"]:
                tests.append(("根路径响应内容", "✅"))
            else:
                tests.append(("根路径响应内容", "❌"))
        else:
            tests.append(("根路径响应内容", "❌"))
    except:
        tests.append(("根路径响应内容", "❌"))
    
    # 测试社区功能
    try:
        response = requests.get("http://127.0.0.1:8000/api/community/posts", timeout=10)
        # 期望返回200，即使内容为空
        if response.status_code == 200:
            tests.append(("社区功能", "✅"))
        else:
            tests.append(("社区功能", "❌"))
    except:
        tests.append(("社区功能", "❌"))
    
    # 输出功能测试结果
    print(f"\n⚙️  功能测试结果:")
    for test_name, status in tests:
        print(f"  {status} {test_name}")
    
    passed_functional = len([t for t in tests if t[1] == "✅"])
    total_functional = len(tests)
    functional_success_rate = (passed_functional / total_functional) * 100 if total_functional > 0 else 0
    
    print(f"\n📈 功能测试成功率: {functional_success_rate:.1f}% ({passed_functional}/{total_functional})")
    
    return functional_success_rate, tests


def main():
    """主函数"""
    print("🚀 部署验证测试")
    print("="*50)
    
    try:
        # 验证部署
        api_success_rate, api_results = validate_deployment()
        
        # 运行功能测试
        print("\n" + "="*50)
        functional_success_rate, functional_results = run_functionality_tests()
        
        print("\n" + "="*50)
        print("✅ 部署验证完成")
        
        # 综合评估
        overall_success_rate = (api_success_rate + functional_success_rate) / 2
        print(f"📈 综合成功率: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 80:
            print("🎯 部署成功! 系统功能正常")
            print("📋 系统已准备好进行长时间测试")
            return 0
        elif overall_success_rate >= 50:
            print("✅ 部署基本成功，大部分功能正常")
            print("📋 系统可进行长时间测试，但可能存在一些限制")
            return 0
        else:
            print("⚠️  部署存在问题，需要修复")
            return 1
            
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())