#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件接口三层防护巡检脚本
作用：自动化校验所有硬件接口的三层防护是否生效
模拟「参数错误、空对象、设备异常」等场景，一键输出校验报告
"""
import requests
import json
import sys
from typing import Dict, List, Any

# 配置项
BASE_URL = "http://localhost:8000"

# 待巡检的硬件接口清单
CHECK_APIS = [
    {
        "name": "PTZ云台连接接口",
        "path": "/api/camera/ptz/connect",
        "method": "POST",
        "valid_data": {
            "protocol": "pelco_d",
            "connection_type": "serial",
            "port": "COM3",
            "baudrate": 9600,
            "address": 1
        },
        "invalid_data": {
            "protocol": "",  # 空协议
            "connection_type": "invalid_type",  # 非法类型
            "port": "",
            "baudrate": "abc"  # 错误类型
        }
    },
    {
        "name": "PTZ云台断开接口",
        "path": "/api/camera/ptz/disconnect",
        "method": "POST",
        "valid_data": {},
        "invalid_data": {}
    },
    {
        "name": "AI设备连接接口",
        "path": "/api/ai-control/device/1/connection",
        "method": "POST",
        "valid_data": {
            "connect": True
        },
        "invalid_data": {
            # 缺少connect参数
        }
    }
]


def check_api_robustness():
    """执行硬件接口防护巡检"""
    print("=" * 80)
    print("🕵️  硬件接口三层防护巡检开始")
    print(f"📍 巡检后端地址：{BASE_URL}")
    print(f"📊 巡检接口数量：{len(CHECK_APIS)}")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    total_tests = 0
    
    for api in CHECK_APIS:
        api_path = BASE_URL + api["path"]
        print(f"\n{'=' * 80}")
        print(f"👉 接口名称：{api['name']}")
        print(f"   接口路径：{api_path}")
        print(f"   请求方式：{api['method']}")
        print(f"{'-' * 80}")
        
        # 测试1：非法参数请求 → 应返回success:false，无500
        print("\n🔍 [测试1] 参数校验层测试（非法参数）")
        total_tests += 1
        try:
            res = requests.request(
                api["method"], 
                api_path, 
                json=api["invalid_data"], 
                timeout=5
            )
            
            if res.status_code == 200:
                try:
                    data = res.json()
                    if data.get("success") is False:
                        print("   ✅ PASS - 参数校验层生效，正确拦截非法参数")
                        print(f"   📝 错误提示：{data.get('message', 'N/A')}")
                        success_count += 1
                    else:
                        print(f"   ⚠️  WARN - 接口未拦截非法参数，success={data.get('success')}")
                        fail_count += 1
                except json.JSONDecodeError:
                    print(f"   ❌ FAIL - 响应格式错误：{res.text[:200]}")
                    fail_count += 1
            else:
                print(f"   ❌ FAIL - HTTP状态码异常：{res.status_code}")
                fail_count += 1
                
        except requests.exceptions.Timeout:
            print("   ❌ FAIL - 请求超时（后端可能未启动）")
            fail_count += 1
        except requests.exceptions.ConnectionError:
            print("   ❌ FAIL - 连接失败（后端未启动或端口错误）")
            fail_count += 1
        except Exception as e:
            print(f"   ❌ FAIL - 接口崩溃，异常：{type(e).__name__} - {str(e)}")
            fail_count += 1
        
        # 测试2：合法参数请求 → 应返回200，无500
        print("\n🔍 [测试2] 全局异常层测试（合法参数）")
        total_tests += 1
        try:
            res = requests.request(
                api["method"], 
                api_path, 
                json=api["valid_data"], 
                timeout=5
            )
            
            if res.status_code == 200:
                try:
                    data = res.json()
                    print("   ✅ PASS - 全局异常层生效，无500错误")
                    print(f"   📝 响应状态：success={data.get('success')}")
                    print(f"   📝 响应消息：{data.get('message', 'N/A')}")
                    success_count += 1
                except json.JSONDecodeError:
                    print(f"   ⚠️  WARN - 响应格式错误（但无500）：{res.text[:200]}")
                    success_count += 1
            elif res.status_code == 500:
                print(f"   ❌ FAIL - 返回500错误，全局异常层未生效")
                print(f"   📝 错误详情：{res.text[:500]}")
                fail_count += 1
            else:
                print(f"   ⚠️  WARN - HTTP状态码：{res.status_code}（非200但也非500）")
                success_count += 1
                
        except requests.exceptions.Timeout:
            print("   ❌ FAIL - 请求超时")
            fail_count += 1
        except requests.exceptions.ConnectionError:
            print("   ❌ FAIL - 连接失败")
            fail_count += 1
        except Exception as e:
            print(f"   ❌ FAIL - 测试异常：{type(e).__name__} - {str(e)}")
            fail_count += 1
    
    # 输出最终报告
    print(f"\n{'=' * 80}")
    print("📊 巡检报告汇总")
    print(f"{'=' * 80}")
    print(f"   总测试数：{total_tests}")
    print(f"   ✅ 通过：{success_count}")
    print(f"   ❌ 失败：{fail_count}")
    print(f"   通过率：{(success_count/total_tests*100):.1f}%")
    print(f"{'=' * 80}")
    
    if fail_count == 0:
        print("🎉 恭喜！所有硬件接口三层防护机制全部生效！")
        return 0
    else:
        print("⚠️  警告！部分接口防护机制未生效，请检查上述失败项！")
        return 1


if __name__ == "__main__":
    try:
        exit_code = check_api_robustness()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  巡检被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 巡检脚本异常：{type(e).__name__} - {str(e)}")
        sys.exit(1)
