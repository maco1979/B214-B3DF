"""
安全修复验证测试脚本
验证SQL注入、XSS攻击防护是否生效
"""

import asyncio
import aiohttp
import urllib.parse
import json
from datetime import datetime
from typing import Dict, Any, List

# 测试配置
BASE_URL = "http://127.0.0.1:8000"

# SQL注入攻击载荷
SQL_INJECTION_PAYLOADS = [
    # 删表攻击
    "'; DROP TABLE users; --",
    # 永真条件
    "1 OR 1=1",
    "1' OR '1'='1",
    # 联合查询
    "' UNION SELECT * FROM users --",
    # 注释绕过
    "admin'--",
    # 多语句
    "1; SELECT * FROM users",
    # 常见变体
    "1/**/OR/**/1=1",
    "' OR ''='",
]

# XSS攻击载荷
XSS_PAYLOADS = [
    # 基础脚本
    "<script>alert('xss')</script>",
    # 图片劫持
    "<img src=x onerror=alert('xss')>",
    # 伪协议
    "javascript:alert('xss')",
    # SVG注入
    "<svg onload=alert('xss')>",
    # 闭合标签
    "'><script>alert('xss')</script>",
    # 事件处理器
    "' onclick=alert('xss')",
]

# 路径遍历载荷
PATH_TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "%2e%2e%2f%2e%2e%2f",
]


class SecurityTester:
    """安全测试器"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {
            "sql_injection": [],
            "xss": [],
            "path_traversal": [],
            "security_headers": [],
        }
        self.start_time = None
    
    async def test_sql_injection_path_param(self, session: aiohttp.ClientSession):
        """测试路径参数SQL注入防护"""
        print("\n📌 测试路径参数SQL注入防护...")
        
        for payload in SQL_INJECTION_PAYLOADS:
            encoded_payload = urllib.parse.quote(payload, safe='')
            url = f"{self.base_url}/api/models/{encoded_payload}"
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    status = resp.status
                    # 如果返回400，说明攻击被拦截
                    # 如果返回404或500，可能存在漏洞
                    blocked = status == 400
                    
                    self.results["sql_injection"].append({
                        "type": "path_param",
                        "payload": payload[:30] + "..." if len(payload) > 30 else payload,
                        "status_code": status,
                        "blocked": blocked,
                        "secure": blocked or status == 404,  # 400或404都是安全的
                    })
                    
                    status_icon = "✅" if blocked else ("⚠️" if status == 404 else "❌")
                    print(f"  {status_icon} 路径参数: {payload[:20]}... -> {status}")
            except Exception as e:
                print(f"  ⚠️ 请求失败: {str(e)[:30]}")
    
    async def test_sql_injection_query_param(self, session: aiohttp.ClientSession):
        """测试查询参数SQL注入防护"""
        print("\n📌 测试查询参数SQL注入防护...")
        
        for payload in SQL_INJECTION_PAYLOADS:
            url = f"{self.base_url}/api/models/"
            params = {"search": payload}
            
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    status = resp.status
                    blocked = status == 400
                    
                    self.results["sql_injection"].append({
                        "type": "query_param",
                        "payload": payload[:30] + "..." if len(payload) > 30 else payload,
                        "status_code": status,
                        "blocked": blocked,
                        "secure": blocked,
                    })
                    
                    status_icon = "✅" if blocked else ("⚠️" if status == 200 else "❌")
                    print(f"  {status_icon} 查询参数: {payload[:20]}... -> {status}")
            except Exception as e:
                print(f"  ⚠️ 请求失败: {str(e)[:30]}")
    
    async def test_xss_attack(self, session: aiohttp.ClientSession):
        """测试XSS攻击防护"""
        print("\n📌 测试XSS攻击防护...")
        
        for payload in XSS_PAYLOADS:
            # 测试查询参数XSS
            url = f"{self.base_url}/api/monitoring/agent/action"
            params = {"agent_id": payload}
            
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    status = resp.status
                    blocked = status == 400
                    
                    self.results["xss"].append({
                        "type": "query_param",
                        "payload": payload[:30] + "..." if len(payload) > 30 else payload,
                        "status_code": status,
                        "blocked": blocked,
                        "secure": blocked or status == 404,
                    })
                    
                    status_icon = "✅" if blocked else ("⚠️" if status == 404 else "❌")
                    print(f"  {status_icon} XSS载荷: {payload[:20]}... -> {status}")
            except Exception as e:
                print(f"  ⚠️ 请求失败: {str(e)[:30]}")
    
    async def test_path_traversal(self, session: aiohttp.ClientSession):
        """测试路径遍历攻击防护"""
        print("\n📌 测试路径遍历攻击防护...")
        
        for payload in PATH_TRAVERSAL_PAYLOADS:
            encoded_payload = urllib.parse.quote(payload, safe='')
            url = f"{self.base_url}/api/models/{encoded_payload}"
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    status = resp.status
                    blocked = status == 400
                    
                    self.results["path_traversal"].append({
                        "payload": payload,
                        "status_code": status,
                        "blocked": blocked,
                        "secure": blocked or status == 404,
                    })
                    
                    status_icon = "✅" if blocked else "⚠️"
                    print(f"  {status_icon} 路径遍历: {payload[:20]}... -> {status}")
            except Exception as e:
                print(f"  ⚠️ 请求失败: {str(e)[:30]}")
    
    async def test_security_headers(self, session: aiohttp.ClientSession):
        """测试安全响应头"""
        print("\n📌 测试安全响应头...")
        
        expected_headers = [
            "X-Frame-Options",
            "X-XSS-Protection",
            "X-Content-Type-Options",
            "Content-Security-Policy",
            "Referrer-Policy",
        ]
        
        try:
            async with session.get(f"{self.base_url}/api/models/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                # 将headers转换为小写key的字典以进行大小写不敏感比较
                headers_lower = {k.lower(): v for k, v in resp.headers.items()}
                
                for header in expected_headers:
                    header_lower = header.lower()
                    present = header_lower in headers_lower
                    value = headers_lower.get(header_lower, "未设置")
                    
                    self.results["security_headers"].append({
                        "header": header,
                        "present": present,
                        "value": value[:50] if len(str(value)) > 50 else value,
                    })
                    
                    status_icon = "✅" if present else "❌"
                    print(f"  {status_icon} {header}: {value[:40] if len(str(value)) > 40 else value}")
        except Exception as e:
            print(f"  ⚠️ 请求失败: {str(e)}")
    
    async def test_rate_limiting(self, session: aiohttp.ClientSession):
        """测试速率限制"""
        print("\n📌 测试速率限制...")
        
        # 快速发送多个请求
        rate_limit_triggered = False
        request_count = 0
        
        for i in range(150):  # 超过每分钟120的限制
            try:
                async with session.get(f"{self.base_url}/api/models/", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    request_count += 1
                    if resp.status == 429:
                        rate_limit_triggered = True
                        print(f"  ✅ 速率限制在第 {request_count} 个请求时触发")
                        break
            except:
                pass
        
        if not rate_limit_triggered:
            print(f"  ⚠️ 速率限制未触发（已发送 {request_count} 个请求）")
    
    async def run_all_tests(self):
        """运行所有安全测试"""
        self.start_time = datetime.now()
        
        print("=" * 60)
        print("🔒 安全修复验证测试")
        print("=" * 60)
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目标: {self.base_url}")
        
        async with aiohttp.ClientSession() as session:
            # 检查服务是否可用
            try:
                async with session.get(f"{self.base_url}/docs", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        print(f"\n❌ 服务不可用: {resp.status}")
                        return
            except Exception as e:
                print(f"\n❌ 无法连接到服务: {e}")
                print("请确保后端服务正在运行: python -m uvicorn src.api:app --host 127.0.0.1 --port 8000")
                return
            
            # 运行各项测试
            await self.test_sql_injection_path_param(session)
            await self.test_sql_injection_query_param(session)
            await self.test_xss_attack(session)
            await self.test_path_traversal(session)
            await self.test_security_headers(session)
            # await self.test_rate_limiting(session)  # 可选：速率限制测试
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("📊 安全测试报告")
        print("=" * 60)
        
        # SQL注入统计
        sql_results = self.results["sql_injection"]
        sql_blocked = sum(1 for r in sql_results if r["blocked"])
        sql_secure = sum(1 for r in sql_results if r["secure"])
        print(f"\n🛡️ SQL注入防护:")
        print(f"   - 测试载荷: {len(sql_results)}")
        print(f"   - 被拦截: {sql_blocked}")
        print(f"   - 安全处理: {sql_secure}")
        print(f"   - 防护率: {sql_secure/len(sql_results)*100:.1f}%" if sql_results else "N/A")
        
        # XSS统计
        xss_results = self.results["xss"]
        xss_blocked = sum(1 for r in xss_results if r["blocked"])
        xss_secure = sum(1 for r in xss_results if r["secure"])
        print(f"\n🛡️ XSS防护:")
        print(f"   - 测试载荷: {len(xss_results)}")
        print(f"   - 被拦截: {xss_blocked}")
        print(f"   - 安全处理: {xss_secure}")
        print(f"   - 防护率: {xss_secure/len(xss_results)*100:.1f}%" if xss_results else "N/A")
        
        # 路径遍历统计
        path_results = self.results["path_traversal"]
        path_secure = sum(1 for r in path_results if r["secure"])
        print(f"\n🛡️ 路径遍历防护:")
        print(f"   - 测试载荷: {len(path_results)}")
        print(f"   - 安全处理: {path_secure}")
        print(f"   - 防护率: {path_secure/len(path_results)*100:.1f}%" if path_results else "N/A")
        
        # 安全头统计
        header_results = self.results["security_headers"]
        headers_present = sum(1 for r in header_results if r["present"])
        print(f"\n🛡️ 安全响应头:")
        print(f"   - 检查项: {len(header_results)}")
        print(f"   - 已配置: {headers_present}")
        print(f"   - 覆盖率: {headers_present/len(header_results)*100:.1f}%" if header_results else "N/A")
        
        # 总体评估
        total_tests = len(sql_results) + len(xss_results) + len(path_results) + len(header_results)
        total_pass = sql_secure + xss_secure + path_secure + headers_present
        
        print("\n" + "=" * 60)
        print("📈 总体安全评估")
        print("=" * 60)
        print(f"测试项总数: {total_tests}")
        print(f"通过项数: {total_pass}")
        print(f"安全得分: {total_pass/total_tests*100:.1f}%" if total_tests else "N/A")
        print(f"测试耗时: {duration:.2f}秒")
        
        # 综合评级
        score = total_pass / total_tests * 100 if total_tests else 0
        if score >= 90:
            rating = "A+ (优秀)"
            emoji = "🌟"
        elif score >= 80:
            rating = "A (良好)"
            emoji = "✅"
        elif score >= 70:
            rating = "B (合格)"
            emoji = "⚠️"
        elif score >= 60:
            rating = "C (需改进)"
            emoji = "🔶"
        else:
            rating = "D (高风险)"
            emoji = "❌"
        
        print(f"\n{emoji} 安全评级: {rating}")
        
        # 保存报告到文件
        report = {
            "test_time": self.start_time.isoformat(),
            "duration_seconds": duration,
            "target": self.base_url,
            "results": self.results,
            "summary": {
                "total_tests": total_tests,
                "total_pass": total_pass,
                "score": score,
                "rating": rating
            }
        }
        
        with open("security_verification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n# The above code is a Python comment. Comments in Python start with a hash symbol
        # (#) and are used to provide explanations or notes within the code. Comments are
        # ignored by the Python interpreter and are not executed as part of the program.
        📄 详细报告已保存到: security_verification_report.json")


async def main():
    tester = SecurityTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
