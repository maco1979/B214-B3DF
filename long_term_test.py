"""
长时间运行测试脚本
验证AI决策系统在长时间运行下的稳定性
"""

import requests
import time
import threading
import json
from datetime import datetime
from pathlib import Path
import signal
import sys


class LongTermTester:
    """长时间测试器"""
    
    def __init__(self):
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "test_cycles": [],
            "errors": [],
            "performance_metrics": []
        }
        self.running = True
        self.test_interval = 30  # 30秒测试一次
        self.test_duration = 0  # 持续时间（秒）
    
    def health_check(self):
        """健康检查"""
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=10)
            return {
                "endpoint": "/health",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "endpoint": "/health",
                "status_code": None,
                "success": False,
                "error": str(e),
                "response_time": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def community_check(self):
        """社区功能检查"""
        try:
            response = requests.get("http://127.0.0.1:8000/api/community/posts", timeout=10)
            return {
                "endpoint": "/api/community/posts",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "endpoint": "/api/community/posts",
                "status_code": None,
                "success": False,
                "error": str(e),
                "response_time": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def auth_check(self):
        """认证功能检查"""
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/auth/login",
                json={"username": "test", "password": "test"},
                timeout=10
            )
            # 期望返回422（参数验证错误）或200（成功），而不是404（不存在）
            success = response.status_code in [422, 200, 401]
            return {
                "endpoint": "/api/auth/login",
                "status_code": response.status_code,
                "success": success,
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "endpoint": "/api/auth/login",
                "status_code": None,
                "success": False,
                "error": str(e),
                "response_time": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_single_test_cycle(self, cycle_num):
        """运行单个测试周期"""
        print(f"🔄 测试周期 #{cycle_num}")
        
        start_time = time.time()
        
        # 运行各种检查
        checks = [
            self.health_check(),
            self.community_check(),
            self.auth_check()
        ]
        
        # 计算性能指标
        successful_checks = [c for c in checks if c["success"]]
        failed_checks = [c for c in checks if not c["success"]]
        
        cycle_result = {
            "cycle_num": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
            "successful_count": len(successful_checks),
            "failed_count": len(failed_checks),
            "success_rate": len(successful_checks) / len(checks) if checks else 0,
            "cycle_duration": time.time() - start_time
        }
        
        # 记录错误
        for check in failed_checks:
            self.test_results["errors"].append(check)
        
        # 记录性能指标
        response_times = [c["response_time"] for c in checks if c["response_time"] is not None]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            self.test_results["performance_metrics"].append({
                "cycle_num": cycle_num,
                "avg_response_time": avg_response_time,
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "timestamp": datetime.now().isoformat()
            })
        
        self.test_results["test_cycles"].append(cycle_result)
        
        # 输出周期结果
        print(f"  ✅ 成功: {len(successful_checks)}, ❌ 失败: {len(failed_checks)}, "
              f"成功率: {cycle_result['success_rate']*100:.1f}%")
        
        return cycle_result
    
    def run_continuous_test(self, max_cycles=None):
        """运行连续测试"""
        print("🚀 开始长时间运行测试...")
        print(f"⏱️  测试间隔: {self.test_interval}秒")
        if max_cycles:
            print(f"📊 最大周期数: {max_cycles}")
        else:
            print("📊 持续运行直到手动停止")
        print("="*60)
        
        cycle_num = 1
        start_time = datetime.now()
        
        try:
            while self.running:
                if max_cycles and cycle_num > max_cycles:
                    break
                
                # 运行测试周期
                self.run_single_test_cycle(cycle_num)
                
                # 输出摘要（每10个周期）
                if cycle_num % 10 == 0:
                    self.print_summary(cycle_num)
                
                # 等待下一个周期
                print(f"⏳ 等待 {self.test_interval} 秒进行下一次测试...")
                for _ in range(self.test_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                
                cycle_num += 1
                
        except KeyboardInterrupt:
            print("\n⏸️  测试被用户中断")
        finally:
            end_time = datetime.now()
            self.test_duration = (end_time - start_time).total_seconds()
            self.print_final_summary()
    
    def print_summary(self, current_cycle):
        """打印测试摘要"""
        total_cycles = len(self.test_results["test_cycles"])
        if total_cycles == 0:
            return
        
        successful_checks = sum(c["successful_count"] for c in self.test_results["test_cycles"])
        total_checks = total_cycles * 3  # 每个周期3个检查
        overall_success_rate = successful_checks / total_checks if total_checks > 0 else 0
        
        print(f"\n📊 摘要 (周期 1-{current_cycle}):")
        print(f"   总周期数: {total_cycles}")
        print(f"   总检查数: {total_checks}")
        print(f"   成功检查: {successful_checks}")
        print(f"   整体成功率: {overall_success_rate*100:.2f}%")
        
        if self.test_results["performance_metrics"]:
            avg_response_times = [pm["avg_response_time"] for pm in self.test_results["performance_metrics"]]
            if avg_response_times:
                avg_response = sum(avg_response_times) / len(avg_response_times)
                print(f"   平均响应时间: {avg_response:.3f}s")
    
    def print_final_summary(self):
        """打印最终摘要"""
        print("\n" + "="*60)
        print("📈 长时间测试完成")
        print("="*60)
        
        total_cycles = len(self.test_results["test_cycles"])
        if total_cycles == 0:
            print("❌ 没有完成任何测试周期")
            return
        
        successful_checks = sum(c["successful_count"] for c in self.test_results["test_cycles"])
        total_checks = total_cycles * 3  # 每个周期3个检查
        overall_success_rate = successful_checks / total_checks if total_checks > 0 else 0
        
        print(f"⏱️  测试持续时间: {self.test_duration/3600:.2f} 小时")
        print(f"🔄 完成周期数: {total_cycles}")
        print(f"✅ 总检查数: {total_checks}")
        print(f"📈 整体成功率: {overall_success_rate*100:.2f}%")
        
        if self.test_results["performance_metrics"]:
            avg_response_times = [pm["avg_response_time"] for pm in self.test_results["performance_metrics"]]
            if avg_response_times:
                avg_response = sum(avg_response_times) / len(avg_response_times)
                min_response = min(pm["min_response_time"] for pm in self.test_results["performance_metrics"])
                max_response = max(pm["max_response_time"] for pm in self.test_results["performance_metrics"])
                
                print(f"⚡ 平均响应时间: {avg_response:.3f}s")
                print(f"⚡ 最快响应时间: {min_response:.3f}s")
                print(f"⚡ 最慢响应时间: {max_response:.3f}s")
        
        error_count = len(self.test_results["errors"])
        print(f"⚠️  总错误数: {error_count}")
        
        # 系统稳定性评估
        print(f"\n🎯 系统稳定性评估:")
        if overall_success_rate >= 0.95:
            print("   🟢 极其稳定 - 系统表现优秀")
        elif overall_success_rate >= 0.90:
            print("   🟢 非常稳定 - 系统表现良好")
        elif overall_success_rate >= 0.80:
            print("   🟡 稳定 - 系统基本正常")
        elif overall_success_rate >= 0.70:
            print("   🟡 需要关注 - 系统存在一些问题")
        else:
            print("   🔴 不稳定 - 系统需要优化")
    
    def save_results(self, filename="long_term_test_results.json"):
        """保存测试结果"""
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["total_duration_seconds"] = self.test_duration
        
        results_path = Path(filename)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 测试结果已保存至: {results_path.absolute()}")
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n🛑 收到信号 {signum}，正在停止测试...")
        self.running = False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='长时间运行测试')
    parser.add_argument('--duration', type=int, default=None,
                       help='测试持续时间（分钟），默认为持续运行')
    parser.add_argument('--cycles', type=int, default=None,
                       help='测试周期数，指定后将覆盖duration参数')
    
    args = parser.parse_args()
    
    tester = LongTermTester()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, tester.signal_handler)
    signal.signal(signal.SIGTERM, tester.signal_handler)
    
    # 如果指定了周期数，使用周期数，否则根据持续时间计算
    max_cycles = args.cycles
    if not max_cycles and args.duration:
        max_cycles = (args.duration * 60) // tester.test_interval
    
    print("🧪 AI决策系统长时间运行测试")
    print("按 Ctrl+C 停止测试")
    
    # 开始测试
    tester.run_continuous_test(max_cycles=max_cycles)
    
    # 保存结果
    tester.save_results()


if __name__ == "__main__":
    main()