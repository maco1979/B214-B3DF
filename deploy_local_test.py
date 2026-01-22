"""
本地部署测试脚本
用于启动完整的AI决策系统进行长时间测试
"""

import os
import sys
import subprocess
import time
import threading
import requests
import signal
import json
from datetime import datetime
from pathlib import Path


class LocalDeploymentTester:
    """本地部署测试器"""
    
    def __init__(self):
        self.project_root = Path("d:\\1.5")
        self.backend_process = None
        self.frontend_process = None
        self.api_gateway_process = None
        self.services_running = False
        
    def check_prerequisites(self):
        """检查先决条件"""
        print("🔍 检查部署先决条件...")
        
        # 检查Python版本
        if sys.version_info < (3, 8):
            print("❌ Python版本过低，需要3.8或更高版本")
            return False
        
        # 检查必需的依赖
        dependencies = ["fastapi", "uvicorn", "requests", "numpy", "jax"]
        missing_deps = []
        
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ 缺少依赖: {', '.join(missing_deps)}")
            return False
        
        print("✅ 先决条件检查通过")
        return True
    
    def start_backend(self):
        """启动后端服务"""
        print("🚀 启动后端服务...")
        
        backend_dir = self.project_root / "backend"
        
        # 检查requirements.txt并安装依赖（如果需要）
        requirements_file = backend_dir / "requirements.txt"
        if requirements_file.exists():
            print("📦 安装后端依赖...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], cwd=backend_dir, check=False)
        
        # 启动后端服务
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.api.simple_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"  # 开发模式下自动重载
        ]
        
        self.backend_process = subprocess.Popen(
            cmd,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ 后端服务启动")
        return True
    
    def start_api_gateway(self):
        """启动API网关"""
        print("🌐 启动API网关...")
        
        gateway_dir = self.project_root / "api-gateway"
        
        # 检查package.json并安装依赖（如果需要）
        package_file = gateway_dir / "package.json"
        if package_file.exists():
            print("📦 安装API网关依赖...")
            subprocess.run([
                "npm", "install"
            ], cwd=gateway_dir, check=False)
        
        # 启动API网关
        cmd = ["npm", "start"]
        
        self.api_gateway_process = subprocess.Popen(
            cmd,
            cwd=gateway_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ API网关启动")
        return True
    
    def start_frontend(self):
        """启动前端服务"""
        print("💻 启动前端服务...")
        
        frontend_dir = self.project_root / "frontend"
        
        # 检查package.json并安装依赖（如果需要）
        package_file = frontend_dir / "package.json"
        if package_file.exists():
            print("📦 安装前端依赖...")
            subprocess.run([
                "npm", "install"
            ], cwd=frontend_dir, check=False)
        
        # 启动前端开发服务器
        cmd = ["npm", "run", "dev"]
        
        self.frontend_process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ 前端服务启动")
        return True
    
    def wait_for_services(self, timeout=60):
        """等待服务启动"""
        print(f"⏳ 等待服务启动 (超时: {timeout}秒)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 检查后端服务
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ 后端服务已就绪")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        else:
            print("❌ 服务启动超时")
            return False
        
        print("✅ 所有服务已就绪")
        return True
    
    def run_health_checks(self):
        """运行健康检查"""
        print("🏥 运行健康检查...")
        
        checks = [
            {
                "name": "后端API",
                "url": "http://localhost:8000/health",
                "method": "GET"
            },
            {
                "name": "模型服务",
                "url": "http://localhost:8000/api/models",
                "method": "GET"
            },
            {
                "name": "用户服务",
                "url": "http://localhost:8000/api/user/stats?user_id=test",
                "method": "GET"
            }
        ]
        
        results = []
        for check in checks:
            try:
                response = requests.request(
                    method=check["method"],
                    url=check["url"],
                    timeout=10
                )
                status = "✅" if response.status_code == 200 else "❌"
                results.append({
                    "name": check["name"],
                    "status": status,
                    "code": response.status_code
                })
                print(f"  {status} {check['name']}: {response.status_code}")
            except Exception as e:
                results.append({
                    "name": check["name"],
                    "status": "❌",
                    "error": str(e)
                })
                print(f"  ❌ {check['name']}: {str(e)}")
        
        return results
    
    def start_long_term_test(self, duration_hours=24):
        """开始长时间测试"""
        print(f"⏱️  开始 {duration_hours} 小时长时间测试...")
        
        start_time = datetime.now()
        end_time = start_time + (duration_hours * 60 * 60)
        
        test_log = {
            "start_time": start_time.isoformat(),
            "duration_hours": duration_hours,
            "checks": []
        }
        
        check_interval = 300  # 每5分钟检查一次
        check_count = 0
        
        try:
            while True:
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()
                remaining = (end_time - current_time).total_seconds()
                
                if remaining <= 0:
                    print(f"✅ 长时间测试完成! 运行时间: {elapsed/3600:.2f} 小时")
                    break
                
                print(f"📊 测试运行中... 已运行 {elapsed/3600:.2f} 小时，剩余 {remaining/3600:.2f} 小时")
                
                # 运行健康检查
                health_results = self.run_health_checks()
                
                # 记录检查结果
                check_result = {
                    "timestamp": current_time.isoformat(),
                    "check_count": check_count,
                    "results": health_results
                }
                test_log["checks"].append(check_result)
                
                # 检查是否有失败的检查
                failed_checks = [r for r in health_results if r["status"] == "❌"]
                if failed_checks:
                    print(f"⚠️  发现 {len(failed_checks)} 个失败的检查")
                
                check_count += 1
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"⏸️  测试被用户中断，运行时间: {elapsed/3600:.2f} 小时")
        
        # 保存测试日志
        log_file = self.project_root / "long_term_test_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(test_log, f, ensure_ascii=False, indent=2)
        
        print(f"📋 测试日志已保存至: {log_file}")
        return test_log
    
    def stop_services(self):
        """停止所有服务"""
        print("🛑 停止所有服务...")
        
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
            print("✅ 后端服务已停止")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
            print("✅ 前端服务已停止")
        
        if self.api_gateway_process:
            self.api_gateway_process.terminate()
            self.api_gateway_process.wait()
            print("✅ API网关已停止")
        
        self.services_running = False
        print("✅ 所有服务已停止")
    
    def deploy_and_test(self, duration_hours=24):
        """部署并开始测试"""
        print("🚀 开始本地部署测试...")
        print("="*60)
        
        try:
            # 检查先决条件
            if not self.check_prerequisites():
                return False
            
            # 启动服务
            self.start_backend()
            time.sleep(5)  # 等待后端启动
            
            self.start_api_gateway()
            time.sleep(3)  # 等待API网关启动
            
            self.start_frontend()
            time.sleep(3)  # 等待前端启动
            
            # 等待服务就绪
            if not self.wait_for_services():
                print("❌ 服务启动失败")
                return False
            
            # 运行初始健康检查
            print("\n🔍 运行初始健康检查...")
            initial_checks = self.run_health_checks()
            
            # 检查初始状态
            failed_initial = [r for r in initial_checks if r["status"] == "❌"]
            if failed_initial:
                print(f"⚠️  初始检查发现 {len(failed_initial)} 个问题")
            else:
                print("✅ 初始健康检查通过")
            
            # 开始长时间测试
            test_log = self.start_long_term_test(duration_hours)
            
            return True
            
        except Exception as e:
            print(f"❌ 部署测试失败: {str(e)}")
            return False
        
        finally:
            # 确保服务被停止
            self.stop_services()
            print("\n" + "="*60)
            print("✅ 本地部署测试完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='本地部署测试')
    parser.add_argument('--duration', type=int, default=24, 
                       help='测试持续时间（小时），默认24小时')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式（仅运行5分钟）')
    
    args = parser.parse_args()
    
    if args.quick:
        duration = 1/12  # 5分钟
        print("🏃‍♂️ 快速测试模式 - 仅运行5分钟")
    else:
        duration = args.duration
        print(f"🏃‍♂️ 标准测试模式 - 运行 {duration} 小时")
    
    tester = LocalDeploymentTester()
    tester.deploy_and_test(duration_hours=duration)


if __name__ == "__main__":
    main()