#!/usr/bin/env python3
"""
主控AI长时间大规模测试脚本
测试AI核心的长时间稳定性、大规模决策能力和系统资源消耗
"""

import requests
import json
import time
import logging
import threading
import random
from datetime import datetime
from typing import Dict, List, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('long_term_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LongTermLargeScaleTester:
    """长时间大规模测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'error_details': {},
            'decision_times': [],
            'risk_analysis_times': [],
            'batch_decision_times': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'resource_usage': []
        }
        self.running = False
        self.lock = threading.Lock()
    
    def _make_request(self, method: str, endpoint: str, data: dict = None, headers: dict = None) -> dict:
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        headers = headers or {"Content-Type": "application/json"}
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=30)
            elif method == "POST":
                response = self.session.post(url, json=data, headers=headers, timeout=30)
            elif method == "PUT":
                response = self.session.put(url, json=data, headers=headers, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_type = type(e).__name__
            with self.lock:
                if error_type not in self.test_results['error_details']:
                    self.test_results['error_details'][error_type] = 0
                self.test_results['error_details'][error_type] += 1
            logger.error(f"请求失败: {url} - {e}")
            return {"success": False, "error": str(e)}
    
    def generate_decision_data(self, test_id: int) -> dict:
        """生成多样化的决策测试数据"""
        crop_types = ['tomato', 'cucumber', 'pepper', 'lettuce', 'strawberry']
        objectives = ['maximize_yield', 'minimize_energy', 'risk_management', 'quality_optimization', 'resource_efficiency']
        task_types = ['routine_monitoring', 'high_priority', 'emergency_response', 'optimization', 'predictive_analysis']
        risk_levels = ['low', 'medium', 'high', 'critical']
        
        return {
            "temperature": 15 + random.uniform(0, 25),  # 15-40度
            "humidity": 40 + random.uniform(0, 50),  # 40-90%
            "co2_level": 400 + random.uniform(0, 1600),  # 400-2000 ppm
            "light_intensity": 500 + random.uniform(0, 2500),  # 500-3000 lux
            "spectrum_config": {
                "uv_380nm": round(random.uniform(0.01, 0.1), 3),
                "far_red_720nm": round(random.uniform(0.05, 0.2), 3),
                "white_light": round(random.uniform(0.5, 0.9), 3),
                "red_660nm": round(random.uniform(0.05, 0.2), 3)
            },
            "crop_type": random.choice(crop_types),
            "growth_day": random.randint(1, 120),  # 1-120天
            "growth_rate": round(random.uniform(0.5, 1.0), 3),  # 0.5-1.0
            "energy_consumption": 300 + random.uniform(0, 1000),  # 300-1300
            "resource_utilization": 50 + random.uniform(0, 45),  # 50-95%
            "health_score": 60 + random.uniform(0, 35),  # 60-95%
            "yield_potential": 60 + random.uniform(0, 35),  # 60-95%
            "objective": random.choice(objectives),
            "task_type": random.choice(task_types),
            "risk_level": random.choice(risk_levels)
        }
    
    def test_single_decision(self, test_id: int):
        """测试单个决策请求"""
        data = self.generate_decision_data(test_id)
        start_time = time.time()
        result = self._make_request("POST", "/api/decision/agriculture", data)
        end_time = time.time()
        
        with self.lock:
            self.test_results['total_requests'] += 1
            if result.get('success'):
                self.test_results['successful_requests'] += 1
                self.test_results['decision_times'].append(end_time - start_time)
            else:
                self.test_results['failed_requests'] += 1
    
    def test_risk_analysis(self, test_id: int):
        """测试风险分析请求"""
        data = self.generate_decision_data(test_id)
        # 调整数据以增加风险因素
        data['temperature'] = 35 + random.uniform(0, 10)  # 35-45度
        data['humidity'] = 80 + random.uniform(0, 20)  # 80-100%
        data['co2_level'] = 1500 + random.uniform(0, 1000)  # 1500-2500 ppm
        data['health_score'] = 40 + random.uniform(0, 30)  # 40-70%
        data['yield_potential'] = 40 + random.uniform(0, 30)  # 40-70%
        data['objective'] = 'risk_management'
        data['task_type'] = 'high_priority'
        
        start_time = time.time()
        result = self._make_request("POST", "/api/decision/risk", data)
        end_time = time.time()
        
        with self.lock:
            self.test_results['total_requests'] += 1
            if result.get('success'):
                self.test_results['successful_requests'] += 1
                self.test_results['risk_analysis_times'].append(end_time - start_time)
            else:
                self.test_results['failed_requests'] += 1
    
    def test_batch_decision(self, batch_size: int = 10):
        """测试批量决策请求"""
        decision_requests = [self.generate_decision_data(i) for i in range(batch_size)]
        
        batch_request = {
            "requests": decision_requests,
            "batch_id": f"long_batch_{int(time.time())}",
            "priority": random.choice(['low', 'normal', 'high'])
        }
        
        start_time = time.time()
        result = self._make_request("POST", "/api/decision/agriculture/batch", batch_request)
        end_time = time.time()
        
        with self.lock:
            self.test_results['total_requests'] += 1
            if result.get('success'):
                self.test_results['successful_requests'] += 1
                self.test_results['batch_decision_times'].append(end_time - start_time)
            else:
                self.test_results['failed_requests'] += 1
    
    def test_system_health(self):
        """测试系统健康状态"""
        result = self._make_request("GET", "/api/system/health")
        metrics_result = self._make_request("GET", "/api/system/metrics")
        
        with self.lock:
            self.test_results['resource_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'health_status': result.get('status', 'unknown'),
                'metrics': metrics_result.get('data', {})
            })
    
    def run_long_term_test(self, duration_hours: float = 1.0, requests_per_minute: int = 60):
        """运行长时间测试"""
        self.running = True
        logger.info(f"开始长时间大规模测试，持续时间: {duration_hours}小时，请求频率: {requests_per_minute}次/分钟")
        
        # 激活主控
        self._make_request("POST", "/api/ai-control/master-control", {"activate": True})
        
        # 激活有机AI迭代
        self._make_request("POST", "/api/decision/organic-core/activate-iteration")
        
        end_time = time.time() + duration_hours * 3600
        request_interval = 60 / requests_per_minute
        test_id = 0
        
        while time.time() < end_time and self.running:
            start_loop = time.time()
            
            # 随机选择测试类型
            test_type = random.choices(
                ['decision', 'risk', 'batch'],
                weights=[0.6, 0.3, 0.1],
                k=1
            )[0]
            
            if test_type == 'decision':
                self.test_single_decision(test_id)
            elif test_type == 'risk':
                self.test_risk_analysis(test_id)
            elif test_type == 'batch':
                self.test_batch_decision(batch_size=random.randint(5, 20))
            
            test_id += 1
            
            # 每100个请求检查一次系统健康
            if test_id % 100 == 0:
                self.test_system_health()
            
            # 每1000个请求打印一次进度
            if test_id % 1000 == 0:
                self.print_progress()
            
            # 控制请求频率
            loop_duration = time.time() - start_loop
            sleep_time = max(0, request_interval - loop_duration)
            time.sleep(sleep_time)
        
        self.running = False
        self.test_results['end_time'] = datetime.now().isoformat()
        
        # 停用所有功能
        self._make_request("POST", "/api/decision/organic-core/deactivate-iteration")
        self._make_request("POST", "/api/ai-control/master-control", {"activate": False})
        
        logger.info("长时间大规模测试完成")
        self.print_final_results()
        self.save_results()
    
    def print_progress(self):
        """打印测试进度"""
        with self.lock:
            total = self.test_results['total_requests']
            successful = self.test_results['successful_requests']
            failed = self.test_results['failed_requests']
            success_rate = (successful / total * 100) if total > 0 else 0
        
        logger.info(f"测试进度: 请求数={total}, 成功={successful}, 失败={failed}, 成功率={success_rate:.2f}%")
    
    def print_final_results(self):
        """打印最终测试结果"""
        with self.lock:
            total = self.test_results['total_requests']
            successful = self.test_results['successful_requests']
            failed = self.test_results['failed_requests']
            success_rate = (successful / total * 100) if total > 0 else 0
            
            # 计算平均响应时间
            avg_decision_time = sum(self.test_results['decision_times']) / len(self.test_results['decision_times']) if self.test_results['decision_times'] else 0
            avg_risk_time = sum(self.test_results['risk_analysis_times']) / len(self.test_results['risk_analysis_times']) if self.test_results['risk_analysis_times'] else 0
            avg_batch_time = sum(self.test_results['batch_decision_times']) / len(self.test_results['batch_decision_times']) if self.test_results['batch_decision_times'] else 0
        
        logger.info("\n=== 长时间大规模测试最终结果 ===")
        logger.info(f"测试开始时间: {self.test_results['start_time']}")
        logger.info(f"测试结束时间: {self.test_results['end_time']}")
        logger.info(f"总请求数: {total}")
        logger.info(f"成功请求数: {successful}")
        logger.info(f"失败请求数: {failed}")
        logger.info(f"成功率: {success_rate:.2f}%")
        logger.info(f"平均决策响应时间: {avg_decision_time:.4f}秒")
        logger.info(f"平均风险分析响应时间: {avg_risk_time:.4f}秒")
        logger.info(f"平均批量决策响应时间: {avg_batch_time:.4f}秒")
        
        if self.test_results['error_details']:
            logger.info("\n错误详情:")
            for error_type, count in self.test_results['error_details'].items():
                logger.info(f"  {error_type}: {count}次")
        
        logger.info("\n=== 测试完成 ===")
    
    def save_results(self):
        """保存测试结果到文件"""
        filename = f"long_term_test_results_{int(time.time())}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"测试结果已保存到 {filename}")
    
    def stop_test(self):
        """停止测试"""
        self.running = False
        logger.info("测试已停止")


if __name__ == "__main__":
    # 创建测试器实例
    tester = LongTermLargeScaleTester()
    
    try:
        # 运行30分钟测试，每分钟60个请求
        tester.run_long_term_test(duration_hours=0.5, requests_per_minute=60)
    except KeyboardInterrupt:
        tester.stop_test()
        tester.print_final_results()
        tester.save_results()
