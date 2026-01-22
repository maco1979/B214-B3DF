#!/usr/bin/env python3
"""
主控AI API测试脚本
测试AI核心的学习能力、决策能力和协同学习能力
"""

import requests
import json
import time
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIApiTester:
    """主控AI API测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {
            'learning_tests': [],
            'decision_tests': [],
            'collaboration_tests': []
        }
    
    def _make_request(self, method: str, endpoint: str, data: dict = None, headers: dict = None) -> dict:
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        headers = headers or {"Content-Type": "application/json"}
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers)
            elif method == "POST":
                response = self.session.post(url, json=data, headers=headers)
            elif method == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {url} - {e}")
            return {"success": False, "error": str(e)}
    
    def test_activate_master_control(self) -> dict:
        """测试激活主控功能"""
        logger.info("测试激活主控功能")
        
        # 激活主控
        result = self._make_request("POST", "/api/ai-control/master-control", {"activate": True})
        logger.info(f"激活主控结果: {result}")
        
        # 验证主控状态
        status_result = self._make_request("GET", "/api/ai-control/master-control/status")
        logger.info(f"主控状态: {status_result}")
        
        return {
            "activate_result": result,
            "status_result": status_result
        }
    
    def test_organic_ai_activation(self) -> dict:
        """测试有机AI激活和迭代"""
        logger.info("测试有机AI激活和迭代")
        
        # 激活有机AI迭代
        activate_result = self._make_request("POST", "/api/decision/organic-core/activate-iteration")
        logger.info(f"激活有机AI迭代结果: {activate_result}")
        
        # 获取有机AI状态
        status_result = self._make_request("GET", "/api/decision/organic-core/status")
        logger.info(f"有机AI状态: {status_result}")
        
        return {
            "activate_result": activate_result,
            "status_result": status_result
        }
    
    def test_organic_ai_evolution(self) -> dict:
        """测试有机AI结构演化"""
        logger.info("测试有机AI结构演化")
        
        # 演化有机AI结构
        evolution_result = self._make_request("POST", "/api/decision/organic-core/evolve-structure")
        logger.info(f"有机AI结构演化结果: {evolution_result}")
        
        return evolution_result
    
    def test_decision_making(self, num_decisions: int = 10) -> dict:
        """测试有机AI决策能力"""
        logger.info(f"测试有机AI决策能力，决策次数: {num_decisions}")
        
        # 生成测试数据
        test_data = []
        for i in range(num_decisions):
            test_data.append({
                "temperature": 25 + (i % 10) - 5,  # 20-30度
                "humidity": 65 + (i % 20) - 10,  # 55-75%
                "co2_level": 600 + (i % 200) - 100,  # 500-700 ppm
                "light_intensity": 1000 + (i % 400) - 200,  # 800-1200 lux
                "spectrum_config": {
                    "uv_380nm": 0.05,
                    "far_red_720nm": 0.1,
                    "white_light": 0.7,
                    "red_660nm": 0.15
                },
                "crop_type": "tomato",
                "growth_day": 30 + i % 10,  # 30-39天
                "growth_rate": 0.8 + (i % 10) * 0.01,  # 0.8-0.89
                "energy_consumption": 500 + (i % 200) - 100,  # 400-600
                "resource_utilization": 70 + (i % 20) - 10,  # 60-80%
                "health_score": 85 + (i % 10) - 5,  # 80-90%
                "yield_potential": 80 + (i % 15) - 7,  # 73-87%
                "objective": "maximize_yield",
                "task_type": "routine_monitoring",
                "risk_level": "low"
            })
        
        # 执行决策测试
        decision_results = []
        for i, data in enumerate(test_data):
            result = self._make_request("POST", "/api/decision/agriculture", data)
            decision_results.append({
                "test_case": i,
                "input_data": data,
                "result": result
            })
            logger.info(f"决策测试用例 {i+1}/{num_decisions} 完成")
        
        return decision_results
    
    def test_risk_analysis(self, num_analyses: int = 5) -> dict:
        """测试风险分析能力"""
        logger.info(f"测试风险分析能力，分析次数: {num_analyses}")
        
        # 生成风险测试数据
        risk_test_data = []
        for i in range(num_analyses):
            risk_level = "low" if i % 2 == 0 else "high"
            risk_test_data.append({
                "temperature": 35 + (i % 10),  # 35-45度（高温风险）
                "humidity": 85 + (i % 15),  # 85-100%（高湿度风险）
                "co2_level": 1500 + (i % 500),  # 1500-2000 ppm（高CO2风险）
                "light_intensity": 1800 + (i % 400),  # 1800-2200 lux（强光照风险）
                "spectrum_config": {
                    "uv_380nm": 0.05,
                    "far_red_720nm": 0.1,
                    "white_light": 0.7,
                    "red_660nm": 0.15
                },
                "crop_type": "tomato",
                "growth_day": 30 + i % 10,  # 30-39天
                "growth_rate": 0.8 + (i % 10) * 0.01,  # 0.8-0.89
                "energy_consumption": 800 + (i % 400),  # 800-1200（高能耗风险）
                "resource_utilization": 90 + (i % 10),  # 90-100%（资源饱和风险）
                "health_score": 60 - (i % 20),  # 40-60%（健康风险）
                "yield_potential": 50 - (i % 20),  # 30-50%（产量风险）
                "objective": "risk_management",
                "task_type": "high_priority",
                "risk_level": risk_level
            })
        
        # 执行风险分析测试
        risk_results = []
        for i, data in enumerate(risk_test_data):
            result = self._make_request("POST", "/api/decision/risk", data)
            risk_results.append({
                "test_case": i,
                "input_data": data,
                "result": result
            })
            logger.info(f"风险分析测试用例 {i+1}/{num_analyses} 完成")
        
        return risk_results
    
    def test_batch_decision_making(self) -> dict:
        """测试批量决策能力"""
        logger.info("测试批量决策能力")
        
        # 生成批量测试数据
        decision_requests = [
            {
                "temperature": 25,
                "humidity": 65,
                "co2_level": 600,
                "light_intensity": 1000,
                "spectrum_config": {
                    "uv_380nm": 0.05,
                    "far_red_720nm": 0.1,
                    "white_light": 0.7,
                    "red_660nm": 0.15
                },
                "crop_type": "tomato",
                "growth_day": 30,
                "growth_rate": 0.8,
                "energy_consumption": 500,
                "resource_utilization": 70,
                "health_score": 85,
                "yield_potential": 80,
                "objective": "maximize_yield",
                "task_type": "routine_monitoring",
                "risk_level": "low"
            },
            {
                "temperature": 28,
                "humidity": 70,
                "co2_level": 650,
                "light_intensity": 1200,
                "spectrum_config": {
                    "uv_380nm": 0.05,
                    "far_red_720nm": 0.1,
                    "white_light": 0.7,
                    "red_660nm": 0.15
                },
                "crop_type": "tomato",
                "growth_day": 32,
                "growth_rate": 0.82,
                "energy_consumption": 550,
                "resource_utilization": 75,
                "health_score": 88,
                "yield_potential": 82,
                "objective": "maximize_yield",
                "task_type": "routine_monitoring",
                "risk_level": "low"
            },
            {
                "temperature": 22,
                "humidity": 60,
                "co2_level": 550,
                "light_intensity": 800,
                "spectrum_config": {
                    "uv_380nm": 0.05,
                    "far_red_720nm": 0.1,
                    "white_light": 0.7,
                    "red_660nm": 0.15
                },
                "crop_type": "tomato",
                "growth_day": 28,
                "growth_rate": 0.78,
                "energy_consumption": 450,
                "resource_utilization": 65,
                "health_score": 82,
                "yield_potential": 78,
                "objective": "maximize_yield",
                "task_type": "routine_monitoring",
                "risk_level": "low"
            }
        ]
        
        # 构建批量决策请求
        batch_request = {
            "requests": decision_requests,
            "batch_id": f"batch_{int(time.time())}",
            "priority": "normal"
        }
        
        # 执行批量决策
        batch_result = self._make_request("POST", "/api/decision/agriculture/batch", batch_request)
        logger.info(f"批量决策结果: {batch_result}")
        
        return batch_result
    
    def test_collaboration_ability(self) -> dict:
        """测试协同学习能力"""
        logger.info("测试协同学习能力")
        
        # 测试区块链模型注册和验证（模拟协同学习）
        blockchain_tests = []
        
        # 测试区块链状态
        blockchain_status = self._make_request("GET", "/api/blockchain/status")
        logger.info(f"区块链状态: {blockchain_status}")
        blockchain_tests.append({
            "test_type": "blockchain_status",
            "result": blockchain_status
        })
        
        # 测试区块链配置
        blockchain_config = self._make_request("GET", "/api/blockchain/config")
        logger.info(f"区块链配置: {blockchain_config}")
        blockchain_tests.append({
            "test_type": "blockchain_config",
            "result": blockchain_config
        })
        
        return blockchain_tests
    
    def test_system_health(self) -> dict:
        """测试系统健康状态"""
        logger.info("测试系统健康状态")
        
        # 获取系统健康状态
        health_result = self._make_request("GET", "/api/system/health")
        logger.info(f"系统健康状态: {health_result}")
        
        # 获取系统指标
        metrics_result = self._make_request("GET", "/api/system/metrics")
        logger.info(f"系统指标: {metrics_result}")
        
        return {
            "health_result": health_result,
            "metrics_result": metrics_result
        }
    
    def test_deactivate_all(self) -> dict:
        """测试停用所有功能"""
        logger.info("测试停用所有功能")
        
        # 停用有机AI迭代
        deactivate_organic = self._make_request("POST", "/api/decision/organic-core/deactivate-iteration")
        logger.info(f"停用有机AI迭代结果: {deactivate_organic}")
        
        # 关闭主控
        deactivate_master = self._make_request("POST", "/api/ai-control/master-control", {"activate": False})
        logger.info(f"关闭主控结果: {deactivate_master}")
        
        # 验证主控状态
        status_result = self._make_request("GET", "/api/ai-control/master-control/status")
        logger.info(f"最终主控状态: {status_result}")
        
        return {
            "deactivate_organic": deactivate_organic,
            "deactivate_master": deactivate_master,
            "status_result": status_result
        }
    
    def run_comprehensive_test(self) -> dict:
        """运行全面测试"""
        logger.info("开始运行全面测试")
        
        test_results = {
            "test_start_time": datetime.now().isoformat(),
            "master_control": self.test_activate_master_control(),
            "organic_ai_activation": self.test_organic_ai_activation(),
            "organic_ai_evolution": self.test_organic_ai_evolution(),
            "decision_making": self.test_decision_making(num_decisions=10),
            "risk_analysis": self.test_risk_analysis(num_analyses=5),
            "batch_decision": self.test_batch_decision_making(),
            "collaboration": self.test_collaboration_ability(),
            "system_health": self.test_system_health(),
            "test_end_time": datetime.now().isoformat()
        }
        
        # 停用所有功能
        test_results["deactivation"] = self.test_deactivate_all()
        
        return test_results
    
    def print_test_summary(self, results: dict):
        """打印测试汇总"""
        logger.info("\n=== 主控AI测试汇总 ===")
        logger.info(f"测试开始时间: {results['test_start_time']}")
        logger.info(f"测试结束时间: {results['test_end_time']}")
        
        # 主控测试结果
        logger.info("\n--- 主控功能测试结果 ---")
        master_control = results.get('master_control', {})
        logger.info(f"主控激活成功: {master_control.get('activate_result', {}).get('success', False)}")
        logger.info(f"主控状态: {master_control.get('status_result', {}).get('master_control_active', False)}")
        
        # 有机AI测试结果
        logger.info("\n--- 有机AI测试结果 ---")
        organic_ai = results.get('organic_ai_activation', {})
        logger.info(f"有机AI激活成功: {organic_ai.get('activate_result', {}).get('success', False)}")
        organic_status = organic_ai.get('status_result', {})
        logger.info(f"有机AI状态: {organic_status.get('state', 'unknown')}")
        logger.info(f"有机AI迭代启用: {organic_status.get('iteration_enabled', False)}")
        
        # 决策测试结果
        logger.info("\n--- 决策能力测试结果 ---")
        decision_results = results.get('decision_making', [])
        successful_decisions = sum(1 for res in decision_results if res.get('result', {}).get('success', False))
        total_decisions = len(decision_results)
        logger.info(f"决策成功率: {successful_decisions}/{total_decisions} ({(successful_decisions/total_decisions*100):.2f}%)")
        
        # 风险分析测试结果
        logger.info("\n--- 风险分析测试结果 ---")
        risk_results = results.get('risk_analysis', [])
        successful_analyses = sum(1 for res in risk_results if res.get('result', {}).get('success', False))
        total_analyses = len(risk_results)
        logger.info(f"风险分析成功率: {successful_analyses}/{total_analyses} ({(successful_analyses/total_analyses*100):.2f}%)")
        
        # 批量决策测试结果
        logger.info("\n--- 批量决策测试结果 ---")
        batch_result = results.get('batch_decision', {})
        logger.info(f"批量决策成功: {batch_result.get('success', False)}")
        if batch_result.get('success') and batch_result.get('data'):
            logger.info(f"批量决策数量: {batch_result['data'].get('total_decisions', 0)}")
            logger.info(f"平均决策时间: {batch_result['data'].get('average_execution_time', 0):.6f}秒")
        else:
            logger.info(f"批量决策数量: 0")
        
        # 协同学习测试结果
        logger.info("\n--- 协同学习测试结果 ---")
        collaboration_results = results.get('collaboration', [])
        successful_collaborations = sum(1 for res in collaboration_results if res.get('result', {}).get('success', False))
        total_collaborations = len(collaboration_results)
        logger.info(f"协同测试成功率: {successful_collaborations}/{total_collaborations} ({(successful_collaborations/total_collaborations*100):.2f}%)")
        
        # 系统健康测试结果
        logger.info("\n--- 系统健康测试结果 ---")
        health_result = results.get('system_health', {}).get('health_result', {})
        logger.info(f"系统健康状态: {health_result.get('status', 'unknown')}")
        
        logger.info("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 创建测试器实例
    tester = AIApiTester()
    
    # 运行全面测试
    results = tester.run_comprehensive_test()
    
    # 打印测试汇总
    tester.print_test_summary(results)
    
    # 保存测试结果到文件
    with open("ai_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("测试结果已保存到 ai_test_results.json")
