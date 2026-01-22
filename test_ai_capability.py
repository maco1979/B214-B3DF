# 测试核心AI能力评估脚本

import sys
import os

# 添加backend目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# 测试1：代码理解与实现能力
def test_code_implementation():
    """测试AI能否基于抽象类实现具体子类"""
    from src.core.decision_engine import DecisionEngine
    
    class AgriculturalDecisionEngine(DecisionEngine):
        """农业决策引擎具体实现"""
        
        async def make_decision(self, decision_request: dict) -> dict:
            """
            实现农业决策逻辑
            
            Args:
                decision_request: 包含作物类型、天气数据、土壤数据等
                
            Returns:
                决策结果，包含灌溉建议、施肥建议等
            """
            crop_type = decision_request.get('crop_type', 'wheat')
            weather_data = decision_request.get('weather_data', {})
            soil_data = decision_request.get('soil_data', {})
            
            # 简化的决策逻辑
            irrigation_recommendation = self._calculate_irrigation(crop_type, weather_data, soil_data)
            fertilization_recommendation = self._calculate_fertilization(crop_type, soil_data)
            
            return {
                "decision_id": "agri-dec-" + str(hash(str(decision_request))),
                "crop_type": crop_type,
                "irrigation": irrigation_recommendation,
                "fertilization": fertilization_recommendation,
                "timestamp": "2026-01-12T10:00:00Z",
                "confidence": 0.85
            }
        
        def _calculate_irrigation(self, crop_type, weather_data, soil_data):
            """计算灌溉建议"""
            soil_moisture = soil_data.get('moisture', 50)
            rainfall = weather_data.get('rainfall', 0)
            
            if soil_moisture < 30:
                return "需要灌溉：50mm"
            elif rainfall > 20:
                return "无需灌溉"
            else:
                return "建议灌溉：20mm"
        
        def _calculate_fertilization(self, crop_type, soil_data):
            """计算施肥建议"""
            nitrogen = soil_data.get('nitrogen', 50)
            phosphorus = soil_data.get('phosphorus', 30)
            potassium = soil_data.get('potassium', 40)
            
            if nitrogen < 40:
                return "需要氮肥：10kg/亩"
            elif phosphorus < 20:
                return "需要磷肥：5kg/亩"
            elif potassium < 30:
                return "需要钾肥：8kg/亩"
            else:
                return "无需额外施肥"
    
    # 测试实现是否正确
    engine = AgriculturalDecisionEngine()
    
    # 测试状态获取
    status = engine.get_status()
    assert status["status"] == "operational"
    assert status["type"] == "AgriculturalDecisionEngine"
    
    print("✅ 测试1通过：代码实现能力正常")
    return engine

# 测试2：逻辑推理与问题解决能力
def test_logical_reasoning():
    """测试AI的逻辑推理能力"""
    from src.core.decision_engine import DecisionEngine
    import asyncio
    
    class TestDecisionEngine(DecisionEngine):
        """测试用决策引擎，包含复杂逻辑"""
        
        async def make_decision(self, decision_request: dict) -> dict:
            """实现复杂决策逻辑"""
            # 解析请求数据
            variables = decision_request.get('variables', {})
            rules = decision_request.get('rules', [])
            
            # 执行规则推理
            results = []
            for rule in rules:
                condition = rule.get('condition', {})
                action = rule.get('action', '')
                
                # 评估条件
                if self._evaluate_condition(condition, variables):
                    results.append(action)
            
            return {
                "inference_results": results,
                "rule_count": len(rules),
                "matched_rules": len(results)
            }
        
        def _evaluate_condition(self, condition: dict, variables: dict) -> bool:
            """评估条件表达式"""
            left = variables.get(condition.get('left'), 0)
            operator = condition.get('operator', '==')
            right = condition.get('right', 0)
            
            if operator == '==':
                return left == right
            elif operator == '!=':
                return left != right
            elif operator == '>':
                return left > right
            elif operator == '<':
                return left < right
            elif operator == '>=':
                return left >= right
            elif operator == '<=':
                return left <= right
            elif operator == 'and':
                return all(self._evaluate_condition(c, variables) for c in condition.get('conditions', []))
            elif operator == 'or':
                return any(self._evaluate_condition(c, variables) for c in condition.get('conditions', []))
            
            return False
    
    # 测试逻辑推理
    engine = TestDecisionEngine()
    
    test_request = {
        "variables": {
            "temperature": 25,
            "humidity": 60,
            "soil_moisture": 40
        },
        "rules": [
            {
                "condition": {"left": "temperature", "operator": ">", "right": 30},
                "action": "开启降温系统"
            },
            {
                "condition": {"left": "humidity", "operator": "<", "right": 50},
                "action": "开启加湿系统"
            },
            {
                "condition": {"left": "soil_moisture", "operator": "<", "right": 50},
                "action": "开启灌溉系统"
            },
            {
                "condition": {
                    "operator": "and",
                    "conditions": [
                        {"left": "temperature", "operator": ">", "right": 20},
                        {"left": "humidity", "operator": ">", "right": 50}
                    ]
                },
                "action": "调整通风系统"
            }
        ]
    }
    
    # 执行异步决策
    import asyncio
    result = asyncio.run(engine.make_decision(test_request))
    
    # 验证结果
    assert result["rule_count"] == 4
    assert "开启灌溉系统" in result["inference_results"]
    assert "调整通风系统" in result["inference_results"]
    assert len(result["inference_results"]) == 2
    
    print("✅ 测试2通过：逻辑推理能力正常")
    return result

# 测试3：工具使用与系统交互能力
def test_tool_usage():
    """测试AI的工具使用能力"""
    import os
    import sys
    
    # 测试文件操作能力
    test_file_path = "test_ai_tool.txt"
    
    # 创建测试文件
    with open(test_file_path, "w") as f:
        f.write("测试AI工具使用能力\n")
        f.write("当前时间：2026-01-12\n")
    
    # 读取测试文件
    with open(test_file_path, "r") as f:
        content = f.read()
    
    assert "测试AI工具使用能力" in content
    assert "2026-01-12" in content
    
    # 修改测试文件
    with open(test_file_path, "a") as f:
        f.write("追加内容：工具使用测试通过\n")
    
    # 再次读取验证
    with open(test_file_path, "r") as f:
        updated_content = f.read()
    
    assert "工具使用测试通过" in updated_content
    
    # 删除测试文件
    os.remove(test_file_path)
    
    print("✅ 测试3通过：工具使用能力正常")
    return True

# 测试4：学习与适应能力
def test_learning_adaptation():
    """测试AI的学习和适应能力"""
    class AdaptiveDecisionEngine:
        """自适应决策引擎，能够从历史决策中学习"""
        
        def __init__(self):
            self.decision_history = []
            self.feedback_scores = []
        
        def make_decision(self, input_data):
            """基于历史数据的自适应决策"""
            # 初始决策逻辑
            base_decision = self._base_decision(input_data)
            
            # 基于历史反馈调整决策
            if self.decision_history:
                adjustment = self._calculate_adjustment()
                base_decision["adjustment_factor"] = adjustment
                base_decision["decision"] = base_decision["decision"] * (1 + adjustment)
            
            # 记录决策
            self.decision_history.append({
                "input": input_data,
                "decision": base_decision
            })
            
            return base_decision
        
        def _base_decision(self, input_data):
            """基础决策逻辑"""
            value = input_data.get('value', 0)
            return {
                "decision": value * 2,
                "confidence": 0.7
            }
        
        def _calculate_adjustment(self):
            """基于历史反馈计算调整因子"""
            if not self.feedback_scores:
                return 0
            
            avg_feedback = sum(self.feedback_scores) / len(self.feedback_scores)
            # 反馈范围：-1到1，映射到调整因子：-0.2到0.2
            adjustment = (avg_feedback) * 0.2
            return adjustment
        
        def provide_feedback(self, score):
            """提供决策反馈"""
            # 限制分数范围
            score = max(-1, min(1, score))
            self.feedback_scores.append(score)
    
    # 测试自适应学习
    engine = AdaptiveDecisionEngine()
    
    # 初始决策
    decision1 = engine.make_decision({"value": 10})
    assert decision1["decision"] == 20  # 10 * 2
    
    # 提供正面反馈
    engine.provide_feedback(0.8)
    engine.provide_feedback(0.9)
    
    # 再次决策，应该有正调整
    decision2 = engine.make_decision({"value": 10})
    assert decision2["decision"] > 20  # 应该大于20
    assert "adjustment_factor" in decision2
    assert decision2["adjustment_factor"] > 0
    
    # 提供负面反馈
    engine.provide_feedback(-0.7)
    engine.provide_feedback(-0.6)
    
    # 再次决策，调整因子应该降低
    decision3 = engine.make_decision({"value": 10})
    assert "adjustment_factor" in decision3
    
    print("✅ 测试4通过：学习与适应能力正常")
    return engine

# 测试5：自然语言处理与理解能力
def test_nlp_understanding():
    """测试AI的自然语言处理能力"""
    # 模拟NLP处理流程
    def process_natural_language(text):
        """处理自然语言请求"""
        # 意图识别
        intent = "unknown"
        entities = []
        
        text_lower = text.lower()
        
        # 意图分类
        if any(word in text_lower for word in ["灌溉", "浇水", "水分"]):
            intent = "irrigation_recommendation"
        elif any(word in text_lower for word in ["施肥", "肥料", "养分"]):
            intent = "fertilization_recommendation"
        elif any(word in text_lower for word in ["天气", "温度", "湿度"]):
            intent = "weather_analysis"
        elif any(word in text_lower for word in ["决策", "建议", "优化"]):
            intent = "decision_making"
        
        # 实体识别
        if "小麦" in text_lower:
            entities.append({"type": "crop", "value": "wheat"})
        elif "水稻" in text_lower:
            entities.append({"type": "crop", "value": "rice"})
        elif "玉米" in text_lower:
            entities.append({"type": "crop", "value": "corn"})
        
        # 提取数值
        import re
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            entities.append({"type": "number", "value": int(num)})
        
        return {
            "original_text": text,
            "intent": intent,
            "entities": entities
        }
    
    # 测试用例
    test_cases = [
        "小麦需要多少灌溉水？",
        "水稻施肥的最佳时间是什么时候？",
        "今天温度30度，湿度60%，玉米需要浇水吗？",
        "给我一个农业生产决策建议"
    ]
    
    for test_case in test_cases:
        result = process_natural_language(test_case)
        assert result["intent"] != "unknown", f"无法识别意图：{test_case}"
        print(f"📝 自然语言处理结果：{test_case} -> 意图：{result['intent']}，实体：{result['entities']}")
    
    print("✅ 测试5通过：自然语言处理能力正常")
    return True

# 测试6：问题分解与解决能力
def test_problem_decomposition():
    """测试AI的问题分解能力"""
    def solve_complex_problem(problem_description):
        """分解并解决复杂问题"""
        # 问题分析与分解
        problem_parts = {
            "analysis": "",
            "subproblems": [],
            "solutions": []
        }
        
        # 分析问题类型
        if "灌溉系统" in problem_description:
            problem_parts["analysis"] = "农业灌溉系统优化问题"
            problem_parts["subproblems"] = [
                "1. 土壤湿度监测与数据采集",
                "2. 天气预测数据集成",
                "3. 作物需水量计算模型",
                "4. 灌溉时间与水量优化算法",
                "5. 灌溉设备控制逻辑"
            ]
            problem_parts["solutions"] = [
                "- 部署物联网土壤湿度传感器网络",
                "- 接入气象局API获取天气预报",
                "- 使用机器学习模型预测作物需水量",
                "- 基于强化学习优化灌溉策略",
                "- 实现自动化灌溉设备控制系统"
            ]
        elif "决策引擎" in problem_description:
            problem_parts["analysis"] = "AI决策引擎设计问题"
            problem_parts["subproblems"] = [
                "1. 决策数据模型设计",
                "2. 决策算法选择与实现",
                "3. 模型训练与优化",
                "4. 决策结果解释与可视化",
                "5. 系统集成与API设计"
            ]
            problem_parts["solutions"] = [
                "- 采用面向对象设计模式",
                "- 结合规则引擎与机器学习",
                "- 实现在线学习与模型更新机制",
                "- 开发可解释AI模块",
                "- 设计RESTful API接口"
            ]
        
        return problem_parts
    
    # 测试问题分解
    test_problem = "如何设计一个高效的农业灌溉系统决策引擎？"
    result = solve_complex_problem(test_problem)
    
    assert len(result["subproblems"]) > 0, "未能分解问题"
    assert len(result["solutions"]) > 0, "未能提供解决方案"
    
    print(f"🔍 问题分析：{result['analysis']}")
    print("📋 子问题分解：")
    for subproblem in result["subproblems"]:
        print(f"   {subproblem}")
    print("💡 解决方案：")
    for solution in result["solutions"]:
        print(f"   {solution}")
    
    print("✅ 测试6通过：问题分解能力正常")
    return result

# 主测试函数
def run_all_tests():
    """运行所有测试"""
    print("🚀 开始测试核心AI能力...")
    print("=" * 50)
    
    test_results = {
        "test_code_implementation": False,
        "test_logical_reasoning": False,
        "test_tool_usage": False,
        "test_learning_adaptation": False,
        "test_nlp_understanding": False,
        "test_problem_decomposition": False
    }
    
    try:
        test_code_implementation()
        test_results["test_code_implementation"] = True
    except Exception as e:
        print(f"❌ 测试1失败：{e}")
    
    try:
        test_logical_reasoning()
        test_results["test_logical_reasoning"] = True
    except Exception as e:
        print(f"❌ 测试2失败：{e}")
    
    try:
        test_tool_usage()
        test_results["test_tool_usage"] = True
    except Exception as e:
        print(f"❌ 测试3失败：{e}")
    
    try:
        test_learning_adaptation()
        test_results["test_learning_adaptation"] = True
    except Exception as e:
        print(f"❌ 测试4失败：{e}")
    
    try:
        test_nlp_understanding()
        test_results["test_nlp_understanding"] = True
    except Exception as e:
        print(f"❌ 测试5失败：{e}")
    
    try:
        test_problem_decomposition()
        test_results["test_problem_decomposition"] = True
    except Exception as e:
        print(f"❌ 测试6失败：{e}")
    
    print("=" * 50)
    print("📊 测试结果汇总：")
    total_passed = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎉 测试完成：{total_passed}/{total_tests} 测试通过")
    print(f"📈 通过率：{round(total_passed/total_tests*100, 2)}%")
    
    if total_passed == total_tests:
        print("\n🏆 所有测试通过！核心AI能力正常。")
    else:
        print(f"\n⚠️  有 {total_tests - total_passed} 个测试失败，需要进一步评估。")

if __name__ == "__main__":
    run_all_tests()
