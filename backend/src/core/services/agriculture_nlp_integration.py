"""
农业领域自然语言交互集成服务
将NLP服务与农业决策引擎集成，实现自然语言驱动的智能农业决策
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .nlp_service import NLPService, IntentResult
from ..decision.agriculture_decision_engine import AgricultureDecisionEngine
from ..models.agriculture_model import AgricultureAIService


@dataclass
class AgricultureNLPChatResult:
    """农业NLP聊天结果"""
    response: str
    intent: str
    confidence: float
    action_taken: Optional[str] = None
    decision_result: Optional[Dict[str, Any]] = None
    conversation_state: Optional[Dict[str, Any]] = None


class AgricultureNLPIntegrationService:
    """农业NLP集成服务"""
    
    def __init__(self):
        # 初始化各个服务
        self.nlp_service = NLPService()
        self.agriculture_ai_service = AgricultureAIService()
        self.decision_engine = AgricultureDecisionEngine(self.agriculture_ai_service)
        
        # 意图到决策目标的映射
        self.intent_to_objective = {
            "QUERY_CROP_GROWTH": "maximize_yield",
            "IRRIGATION_ADVICE": "optimize_efficiency",
            "FERTILIZATION_ADVICE": "maximize_yield",
            "WEATHER_IMPACT": "enhance_resistance",
            "PEST_CONTROL_ADVICE": "enhance_resistance",
            "YIELD_PREDICTION": "maximize_yield",
            "GROWTH_STAGE_ADVICE": "maximize_yield",
            "SOIL_MANAGEMENT_ADVICE": "maximize_yield",
            "CROP_VARIETY_RECOMMENDATION": "improve_quality",
            "HARVEST_TIME_ADVICE": "improve_quality"
        }
        
        # 作物名称映射
        self.crop_name_mapping = {
            "小麦": "wheat",
            "水稻": "rice",
            "玉米": "corn",
            "番茄": "tomato",
            "黄瓜": "cucumber",
            "大豆": "soybean",
            "棉花": "cotton",
            "土豆": "potato",
            "胡萝卜": "carrot",
            "生菜": "lettuce"
        }
    
    def process_chat_input(self, user_input: str, context: Dict[str, Any] = None) -> AgricultureNLPChatResult:
        """
        处理用户聊天输入，生成响应
        
        Args:
            user_input: 用户输入的自然语言文本
            context: 上下文信息
            
        Returns:
            聊天结果，包含响应文本、意图、置信度等
        """
        # 使用NLP服务处理输入
        intent_result = self.nlp_service.process_text(user_input, context)
        
        # 获取对话状态
        conversation_state = self.nlp_service.get_conversation_state()
        
        # 基于意图生成响应
        response = ""
        action_taken = None
        decision_result = None
        
        # 如果需要澄清，直接返回澄清问题
        if conversation_state["clarification_needed"] and conversation_state["clarification_question"]:
            return AgricultureNLPChatResult(
                response=conversation_state["clarification_question"],
                intent=intent_result.intent,
                confidence=intent_result.confidence,
                conversation_state=conversation_state
            )
        
        # 基于意图处理
        if intent_result.intent == "UNKNOWN":
            response = "对不起，我不太理解您的意思。您能换一种方式表达吗？"
        else:
            # 根据意图调用相应的处理方法
            if hasattr(self, f"_handle_{intent_result.intent.lower()}"):
                handler = getattr(self, f"_handle_{intent_result.intent.lower()}")
                response, action_taken, decision_result = handler(intent_result, conversation_state)
            else:
                # 默认响应
                response = f"我已经理解您的意图：{intent_result.intent}，正在为您处理..."
        
        return AgricultureNLPChatResult(
            response=response,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            action_taken=action_taken,
            decision_result=decision_result,
            conversation_state=conversation_state
        )
    
    def _handle_query_crop_growth(self, intent_result: IntentResult, conversation_state: Dict[str, Any]) -> tuple:
        """处理查询作物生长状态意图"""
        # 从实体中提取作物类型
        crop_name = self._extract_crop_name(intent_result.entities)
        
        if not crop_name:
            return "请问您想查询哪种作物的生长状态？", None, None
        
        # 构建决策请求
        decision_request = self._build_decision_request(crop_name, "maximize_yield")
        
        # 调用决策引擎
        decision_result = self.decision_engine.make_decision(decision_request)
        
        # 生成自然语言响应
        response = self._generate_growth_response(crop_name, decision_result)
        
        return response, "query_growth", decision_result
    
    def _handle_irrigation_advice(self, intent_result: IntentResult, conversation_state: Dict[str, Any]) -> tuple:
        """处理灌溉建议意图"""
        crop_name = self._extract_crop_name(intent_result.entities)
        
        if not crop_name:
            return "请问您想为哪种作物获取灌溉建议？", None, None
        
        decision_request = self._build_decision_request(crop_name, "optimize_efficiency")
        decision_result = self.decision_engine.make_decision(decision_request)
        
        response = self._generate_irrigation_response(crop_name, decision_result)
        
        return response, "irrigation_advice", decision_result
    
    def _handle_fertilization_advice(self, intent_result: IntentResult, conversation_state: Dict[str, Any]) -> tuple:
        """处理施肥建议意图"""
        crop_name = self._extract_crop_name(intent_result.entities)
        
        if not crop_name:
            return "请问您想为哪种作物获取施肥建议？", None, None
        
        decision_request = self._build_decision_request(crop_name, "maximize_yield")
        decision_result = self.decision_engine.make_decision(decision_request)
        
        response = self._generate_fertilization_response(crop_name, decision_result)
        
        return response, "fertilization_advice", decision_result
    
    def _handle_weather_impact(self, intent_result: IntentResult, conversation_state: Dict[str, Any]) -> tuple:
        """处理天气影响分析意图"""
        crop_name = self._extract_crop_name(intent_result.entities)
        
        if not crop_name:
            return "请问您想了解哪种作物的天气影响分析？", None, None
        
        decision_request = self._build_decision_request(crop_name, "enhance_resistance")
        decision_result = self.decision_engine.make_decision(decision_request)
        
        response = self._generate_weather_impact_response(crop_name, decision_result)
        
        return response, "weather_impact_analysis", decision_result
    
    def _handle_pest_control_advice(self, intent_result: IntentResult, conversation_state: Dict[str, Any]) -> tuple:
        """处理病虫害防治建议意图"""
        crop_name = self._extract_crop_name(intent_result.entities)
        
        if not crop_name:
            return "请问您想为哪种作物获取病虫害防治建议？", None, None
        
        decision_request = self._build_decision_request(crop_name, "enhance_resistance")
        decision_result = self.decision_engine.make_decision(decision_request)
        
        response = self._generate_pest_control_response(crop_name, decision_result)
        
        return response, "pest_control_advice", decision_result
    
    def _handle_yield_prediction(self, intent_result: IntentResult, conversation_state: Dict[str, Any]) -> tuple:
        """处理产量预测意图"""
        crop_name = self._extract_crop_name(intent_result.entities)
        
        if not crop_name:
            return "请问您想预测哪种作物的产量？", None, None
        
        decision_request = self._build_decision_request(crop_name, "maximize_yield")
        decision_result = self.decision_engine.make_decision(decision_request)
        
        response = self._generate_yield_prediction_response(crop_name, decision_result)
        
        return response, "yield_prediction", decision_result
    
    def _extract_crop_name(self, entities: List) -> Optional[str]:
        """从实体列表中提取作物名称"""
        for entity in entities:
            if entity.type == "crop":
                return entity.value
        return None
    
    def _build_decision_request(self, crop_name: str, objective: str) -> Dict[str, Any]:
        """
        构建决策请求
        
        Args:
            crop_name: 作物名称（中文）
            objective: 决策目标
            
        Returns:
            决策请求字典
        """
        # 转换为英文作物名称
        crop_type = self.crop_name_mapping.get(crop_name, "tomato")
        
        # 构建默认的决策请求
        return {
            "temperature": 25.0,
            "humidity": 60.0,
            "co2_level": 400.0,
            "light_intensity": 500.0,
            "soil_temperature": 22.0,
            "soil_moisture": 50.0,
            "soil_ph": 6.5,
            "wind_speed": 5.0,
            "rainfall": 0.0,
            "uv_index": 5.0,
            "spectrum_config": {
                "uv_380nm": 0.05,
                "far_red_720nm": 0.1,
                "white_red_ratio": 0.5,
                "white_light": 0.4
            },
            "crop_type": crop_type,
            "growth_day": 30,
            "growth_rate": 0.8,
            "health_score": 0.9,
            "yield_potential": 0.85,
            "leaf_area_index": 1.0,
            "nutrient_level": {
                "n": 50,
                "p": 30,
                "k": 40
            },
            "pest_risk": 0.3,
            "disease_risk": 0.2,
            "energy_consumption": 50.0,
            "resource_utilization": 0.7,
            "water_consumption": 20.0,
            "fertilizer_consumption": 5.0,
            "equipment_status": {
                "pump": "normal",
                "heater": "normal"
            },
            "objective": objective
        }
    
    def _generate_growth_response(self, crop_name: str, decision_result: Dict[str, Any]) -> str:
        """生成作物生长状态响应"""
        visualization = decision_result.get("visualization", {})
        state_comparison = visualization.get("state_comparison", {})
        
        response = f"{crop_name}的当前生长状态如下：\n"
        
        # 温度状态
        temp = state_comparison.get("temperature", {})
        if temp:
            response += f"- 温度：{temp['current']}°C（最优：{temp['optimal']}°C）\n"
        
        # 湿度状态
        humidity = state_comparison.get("humidity", {})
        if humidity:
            response += f"- 湿度：{humidity['current']}%（最优：{humidity['optimal']}%）\n"
        
        # 土壤湿度
        soil_moisture = state_comparison.get("soil_moisture", {})
        if soil_moisture:
            response += f"- 土壤湿度：{soil_moisture['current']}%（最优：{soil_moisture['optimal']}%）\n"
        
        # 建议动作
        action = decision_result.get("action", "")
        if action:
            response += f"\n建议：{self._action_to_chinese(action)}\n"
        
        return response
    
    def _generate_irrigation_response(self, crop_name: str, decision_result: Dict[str, Any]) -> str:
        """生成灌溉建议响应"""
        visualization = decision_result.get("visualization", {})
        soil_moisture = visualization.get("state_comparison", {}).get("soil_moisture", {})
        
        response = f"{crop_name}的灌溉建议：\n"
        
        if soil_moisture:
            current = soil_moisture['current']
            optimal = soil_moisture['optimal']
            
            if current < optimal:
                response += f"当前土壤湿度为{current}%，低于最优值{optimal}%，建议增加灌溉量。\n"
            elif current > optimal:
                response += f"当前土壤湿度为{current}%，高于最优值{optimal}%，建议减少灌溉量。\n"
            else:
                response += f"当前土壤湿度为{current}%，处于最优范围，无需调整。\n"
        
        # 预期效果
        expected_effects = visualization.get("expected_effects", {})
        if expected_effects:
            water_savings = expected_effects.get("resource_savings", {}).get("water", 0)
            if water_savings > 0:
                response += f"\n按照建议调整后，预计可节省{water_savings*100}%的水资源。\n"
        
        return response
    
    def _generate_fertilization_response(self, crop_name: str, decision_result: Dict[str, Any]) -> str:
        """生成施肥建议响应"""
        action = decision_result.get("action", "")
        parameters = decision_result.get("parameters", {})
        
        response = f"{crop_name}的施肥建议：\n"
        
        if action == "adjust_nutrients":
            npk_adjustment = parameters.get("npk_ratio_adjustment", 0)
            if npk_adjustment > 0:
                response += "建议增加养分供应，调整氮磷钾比例。\n"
            elif npk_adjustment < 0:
                response += "建议减少养分供应，调整氮磷钾比例。\n"
        
        # 预期效果
        visualization = decision_result.get("visualization", {})
        expected_effects = visualization.get("expected_effects", {})
        if expected_effects:
            growth_improvement = expected_effects.get("growth_rate_improvement", 0)
            if growth_improvement > 0:
                response += f"\n按照建议调整后，预计生长率可提高{growth_improvement*100}%。\n"
        
        return response
    
    def _generate_weather_impact_response(self, crop_name: str, decision_result: Dict[str, Any]) -> str:
        """生成天气影响分析响应"""
        response = f"天气条件对{crop_name}的影响分析：\n"
        
        # 关键影响因素
        visualization = decision_result.get("visualization", {})
        factors = visualization.get("decision_rationale", {}).get("key_factors", [])
        if factors:
            response += "关键影响因素：\n"
            for factor in factors:
                response += f"- {factor}\n"
        
        # 建议
        action = decision_result.get("action", "")
        if action:
            response += f"\n应对建议：{self._action_to_chinese(action)}\n"
        
        return response
    
    def _generate_pest_control_response(self, crop_name: str, decision_result: Dict[str, Any]) -> str:
        """生成病虫害防治建议响应"""
        visualization = decision_result.get("visualization", {})
        decision_rationale = visualization.get("decision_rationale", {})
        
        response = f"{crop_name}的病虫害防治建议：\n"
        
        # 风险评估
        factors = decision_rationale.get("key_factors", [])
        if factors:
            response += "当前风险因素：\n"
            for factor in factors:
                response += f"- {factor}\n"
        
        # 建议动作
        action = decision_result.get("action", "")
        if action:
            response += f"\n建议措施：{self._action_to_chinese(action)}\n"
        
        return response
    
    def _generate_yield_prediction_response(self, crop_name: str, decision_result: Dict[str, Any]) -> str:
        """生成产量预测响应"""
        visualization = decision_result.get("visualization", {})
        expected_effects = visualization.get("expected_effects", {})
        
        response = f"{crop_name}的产量预测：\n"
        
        # 预期产量提升
        yield_improvement = expected_effects.get("yield_improvement", 0)
        if yield_improvement > 0:
            response += f"按照当前管理方式，预计产量可提高{yield_improvement*100}%。\n"
        
        # 建议优化
        action = decision_result.get("action", "")
        if action:
            response += f"\n建议优化措施：{self._action_to_chinese(action)}\n"
        
        return response
    
    def _action_to_chinese(self, action: str) -> str:
        """将英文动作转换为中文"""
        action_mapping = {
            "adjust_spectrum": "调整光谱参数",
            "adjust_temperature": "调整温度",
            "adjust_humidity": "调整湿度",
            "adjust_nutrients": "调整养分供应",
            "control_camera": "启动监测设备",
            "no_action": "无需调整"
        }
        return action_mapping.get(action, action)
    
    def get_intent_stats(self) -> Dict[str, Any]:
        """获取意图统计信息"""
        return {
            "available_intents": list(self.intent_to_objective.keys()),
            "supported_crops": list(self.crop_name_mapping.keys())
        }
    
    def clear_context(self) -> None:
        """清除上下文"""
        self.nlp_service.clear_context()
