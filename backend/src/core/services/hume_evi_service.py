"""Hume AI EVI Service

集成Hume AI的情感识别系统(EVI)，支持53种情绪识别
"""

import os
import logging
from typing import Dict, List, Optional, Any
import aiohttp

logger = logging.getLogger(__name__)

class HumeEVIService:
    """Hume AI EVI服务类"""
    
    def __init__(self):
        self.api_key = os.getenv("HUME_API_KEY")
        self.api_url = "https://api.hume.ai/v1/analyze"
        self.is_enabled = self.api_key is not None
        self.emotion_categories = self._get_emotion_categories()
        
        if self.is_enabled:
            logger.info("✅ Hume AI EVI服务初始化成功")
        else:
            logger.warning("⚠️ Hume AI API密钥未配置，情感识别功能将被禁用")
    
    def _get_emotion_categories(self) -> List[str]:
        """获取Hume AI支持的53种情绪类别"""
        return [
            "admiration", "adoration", "aesthetic_appreciation", "amusement", "anger",
            "anxiety", "awe", "awkwardness", "boredom", "calmness",
            "concentration", "confusion", "contemplation", "contempt", "contentment",
            "craving", "desire", "determination", "disappointment", "disapproval",
            "disgust", "distress", "doubt", "ecstasy", "embarrassment",
            "empathic_pain", "envy", "excitement", "fear", "guilt",
            "horror", "interest", "joy", "love", "nostalgia",
            "pain", "pride", "realization", "relief", "romance",
            "sadness", "satisfaction", "shame", "surprise", "sympathy",
            "tiredness", "triumph", "trust", "anticipation", "annoyance",
            "apprehension", "hopefulness", "irritation", "pleasure", "sensitivity"
        ]
    
    async def analyze_emotions(self, text: str) -> Optional[Dict[str, float]]:
        """分析文本中的情绪
        
        Args:
            text: 要分析的文本
            
        Returns:
            情绪分析结果，键为情绪名称，值为置信度
        """
        if not self.is_enabled or not text:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Hume-Api-Key": self.api_key
            }
            
            payload = {
                "text": text,
                "models": {
                    "emotion": {}
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_emotion_result(result)
                    else:
                        logger.error(f"Hume AI API调用失败，状态码: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Hume AI情绪分析失败: {e}")
            return None
    
    def _parse_emotion_result(self, result: Dict) -> Dict[str, float]:
        """解析Hume AI API返回的情绪结果
        
        Args:
            result: API返回的原始结果
            
        Returns:
            格式化的情绪分析结果
        """
        emotions = {}
        
        try:
            # 解析API响应格式
            if "results" in result and len(result["results"]) > 0:
                prediction = result["results"][0].get("predictions", [])
                if prediction and len(prediction) > 0:
                    emotion_predictions = prediction[0].get("emotions", [])
                    for emotion in emotion_predictions:
                        emotion_name = emotion.get("name", "")
                        emotion_score = emotion.get("score", 0.0)
                        if emotion_name and emotion_score > 0:
                            emotions[emotion_name] = round(emotion_score, 4)
        except Exception as e:
            logger.error(f"解析Hume AI结果失败: {e}")
        
        return emotions
    
    def get_dominant_emotions(self, emotions: Dict[str, float], top_n: int = 3) -> List[Dict[str, Any]]:
        """获取主导情绪
        
        Args:
            emotions: 情绪分析结果
            top_n: 返回前N个主导情绪
            
        Returns:
            主导情绪列表，包含情绪名称和置信度
        """
        if not emotions:
            return []
        
        # 按置信度排序，返回前N个
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{
            "name": emotion[0],
            "score": emotion[1]
        } for emotion in sorted_emotions]
    
    def generate_emotional_response(self, emotions: Dict[str, float], context: str) -> str:
        """基于情绪分析结果生成情感响应
        
        Args:
            emotions: 情绪分析结果
            context: 对话上下文
            
        Returns:
            情感响应文本
        """
        if not emotions:
            return ""
        
        dominant_emotions = self.get_dominant_emotions(emotions)
        if not dominant_emotions:
            return ""
        
        # 生成情感响应
        response = "我注意到你现在的情绪："
        
        for i, emotion in enumerate(dominant_emotions):
            if i > 0:
                if i == len(dominant_emotions) - 1:
                    response += " 和"
                else:
                    response += "、"
            response += f"{emotion['name']}（{emotion['score']:.2f}）"
        
        response += "。"
        
        # 根据情绪添加相应的回应
        primary_emotion = dominant_emotions[0]["name"]
        if primary_emotion in ["anger", "irritation", "annoyance"]:
            response += " 我理解你现在可能感到愤怒，我会尽力帮助你解决问题。"
        elif primary_emotion in ["sadness", "disappointment", "distress"]:
            response += " 我很抱歉听到这个消息，我在这里支持你。"
        elif primary_emotion in ["joy", "happiness", "satisfaction"]:
            response += " 看到你这么开心，我也很高兴！"
        elif primary_emotion in ["fear", "anxiety", "apprehension"]:
            response += " 别担心，我会陪你一起面对这个问题。"
        elif primary_emotion in ["interest", "curiosity", "excitement"]:
            response += " 我也对这个话题很感兴趣，让我们一起深入探讨吧！"
        
        return response

# 创建单例实例
hume_evi_service = HumeEVIService()
