"""NELLIE Neural Symbolic Reasoning Service

集成NELLIE系统，实现神经符号推理功能
"""

import os
import logging
from typing import Dict, List, Optional, Any
import aiohttp

logger = logging.getLogger(__name__)

class NELLIEService:
    """NELLIE神经符号推理服务类"""
    
    def __init__(self):
        self.api_key = os.getenv("NELLIE_API_KEY")
        self.api_url = "https://api.nellie.ai/v1/reason"
        self.is_enabled = self.api_key is not None
        
        if self.is_enabled:
            logger.info("✅ NELLIE神经符号推理服务初始化成功")
        else:
            logger.warning("⚠️ NELLIE API密钥未配置，神经符号推理功能将被禁用")
    
    async def perform_symbolic_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """执行神经符号推理
        
        Args:
            query: 推理查询
            context: 推理上下文信息
            
        Returns:
            推理结果，包含符号表示和推理过程
        """
        if not self.is_enabled or not query:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "query": query,
                "context": context or {},
                "return_symbolic_form": True,
                "return_reasoning_path": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_reasoning_result(result)
                    else:
                        logger.error(f"NELLIE API调用失败，状态码: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"NELLIE神经符号推理失败: {e}")
            return None
    
    def _parse_reasoning_result(self, result: Dict) -> Dict[str, Any]:
        """解析NELLIE API返回的推理结果
        
        Args:
            result: API返回的原始结果
            
        Returns:
            格式化的推理结果
        """
        parsed_result = {
            "symbolic_form": result.get("symbolic_form", ""),
            "reasoning_path": result.get("reasoning_path", []),
            "conclusion": result.get("conclusion", ""),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", "")
        }
        
        return parsed_result
    
    async def symbolize_text(self, text: str) -> Optional[Dict[str, Any]]:
        """将文本转换为符号表示
        
        Args:
            text: 要转换的文本
            
        Returns:
            符号化结果
        """
        try:
            logger.info(f"正在符号化文本: {text[:50]}...")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "text": text,
                "task": "symbolic_representation"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/symbolize", headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"NELLIE符号化调用失败，状态码: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"NELLIE符号化失败: {e}")
            return None
    
    async def validate_reasoning(self, symbolic_form: str, conclusion: str) -> Optional[Dict[str, Any]]:
        """验证推理过程的正确性
        
        Args:
            symbolic_form: 符号表示
            conclusion: 结论
            
        Returns:
            验证结果
        """
        if not self.is_enabled or not symbolic_form:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "symbolic_form": symbolic_form,
                "conclusion": conclusion,
                "task": "validate_reasoning"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/validate", headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"NELLIE推理验证失败，状态码: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"NELLIE推理验证失败: {e}")
            return None
    
    def enhance_explanation(self, conclusion: str, reasoning_path: List[Dict]) -> str:
        """增强推理解释的可读性
        
        Args:
            conclusion: 推理结论
            reasoning_path: 推理路径
            
        Returns:
            增强后的解释文本
        """
        if not conclusion or not reasoning_path:
            return conclusion
        
        explanation = f"结论: {conclusion}\n\n推理过程:\n"
        
        for i, step in enumerate(reasoning_path, 1):
            explanation += f"{i}. {step.get('description', '')}" 
            if 'confidence' in step:
                explanation += f" (可信度: {step['confidence']:.2f})"
            explanation += "\n"
        
        return explanation

# 创建单例实例
nellie_service = NELLIEService()
