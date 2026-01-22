"""LangSmith服务
提供LangSmith跟踪、评估和OpenAI判断功能
"""

from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI
import os
import logging
from typing import Optional, Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

class LangSmithService:
    """LangSmith服务类"""
    
    def __init__(self):
        self.client = None
        self.openai_client = None
        self.llm_judge = None
        self._initialize()
    
    def _initialize(self):
        """初始化LangSmith和OpenAI客户端"""
        try:
            # 检查环境变量
            if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
                logger.info("正在初始化LangSmith客户端...")
                self.client = Client(
                    api_url=os.getenv("LANGSMITH_ENDPOINT"),
                    api_key=os.getenv("LANGSMITH_API_KEY")
                )
                logger.info("✅ LangSmith客户端初始化成功")
            else:
                logger.info("LangSmith跟踪已禁用")
            
            # 初始化OpenAI客户端
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key and openai_api_key != "<your-openai-api-key>":
                logger.info("正在初始化OpenAI客户端...")
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI客户端初始化成功")
                
                # 创建LLM评估器
                self.llm_judge = create_llm_as_judge(
                    llm=self.openai_client,
                    prompt_template=CORRECTNESS_PROMPT
                )
                logger.info("✅ LLM评估器初始化成功")
            else:
                logger.warning("OpenAI API密钥未配置，LLM评估功能将不可用")
        except Exception as e:
            logger.error(f"初始化LangSmith服务失败: {str(e)}")
    
    def trace_ai_response(self, input_text: str, output_text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """跟踪AI响应
        
        Args:
            input_text: 用户输入文本
            output_text: AI输出文本
            metadata: 元数据
            
        Returns:
            跟踪ID
        """
        if not self.client:
            return None
        
        try:
            trace = self.client.create_trace(
                name="AI Assistant Response",
                metadata=metadata or {}
            )
            
            # 创建输入和输出步骤
            self.client.create_span(
                name="user_input",
                trace_id=trace.id,
                inputs={"input": input_text}
            )
            
            self.client.create_span(
                name="ai_output",
                trace_id=trace.id,
                inputs={"input": input_text},
                outputs={"output": output_text}
            )
            
            return trace.id
        except Exception as e:
            logger.error(f"跟踪AI响应失败: {str(e)}")
            return None
    
    def evaluate_response_correctness(self, input_text: str, output_text: str, expected: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """评估响应正确性
        
        Args:
            input_text: 用户输入文本
            output_text: AI输出文本
            expected: 期望输出
            
        Returns:
            评估结果
        """
        if not self.llm_judge:
            return None
        
        try:
            # 构建评估请求
            eval_data = {
                "input": input_text,
                "output": output_text,
                "expected": expected
            }
            
            # 执行评估
            result = self.llm_judge(eval_data)
            
            return {
                "correctness": result["score"],
                "reasoning": result["reasoning"],
                "eval_data": eval_data
            }
        except Exception as e:
            logger.error(f"评估响应正确性失败: {str(e)}")
            return None
    
    def get_client(self) -> Optional[Client]:
        """获取LangSmith客户端
        
        Returns:
            LangSmith客户端实例
        """
        return self.client
    
    def get_openai_client(self) -> Optional[OpenAI]:
        """获取OpenAI客户端
        
        Returns:
            OpenAI客户端实例
        """
        return self.openai_client
    
    def get_llm_judge(self) -> Optional[Any]:
        """获取LLM评估器
        
        Returns:
            LLM评估器实例
        """
        return self.llm_judge


# 单例模式
langsmith_service = LangSmithService()