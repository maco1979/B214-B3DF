"""AI学习服务
实现AI模型的持续学习和迭代功能，包括数据收集、模型更新等
"""

from typing import Dict, List, Optional
import json
import os
import logging
import time
from datetime import datetime
import re

# 配置日志
logger = logging.getLogger(__name__)


class LearningService:
    """AI学习服务类，提供持续学习和迭代功能"""
    
    def __init__(self, data_dir: str = "./learning_data"):
        """初始化学习服务
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.interaction_data_file = os.path.join(data_dir, "interaction_data.jsonl")
        self.feedback_data_file = os.path.join(data_dir, "feedback_data.jsonl")
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"AI学习服务初始化完成，数据目录: {data_dir}")
    
    def save_interaction_data(self, interaction: Dict[str, any]):
        """保存交互数据
        
        Args:
            interaction: 交互数据，包含用户输入、AI响应、意图识别结果等
        """
        try:
            # 添加时间戳
            interaction["timestamp"] = datetime.now().isoformat()
            
            # 保存到JSONL文件
            with open(self.interaction_data_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
            
            logger.debug(f"交互数据保存成功")
            return True
        except Exception as e:
            logger.error(f"保存交互数据失败: {str(e)}")
            return False
    
    def save_feedback_data(self, feedback: Dict[str, any]):
        """保存反馈数据
        
        Args:
            feedback: 反馈数据，包含用户反馈、交互ID等
        """
        try:
            # 添加时间戳
            feedback["timestamp"] = datetime.now().isoformat()
            
            # 保存到JSONL文件
            with open(self.feedback_data_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
            
            logger.debug(f"反馈数据保存成功")
            return True
        except Exception as e:
            logger.error(f"保存反馈数据失败: {str(e)}")
            return False
    
    def get_interaction_data(self, limit: Optional[int] = None) -> List[Dict[str, any]]:
        """获取交互数据
        
        Args:
            limit: 返回数据的数量限制
            
        Returns:
            交互数据列表
        """
        try:
            data = []
            if os.path.exists(self.interaction_data_file):
                with open(self.interaction_data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            
            # 限制返回数量
            if limit:
                data = data[-limit:]
            
            return data
        except Exception as e:
            logger.error(f"获取交互数据失败: {str(e)}")
            return []
    
    def get_feedback_data(self, limit: Optional[int] = None) -> List[Dict[str, any]]:
        """获取反馈数据
        
        Args:
            limit: 返回数据的数量限制
            
        Returns:
            反馈数据列表
        """
        try:
            data = []
            if os.path.exists(self.feedback_data_file):
                with open(self.feedback_data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            
            # 限制返回数量
            if limit:
                data = data[-limit:]
            
            return data
        except Exception as e:
            logger.error(f"获取反馈数据失败: {str(e)}")
            return []
    
    def analyze_interaction_data(self) -> Dict[str, any]:
        """分析交互数据，生成分析报告
        
        Returns:
            分析报告，包含意图分布、交互次数等统计信息
        """
        try:
            interactions = self.get_interaction_data()
            if not interactions:
                return {"message": "没有交互数据"}
            
            # 统计意图分布
            intent_counts = {}
            for interaction in interactions:
                intent = interaction.get("intent", "unknown")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # 统计交互次数
            total_interactions = len(interactions)
            
            # 统计平均响应时间
            response_times = []
            for interaction in interactions:
                if "response_time" in interaction:
                    response_times.append(interaction["response_time"])
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # 统计不同输入类型的分布
            input_type_counts = {}
            for interaction in interactions:
                input_type = interaction.get("input_type", "text")
                input_type_counts[input_type] = input_type_counts.get(input_type, 0) + 1
            
            # 生成分析报告
            report = {
                "total_interactions": total_interactions,
                "intent_distribution": intent_counts,
                "input_type_distribution": input_type_counts,
                "avg_response_time": avg_response_time,
                "latest_interaction": interactions[-1] if interactions else None
            }
            
            return report
        except Exception as e:
            logger.error(f"分析交互数据失败: {str(e)}")
            return {"error": str(e)}
    
    def extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取，实际项目中可以使用更复杂的算法
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 分割成单词
        words = text.split()
        # 过滤掉常见停用词
        stop_words = set(["的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"])
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords
    
    def analyze_user_feedback(self) -> Dict[str, List[str]]:
        """分析用户反馈，提取改进建议
        
        Returns:
            反馈分析结果，包含需要改进的意图和建议的关键词
        """
        try:
            feedbacks = self.get_feedback_data()
            if not feedbacks:
                return {}
            
            # 分析负面反馈
            negative_feedbacks = [f for f in feedbacks if f.get("type") == "negative"]
            if not negative_feedbacks:
                return {}
            
            # 从负面反馈相关的交互中提取改进建议
            interaction_ids = [f.get("interaction_id") for f in negative_feedbacks if f.get("interaction_id")]
            interactions = self.get_interaction_data()
            related_interactions = [i for i in interactions if i.get("context_id") in interaction_ids]
            
            # 分析交互数据，提取可能的关键词
            intent_keyword_suggestions = {}
            for interaction in related_interactions:
                intent = interaction.get("intent")
                if intent == "unknown":
                    # 对于未知意图，尝试提取关键词
                    user_input = interaction.get("user_input", "")
                    keywords = self.extract_keywords(user_input)
                    if keywords:
                        # 尝试确定正确的意图（这里可以根据上下文或其他方式改进）
                        # 暂时使用CHAT作为默认意图
                        suggested_intent = "chat"
                        if suggested_intent not in intent_keyword_suggestions:
                            intent_keyword_suggestions[suggested_intent] = []
                        intent_keyword_suggestions[suggested_intent].extend(keywords)
            
            # 去重并返回结果
            for intent, keywords in intent_keyword_suggestions.items():
                intent_keyword_suggestions[intent] = list(set(keywords))
            
            return intent_keyword_suggestions
        except Exception as e:
            logger.error(f"分析用户反馈失败: {str(e)}")
            return {}
    
    def update_model(self):
        """更新AI模型
        
        基于收集的数据更新AI模型，包括：
        1. 分析用户反馈
        2. 提取新的关键词
        3. 更新意图识别规则
        4. 优化响应生成
        
        Returns:
            更新结果信息
        """
        try:
            logger.info("开始更新AI模型...")
            
            # 1. 导入AI模型服务
            from src.core.services.ai_model_service import aimodel_service
            
            # 2. 分析用户反馈，提取关键词建议
            feedback_analysis = self.analyze_user_feedback()
            
            # 3. 分析交互数据，识别高频未知意图
            interactions = self.get_interaction_data(limit=100)
            unknown_intent_interactions = [i for i in interactions if i.get("intent") == "unknown"]
            
            # 4. 从高频未知意图中提取关键词
            if unknown_intent_interactions:
                logger.info(f"发现 {len(unknown_intent_interactions)} 条未知意图交互")
                
                # 统计高频词
                word_counts = {}
                for interaction in unknown_intent_interactions:
                    user_input = interaction.get("user_input", "")
                    keywords = self.extract_keywords(user_input)
                    for keyword in keywords:
                        word_counts[keyword] = word_counts.get(keyword, 0) + 1
                
                # 提取高频关键词（出现次数大于2）
                high_freq_keywords = [word for word, count in word_counts.items() if count > 2]
                
                if high_freq_keywords:
                    logger.info(f"提取到高频关键词: {high_freq_keywords}")
                    # 为CHAT意图添加这些关键词
                    aimodel_service.add_intent_keywords("chat", high_freq_keywords)
            
            # 5. 根据反馈分析结果更新意图规则
            if feedback_analysis:
                logger.info(f"根据用户反馈更新意图规则: {feedback_analysis}")
                for intent, keywords in feedback_analysis.items():
                    aimodel_service.add_intent_keywords(intent, keywords)
            
            # 6. 保存模型更新记录
            update_record = {
                "timestamp": datetime.now().isoformat(),
                "feedback_analysis": feedback_analysis,
                "unknown_intent_count": len(unknown_intent_interactions),
                "status": "success"
            }
            
            update_log_file = os.path.join(self.data_dir, "model_updates.jsonl")
            with open(update_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(update_record, ensure_ascii=False) + "\n")
            
            logger.info("AI模型更新完成")
            return {
                "status": "success", 
                "message": "AI模型更新完成",
                "details": {
                    "feedback_analysis_used": len(feedback_analysis) > 0,
                    "unknown_intents_processed": len(unknown_intent_interactions),
                    "timestamp": update_record["timestamp"]
                }
            }
        except Exception as e:
            logger.error(f"更新AI模型失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def clear_old_data(self, days: int = 30):
        """清理旧数据
        
        Args:
            days: 保留最近多少天的数据
            
        Returns:
            清理结果信息
        """
        try:
            cutoff_time = time.time() - days * 86400
            
            # 清理交互数据
            if os.path.exists(self.interaction_data_file):
                with open(self.interaction_data_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # 保留最近days天的数据
                new_lines = []
                for line in lines:
                    interaction = json.loads(line.strip())
                    interaction_time = datetime.fromisoformat(interaction["timestamp"]).timestamp()
                    if interaction_time >= cutoff_time:
                        new_lines.append(line)
                
                # 写回文件
                with open(self.interaction_data_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            
            # 清理反馈数据
            if os.path.exists(self.feedback_data_file):
                with open(self.feedback_data_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # 保留最近days天的数据
                new_lines = []
                for line in lines:
                    feedback = json.loads(line.strip())
                    feedback_time = datetime.fromisoformat(feedback["timestamp"]).timestamp()
                    if feedback_time >= cutoff_time:
                        new_lines.append(line)
                
                # 写回文件
                with open(self.feedback_data_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            
            logger.info(f"清理旧数据完成，保留最近{days}天的数据")
            return {"status": "success", "message": f"清理旧数据完成，保留最近{days}天的数据"}
        except Exception as e:
            logger.error(f"清理旧数据失败: {str(e)}")
            return {"status": "error", "message": str(e)}


# 单例模式
learning_service = LearningService()