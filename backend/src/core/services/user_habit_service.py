"""用户习惯服务
提供用户行为记录、习惯分析、习惯预测、自动推荐等功能
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import uuid
from collections import defaultdict

# 配置日志
logger = logging.getLogger(__name__)


class HabitCategory(str):
    """习惯类别枚举"""
    WORK = "work"  # 工作相关
    LEISURE = "leisure"  # 休闲娱乐
    HEALTH = "health"  # 健康生活
    PRODUCTIVITY = "productivity"  # 生产力
    COMMUNICATION = "communication"  # 沟通交流


class UserHabitService:
    """用户习惯服务类
    
    功能：
    1. 记录用户行为
    2. 分析用户习惯
    3. 预测用户行为
    4. 提供个性化推荐
    """
    
    def __init__(self, data_dir: str = "data/habits"):
        """初始化用户习惯服务
        
        Args:
            data_dir: 习惯数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.habit_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # 加载现有数据
        self._load_habit_data()
        self._load_user_profiles()
        
        logger.info("用户习惯服务初始化完成")
    
    def _load_habit_data(self):
        """加载习惯数据"""
        for habit_file in self.data_dir.glob("habit_*.json"):
            try:
                with open(habit_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    user_id = habit_file.stem.split("_", 1)[1]  # 从文件名提取用户ID
                    self.habit_data[user_id] = data
            except Exception as e:
                logger.error(f"加载习惯数据失败: {habit_file}，错误: {e}")
    
    def _load_user_profiles(self):
        """加载用户画像"""
        profile_file = self.data_dir / "user_profiles.json"
        if profile_file.exists():
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    self.user_profiles = json.load(f)
            except Exception as e:
                logger.error(f"加载用户画像失败: {e}")
    
    def _save_habit_data(self, user_id: str):
        """保存用户习惯数据"""
        habit_file = self.data_dir / f"habit_{user_id}.json"
        try:
            with open(habit_file, "w", encoding="utf-8") as f:
                json.dump(self.habit_data[user_id], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存习惯数据失败: {user_id}，错误: {e}")
    
    def _save_user_profiles(self):
        """保存用户画像"""
        profile_file = self.data_dir / "user_profiles.json"
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                json.dump(self.user_profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")
    
    def record_user_behavior(self, user_id: str, behavior_type: str, 
                           params: Dict[str, Any], timestamp: Optional[float] = None):
        """记录用户行为
        
        Args:
            user_id: 用户ID
            behavior_type: 行为类型（如：search, voice, device_control, etc.）
            params: 行为参数
            timestamp: 时间戳（可选）
        """
        if not timestamp:
            timestamp = datetime.now().timestamp()
        
        # 创建行为记录
        behavior_record = {
            "behavior_id": f"behavior_{uuid.uuid4()}",
            "user_id": user_id,
            "behavior_type": behavior_type,
            "params": params,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat()
        }
        
        # 添加到用户习惯数据
        self.habit_data[user_id].append(behavior_record)
        
        # 保存数据
        self._save_habit_data(user_id)
        
        # 更新用户画像
        self._update_user_profile(user_id, behavior_type, params)
        
        logger.info(f"记录用户行为: {user_id} - {behavior_type}")
        return behavior_record
    
    def _update_user_profile(self, user_id: str, behavior_type: str, params: Dict[str, Any]):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "behavior_counts": defaultdict(int),
                "preferences": {},
                "active_hours": defaultdict(int),
                "weekly_pattern": defaultdict(int),
                # 添加智能体使用统计
                "agent_interactions": defaultdict(lambda: {
                    "count": 0,
                    "success_count": 0,
                    "satisfaction": 0,
                    "total_score": 0
                }),
                "agent_preferences": {},
                "last_agent_used": None,
                "last_agent_time": None
            }
        
        profile = self.user_profiles[user_id]
        
        # 更新行为计数
        profile["behavior_counts"][behavior_type] += 1
        
        # 更新活跃时间
        hour = datetime.now().hour
        profile["active_hours"][hour] += 1
        
        # 更新周模式
        weekday = datetime.now().weekday()
        profile["weekly_pattern"][weekday] += 1
        
        # 更新偏好，示例：如果是设备控制，记录设备使用偏好
        if behavior_type == "device_control" and "device_type" in params:
            device_type = params["device_type"]
            if device_type not in profile["preferences"]:
                profile["preferences"][device_type] = defaultdict(int)
            
            if "action" in params:
                profile["preferences"][device_type][params["action"]] += 1
        
        # 更新智能体交互记录
        if behavior_type == "agent_interaction" and "agent_id" in params:
            agent_id = params["agent_id"]
            interaction_data = profile["agent_interactions"][agent_id]
            interaction_data["count"] += 1
            
            # 更新成功计数
            if params.get("success", True):
                interaction_data["success_count"] += 1
            
            # 更新满意度
            if "satisfaction" in params:
                satisfaction = params["satisfaction"]
                interaction_data["total_score"] += satisfaction
                interaction_data["satisfaction"] = interaction_data["total_score"] / interaction_data["count"]
            
            # 更新最后使用的智能体
            profile["last_agent_used"] = agent_id
            profile["last_agent_time"] = datetime.now().isoformat()
        
        # 更新最后更新时间
        profile["last_updated"] = datetime.now().isoformat()
        
        # 保存用户画像
        self._save_user_profiles()
    
    def analyze_user_habits(self, user_id: str, time_range_days: int = 7) -> Dict[str, Any]:
        """分析用户习惯
        
        Args:
            user_id: 用户ID
            time_range_days: 分析的时间范围（天）
            
        Returns:
            习惯分析结果
        """
        logger.info(f"分析用户习惯: {user_id}，时间范围: {time_range_days}天")
        
        # 获取用户行为数据
        user_behavior = self.habit_data.get(user_id, [])
        
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        start_timestamp = start_time.timestamp()
        
        # 过滤指定时间范围内的数据
        recent_behavior = [
            b for b in user_behavior 
            if b["timestamp"] >= start_timestamp
        ]
        
        if not recent_behavior:
            return {
                "user_id": user_id,
                "time_range": f"最近{time_range_days}天",
                "behavior_count": 0,
                "patterns": [],
                "preferences": {},
                "active_hours": []
            }
        
        # 行为类型统计
        behavior_counts = defaultdict(int)
        for b in recent_behavior:
            behavior_counts[b["behavior_type"]] += 1
        
        # 活跃时间分析
        hour_counts = defaultdict(int)
        for b in recent_behavior:
            hour = datetime.fromtimestamp(b["timestamp"]).hour
            hour_counts[hour] += 1
        
        # 找出最活跃的小时
        active_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        top_active_hours = [hour for hour, count in active_hours[:3]]
        
        # 分析设备使用偏好
        device_preferences = defaultdict(int)
        for b in recent_behavior:
            if b["behavior_type"] == "device_control" and "device_type" in b["params"]:
                device_preferences[b["params"]["device_type"]] += 1
        
        # 生成习惯模式
        patterns = []
        if behavior_counts.get("device_control", 0) > 3:  # 如果设备控制行为超过3次
            patterns.append({
                "pattern_type": "device_usage",
                "description": "频繁使用设备控制功能",
                "confidence": 0.8
            })
        
        if behavior_counts.get("search", 0) > 5:  # 如果搜索行为超过5次
            patterns.append({
                "pattern_type": "search_usage",
                "description": "频繁使用搜索功能",
                "confidence": 0.9
            })
        
        analysis_result = {
            "user_id": user_id,
            "time_range": f"最近{time_range_days}天",
            "behavior_count": len(recent_behavior),
            "behavior_distribution": dict(behavior_counts),
            "active_hours": top_active_hours,
            "device_preferences": dict(device_preferences),
            "patterns": patterns
        }
        
        logger.info(f"完成用户习惯分析: {user_id}")
        return analysis_result
    
    def predict_user_behavior(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """预测用户行为
        
        Args:
            user_id: 用户ID
            context: 上下文信息（如：时间、位置、当前活动等）
            
        Returns:
            预测结果
        """
        logger.info(f"预测用户行为: {user_id}，上下文: {context}")
        
        # 获取用户画像
        if user_id not in self.user_profiles:
            return {
                "success": False,
                "reason": "用户不存在"
            }
        
        profile = self.user_profiles[user_id]
        
        # 基于时间预测
        predictions = []
        
        # 当前时间
        now = datetime.now()
        current_hour = now.hour
        weekday = now.weekday()
        
        # 检查活跃时间
        if profile["active_hours"].get(current_hour, 0) > 2:  # 如果该时间段活跃次数超过2次
            # 检查设备使用偏好
            for device_type, actions in profile["preferences"].items():
                if actions:
                    # 找出该设备类型最常用的动作
                    top_action = max(actions.items(), key=lambda x: x[1])[0]
                    predictions.append({
                        "type": "device_action",
                        "device_type": device_type,
                        "action": top_action,
                        "confidence": 0.7,
                        "reason": f"该时段您通常使用{device_type}设备执行{top_action}操作"
                    })
        
        # 检查工作模式
        if 9 <= current_hour <= 18 and weekday < 5:  # 工作日9-18点
            if profile["behavior_counts"].get("work", 0) > profile["behavior_counts"].get("leisure", 0):
                predictions.append({
                    "type": "activity_suggestion",
                    "activity": "work_mode",
                    "confidence": 0.8,
                    "reason": "工作日期间您通常处于工作模式"
                })
        
        return {
            "success": True,
            "user_id": user_id,
            "predictions": predictions,
            "context": context
        }
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像"""
        return self.user_profiles.get(user_id)
    
    def analyze_agent_preferences(self, user_id: str) -> Dict[str, Any]:
        """分析用户的智能体偏好
        
        Args:
            user_id: 用户ID
            
        Returns:
            智能体偏好分析结果
        """
        logger.info(f"分析智能体偏好: {user_id}")
        
        # 获取用户画像
        profile = self.get_user_profile(user_id)
        if not profile:
            return {
                "user_id": user_id,
                "agent_preferences": {},
                "top_agents": [],
                "total_interactions": 0
            }
        
        agent_interactions = profile.get("agent_interactions", {})
        total_interactions = sum(interaction["count"] for interaction in agent_interactions.values())
        
        # 计算每个智能体的评分
        agent_scores = {}
        for agent_id, interaction in agent_interactions.items():
            count = interaction["count"]
            success_count = interaction["success_count"]
            satisfaction = interaction["satisfaction"]
            
            # 计算成功率
            success_rate = success_count / count if count > 0 else 0
            
            # 计算使用频率占比
            frequency = count / total_interactions if total_interactions > 0 else 0
            
            # 综合评分（成功率占50%，满意度占30%，频率占20%）
            score = (success_rate * 0.5) + (satisfaction * 0.3) + (frequency * 0.2)
            
            agent_scores[agent_id] = {
                "score": round(score, 3),
                "success_rate": round(success_rate, 3),
                "satisfaction": round(satisfaction, 3),
                "frequency": round(frequency, 3),
                "interaction_count": count,
                "success_count": success_count
            }
        
        # 对智能体进行排序
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top_agents = [{
            "agent_id": agent_id,
            "score": score_info["score"],
            "success_rate": score_info["success_rate"],
            "satisfaction": score_info["satisfaction"],
            "interaction_count": score_info["interaction_count"]
        } for agent_id, score_info in sorted_agents]
        
        # 更新用户画像中的智能体偏好
        profile["agent_preferences"] = agent_scores
        self._save_user_profiles()
        
        return {
            "user_id": user_id,
            "agent_preferences": agent_scores,
            "top_agents": top_agents,
            "total_interactions": total_interactions,
            "last_agent_used": profile.get("last_agent_used"),
            "last_agent_time": profile.get("last_agent_time")
        }
    
    def predict_agent_preference(self, user_id: str, task_type: str = "general", context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """预测用户在特定场景下偏好的智能体
        
        Args:
            user_id: 用户ID
            task_type: 任务类型
            context: 上下文信息
            
        Returns:
            智能体推荐列表
        """
        logger.info(f"预测智能体偏好: {user_id}，任务类型: {task_type}")
        
        # 分析用户的智能体偏好
        preference_analysis = self.analyze_agent_preferences(user_id)
        top_agents = preference_analysis.get("top_agents", [])
        
        # 如果没有足够的历史数据，返回空列表
        if not top_agents:
            return []
        
        # 基于任务类型和上下文进行调整
        # 这里可以添加更复杂的逻辑，比如根据任务类型匹配智能体的专长
        recommendations = []
        for agent in top_agents[:3]:  # 只返回前3个
            recommendations.append({
                "agent_id": agent["agent_id"],
                "score": agent["score"],
                "confidence": agent["score"],
                "reason": f"根据您的使用习惯，推荐该智能体（成功率: {agent['success_rate']:.2%}，满意度: {agent['satisfaction']:.2f}）"
            })
        
        return recommendations
    
    def get_habit_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """获取习惯推荐"""
        logger.info(f"生成习惯推荐: {user_id}")
        
        # 获取用户画像
        profile = self.get_user_profile(user_id)
        if not profile:
            return []
        
        recommendations = []
        
        # 基于活跃时间推荐
        now = datetime.now()
        current_hour = now.hour
        
        # 如果当前是活跃时间
        if profile["active_hours"].get(current_hour, 0) > 2:
            # 检查设备偏好
            for device_type, actions in profile["preferences"].items():
                if actions:
                    top_action = max(actions.items(), key=lambda x: x[1])[0]
                    recommendations.append({
                        "type": "device_command",
                        "device_type": device_type,
                        "action": top_action,
                        "reason": f"当前是您的活跃时段，建议执行常用的{device_type} {top_action}操作"
                    })
        
        # 基于周模式推荐
        weekday = now.weekday()
        if profile["weekly_pattern"].get(weekday, 0) > 3:  # 如果该工作日活跃超过3次
            # 检查是否有工作相关行为
            if profile["behavior_counts"].get("work", 0) > profile["behavior_counts"].get("leisure", 0):
                recommendations.append({
                    "type": "workflow",
                    "workflow": "work_mode",
                    "reason": f"工作日{weekday+1}您通常处于工作模式"
                })
        
        # 基于智能体偏好推荐
        agent_preferences = self.analyze_agent_preferences(user_id)
        top_agents = agent_preferences.get("top_agents", [])
        if top_agents:
            recommendations.append({
                "type": "agent_recommendation",
                "agent_id": top_agents[0]["agent_id"],
                "reason": f"根据您的使用习惯，推荐使用该智能体"
            })
        
        return recommendations


# 单例模式
user_habit_service = UserHabitService()
