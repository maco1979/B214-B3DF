"""Combination Creativity Framework Service

实现组合创造力框架，用于生成创造性想法和解决方案
"""

import logging
from typing import Dict, List, Optional, Any
import random
import uuid

logger = logging.getLogger(__name__)

class CreativityService:
    """组合创造力框架服务类"""
    
    def __init__(self):
        self.creativity_templates = self._load_creativity_templates()
        self.domain_knowledge = self._load_domain_knowledge()
        logger.info("✅ 组合创造力框架服务初始化成功")
    
    def _load_creativity_templates(self) -> List[Dict[str, Any]]:
        """加载创造力模板
        
        Returns:
            创造力模板列表
        """
        return [
            {
                "id": "template_1",
                "name": "领域交叉",
                "description": "将不同领域的概念结合",
                "structure": "{领域A} + {领域B} = {新创意}",
                "examples": ["人工智能 + 医疗 = 智能诊断", "区块链 + 供应链 = 透明溯源"]
            },
            {
                "id": "template_2",
                "name": "功能组合",
                "description": "将不同功能结合",
                "structure": "{功能A} + {功能B} + {功能C} = {新创意}",
                "examples": ["拍照 + 翻译 + 导航 = 智能旅行助手", "音乐 + 健身 + 社交 = 音乐健身社区"]
            },
            {
                "id": "template_3",
                "name": "属性转移",
                "description": "将一个事物的属性转移到另一个事物",
                "structure": "{事物A}的{属性}应用到{事物B} = {新创意}",
                "examples": ["鸟类的飞行能力应用到汽车 = 飞行汽车", "荷叶的自洁性应用到建筑材料 = 自洁涂料"]
            },
            {
                "id": "template_4",
                "name": "问题重构",
                "description": "从不同角度重构问题",
                "structure": "与其{传统方法}，不如{新角度} = {新创意}",
                "examples": ["与其花钱买水，不如从空气中取水 = 空气制水机", "与其开车上班，不如让工作来找你 = 远程办公"]
            },
            {
                "id": "template_5",
                "name": "尺度变换",
                "description": "改变事物的尺度",
                "structure": "{事物}的{尺度变化} = {新创意}",
                "examples": ["缩小的计算机 = 智能手机", "放大的分子 = 纳米材料"]
            }
        ]
    
    def _load_domain_knowledge(self) -> Dict[str, List[str]]:
        """加载领域知识
        
        Returns:
            领域知识字典，键为领域，值为该领域的概念列表
        """
        return {
            "人工智能": ["机器学习", "深度学习", "自然语言处理", "计算机视觉", "强化学习", "生成式AI", "大语言模型"],
            "医疗健康": ["诊断", "治疗", "预防", "康复", "药物研发", "远程医疗", "个性化医疗"],
            "教育": ["在线学习", "个性化教育", "教育游戏", "职业培训", "终身学习", "STEAM教育"],
            "环境保护": ["可再生能源", "碳中和", "垃圾分类", "水资源保护", "生物多样性", "可持续发展"],
            "金融科技": ["数字支付", "区块链", "人工智能金融", "普惠金融", "风险管理", "智能投顾"],
            "交通出行": ["自动驾驶", "共享出行", "新能源汽车", "智能交通", "飞行汽车", "超级高铁"],
            "制造业": ["工业4.0", "智能制造", "3D打印", "机器人", "供应链优化", "数字孪生"],
            "娱乐媒体": ["元宇宙", "虚拟现实", "增强现实", "流媒体", "游戏化", "内容创作"],
            "农业": ["精准农业", "智慧农业", "垂直农业", "基因编辑", "农业机器人", "可持续农业"],
            "航天科技": ["太空探索", "卫星通信", "商业航天", "火星殖民", "太空旅游", "太空资源开发"]
        }
    
    def generate_creative_ideas(self, prompt: str, num_ideas: int = 5, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """生成创造性想法
        
        Args:
            prompt: 创意生成提示
            num_ideas: 生成的想法数量
            context: 上下文信息
            
        Returns:
            创造性想法列表
        """
        if not prompt:
            return []
        
        ideas = []
        
        try:
            for i in range(num_ideas):
                # 随机选择一个创造力模板
                template = random.choice(self.creativity_templates)
                
                # 根据模板生成创意
                idea = self._generate_idea_from_template(template, prompt, context)
                
                if idea:
                    # 评估创意
                    evaluation = self._evaluate_creativity(idea, prompt)
                    idea.update(evaluation)
                    ideas.append(idea)
        except Exception as e:
            logger.error(f"生成创造性想法失败: {e}")
        
        return ideas
    
    def _generate_idea_from_template(self, template: Dict[str, Any], prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """从模板生成创意
        
        Args:
            template: 创造力模板
            prompt: 创意生成提示
            context: 上下文信息
            
        Returns:
            生成的创意
        """
        try:
            if template["id"] == "template_1":
                # 领域交叉
                domains = list(self.domain_knowledge.keys())
                domain_a = random.choice(domains)
                domain_b = random.choice([d for d in domains if d != domain_a])
                concept_a = random.choice(self.domain_knowledge[domain_a])
                concept_b = random.choice(self.domain_knowledge[domain_b])
                
                idea = f"{concept_a} + {concept_b} = {concept_a}{concept_b}融合创新"
                
            elif template["id"] == "template_2":
                # 功能组合
                domains = list(self.domain_knowledge.keys())
                selected_domains = random.sample(domains, 3)
                concepts = [random.choice(self.domain_knowledge[domain]) for domain in selected_domains]
                
                idea = f"{concepts[0]} + {concepts[1]} + {concepts[2]} = {concepts[0]}{concepts[1]}{concepts[2]}综合服务"
                
            elif template["id"] == "template_3":
                # 属性转移
                domains = list(self.domain_knowledge.keys())
                domain_a = random.choice(domains)
                domain_b = random.choice([d for d in domains if d != domain_a])
                concept_a = random.choice(self.domain_knowledge[domain_a])
                concept_b = random.choice(self.domain_knowledge[domain_b])
                
                # 随机属性
                attributes = ["效率", "速度", "安全性", "灵活性", "可持续性", "智能性", "便携性", "易用性"]
                attribute = random.choice(attributes)
                
                idea = f"{concept_a}的{attribute}应用到{concept_b} = {attribute}{concept_b}"
                
            elif template["id"] == "template_4":
                # 问题重构
                traditional_methods = ["传统方法", "常规思路", "现有技术", "传统模式", "常见做法"]
                new_angles = ["创新角度", "逆向思维", "跨界整合", "技术融合", "用户中心"]
                
                traditional = random.choice(traditional_methods)
                new_angle = random.choice(new_angles)
                
                idea = f"与其{traditional}，不如{new_angle} = {prompt}创新解决方案"
                
            elif template["id"] == "template_5":
                # 尺度变换
                scales = ["微型化", "大型化", "轻量化", "高效化", "智能化", "模块化"]
                scale = random.choice(scales)
                
                domains = list(self.domain_knowledge.keys())
                domain = random.choice(domains)
                concept = random.choice(self.domain_knowledge[domain])
                
                idea = f"{concept}的{scale} = {scale}{concept}"
                
            else:
                # 默认创意
                idea = f"基于{template['name']}的{prompt}创意"
            
            return {
                "id": str(uuid.uuid4()),
                "idea": idea,
                "template_id": template["id"],
                "template_name": template["name"],
                "prompt": prompt
            }
        except Exception as e:
            logger.error(f"从模板生成创意失败: {e}")
            return {}
    
    def _evaluate_creativity(self, idea: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """评估创意的创造性
        
        Args:
            idea: 创意
            prompt: 原始提示
            
        Returns:
            创意评估结果
        """
        try:
            # 简单评估逻辑（实际应使用更复杂的算法）
            relevance = random.uniform(0.7, 1.0)
            originality = random.uniform(0.6, 0.95)
            feasibility = random.uniform(0.5, 0.9)
            impact = random.uniform(0.6, 0.9)
            
            overall_score = (relevance + originality + feasibility + impact) / 4
            
            return {
                "relevance": round(relevance, 2),
                "originality": round(originality, 2),
                "feasibility": round(feasibility, 2),
                "impact": round(impact, 2),
                "overall_score": round(overall_score, 2)
            }
        except Exception as e:
            logger.error(f"评估创意失败: {e}")
            return {
                "relevance": 0.0,
                "originality": 0.0,
                "feasibility": 0.0,
                "impact": 0.0,
                "overall_score": 0.0
            }
    
    def enhance_creative_idea(self, idea: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """增强创造性想法
        
        Args:
            idea: 原始创意
            context: 上下文信息
            
        Returns:
            增强后的创意
        """
        if not idea:
            return {
                "enhanced_idea": "",
                "details": [],
                "implementation_steps": []
            }
        
        try:
            # 生成创意详情
            details = [
                f"核心价值: {idea}的创新点在于...",
                f"目标用户: {idea}适合...",
                f"市场定位: {idea}在市场中...",
                f"差异化优势: 与现有解决方案相比，{idea}..."
            ]
            
            # 生成实施步骤
            implementation_steps = [
                "1. 进行市场调研和用户需求分析",
                "2. 设计原型和进行初步测试",
                "3. 迭代优化和完善产品功能",
                "4. 小规模试点和用户反馈收集",
                "5. 大规模推广和商业化"
            ]
            
            return {
                "enhanced_idea": f"增强版: {idea}",
                "details": details,
                "implementation_steps": implementation_steps,
                "enhancement_id": str(uuid.uuid4())
            }
        except Exception as e:
            logger.error(f"增强创意失败: {e}")
            return {
                "enhanced_idea": idea,
                "details": [],
                "implementation_steps": []
            }
    
    def generate_creative_story(self, prompt: str, genre: str = "科幻", length: int = 200) -> Dict[str, Any]:
        """生成创造性故事
        
        Args:
            prompt: 故事生成提示
            genre: 故事类型
            length: 故事长度
            
        Returns:
            生成的故事
        """
        if not prompt:
            return {
                "story": "",
                "genre": genre,
                "prompt": prompt
            }
        
        try:
            # 生成故事开头
            story_openings = [
                f"在{genre}的世界里，{prompt}...",
                f"当{prompt}发生时，{genre}的大门悄然开启...",
                f"{prompt}，这个看似平凡的事物，在{genre}的视角下却...",
                f"在{genre}的未来，{prompt}已经成为了..."
            ]
            
            opening = random.choice(story_openings)
            
            # 生成故事发展
            story_developments = [
                "随着时间的推移，事情开始变得不同寻常...",
                "突然，一个意外的发现改变了一切...",
                "人们开始意识到，这不仅仅是一个简单的现象...",
                "科学家们经过研究，发现了背后惊人的秘密..."
            ]
            
            development = random.choice(story_developments)
            
            # 生成故事结尾
            story_endings = [
                "最终，人们从中学到了宝贵的教训...",
                "这个发现彻底改变了人类的未来...",
                "故事并没有结束，新的篇章即将开始...",
                "也许，这只是宇宙中无数奇迹之一..."
            ]
            
            ending = random.choice(story_endings)
            
            story = f"{opening} {development} {ending}"
            
            return {
                "story": story,
                "genre": genre,
                "prompt": prompt,
                "story_id": str(uuid.uuid4())
            }
        except Exception as e:
            logger.error(f"生成创造性故事失败: {e}")
            return {
                "story": "",
                "genre": genre,
                "prompt": prompt
            }

# 创建单例实例
creativity_service = CreativityService()
