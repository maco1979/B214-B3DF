"""Cognitive Architecture Module
Based on ACT-R and Soar principles for general intelligence
Implementing modular structure, symbolic processing, and incremental learning mechanisms
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class CognitiveModuleType(str, Enum):
    """认知模块类型枚举"""
    DECLARATIVE_MEMORY = "declarative_memory"
    PROCEDURAL_MEMORY = "procedural_memory"
    GOAL_MODULE = "goal_module"
    WORKING_MEMORY = "working_memory"
    PERCEPTUAL_MODULE = "perceptual_module"
    MOTOR_MODULE = "motor_module"
    PRODUCTION_SYSTEM = "production_system"
    LEARNING_MODULE = "learning_module"
    METACOGNITIVE_MODULE = "metacognitive_module"
    EMOTIONAL_PROCESSING = "emotional_processing"


class ProductionRuleStatus(str, Enum):
    """产生式规则状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LEARNED = "learned"
    CONFLICTED = "conflicted"


@dataclass
class DeclarativeMemoryChunk:
    """陈述性记忆块数据结构"""
    chunk_id: str
    chunk_type: str
    attributes: Dict[str, Any]
    activation: float  # 激活度，影响回忆概率
    creation_time: datetime
    last_access_time: datetime
    usage_count: int = 0
    source: str = "system"


@dataclass
class ProceduralRule:
    """产生式规则数据结构"""
    rule_id: str
    condition: List[str]  # 条件列表，格式: ["goal(state=active)", "declarative(fact=rain)"]
    action: List[str]  # 动作列表，格式: ["set_goal(state=planning)", "retrieve_memory(fact=umbrella)"]
    utility: float  # 规则效用值
    status: ProductionRuleStatus = ProductionRuleStatus.ACTIVE
    creation_time: datetime = datetime.now()
    last_fired_time: Optional[datetime] = None
    firing_count: int = 0
    source: str = "system"


@dataclass
class Goal:
    """目标数据结构"""
    goal_id: str
    goal_type: str
    state: str  # active, completed, failed, suspended
    attributes: Dict[str, Any]
    priority: int  # 优先级，数值越高优先级越高
    creation_time: datetime
    parent_goal_id: Optional[str] = None  # 父目标ID，用于子目标结构
    subgoals: List[str] = None  # 子目标ID列表
    completion_time: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.subgoals is None:
            self.subgoals = []


@dataclass
class WorkingMemoryElement:
    """工作记忆元素数据结构"""
    wme_id: str
    content: Any
    activation: float
    source: str  # declarative, procedural, perceptual, goal
    creation_time: datetime
    last_access_time: datetime
    attributes: Optional[Dict[str, Any]] = None


class CognitiveModule:
    """认知模块基类"""
    
    def __init__(self, module_type: CognitiveModuleType, name: str):
        self.module_type = module_type
        self.name = name
        self.initialized = False
        self.last_update = datetime.now()
        logger.info(f"初始化认知模块: {module_type.value} - {name}")
    
    async def initialize(self):
        """初始化模块"""
        self.initialized = True
        self.last_update = datetime.now()
        logger.info(f"认知模块已初始化: {self.module_type.value} - {self.name}")
    
    async def process(self, input_data: Any) -> Any:
        """处理输入数据"""
        raise NotImplementedError("子类必须实现process方法")
    
    def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            "type": self.module_type.value,
            "name": self.name,
            "initialized": self.initialized,
            "last_update": self.last_update.isoformat()
        }


class DeclarativeMemory(CognitiveModule):
    """陈述性记忆模块，存储事实、事件和概念"""
    
    def __init__(self, name: str = "declarative_memory"):
        super().__init__(CognitiveModuleType.DECLARATIVE_MEMORY, name)
        self.chunks: Dict[str, DeclarativeMemoryChunk] = {}  # 记忆块字典
        self.chunk_type_index: Dict[str, List[str]] = {}  # 按类型索引记忆块
    
    async def initialize(self):
        """初始化陈述性记忆"""
        # 添加一些默认记忆块
        default_chunks = [
            DeclarativeMemoryChunk(
                chunk_id="chunk_weather_rain",
                chunk_type="weather",
                attributes={"type": "rain", "action": "need_umbrella", "confidence": 0.99},
                activation=0.8,
                creation_time=datetime.now(),
                last_access_time=datetime.now(),
                source="system"
            ),
            DeclarativeMemoryChunk(
                chunk_id="chunk_health_sleep",
                chunk_type="health",
                attributes={"activity": "sleep", "effect": "reduce_fatigue", "recommendation": "8_hours_per_day"},
                activation=0.75,
                creation_time=datetime.now(),
                last_access_time=datetime.now(),
                source="system"
            ),
            DeclarativeMemoryChunk(
                chunk_id="chunk_activity_exercise",
                chunk_type="activity",
                attributes={"type": "exercise", "benefit": "improve_health", "frequency": "3_times_per_week"},
                activation=0.7,
                creation_time=datetime.now(),
                last_access_time=datetime.now(),
                source="system"
            )
        ]
        
        for chunk in default_chunks:
            self.add_chunk(chunk)
        
        await super().initialize()
    
    def add_chunk(self, chunk: DeclarativeMemoryChunk):
        """添加记忆块"""
        self.chunks[chunk.chunk_id] = chunk
        
        # 更新类型索引
        if chunk.chunk_type not in self.chunk_type_index:
            self.chunk_type_index[chunk.chunk_type] = []
        if chunk.chunk_id not in self.chunk_type_index[chunk.chunk_type]:
            self.chunk_type_index[chunk.chunk_type].append(chunk.chunk_id)
        
        logger.debug(f"添加陈述性记忆块: {chunk.chunk_id} - {chunk.chunk_type}")
    
    def retrieve_chunks(self, chunk_type: Optional[str] = None, 
                       attributes: Optional[Dict[str, Any]] = None) -> List[DeclarativeMemoryChunk]:
        """检索记忆块"""
        results = []
        
        # 按类型过滤
        if chunk_type:
            if chunk_type in self.chunk_type_index:
                candidate_chunk_ids = self.chunk_type_index[chunk_type]
            else:
                return []
        else:
            candidate_chunk_ids = list(self.chunks.keys())
        
        # 按属性过滤
        for chunk_id in candidate_chunk_ids:
            chunk = self.chunks[chunk_id]
            if attributes:
                # 检查所有属性是否匹配
                match = True
                for key, value in attributes.items():
                    if key not in chunk.attributes or chunk.attributes[key] != value:
                        match = False
                        break
                if match:
                    results.append(chunk)
            else:
                results.append(chunk)
        
        # 按激活度排序
        results.sort(key=lambda x: x.activation, reverse=True)
        
        # 更新访问时间和使用次数
        for chunk in results:
            chunk.last_access_time = datetime.now()
            chunk.usage_count += 1
            # 增加激活度
            chunk.activation = min(1.0, chunk.activation + 0.05)
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Optional[DeclarativeMemoryChunk]:
        """根据ID获取记忆块"""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            # 更新访问信息
            chunk.last_access_time = datetime.now()
            chunk.usage_count += 1
            chunk.activation = min(1.0, chunk.activation + 0.05)
            return chunk
        return None
    
    async def process(self, input_data: Dict[str, Any]) -> List[DeclarativeMemoryChunk]:
        """处理输入，检索相关记忆"""
        chunk_type = input_data.get("type")
        attributes = input_data.get("attributes", {})
        return self.retrieve_chunks(chunk_type, attributes)


class ProceduralMemory(CognitiveModule):
    """程序性记忆模块，存储产生式规则"""
    
    def __init__(self, name: str = "procedural_memory"):
        super().__init__(CognitiveModuleType.PROCEDURAL_MEMORY, name)
        self.rules: Dict[str, ProceduralRule] = {}  # 产生式规则字典
        self.rule_status_index: Dict[ProductionRuleStatus, List[str]] = {}  # 按状态索引规则
    
    async def initialize(self):
        """初始化程序性记忆"""
        # 添加默认产生式规则
        default_rules = [
            ProceduralRule(
                rule_id="rule_rain_umbrella",
                condition=["goal(state=active)", "perceptual(weather=rain)"],
                action=["retrieve_memory(type=weather, attributes={'type':'rain'})"],
                utility=0.8
            ),
            ProceduralRule(
                rule_id="rule_goal_planning",
                condition=["goal(state=planning)", "working_memory(has_data=true)"],
                action=["execute_plan()", "set_goal(state=executing)"],
                utility=0.7
            ),
            ProceduralRule(
                rule_id="rule_learning_update",
                condition=["learning(need_update=true)", "goal(state=completed)"],
                action=["update_knowledge()", "set_goal(state=learning)"],
                utility=0.9
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
        
        await super().initialize()
    
    def add_rule(self, rule: ProceduralRule):
        """添加产生式规则"""
        self.rules[rule.rule_id] = rule
        
        # 更新状态索引
        if rule.status not in self.rule_status_index:
            self.rule_status_index[rule.status] = []
        self.rule_status_index[rule.status].append(rule.rule_id)
        
        logger.debug(f"添加产生式规则: {rule.rule_id} - {rule.status.value}")
    
    def get_active_rules(self) -> List[ProceduralRule]:
        """获取所有激活的规则"""
        if ProductionRuleStatus.ACTIVE in self.rule_status_index:
            return [self.rules[rule_id] for rule_id in self.rule_status_index[ProductionRuleStatus.ACTIVE]]
        return []
    
    def match_rules(self, conditions: List[str]) -> List[ProceduralRule]:
        """匹配满足条件的规则"""
        matched_rules = []
        
        for rule in self.get_active_rules():
            # 检查规则的所有条件是否都满足
            matched = True
            for rule_condition in rule.condition:
                if rule_condition not in conditions:
                    matched = False
                    break
            if matched:
                matched_rules.append(rule)
        
        # 按效用值排序
        matched_rules.sort(key=lambda x: x.utility, reverse=True)
        
        return matched_rules
    
    def fire_rule(self, rule_id: str) -> bool:
        """触发产生式规则"""
        if rule_id not in self.rules:
            logger.error(f"规则不存在: {rule_id}")
            return False
        
        rule = self.rules[rule_id]
        if rule.status != ProductionRuleStatus.ACTIVE:
            logger.error(f"规则未激活: {rule_id}")
            return False
        
        # 更新规则信息
        rule.last_fired_time = datetime.now()
        rule.firing_count += 1
        # 增加规则效用值
        rule.utility = min(1.0, rule.utility + 0.02)
        
        logger.debug(f"触发产生式规则: {rule_id} - 效用值: {rule.utility}")
        return True
    
    async def process(self, input_data: List[str]) -> List[ProceduralRule]:
        """处理输入条件，匹配规则"""
        return self.match_rules(input_data)


class GoalModule(CognitiveModule):
    """目标模块，管理目标的创建、更新和完成"""
    
    def __init__(self, name: str = "goal_module"):
        super().__init__(CognitiveModuleType.GOAL_MODULE, name)
        self.goals: Dict[str, Goal] = {}  # 目标字典
        self.active_goals: List[str] = []  # 活跃目标ID列表
        self.goal_hierarchy: Dict[str, List[str]] = {}  # 目标层次结构
    
    async def initialize(self):
        """初始化目标模块"""
        # 创建默认目标
        default_goal = Goal(
            goal_id="goal_default",
            goal_type="system",
            state="active",
            attributes={"purpose": "system_maintenance", "priority": 1},
            priority=1,
            creation_time=datetime.now()
        )
        
        self.create_goal(default_goal)
        
        await super().initialize()
    
    def create_goal(self, goal: Goal) -> str:
        """创建目标"""
        # 生成唯一ID如果没有提供
        if not goal.goal_id:
            goal.goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        
        self.goals[goal.goal_id] = goal
        
        # 如果目标是活跃状态，添加到活跃目标列表
        if goal.state == "active":
            self.active_goals.append(goal.goal_id)
        
        # 更新目标层次结构
        if goal.parent_goal_id:
            if goal.parent_goal_id not in self.goal_hierarchy:
                self.goal_hierarchy[goal.parent_goal_id] = []
            self.goal_hierarchy[goal.parent_goal_id].append(goal.goal_id)
        
        logger.debug(f"创建目标: {goal.goal_id} - {goal.goal_type}")
        return goal.goal_id
    
    def update_goal(self, goal_id: str, updates: Dict[str, Any]):
        """更新目标"""
        if goal_id not in self.goals:
            logger.error(f"目标不存在: {goal_id}")
            return False
        
        goal = self.goals[goal_id]
        
        # 更新目标属性
        for key, value in updates.items():
            if key == "state":
                # 处理状态变化
                old_state = goal.state
                goal.state = value
                
                if value == "active" and goal_id not in self.active_goals:
                    self.active_goals.append(goal_id)
                elif value != "active" and goal_id in self.active_goals:
                    self.active_goals.remove(goal_id)
                    
                if value == "completed":
                    goal.completion_time = datetime.now()
                    
                logger.debug(f"目标状态变化: {goal_id} - {old_state} -> {value}")
            elif key == "attributes":
                # 更新属性
                goal.attributes.update(value)
            elif hasattr(goal, key):
                setattr(goal, key, value)
        
        return True
    
    def get_active_goals(self) -> List[Goal]:
        """获取活跃目标"""
        return [self.goals[goal_id] for goal_id in self.active_goals]
    
    def get_goal_hierarchy(self, goal_id: str) -> Dict[str, Any]:
        """获取目标层次结构"""
        if goal_id not in self.goals:
            return {}
        
        goal = self.goals[goal_id]
        children = []
        
        if goal_id in self.goal_hierarchy:
            for child_id in self.goal_hierarchy[goal_id]:
                children.append(self.get_goal_hierarchy(child_id))
        
        return {
            "goal": goal,
            "children": children
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Goal:
        """处理目标相关请求"""
        action = input_data.get("action", "create")
        
        if action == "create":
            goal_data = input_data.get("goal_data", {})
            goal = Goal(
                goal_id=goal_data.get("goal_id", f"goal_{uuid.uuid4().hex[:8]}"),
                goal_type=goal_data.get("goal_type", "user"),
                state=goal_data.get("state", "active"),
                attributes=goal_data.get("attributes", {}),
                priority=goal_data.get("priority", 5),
                creation_time=datetime.now(),
                parent_goal_id=goal_data.get("parent_goal_id")
            )
            self.create_goal(goal)
            return goal
        elif action == "update":
            goal_id = input_data.get("goal_id")
            updates = input_data.get("updates", {})
            self.update_goal(goal_id, updates)
            return self.goals.get(goal_id) if goal_id in self.goals else None
        
        return None


class WorkingMemory(CognitiveModule):
    """工作记忆模块，用于活跃信息处理"""
    
    def __init__(self, name: str = "working_memory"):
        super().__init__(CognitiveModuleType.WORKING_MEMORY, name)
        self.elements: Dict[str, WorkingMemoryElement] = {}  # 工作记忆元素字典
        self.max_elements = 100  # 工作记忆容量限制
    
    async def initialize(self):
        """初始化工作记忆"""
        # 工作记忆初始为空
        await super().initialize()
    
    def add_element(self, content: Any, source: str, activation: float = 0.8) -> str:
        """添加工作记忆元素"""
        # 生成唯一ID
        element_id = f"wme_{uuid.uuid4().hex[:8]}"
        
        # 如果超过容量，移除激活度最低的元素
        if len(self.elements) >= self.max_elements:
            # 按激活度排序，移除最低的
            sorted_elements = sorted(self.elements.items(), key=lambda x: x[1].activation)
            remove_id = sorted_elements[0][0]
            del self.elements[remove_id]
            logger.debug(f"工作记忆已满，移除元素: {remove_id}")
        
        # 创建新元素
        element = WorkingMemoryElement(
            wme_id=element_id,
            content=content,
            activation=activation,
            source=source,
            creation_time=datetime.now(),
            last_access_time=datetime.now()
        )
        
        self.elements[element_id] = element
        logger.debug(f"添加工作记忆元素: {element_id} - {source}")
        return element_id
    
    def get_element(self, element_id: str) -> Optional[WorkingMemoryElement]:
        """获取工作记忆元素"""
        if element_id in self.elements:
            element = self.elements[element_id]
            # 更新访问时间
            element.last_access_time = datetime.now()
            # 增加激活度
            element.activation = min(1.0, element.activation + 0.1)
            return element
        return None
    
    def get_active_elements(self, limit: int = 10) -> List[WorkingMemoryElement]:
        """获取最活跃的工作记忆元素"""
        # 按激活度排序
        sorted_elements = sorted(self.elements.values(), key=lambda x: x.activation, reverse=True)
        return sorted_elements[:limit]
    
    def clear_working_memory(self):
        """清空工作记忆"""
        self.elements.clear()
        logger.debug("清空工作记忆")
    
    async def process(self, input_data: Any) -> str:
        """处理输入，添加到工作记忆"""
        return self.add_element(input_data, source="external")


class ProductionSystem(CognitiveModule):
    """产生式系统，处理规则匹配和执行"""
    
    def __init__(self, procedural_memory: ProceduralMemory, working_memory: WorkingMemory,
                 name: str = "production_system"):
        super().__init__(CognitiveModuleType.PRODUCTION_SYSTEM, name)
        self.procedural_memory = procedural_memory
        self.working_memory = working_memory
    
    async def initialize(self):
        """初始化产生式系统"""
        await super().initialize()
    
    def conflict_resolution(self, matched_rules: List[ProceduralRule]) -> Optional[ProceduralRule]:
        """冲突消解，选择最佳规则"""
        if not matched_rules:
            return None
        
        # 简单的冲突消解：选择效用值最高的规则
        return max(matched_rules, key=lambda x: x.utility)
    
    def execute_rule(self, rule: ProceduralRule) -> List[str]:
        """执行产生式规则"""
        if not self.procedural_memory.fire_rule(rule.rule_id):
            return []
        
        # 执行规则动作
        results = []
        for action in rule.action:
            # 简单的动作执行，实际应用中可以扩展
            results.append(f"执行动作: {action}")
            logger.debug(f"执行规则动作: {action}")
        
        return results
    
    def production_cycle(self, conditions: List[str]) -> List[str]:
        """产生式系统周期：匹配-选择-执行"""
        # 1. 匹配：寻找满足条件的规则
        matched_rules = self.procedural_memory.match_rules(conditions)
        
        if not matched_rules:
            return ["没有匹配的规则"]
        
        # 2. 选择：冲突消解
        selected_rule = self.conflict_resolution(matched_rules)
        if not selected_rule:
            return ["冲突消解失败"]
        
        # 3. 执行：执行选中的规则
        return self.execute_rule(selected_rule)
    
    async def process(self, input_data: List[str]) -> List[str]:
        """处理输入条件，执行产生式系统周期"""
        return self.production_cycle(input_data)


class EmotionalProcessing(CognitiveModule):
    """情感处理模块，用于识别和处理情感信息"""
    
    def __init__(self, name: str = "emotional_processing"):
        super().__init__(CognitiveModuleType.EMOTIONAL_PROCESSING, name)
        from .services.hume_evi_service import hume_evi_service
        self.hume_evi_service = hume_evi_service
        logger.info(f"初始化情感处理模块: {name}")
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """处理输入数据，识别情感"""
        text = input_data.get("text", "") if isinstance(input_data, dict) else str(input_data)
        emotions = await self.hume_evi_service.analyze_emotions(text)
        
        result = {
            "text": text,
            "emotions": emotions,
            "dominant_emotions": self.hume_evi_service.get_dominant_emotions(emotions) if emotions else []
        }
        
        return result


class CognitiveArchitecture:
    """认知架构主类，整合所有认知模块"""
    
    def __init__(self):
        self.modules: Dict[CognitiveModuleType, CognitiveModule] = {}
        self.initialized = False
        
        # 初始化核心认知模块
        self._initialize_modules()
        
        logger.info("认知架构初始化完成")
    
    def _initialize_modules(self):
        """初始化所有认知模块"""
        # 创建核心模块
        self.declarative_memory = DeclarativeMemory()
        self.procedural_memory = ProceduralMemory()
        self.goal_module = GoalModule()
        self.working_memory = WorkingMemory()
        self.emotional_processing = EmotionalProcessing()
        
        # 创建依赖其他模块的模块
        self.production_system = ProductionSystem(self.procedural_memory, self.working_memory)
        
        # 注册模块
        self.modules[CognitiveModuleType.DECLARATIVE_MEMORY] = self.declarative_memory
        self.modules[CognitiveModuleType.PROCEDURAL_MEMORY] = self.procedural_memory
        self.modules[CognitiveModuleType.GOAL_MODULE] = self.goal_module
        self.modules[CognitiveModuleType.WORKING_MEMORY] = self.working_memory
        self.modules[CognitiveModuleType.EMOTIONAL_PROCESSING] = self.emotional_processing
        self.modules[CognitiveModuleType.PRODUCTION_SYSTEM] = self.production_system
    
    async def initialize(self):
        """初始化所有认知模块"""
        for module in self.modules.values():
            await module.initialize()
        
        self.initialized = True
        logger.info("所有认知模块初始化完成")
    
    def get_module(self, module_type: CognitiveModuleType) -> Optional[CognitiveModule]:
        """获取认知模块"""
        return self.modules.get(module_type)
    
    def process_input(self, input_data: Any, input_type: str = "perceptual") -> Any:
        """处理输入数据"""
        # 简单的输入处理流程
        # 1. 将输入添加到工作记忆
        wme_id = self.working_memory.add_element(input_data, source=input_type)
        
        # 2. 获取活跃的工作记忆元素
        active_elements = self.working_memory.get_active_elements()
        
        # 3. 从陈述性记忆中检索相关信息
        declarative_chunks = self.declarative_memory.retrieve_chunks()
        
        # 4. 情感识别处理
        # 初始化默认的情感分析结果
        emotion_result = {
            "text": input_data if isinstance(input_data, str) else str(input_data),
            "emotions": None,
            "dominant_emotions": []
        }
        
        # 构建产生式系统条件
        conditions = [f"input(type={input_type})"]
        for element in active_elements:
            conditions.append(f"working_memory(content={str(element.content)[:50]})")
        for chunk in declarative_chunks[:3]:  # 只使用前3个最相关的记忆块
            conditions.append(f"declarative(type={chunk.chunk_type})")
        
        # 6. 执行产生式系统周期
        results = self.production_system.production_cycle(conditions)
        
        # 7. 返回结果
        return {
            "wme_id": wme_id,
            "active_elements": len(active_elements),
            "declarative_chunks": len(declarative_chunks),
            "emotion_analysis": emotion_result,
            "production_results": results,
            "timestamp": datetime.now()
        }
    
    def add_knowledge(self, knowledge_type: str, knowledge_content: Dict[str, Any]):
        """添加知识到认知架构"""
        # 根据知识类型添加到不同的记忆模块
        if knowledge_type == "declarative":
            # 添加到陈述性记忆
            chunk = DeclarativeMemoryChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                chunk_type=knowledge_content.get("type", "general"),
                attributes=knowledge_content.get("attributes", {}),
                activation=0.7,
                creation_time=datetime.now(),
                last_access_time=datetime.now(),
                source=knowledge_content.get("source", "external")
            )
            self.declarative_memory.add_chunk(chunk)
            return chunk.chunk_id
        elif knowledge_type == "procedural":
            # 添加到程序性记忆
            rule = ProceduralRule(
                rule_id=f"rule_{uuid.uuid4().hex[:8]}",
                condition=knowledge_content.get("condition", []),
                action=knowledge_content.get("action", []),
                utility=knowledge_content.get("utility", 0.5)
            )
            self.procedural_memory.add_rule(rule)
            return rule.rule_id
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取认知架构状态"""
        status = {
            "initialized": self.initialized,
            "modules": {},
            "active_goals": len(self.goal_module.active_goals),
            "working_memory_elements": len(self.working_memory.elements),
            "declarative_chunks": len(self.declarative_memory.chunks),
            "procedural_rules": len(self.procedural_memory.rules)
        }
        
        # 添加每个模块的状态
        for module_type, module in self.modules.items():
            status["modules"][module_type.value] = module.get_status()
        
        return status


# 创建全局认知架构实例
cognitive_architecture = CognitiveArchitecture()