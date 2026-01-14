"""
NLP服务类
实现自然语言处理功能，包括意图识别、语义理解、上下文管理等
"""

from typing import Dict, List, Any, Optional
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 定义意图类型
IntentType = \
    str

# 定义实体类型
@dataclass
class Entity:
    type: str
    value: Any
    start: int
    end: int
    parent: Optional['Entity'] = None
    children: List['Entity'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

# 定义意图识别结果
@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: List[Entity]
    text: str
    context: Dict[str, Any]

# 定义意图规则
@dataclass
class IntentRule:
    pattern: re.Pattern
    intent: str
    extract_entities: Optional[callable] = None

# 对话状态枚举
class ConversationState(Enum):
    INITIAL = "initial"
    ACTIVE = "active"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    WAITING_FOR_DETAILS = "waiting_for_details"
    CLARIFICATION_NEEDED = "clarification_needed"
    COMPLETED = "completed"

# 上下文管理
@dataclass
class Context:
    last_intent: Optional[str] = None
    last_entities: Optional[List[Entity]] = None
    conversation_history: List[str] = None
    timestamp: float = 0.0
    
    # 增强的上下文属性
    conversation_state: ConversationState = ConversationState.INITIAL
    intent_history: List[str] = None  # 意图历史
    entity_history: List[List[Entity]] = None  # 实体历史
    current_topic: Optional[str] = None  # 当前主题
    pending_questions: List[str] = None  # 待回答的问题
    clarification_needed: bool = False  # 是否需要澄清
    clarification_question: Optional[str] = None  # 澄清问题
    context_variables: Dict[str, Any] = None  # 上下文变量
    session_id: str = None  # 会话ID
    last_response: Optional[str] = None  # 最后一次响应
    
    # 对话管理增强字段
    user_preferences: Dict[str, Any] = None  # 用户偏好
    topic_history: List[str] = None  # 主题历史
    entity_references: Dict[str, List[Entity]] = None  # 实体引用映射，用于指代消解
    coreference_chains: List[Dict[str, Any]] = None  # 指代链，用于记录指代关系
    conversation_turns: int = 0  # 对话轮次
    
    # 记忆增强字段
    long_term_memory: List[Dict[str, Any]] = None  # 长期记忆，存储重要对话内容
    short_term_memory: Dict[str, Any] = None  # 短期记忆，存储当前对话上下文
    recent_entities: List[Entity] = None  # 最近提到的实体，用于指代消解


class NLPService:
    """自然语言处理服务"""
    
    def __init__(self):
        self.intent_rules = self._initialize_intent_rules()
        self.context = self._initialize_context()
        self.max_history_length = 10
        
        # 初始化TF-IDF相关组件
        self.tfidf_vectorizer = TfidfVectorizer()
        self.intent_examples = self._initialize_intent_examples()
        self.intent_tfidf_matrix = None
        self._train_tfidf_model()
    
    def _initialize_context(self, session_id: str = None) -> Context:
        """初始化上下文"""
        return Context(
            conversation_history=[],
            timestamp=datetime.now().timestamp(),
            conversation_state=ConversationState.INITIAL,
            intent_history=[],
            entity_history=[],
            current_topic=None,
            pending_questions=[],
            clarification_needed=False,
            clarification_question=None,
            context_variables={},
            session_id=session_id or str(uuid.uuid4()),
            last_response=None,
            
            # 初始化新增的上下文字段
            user_preferences={},
            topic_history=[],
            entity_references={},
            coreference_chains=[],
            conversation_turns=0,
            long_term_memory=[],
            short_term_memory={},
            recent_entities=[]
        )
    
    def _initialize_intent_rules(self) -> List[IntentRule]:
        """初始化意图规则"""
        return [
            # 原有规则
            IntentRule(
                pattern=re.compile(r'(开启|启动|打开).*主控', re.IGNORECASE),
                intent="TOGGLE_MASTER_CONTROL",
                extract_entities=self._extract_action_entity
            ),
            IntentRule(
                pattern=re.compile(r'(关闭|停止).*主控', re.IGNORECASE),
                intent="TOGGLE_MASTER_CONTROL",
                extract_entities=self._extract_action_entity
            ),
            IntentRule(
                pattern=re.compile(r'(打开|开启).*摄像头', re.IGNORECASE),
                intent="OPEN_CAMERA",
                extract_entities=self._extract_camera_index_entity
            ),
            IntentRule(
                pattern=re.compile(r'(关闭|停止).*摄像头', re.IGNORECASE),
                intent="CLOSE_CAMERA"
            ),
            IntentRule(
                pattern=re.compile(r'(左转|右转|上转|下转|放大|缩小)', re.IGNORECASE),
                intent="PTZ_CONTROL",
                extract_entities=self._extract_ptz_action_entity
            ),
            IntentRule(
                pattern=re.compile(r'(开始|启动).*AI', re.IGNORECASE),
                intent="START_AI"
            ),
            IntentRule(
                pattern=re.compile(r'(停止|关闭).*AI', re.IGNORECASE),
                intent="STOP_AI"
            ),
            
            # 农业相关规则
            IntentRule(
                pattern=re.compile(r'(查询|查看|了解|获取).*(作物|植物|庄稼).*生长|生长.*状态', re.IGNORECASE),
                intent="QUERY_CROP_GROWTH",
                extract_entities=self._extract_crop_entity
            ),
            IntentRule(
                pattern=re.compile(r'(需要|应该|建议).*(灌溉|浇水)', re.IGNORECASE),
                intent="IRRIGATION_ADVICE",
                extract_entities=self._extract_crop_entity
            ),
            IntentRule(
                pattern=re.compile(r'(需要|应该|建议).*(施肥|肥料)', re.IGNORECASE),
                intent="FERTILIZATION_ADVICE",
                extract_entities=self._extract_crop_entity
            ),
            IntentRule(
                pattern=re.compile(r'(天气|温度|湿度|光照).*影响|影响.*作物', re.IGNORECASE),
                intent="WEATHER_IMPACT",
                extract_entities=self._extract_weather_entity
            ),
            IntentRule(
                pattern=re.compile(r'(病虫害|病害|虫害|防治).*建议', re.IGNORECASE),
                intent="PEST_CONTROL_ADVICE",
                extract_entities=self._extract_pest_entity
            ),
            IntentRule(
                pattern=re.compile(r'(预测|预计).*产量', re.IGNORECASE),
                intent="YIELD_PREDICTION",
                extract_entities=self._extract_crop_entity
            ),
            IntentRule(
                pattern=re.compile(r'(生长|发育).*阶段', re.IGNORECASE),
                intent="GROWTH_STAGE_ADVICE",
                extract_entities=self._extract_crop_entity
            ),
            IntentRule(
                pattern=re.compile(r'(土壤|土质).*管理|管理.*土壤', re.IGNORECASE),
                intent="SOIL_MANAGEMENT_ADVICE",
                extract_entities=self._extract_soil_entity
            ),
            IntentRule(
                pattern=re.compile(r'(推荐|建议).*品种|品种.*推荐', re.IGNORECASE),
                intent="CROP_VARIETY_RECOMMENDATION",
                extract_entities=self._extract_crop_entity
            ),
            IntentRule(
                pattern=re.compile(r'(收获|采摘).*时间|时间.*收获', re.IGNORECASE),
                intent="HARVEST_TIME_ADVICE",
                extract_entities=self._extract_crop_entity
            )
        ]
    
    def _extract_action_entity(self, text: str) -> List[Entity]:
        """提取动作实体"""
        entities = []
        if re.search(r'(开启|启动|打开)', text, re.IGNORECASE):
            entities.append(Entity(
                type="action",
                value="start",
                start=text.index(re.search(r'(开启|启动|打开)', text, re.IGNORECASE).group()),
                end=text.index(re.search(r'(开启|启动|打开)', text, re.IGNORECASE).group()) + 2
            ))
        elif re.search(r'(关闭|停止)', text, re.IGNORECASE):
            entities.append(Entity(
                type="action",
                value="stop",
                start=text.index(re.search(r'(关闭|停止)', text, re.IGNORECASE).group()),
                end=text.index(re.search(r'(关闭|停止)', text, re.IGNORECASE).group()) + 2
            ))
        return entities
    
    def _extract_camera_index_entity(self, text: str) -> List[Entity]:
        """提取摄像头索引实体"""
        entities = []
        match = re.search(r'(\d+)', text)
        if match:
            entities.append(Entity(
                type="camera_index",
                value=int(match.group()),
                start=match.start(),
                end=match.end()
            ))
        return entities
    
    def _extract_ptz_action_entity(self, text: str) -> List[Entity]:
        """提取PTZ动作实体"""
        entities = []
        ptz_actions = {
            "左转": "left",
            "右转": "right",
            "上转": "up",
            "下转": "down",
            "放大": "zoom_in",
            "缩小": "zoom_out"
        }
        
        for action, value in ptz_actions.items():
            match = re.search(action, text)
            if match:
                entities.append(Entity(
                    type="ptz_action",
                    value=value,
                    start=match.start(),
                    end=match.end()
                ))
                break
        
        return entities
    
    def _resolve_coreferences(self, text: str) -> str:
        """指代消解：将代词映射到之前提到的实体"""
        resolved_text = text
        
        # 常见的代词列表
        pronouns = ["它", "它们", "这个", "那个", "这些", "那些", "该", "其"]
        
        # 检查是否有代词
        has_pronoun = any(pronoun in resolved_text for pronoun in pronouns)
        if not has_pronoun or not self.context.recent_entities:
            return resolved_text
        
        # 找到最近的实体
        recent_entities = sorted(self.context.recent_entities, 
                                key=lambda x: x.end, reverse=True)
        
        # 对于每个代词，尝试替换为最近的实体
        for pronoun in pronouns:
            if pronoun in resolved_text:
                # 找到最近的实体
                if recent_entities:
                    # 使用最近的实体值替换代词
                    recent_entity = recent_entities[0]
                    resolved_text = resolved_text.replace(pronoun, recent_entity.value)
                    
                    # 记录指代关系
                    coreference = {
                        "pronoun": pronoun,
                        "resolved_to": recent_entity.value,
                        "entity_type": recent_entity.type,
                        "timestamp": datetime.now().timestamp()
                    }
                    self.context.coreference_chains.append(coreference)
        
        return resolved_text
    
    def _update_recent_entities(self, entities: List[Entity]) -> None:
        """更新最近提到的实体列表"""
        if entities:
            # 添加新实体到最近实体列表
            self.context.recent_entities.extend(entities)
            
            # 移除重复实体（根据类型和值）
            seen = set()
            unique_entities = []
            for entity in self.context.recent_entities:
                key = (entity.type, str(entity.value))
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            self.context.recent_entities = unique_entities
            
            # 只保留最近的5个实体
            if len(self.context.recent_entities) > 5:
                self.context.recent_entities = self.context.recent_entities[-5:]
            
            # 按出现顺序排序（最新的在前）
            self.context.recent_entities.sort(key=lambda x: x.end, reverse=True)
    
    def process_text(self, text: str, context: Dict[str, Any] = None) -> IntentResult:
        """处理文本，识别意图和实体"""
        # 指代消解
        resolved_text = self._resolve_coreferences(text)
        
        # 更新上下文
        self._update_context(resolved_text)
        
        # 识别意图
        result = self._recognize_intent(resolved_text)
        
        # 更新上下文的最后意图和实体
        self.context.last_intent = result.intent
        self.context.last_entities = result.entities
        self.context.timestamp = datetime.now().timestamp()
        
        # 更新对话轮次
        self.context.conversation_turns += 1
        
        # 更新最近实体列表
        self._update_recent_entities(result.entities)
        
        # 更新对话状态
        self._update_conversation_state(result.intent, result.entities)
        
        return result
    
    def _recognize_intent(self, text: str) -> IntentResult:
        """识别意图"""
        best_intent: str = "UNKNOWN"
        best_confidence: float = 0.0
        best_entities: List[Entity] = []
        
        # 1. 首先使用正则表达式规则进行匹配
        for rule in self.intent_rules:
            if rule.pattern.search(text):
                confidence = 0.9  # 正则匹配的置信度较高
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = rule.intent
                    
                    # 提取实体
                    if rule.extract_entities:
                        best_entities = rule.extract_entities(text)
                    else:
                        best_entities = []
        
        # 2. 如果没有找到匹配的规则，使用TF-IDF进行语义匹配
        if best_intent == "UNKNOWN":
            # 将输入文本转换为TF-IDF向量
            text_vector = self.tfidf_vectorizer.transform([text])
            
            # 计算输入文本与所有训练样本的余弦相似度
            similarities = cosine_similarity(text_vector, self.intent_tfidf_matrix)
            
            # 找到相似度最高的样本
            max_similarity_index = np.argmax(similarities)
            max_similarity = similarities[0][max_similarity_index]
            
            # 如果相似度足够高，使用对应的意图
            if max_similarity > 0.5:  # 设置相似度阈值
                best_intent = self.intent_labels[max_similarity_index]
                best_confidence = float(max_similarity)
                
                # 尝试提取实体
                for rule in self.intent_rules:
                    if rule.intent == best_intent and rule.extract_entities:
                        best_entities = rule.extract_entities(text)
                        break
        
        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            entities=best_entities,
            text=text,
            context=self._get_context_dict()
        )
    
    def _update_context(self, text: str) -> None:
        """更新上下文"""
        # 添加到对话历史
        if self.context.conversation_history is None:
            self.context.conversation_history = []
        self.context.conversation_history.append(text)
        
        # 限制历史长度
        if len(self.context.conversation_history) > self.max_history_length:
            self.context.conversation_history.pop(0)
        
        # 更新主题历史
        if self.context.current_topic and (not self.context.topic_history or 
                                         self.context.current_topic != self.context.topic_history[-1]):
            if self.context.topic_history is None:
                self.context.topic_history = []
            self.context.topic_history.append(self.context.current_topic)
            # 限制主题历史长度
            if len(self.context.topic_history) > 5:
                self.context.topic_history.pop(0)
        
        # 更新短期记忆
        self.context.short_term_memory = {
            "last_text": text,
            "last_turn_timestamp": datetime.now().timestamp(),
            "recent_topics": self.context.topic_history.copy() if self.context.topic_history else [],
            "recent_entities": self.context.recent_entities.copy() if self.context.recent_entities else []
        }
        
        # 更新时间戳
        self.context.timestamp = datetime.now().timestamp()
    
    def _update_conversation_state(self, intent: str, entities: List[Entity]) -> None:
        """更新对话状态"""
        # 根据意图和实体更新对话状态
        if intent == "UNKNOWN":
            if not self.context.clarification_needed:
                self.context.conversation_state = ConversationState.CLARIFICATION_NEEDED
                self.context.clarification_needed = True
                self.context.clarification_question = "对不起，我不太理解您的意思。您能换一种方式表达吗？"
        else:
            # 添加到意图历史
            if self.context.intent_history is None:
                self.context.intent_history = []
            self.context.intent_history.append(intent)
            
            # 添加到实体历史
            if self.context.entity_history is None:
                self.context.entity_history = []
            self.context.entity_history.append(entities)
            
            # 更新当前主题
            if entities:
                # 查找作物实体
                crop_entities = [e for e in entities if e.type == "crop"]
                if crop_entities:
                    self.context.current_topic = crop_entities[0].value
                # 查找天气条件实体
                weather_entities = [e for e in entities if e.type == "weather_condition"]
                if weather_entities and not self.context.current_topic:
                    self.context.current_topic = f"天气条件_{weather_entities[0].value['factor']}"
                # 查找病虫害实体
                pest_entities = [e for e in entities if e.type == "pest"]
                if pest_entities and not self.context.current_topic:
                    self.context.current_topic = pest_entities[0].value
                # 查找土壤条件实体
                soil_entities = [e for e in entities if e.type == "soil_condition"]
                if soil_entities and not self.context.current_topic:
                    self.context.current_topic = f"土壤条件_{soil_entities[0].value['factor']}"
            
            # 更新对话状态
            self.context.conversation_state = ConversationState.ACTIVE
            self.context.clarification_needed = False
            self.context.clarification_question = None
            
            # 更新用户偏好（基于频繁查询的意图）
            if self.context.user_preferences is None:
                self.context.user_preferences = {}
            self.context.user_preferences[intent] = self.context.user_preferences.get(intent, 0) + 1
            
            # 更新长期记忆（每5轮对话保存一次重要信息）
            if self.context.conversation_turns % 5 == 0:
                if self.context.long_term_memory is None:
                    self.context.long_term_memory = []
                
                # 保存当前主题、意图和实体到长期记忆
                memory_item = {
                    "timestamp": datetime.now().timestamp(),
                    "conversation_turns": self.context.conversation_turns,
                    "current_topic": self.context.current_topic,
                    "last_intent": intent,
                    "last_entities": [{
                        "type": e.type,
                        "value": e.value
                    } for e in entities],
                    "user_preferences": dict(self.context.user_preferences)
                }
                self.context.long_term_memory.append(memory_item)
                
                # 限制长期记忆长度
                if len(self.context.long_term_memory) > 20:
                    self.context.long_term_memory.pop(0)
    
    def _initialize_intent_examples(self) -> Dict[str, List[str]]:
        """初始化意图示例句子"""
        return {
            # 原有意图
            "TOGGLE_MASTER_CONTROL": [
                "开启主控", "启动主控", "打开主控", "关闭主控", "停止主控"
            ],
            "OPEN_CAMERA": [
                "打开摄像头", "开启摄像头", "启动摄像头", "我想看摄像头画面"
            ],
            "CLOSE_CAMERA": [
                "关闭摄像头", "停止摄像头", "关闭所有摄像头"
            ],
            "PTZ_CONTROL": [
                "左转", "右转", "上转", "下转", "放大", "缩小", "调整摄像头角度"
            ],
            "START_AI": [
                "开始AI", "启动AI", "开启AI功能", "激活AI"
            ],
            "STOP_AI": [
                "停止AI", "关闭AI", "停止AI功能", "停用AI"
            ],
            
            # 农业相关意图
            "QUERY_CROP_GROWTH": [
                "查询作物生长状态", "查看植物生长情况", "了解庄稼生长情况", "获取番茄生长信息", "我的小麦长得怎么样"
            ],
            "IRRIGATION_ADVICE": [
                "作物需要灌溉吗", "应该什么时候浇水", "建议浇多少水", "番茄需要浇水吗", "小麦浇水的最佳时间"
            ],
            "FERTILIZATION_ADVICE": [
                "需要施肥吗", "应该施什么肥", "建议施肥量是多少", "番茄应该用什么肥料", "小麦施肥时间"
            ],
            "WEATHER_IMPACT": [
                "天气对作物的影响", "温度如何影响植物", "湿度太高会影响生长吗", "光照不足怎么办", "降雨对庄稼的影响"
            ],
            "PEST_CONTROL_ADVICE": [
                "病虫害防治建议", "如何防治蚜虫", "白粉病怎么处理", "番茄得了灰霉病怎么办", "小麦锈病防治方法"
            ],
            "YIELD_PREDICTION": [
                "预测今年的产量", "预计小麦产量", "今年番茄能产多少", "产量预测", "估算收成"
            ],
            "GROWTH_STAGE_ADVICE": [
                "作物生长阶段", "番茄现在处于什么生长阶段", "小麦的生长周期", "植物发育阶段", "生长阶段管理"
            ],
            "SOIL_MANAGEMENT_ADVICE": [
                "土壤管理建议", "如何改良土壤", "土壤pH值调整", "土壤肥力提升", "氮磷钾的作用"
            ],
            "CROP_VARIETY_RECOMMENDATION": [
                "推荐什么作物品种", "适合本地种植的番茄品种", "小麦品种推荐", "高产作物品种", "抗病虫的作物品种"
            ],
            "HARVEST_TIME_ADVICE": [
                "收获时间建议", "什么时候可以采摘番茄", "小麦收割时间", "最佳收获期", "采摘时机"
            ]
        }
    
    def _train_tfidf_model(self) -> None:
        """训练TF-IDF模型"""
        # 准备训练数据：将所有意图的示例句子收集起来
        all_examples = []
        intent_labels = []
        
        for intent, examples in self.intent_examples.items():
            for example in examples:
                all_examples.append(example)
                intent_labels.append(intent)
        
        # 训练TF-IDF模型
        self.intent_tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_examples)
        self.intent_labels = intent_labels
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """获取对话状态"""
        return {
            "state": self.context.conversation_state.value,
            "current_topic": self.context.current_topic,
            "intent_history": self.context.intent_history,
            "entity_history": [[{"type": e.type, "value": e.value} for e in entities] for entities in self.context.entity_history or []],
            "pending_questions": self.context.pending_questions,
            "clarification_needed": self.context.clarification_needed,
            "clarification_question": self.context.clarification_question
        }
    
    def set_context_variable(self, key: str, value: Any) -> None:
        """设置上下文变量"""
        if self.context.context_variables is None:
            self.context.context_variables = {}
        self.context.context_variables[key] = value
    
    def get_context_variable(self, key: str) -> Any:
        """获取上下文变量"""
        if self.context.context_variables is None:
            self.context.context_variables = {}
        return self.context.context_variables.get(key)
    
    def add_pending_question(self, question: str) -> None:
        """添加待回答的问题"""
        if self.context.pending_questions is None:
            self.context.pending_questions = []
        self.context.pending_questions.append(question)
    
    def remove_pending_question(self, question: str) -> None:
        """移除待回答的问题"""
        if self.context.pending_questions and question in self.context.pending_questions:
            self.context.pending_questions.remove(question)
    
    def clear_pending_questions(self) -> None:
        """清除所有待回答的问题"""
        self.context.pending_questions = []
    
    def _get_context_dict(self) -> Dict[str, Any]:
        """获取上下文字典"""
        return {
            "last_intent": self.context.last_intent,
            "last_entities": [
                {
                    "type": entity.type,
                    "value": entity.value,
                    "start": entity.start,
                    "end": entity.end
                }
                for entity in self.context.last_entities or []
            ],
            "conversation_history": self.context.conversation_history.copy(),
            "timestamp": self.context.timestamp
        }
    
    def clear_context(self) -> None:
        """清除上下文"""
        self.context = self._initialize_context()
    
    def map_intent_to_action(self, intent: str) -> str:
        """将意图映射到具体动作"""
        action_map = {
            # 原有意图
            "TOGGLE_MASTER_CONTROL": "toggle_master_control",
            "OPEN_CAMERA": "open_camera",
            "CLOSE_CAMERA": "close_camera",
            "PTZ_CONTROL": "ptz_control",
            "START_AI": "start_ai",
            "STOP_AI": "stop_ai",
            
            # 农业相关意图
            "QUERY_CROP_GROWTH": "query_crop_growth",
            "IRRIGATION_ADVICE": "irrigation_advice",
            "FERTILIZATION_ADVICE": "fertilization_advice",
            "WEATHER_IMPACT": "weather_impact_analysis",
            "PEST_CONTROL_ADVICE": "pest_control_advice",
            "YIELD_PREDICTION": "yield_prediction",
            "GROWTH_STAGE_ADVICE": "growth_stage_advice",
            "SOIL_MANAGEMENT_ADVICE": "soil_management_advice",
            "CROP_VARIETY_RECOMMENDATION": "crop_variety_recommendation",
            "HARVEST_TIME_ADVICE": "harvest_time_advice",
            
            # 默认
            "UNKNOWN": "unknown_action"
        }
        return action_map.get(intent, "unknown_action")
    
    def add_intent_rule(self, rule: Dict[str, Any]) -> None:
        """添加自定义意图规则"""
        pattern = re.compile(rule.get("pattern", ""), re.IGNORECASE)
        intent = rule.get("intent", "UNKNOWN")
        
        # 创建实体提取函数
        extract_entities = None
        if "entity_extractor" in rule:
            # 这里简化处理，实际可以根据规则创建动态的实体提取函数
            extract_entities = self._extract_generic_entity
        
        self.intent_rules.append(IntentRule(
            pattern=pattern,
            intent=intent,
            extract_entities=extract_entities
        ))
    
    def _extract_crop_entity(self, text: str) -> List[Entity]:
        """提取作物实体，支持复合实体"""
        entities = []
        crops = ["小麦", "水稻", "玉米", "番茄", "黄瓜", "大豆", "棉花", "土豆", "胡萝卜", "生菜", "青椒", "茄子", "草莓", "西瓜", "南瓜"]
        
        # 查找所有匹配的作物
        for crop in crops:
            for match in re.finditer(crop, text, re.IGNORECASE):
                entities.append(Entity(
                    type="crop",
                    value=crop,
                    start=match.start(),
                    end=match.end()
                ))
        
        # 如果找到多个作物，创建复合实体
        if len(entities) > 1:
            # 找到最早和最晚的位置，创建复合实体
            start = min(entity.start for entity in entities)
            end = max(entity.end for entity in entities)
            
            # 创建复合作物实体
            composite_entity = Entity(
                type="composite_crop",
                value=[entity.value for entity in entities],
                start=start,
                end=end
            )
            
            # 设置父实体关系
            for entity in entities:
                entity.parent = composite_entity
                composite_entity.children.append(entity)
            
            # 将复合实体添加到结果列表
            entities.append(composite_entity)
        
        return entities
    
    def _extract_weather_entity(self, text: str) -> List[Entity]:
        """提取天气实体，支持复合实体和嵌套实体"""
        entities = []
        
        # 天气因素列表
        weather_factors = ["温度", "湿度", "光照", "降雨", "风速", "紫外线"]
        
        # 提取所有天气因素
        weather_factor_entities = []
        for factor in weather_factors:
            for match in re.finditer(factor, text, re.IGNORECASE):
                factor_entity = Entity(
                    type="weather_factor",
                    value=factor,
                    start=match.start(),
                    end=match.end()
                )
                weather_factor_entities.append(factor_entity)
                entities.append(factor_entity)
        
        # 提取所有数值和范围
        number_entities = []
        # 匹配单个数值或范围
        for match in re.finditer(r'\d+(\.\d+)?(?:\s*[-~]\s*\d+(\.\d+)?)?', text):
            value_str = match.group()
            
            # 检查是否是范围
            if '-' in value_str or '~' in value_str:
                # 处理范围
                range_parts = re.split(r'\s*[-~]\s*', value_str)
                if len(range_parts) == 2:
                    # 创建范围实体
                    range_entity = Entity(
                        type="number_range",
                        value={
                            "min": float(range_parts[0]),
                            "max": float(range_parts[1])
                        },
                        start=match.start(),
                        end=match.end()
                    )
                    number_entities.append(range_entity)
                    entities.append(range_entity)
            else:
                # 处理单个数值
                number_entity = Entity(
                    type="number",
                    value=float(value_str),
                    start=match.start(),
                    end=match.end()
                )
                number_entities.append(number_entity)
                entities.append(number_entity)
        
        # 提取单位
        unit_entities = []
        units = ["度", "%", "摄氏度", "华氏度", "毫米", "米/秒", "勒克斯"]
        for unit in units:
            for match in re.finditer(unit, text):
                unit_entity = Entity(
                    type="unit",
                    value=unit,
                    start=match.start(),
                    end=match.end()
                )
                unit_entities.append(unit_entity)
                entities.append(unit_entity)
        
        # 创建天气条件实体（复合实体，包含因素、数值和单位）
        # 这里简化处理，实际可以根据位置关系建立更复杂的关联
        if weather_factor_entities and number_entities:
            # 为每个天气因素创建复合天气条件实体
            for factor_entity in weather_factor_entities:
                # 找到最接近的数值实体
                closest_number = min(number_entities, key=lambda x: abs(x.start - factor_entity.end))
                
                # 查找对应的单位
                unit_entity = None
                if unit_entities:
                    unit_entity = min(unit_entities, key=lambda x: abs(x.start - closest_number.end))
                
                # 创建复合天气条件实体
                weather_condition = Entity(
                    type="weather_condition",
                    value={
                        "factor": factor_entity.value,
                        "value": closest_number.value,
                        "unit": unit_entity.value if unit_entity else ""
                    },
                    start=factor_entity.start,
                    end=unit_entity.end if unit_entity else closest_number.end
                )
                
                # 设置嵌套关系
                factor_entity.parent = weather_condition
                closest_number.parent = weather_condition
                weather_condition.children.append(factor_entity)
                weather_condition.children.append(closest_number)
                
                if unit_entity:
                    unit_entity.parent = weather_condition
                    weather_condition.children.append(unit_entity)
                
                entities.append(weather_condition)
        
        return entities
    
    def _extract_pest_entity(self, text: str) -> List[Entity]:
        """提取病虫害实体，支持复合实体和嵌套实体"""
        entities = []
        pests = ["蚜虫", "红蜘蛛", "白粉病", "灰霉病", "螟虫", "锈病", "病毒病", "叶斑病", "炭疽病", "根腐病", "蓟马", "介壳虫"]
        
        # 查找所有匹配的病虫害
        pest_entities = []
        for pest in pests:
            for match in re.finditer(pest, text, re.IGNORECASE):
                pest_entity = Entity(
                    type="pest",
                    value=pest,
                    start=match.start(),
                    end=match.end()
                )
                pest_entities.append(pest_entity)
                entities.append(pest_entity)
        
        # 如果找到多个病虫害，创建复合实体
        if len(pest_entities) > 1:
            # 找到最早和最晚的位置
            start = min(entity.start for entity in pest_entities)
            end = max(entity.end for entity in pest_entities)
            
            # 创建复合病虫害实体
            composite_entity = Entity(
                type="composite_pest",
                value=[entity.value for entity in pest_entities],
                start=start,
                end=end
            )
            
            # 设置父实体关系
            for entity in pest_entities:
                entity.parent = composite_entity
                composite_entity.children.append(entity)
            
            # 将复合实体添加到结果列表
            entities.append(composite_entity)
        
        return entities
    
    def _extract_soil_entity(self, text: str) -> List[Entity]:
        """提取土壤实体，支持复合实体和嵌套实体"""
        entities = []
        soil_factors = ["pH", "湿度", "温度", "肥力", "氮", "磷", "钾", "有机质", "盐分", "透气性"]
        
        # 提取所有土壤因素
        soil_factor_entities = []
        for factor in soil_factors:
            for match in re.finditer(factor, text, re.IGNORECASE):
                factor_entity = Entity(
                    type="soil_factor",
                    value=factor,
                    start=match.start(),
                    end=match.end()
                )
                soil_factor_entities.append(factor_entity)
                entities.append(factor_entity)
        
        # 提取所有数值和范围
        number_entities = []
        # 匹配单个数值、范围或文本描述
        for match in re.finditer(r'\d+\.\d+|\d+|(?:\d+\s*[-~]\s*\d+\.\d+)|(?:\d+\s*[-~]\s*\d+)|(?:高|中|低|肥沃|贫瘠)', text):
            value_str = match.group()
            
            # 处理数值范围
            if '-' in value_str or '~' in value_str:
                range_parts = re.split(r'\s*[-~]\s*', value_str)
                if len(range_parts) == 2:
                    range_entity = Entity(
                        type="number_range",
                        value={
                            "min": float(range_parts[0]),
                            "max": float(range_parts[1])
                        },
                        start=match.start(),
                        end=match.end()
                    )
                    number_entities.append(range_entity)
                    entities.append(range_entity)
            # 处理文本描述
            elif value_str in ["高", "中", "低", "肥沃", "贫瘠"]:
                text_entity = Entity(
                    type="text_value",
                    value=value_str,
                    start=match.start(),
                    end=match.end()
                )
                number_entities.append(text_entity)
                entities.append(text_entity)
            # 处理单个数值
            else:
                try:
                    number_entity = Entity(
                        type="number",
                        value=float(value_str),
                        start=match.start(),
                        end=match.end()
                    )
                    number_entities.append(number_entity)
                    entities.append(number_entity)
                except ValueError:
                    pass
        
        # 提取单位
        unit_entities = []
        units = ["%", "度", "mg/kg", "ppm"]
        for unit in units:
            for match in re.finditer(unit, text):
                unit_entity = Entity(
                    type="unit",
                    value=unit,
                    start=match.start(),
                    end=match.end()
                )
                unit_entities.append(unit_entity)
                entities.append(unit_entity)
        
        # 创建土壤条件实体（复合实体，包含因素、数值和单位）
        if soil_factor_entities and number_entities:
            # 为每个土壤因素创建复合土壤条件实体
            for factor_entity in soil_factor_entities:
                # 找到最接近的数值实体
                closest_number = min(number_entities, key=lambda x: abs(x.start - factor_entity.end))
                
                # 查找对应的单位
                unit_entity = None
                if unit_entities:
                    unit_entity = min(unit_entities, key=lambda x: abs(x.start - closest_number.end))
                
                # 创建复合土壤条件实体
                soil_condition = Entity(
                    type="soil_condition",
                    value={
                        "factor": factor_entity.value,
                        "value": closest_number.value,
                        "unit": unit_entity.value if unit_entity else ""
                    },
                    start=factor_entity.start,
                    end=unit_entity.end if unit_entity else closest_number.end
                )
                
                # 设置嵌套关系
                factor_entity.parent = soil_condition
                closest_number.parent = soil_condition
                soil_condition.children.append(factor_entity)
                soil_condition.children.append(closest_number)
                
                if unit_entity:
                    unit_entity.parent = soil_condition
                    soil_condition.children.append(unit_entity)
                
                entities.append(soil_condition)
        
        # 如果找到多个土壤条件，创建复合土壤实体
        soil_condition_entities = [e for e in entities if e.type == "soil_condition"]
        if len(soil_condition_entities) > 1:
            # 找到最早和最晚的位置
            start = min(entity.start for entity in soil_condition_entities)
            end = max(entity.end for entity in soil_condition_entities)
            
            # 创建复合土壤实体
            composite_soil_entity = Entity(
                type="composite_soil",
                value=[entity.value for entity in soil_condition_entities],
                start=start,
                end=end
            )
            
            # 设置父实体关系
            for entity in soil_condition_entities:
                entity.parent = composite_soil_entity
                composite_soil_entity.children.append(entity)
            
            # 将复合实体添加到结果列表
            entities.append(composite_soil_entity)
        
        return entities
    
    def _extract_generic_entity(self, text: str) -> List[Entity]:
        """通用实体提取函数"""
        # 简化实现，实际可以根据规则动态提取
        return []
    
    def get_intent_rules(self) -> List[Dict[str, Any]]:
        """获取意图规则"""
        return [
            {
                "pattern": rule.pattern.pattern,
                "intent": rule.intent
            }
            for rule in self.intent_rules
        ]
    
    def process_complex_command(self, text: str) -> List[IntentResult]:
        """处理复杂指令，例如："打开客厅的灯并把空调调到26度"""
        # 简单实现：按连接词分割命令
        commands = re.split(r'(并|和|然后)', text)
        # 过滤掉连接词
        commands = [cmd.strip() for cmd in commands if cmd.strip() and not re.match(r'(并|和|然后)', cmd.strip())]
        
        return [self.process_text(cmd) for cmd in commands]
