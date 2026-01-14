"""AI模型服务
提供AI模型加载、意图识别、对话生成等功能
"""

from typing import Optional, List, Dict, Callable
from enum import Enum
import os
import logging
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# 配置日志
logger = logging.getLogger(__name__)

# 导入Hume AI EVI服务
from src.core.services.hume_evi_service import hume_evi_service
# 导入NELLIE神经符号推理服务
from src.core.services.nellie_service import nellie_service
# 导入问题树框架服务
from src.core.services.tree_of_problems_service import tree_of_problems_service
# 导入组合创造力框架服务
from src.core.services.creativity_service import creativity_service
# 导入NACS意识模拟服务
from src.core.services.nacs_service import nacs_service
# 导入ComVas动态价值对齐服务
from src.core.services.comvas_service import comvas_service

# 意图类型枚举
class IntentType(str, Enum):
    """意图类型枚举"""
    CHAT = "chat"  # 普通聊天
    OPEN_FILE = "open_file"  # 打开文件
    TAKE_PHOTO = "take_photo"  # 拍照
    RUN_COMMAND = "run_command"  # 执行命令
    LIST_FILES = "list_files"  # 列出文件
    GET_SYSTEM_INFO = "get_system_info"  # 获取系统信息
    NAVIGATE = "navigate"  # 导航
    OPEN_URL = "open_url"  # 打开URL
    SCREENSHOT = "screenshot"  # 截图
    CREATE_FILE = "create_file"  # 创建文件
    COPY_FILE = "copy_file"  # 复制文件
    DELETE_FILE = "delete_file"  # 删除文件
    START_APPLICATION = "start_application"  # 启动应用程序
    GET_PROCESS_LIST = "get_process_list"  # 获取进程列表
    # 新增意图类型
    SEARCH_INTERNET = "search_internet"  # 搜索互联网
    RETRIEVE_DATA = "retrieve_data"  # 检索数据
    CALL_API = "call_api"  # 调用API
    DOWNLOAD_FILE = "download_file"  # 下载文件
    ORCHESTRATE_AGENT = "orchestrate_agent"  # 编排智能体
    MANAGE_AGENT = "manage_agent"  # 管理智能体
    CONTROL_DEVICE = "control_device"  # 控制设备
    DISCOVER_DEVICE = "discover_device"  # 发现设备
    MONITOR_DEVICE = "monitor_device"  # 监控设备
    ANALYZE_HABITS = "analyze_habits"  # 分析习惯
    PREDICT_BEHAVIOR = "predict_behavior"  # 预测行为
    GET_RECOMMENDATIONS = "get_recommendations"  # 获取推荐
    CHECK_UPDATE = "check_update"  # 检查更新
    UPDATE_APPLICATION = "update_application"  # 更新应用
    GENERATE_INSTALL_PACKAGE = "generate_install_package"  # 生成安装包
    GET_INSTALL_GUIDE = "get_install_guide"  # 获取安装指南
    # 生成式AI相关意图
    GENERATE_CREATIVE_IDEA = "generate_creative_idea"  # 生成创造性想法
    GENERATE_STORY = "generate_story"  # 生成故事
    GENERATE_POEM = "generate_poem"  # 生成诗歌
    GENERATE_CODE = "generate_code"  # 生成代码
    GENERATE_IMAGE = "generate_image"  # 生成图像
    GENERATE_DESIGN = "generate_design"  # 生成设计
    GENERATE_REPORT = "generate_report"  # 生成报告
    GENERATE_PRESENTATION = "generate_presentation"  # 生成演示文稿
    UNKNOWN = "unknown"  # 未知意图

# 对话上下文类
class ConversationContext:
    """对话上下文"""
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """添加对话消息"""
        import datetime
        self.history.append({
            "role": role, 
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        # 保持历史记录不超过最大值
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.history
    
    def clear(self):
        """清空对话历史"""
        self.history = []

# 聊天扩展统一管理类
class ChatExtensionManager:
    """聊天扩展统一管理器"""
    
    def __init__(self):
        self.extensions: Dict[str, Callable] = {}
        self.enabled = True  # 默认启用
    
    def register_extension(self, extension_name: str, extension_func: Callable):
        """注册聊天扩展
        
        Args:
            extension_name: 扩展名称
            extension_func: 扩展函数，接收文本和上下文，返回处理后的文本
        """
        self.extensions[extension_name] = extension_func
        logger.info(f"聊天扩展已注册: {extension_name}")
    
    def unregister_extension(self, extension_name: str):
        """注销聊天扩展
        
        Args:
            extension_name: 扩展名称
        """
        if extension_name in self.extensions:
            del self.extensions[extension_name]
            logger.info(f"聊天扩展已注销: {extension_name}")
    
    def get_extensions(self) -> List[str]:
        """获取所有已注册的扩展
        
        Returns:
            扩展名称列表
        """
        return list(self.extensions.keys())
    
    def process_with_extensions(self, text: str, context: ConversationContext) -> str:
        """使用所有扩展处理文本
        
        Args:
            text: 用户输入文本
            context: 对话上下文
            
        Returns:
            处理后的文本
        """
        if not self.enabled or not self.extensions:
            return text
        
        processed_text = text
        for extension_name, extension_func in self.extensions.items():
            try:
                processed_text = extension_func(processed_text, context)
                logger.debug(f"扩展 {extension_name} 处理完成")
            except Exception as e:
                logger.error(f"扩展 {extension_name} 处理失败: {str(e)}")
        
        return processed_text
    
    def set_enabled(self, enabled: bool):
        """设置扩展统一处理是否启用
        
        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        logger.info(f"聊天扩展统一处理已{'启用' if enabled else '禁用'}")
    
    def is_enabled(self) -> bool:
        """检查扩展统一处理是否启用
        
        Returns:
            是否启用
        """
        return self.enabled

# AI模型服务类
class AIModelService:
    """AI模型服务"""
    
    def __init__(self):
        self.model_loaded = False
        self.context = ConversationContext()
        self.model_dir = os.path.join(os.path.dirname(__file__), "../models")
        self.intent_rules_file = os.path.join(self.model_dir, "intent_rules.json")
        self.intent_rules = self._load_intent_rules()
        self.openai_client = None
        self.use_openai = False
        
        # 初始化聊天扩展管理器
        self.chat_extension_manager = ChatExtensionManager()
        
        # 初始化OpenAI客户端
        self._init_openai_client()
        
        self.initialize_model()
        
        # 注册默认扩展（示例）
        self._register_default_extensions()
    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and openai_api_key and openai_api_key != "<your-openai-api-key>":
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.use_openai = True
                logger.info("✅ OpenAI客户端初始化成功")
            except Exception as e:
                logger.error(f"❌ OpenAI客户端初始化失败: {str(e)}")
                self.use_openai = False
        else:
            logger.warning("⚠️ OpenAI API密钥未配置，将使用本地规则匹配")
            self.use_openai = False
    
    def _load_intent_rules(self) -> Dict[str, List[str]]:
        """加载意图识别规则
        
        Returns:
            意图规则字典，键为意图类型，值为关键词列表
        """
        # 默认意图规则
        default_rules = {
            "open_file": ["打开文件"],
            "take_photo": ["拍照"],
            "run_command": ["执行命令", "运行命令"],
            "list_files": ["列出文件"],
            "get_system_info": ["系统信息"],
            "navigate": ["导航到", "打开页面", "进入"],
            "open_url": ["打开网页", "访问网站", "打开链接"],
            "screenshot": ["截图", "截屏"],
            "create_file": ["创建文件"],
            "copy_file": ["复制文件"],
            "delete_file": ["删除文件"],
            "start_application": ["启动应用", "打开应用"],
            "get_process_list": ["进程列表", "运行的程序"],
            "chat": ["你好", "hello", "hi", "在吗"],
            # 新增意图规则
            "search_internet": ["搜索", "查一下", "找一找", "上网搜索"],
            "retrieve_data": ["获取数据", "检索数据", "查询数据"],
            "call_api": ["调用API", "请求接口", "调用接口"],
            "download_file": ["下载文件", "保存文件", "下载"],
            "orchestrate_agent": ["编排智能体", "调用智能体", "指派任务"],
            "manage_agent": ["管理智能体", "注册智能体", "列出智能体"],
            "control_device": ["控制设备", "操作设备", "设备控制"],
            "discover_device": ["发现设备", "搜索设备", "查找设备"],
            "monitor_device": ["监控设备", "查看设备状态", "设备状态"],
            "analyze_habits": ["分析习惯", "我的习惯", "习惯分析"],
            "predict_behavior": ["预测行为", "行为预测", "预测"],
            "get_recommendations": ["推荐", "建议", "推荐内容"],
            "check_update": ["检查更新", "更新检查", "版本检查"],
            "update_application": ["更新应用", "应用更新", "升级"],
            "generate_install_package": ["生成安装包", "安装包", "本地安装"],
            "get_install_guide": ["安装指南", "如何安装", "安装说明"],
            # 生成式AI相关意图规则
            "generate_creative_idea": ["生成创意", "创意想法", "创新方案", "创意点子", "头脑风暴"],
            "generate_story": ["写故事", "生成故事", "编故事", "创作故事", "故事生成"],
            "generate_poem": ["写诗歌", "生成诗歌", "创作诗歌", "诗歌生成", "写诗句"],
            "generate_code": ["生成代码", "写代码", "编程", "代码生成", "开发程序"],
            "generate_image": ["生成图像", "生成图片", "创建图片", "绘制图像", "生成视觉内容"],
            "generate_design": ["生成设计", "设计方案", "设计创意", "创意设计", "设计生成"],
            "generate_report": ["生成报告", "写报告", "报告生成", "创建报告", "生成文档"],
            "generate_presentation": ["生成演示文稿", "创建幻灯片", "幻灯片生成", "演示文稿生成", "制作PPT"]
        }
        
        # 如果规则文件存在，则加载
        if os.path.exists(self.intent_rules_file):
            try:
                with open(self.intent_rules_file, "r", encoding="utf-8") as f:
                    loaded_rules = json.load(f)
                logger.info(f"从文件加载意图规则: {self.intent_rules_file}")
                return loaded_rules
            except Exception as e:
                logger.error(f"加载意图规则失败: {str(e)}")
                logger.info("使用默认意图规则")
        else:
            # 创建模型目录
            os.makedirs(self.model_dir, exist_ok=True)
            # 保存默认规则
            self._save_intent_rules(default_rules)
        
        return default_rules
    
    def _save_intent_rules(self, rules: Dict[str, List[str]]):
        """保存意图识别规则到文件
        
        Args:
            rules: 意图规则字典
        """
        try:
            with open(self.intent_rules_file, "w", encoding="utf-8") as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)
            logger.info(f"意图规则已保存到文件: {self.intent_rules_file}")
        except Exception as e:
            logger.error(f"保存意图规则失败: {str(e)}")
    
    def update_intent_rules(self, new_rules: Dict[str, List[str]]):
        """更新意图识别规则
        
        Args:
            new_rules: 新的意图规则字典
        """
        # 合并规则，新规则覆盖旧规则
        for intent, keywords in new_rules.items():
            if intent in self.intent_rules:
                # 合并关键词，去重
                existing_keywords = self.intent_rules[intent]
                self.intent_rules[intent] = list(set(existing_keywords + keywords))
            else:
                self.intent_rules[intent] = keywords
        
        # 保存更新后的规则
        self._save_intent_rules(self.intent_rules)
        logger.info("意图规则已更新")
    
    def add_intent_keywords(self, intent: str, keywords: List[str]):
        """添加意图关键词
        
        Args:
            intent: 意图类型
            keywords: 关键词列表
        """
        if intent in self.intent_rules:
            # 去重添加
            existing_keywords = self.intent_rules[intent]
            self.intent_rules[intent] = list(set(existing_keywords + keywords))
        else:
            self.intent_rules[intent] = keywords
        
        # 保存更新后的规则
        self._save_intent_rules(self.intent_rules)
        logger.info(f"为意图 {intent} 添加了关键词: {keywords}")
    
    def _register_default_extensions(self):
        """注册默认聊天扩展"""
        # 示例扩展1: 简单的文本转换扩展
        def example_extension(text: str, context: ConversationContext) -> str:
            """示例扩展：将所有问号转换为感叹号"""
            return text.replace("?", "!")
        
        # 示例扩展2: 上下文感知扩展
        def context_extension(text: str, context: ConversationContext) -> str:
            """示例扩展：根据上下文添加问候语"""
            history = context.get_history()
            if len(history) <= 1 and any(greeting in text.lower() for greeting in ["你好", "hello", "hi"]):
                return f"{text} 很高兴见到你！"
            return text
        
        # 注册默认扩展
        self.chat_extension_manager.register_extension("example", example_extension)
        self.chat_extension_manager.register_extension("context", context_extension)
        
        logger.info("默认聊天扩展注册完成")
    
    def initialize_model(self):
        """初始化AI模型"""
        try:
            # 这里可以加载真正的AI模型
            # 目前使用简单的规则引擎作为示例
            logger.info("AI模型服务初始化完成")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"AI模型加载失败: {str(e)}")
            self.model_loaded = False
    
    def recognize_intent(self, text: str) -> IntentType:
        """识别用户意图
        
        Args:
            text: 用户输入文本
            
        Returns:
            意图类型
        """
        text_lower = text.lower()
        
        # 使用动态加载的意图规则进行匹配
        for intent_str, keywords in self.intent_rules.items():
            if any(keyword in text_lower for keyword in keywords):
                # 检查意图类型是否有效
                if intent_str in [intent.value for intent in IntentType]:
                    return IntentType(intent_str)
        
        # 如果没有匹配到任何意图，返回未知意图
        return IntentType.UNKNOWN
    
    def extract_entities(self, text: str, intent: IntentType) -> Dict[str, str]:
        """提取实体
        
        Args:
            text: 用户输入文本
            intent: 意图类型
            
        Returns:
            提取的实体
        """
        entities = {}
        
        if intent == IntentType.OPEN_FILE:
            # 提取文件路径
            if "打开文件" in text:
                file_path = text.split("打开文件")[-1].strip()
                entities["file_path"] = file_path
        elif intent == IntentType.RUN_COMMAND:
            # 提取命令
            if "执行命令" in text:
                command = text.split("执行命令")[-1].strip()
                entities["command"] = command
            elif "运行命令" in text:
                command = text.split("运行命令")[-1].strip()
                entities["command"] = command
        elif intent == IntentType.LIST_FILES:
            # 提取目录路径
            if "列出文件" in text:
                directory = text.split("列出文件")[-1].strip()
                entities["directory"] = directory if directory else "."
            else:
                entities["directory"] = "."
        elif intent == IntentType.NAVIGATE:
            # 提取导航目标
            for keyword in ["导航到", "打开页面", "进入"]:
                if keyword in text:
                    target = text.split(keyword)[-1].strip()
                    entities["target"] = target
                    break
        elif intent == IntentType.OPEN_URL:
            # 提取URL
            for keyword in ["打开网页", "访问网站", "打开链接"]:
                if keyword in text:
                    url = text.split(keyword)[-1].strip()
                    entities["url"] = url
                    break
        elif intent == IntentType.SCREENSHOT:
            # 提取保存路径（可选）
            if "保存到" in text:
                save_path = text.split("保存到")[-1].strip()
                entities["save_path"] = save_path
        elif intent == IntentType.CREATE_FILE:
            # 提取文件路径和内容
            if "创建文件" in text:
                content_part = text.split("创建文件")[-1].strip()
                if "内容是" in content_part:
                    file_path = content_part.split("内容是")[0].strip()
                    content = content_part.split("内容是")[-1].strip()
                    entities["file_path"] = file_path
                    entities["content"] = content
                else:
                    entities["file_path"] = content_part
        elif intent == IntentType.COPY_FILE:
            # 提取源文件路径和目标文件路径
            if "复制文件" in text:
                copy_part = text.split("复制文件")[-1].strip()
                if "到" in copy_part:
                    source = copy_part.split("到")[0].strip()
                    destination = copy_part.split("到")[-1].strip()
                    entities["source"] = source
                    entities["destination"] = destination
        elif intent == IntentType.DELETE_FILE:
            # 提取文件路径
            if "删除文件" in text:
                file_path = text.split("删除文件")[-1].strip()
                entities["file_path"] = file_path
        elif intent == IntentType.START_APPLICATION:
            # 提取应用程序名称
            for keyword in ["启动应用", "打开应用"]:
                if keyword in text:
                    app_name = text.split(keyword)[-1].strip()
                    entities["app_name"] = app_name
                    break
        
        return entities
    
    def generate_response(self, input_data: Dict[str, any], context: Optional[ConversationContext] = None) -> str:
        """生成AI响应
        
        Args:
            input_data: 包含文本、图像、音频等多模态输入的数据
            context: 对话上下文
            
        Returns:
            AI生成的响应
        """
        # 如果没有提供上下文，使用默认上下文
        if not context:
            context = self.context
        
        # 更新NACS意识状态
        nacs_service.update_awareness(input_data, {
            "context": str(context.get_history()),
            "timestamp": str(context.get_history()[-1]["timestamp"]) if context.get_history() else ""
        })
        
        # 构建消息内容，支持多模态
        message_content = []
        
        # 处理文本输入
        text_input = input_data.get("text", "")
        if text_input:
            message_content.append({"type": "text", "text": text_input})
        
        # 处理图像输入
        if "image" in input_data and input_data["image"]:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": input_data["image"]
                }
            })
        
        # 处理音频输入
        if "audio" in input_data and input_data["audio"]:
            message_content.append({"type": "audio", "audio_url": input_data["audio"]})
        
        # 构建消息对象
        user_message = {
            "role": "user",
            "content": message_content
        }
        
        # 添加用户消息到上下文
        context.add_message("user", str(input_data))
        
        # 分析用户情绪（如果有文本输入）
        emotional_response = ""
        if text_input:
            try:
                # 使用Hume AI EVI服务分析情绪
                emotions = hume_evi_service.analyze_emotions(text_input)
                if emotions:
                    # 生成情感响应
                    emotional_response = hume_evi_service.generate_emotional_response(emotions, str(context.get_history()))
            except Exception as e:
                logger.error(f"情绪分析失败: {str(e)}")
        
        # 执行神经符号推理（如果有文本输入且需要推理）
        reasoning_enhancement = ""
        if text_input:
            try:
                # 检测是否需要符号推理（包含疑问词、逻辑关系等）
                reasoning_triggers = ["为什么", "因为", "所以", "如果", "那么", "是否", "能否", "如何", "推理", "逻辑"]
                if any(trigger in text_input for trigger in reasoning_triggers):
                    # 执行神经符号推理
                    reasoning_result = nellie_service.perform_symbolic_reasoning(text_input, {
                        "context": str(context.get_history()),
                        "query_type": "reasoning"
                    })
                    if reasoning_result:
                        # 增强AI响应
                        reasoning_enhancement = f"\n\n推理增强: {reasoning_result['explanation']}"
            except Exception as e:
                logger.error(f"神经符号推理失败: {str(e)}")
        
        # 执行问题树解决（如果有文本输入且是问题）
        problem_solution = ""
        if text_input:
            try:
                # 检测是否是问题（包含问号或疑问词）
                is_question = "?" in text_input or any(word in text_input for word in ["如何", "怎样", "为什么", "为何", "什么", "哪个", "哪里", "何时", "谁", "怎么"])
                
                # 检测是否是需要解决的问题
                problem_triggers = ["解决", "处理", "应对", "解决方法", "解决方案", "如何解决"]
                is_problem_solving = any(trigger in text_input for trigger in problem_triggers)
                
                if is_question or is_problem_solving:
                    # 使用问题树框架生成解决方案
                    solution = tree_of_problems_service.generate_problem_solution(text_input, {
                        "context": str(context.get_history())
                    })
                    if solution:
                        problem_solution = f"\n\n问题解决分析:\n{solution}"
            except Exception as e:
                logger.error(f"问题树解决失败: {str(e)}")
        
        # 执行创造性想法生成（如果有文本输入且需要创意）
        creative_ideas = ""
        if text_input:
            try:
                # 检测是否需要创造性想法
                creativity_triggers = ["创意", "创新", "想法", "点子", "创造", "设计", "发明", "想象", "灵感", "创意解决方案"]
                is_creative_request = any(trigger in text_input for trigger in creativity_triggers)
                
                # 检测是否是故事生成请求
                story_triggers = ["故事", "写故事", "编故事", "创作故事", "故事生成"]
                is_story_request = any(trigger in text_input for trigger in story_triggers)
                
                if is_creative_request:
                    # 生成创造性想法
                    ideas = creativity_service.generate_creative_ideas(text_input, num_ideas=3)
                    if ideas:
                        creative_ideas = "\n\n创意想法:"
                        for i, idea in enumerate(ideas, 1):
                            creative_ideas += f"\n{i}. {idea['idea']} (评分: {idea['overall_score']:.2f})"
                elif is_story_request:
                    # 生成创造性故事
                    story_result = creativity_service.generate_creative_story(text_input, genre="科幻")
                    if story_result["story"]:
                        creative_ideas = f"\n\n生成故事: {story_result['story']}"
            except Exception as e:
                logger.error(f"创造性想法生成失败: {str(e)}")
        
        # 如果OpenAI可用，使用OpenAI API
        if self.use_openai and self.openai_client:
            try:
                # 构建消息历史
                messages = [
                    {"role": "system", "content": "你是一个智能AI助手，可以帮助用户执行各种任务，如打开文件、拍照、执行命令等。请用中文回答。"}
                ]
                
                # 添加历史对话
                for msg in context.get_history():
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # 添加当前多模态消息
                messages.append(user_message)
                
                # 调用OpenAI API，使用GPT-4o支持多模态
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                ai_response = response.choices[0].message.content
                
                # 融合情感响应
                if emotional_response:
                    ai_response = f"{emotional_response} {ai_response}"
                
                # 融合推理增强
                if reasoning_enhancement:
                    ai_response = f"{ai_response}{reasoning_enhancement}"
                
                # 融合问题解决分析
                if problem_solution:
                    ai_response = f"{ai_response}{problem_solution}"
                
                # 融合创造性想法
                if creative_ideas:
                    ai_response = f"{ai_response}{creative_ideas}"
            except Exception as e:
                logger.error(f"OpenAI API调用失败: {str(e)}")
                # 回退到本地规则匹配
                ai_response = f"抱歉，我现在无法处理您的请求。错误：{str(e)}"
        else:
            # 本地规则匹配，只处理文本
            intent = self.recognize_intent(text_input)
            entities = self.extract_entities(text_input, intent)
            ai_response = self._generate_response_by_intent(text_input, intent, entities, context)
            
            # 融合情感响应
            if emotional_response:
                ai_response = f"{emotional_response} {ai_response}"
            
            # 融合推理增强
            if reasoning_enhancement:
                ai_response = f"{ai_response}{reasoning_enhancement}"
            
            # 融合问题解决分析
            if problem_solution:
                ai_response = f"{ai_response}{problem_solution}"
            
            # 融合创造性想法
            if creative_ideas:
                ai_response = f"{ai_response}{creative_ideas}"
        
        # 伦理决策对齐
        ethical_alignment = comvas_service.align_decision(ai_response, {
            "input_data": input_data,
            "context": str(context.get_history()),
            "intent": str(intent) if 'intent' in locals() else "unknown"
        })
        
        # 如果伦理分数较低，调整响应
        if ethical_alignment["ethical_evaluation"]["ethical_score"] < 0.7:
            ai_response = ethical_alignment["aligned_action"]
            ai_response += f"\n\n伦理提示: {', '.join(ethical_alignment['ethical_evaluation']['suggestions'])}"
        
        # 添加AI响应到上下文
        context.add_message("assistant", ai_response)
        
        return ai_response
    
    def _generate_response_by_intent(self, text: str, intent: IntentType, entities: Dict[str, str], context: ConversationContext) -> str:
        """根据意图生成响应
        
        Args:
            text: 用户输入文本
            intent: 意图类型
            entities: 提取的实体
            context: 对话上下文
            
        Returns:
            AI生成的响应
        """
        if intent == IntentType.CHAT:
            # 普通聊天
            return self._generate_chat_response(text, context)
        elif intent == IntentType.OPEN_FILE:
            # 打开文件
            file_path = entities.get("file_path", "")
            return f"我将帮你打开文件: {file_path}"
        elif intent == IntentType.TAKE_PHOTO:
            # 拍照
            return "我将帮你调用摄像头拍照"
        elif intent == IntentType.RUN_COMMAND:
            # 执行命令
            command = entities.get("command", "")
            return f"我将帮你执行命令: {command}"
        elif intent == IntentType.LIST_FILES:
            # 列出文件
            directory = entities.get("directory", ".")
            return f"我将帮你列出目录 {directory} 中的文件"
        elif intent == IntentType.GET_SYSTEM_INFO:
            # 获取系统信息
            return "我将帮你获取系统信息"
        elif intent == IntentType.NAVIGATE:
            # 导航
            target = entities.get("target", "")
            return f"我将帮你导航到 {target}"
        elif intent == IntentType.OPEN_URL:
            # 打开URL
            url = entities.get("url", "")
            return f"我将帮你打开URL: {url}"
        elif intent == IntentType.SCREENSHOT:
            # 截图
            save_path = entities.get("save_path", "")
            if save_path:
                return f"我将帮你截图并保存到: {save_path}"
            else:
                return "我将帮你截图"
        elif intent == IntentType.CREATE_FILE:
            # 创建文件
            file_path = entities.get("file_path", "")
            return f"我将帮你创建文件: {file_path}"
        elif intent == IntentType.COPY_FILE:
            # 复制文件
            source = entities.get("source", "")
            destination = entities.get("destination", "")
            return f"我将帮你复制文件，从 {source} 到 {destination}"
        elif intent == IntentType.DELETE_FILE:
            # 删除文件
            file_path = entities.get("file_path", "")
            return f"我将帮你删除文件: {file_path}"
        elif intent == IntentType.START_APPLICATION:
            # 启动应用程序
            app_name = entities.get("app_name", "")
            return f"我将帮你启动应用程序: {app_name}"
        elif intent == IntentType.GET_PROCESS_LIST:
            # 获取进程列表
            return "我将帮你获取进程列表"
        # 新增意图响应
        elif intent == IntentType.SEARCH_INTERNET:
            # 搜索互联网
            return "我将帮你搜索互联网上的资料"
        elif intent == IntentType.RETRIEVE_DATA:
            # 检索数据
            return "我将帮你检索所需的数据"
        elif intent == IntentType.CALL_API:
            # 调用API
            return "我将帮你调用指定的API接口"
        elif intent == IntentType.DOWNLOAD_FILE:
            # 下载文件
            return "我将帮你下载指定的文件"
        elif intent == IntentType.ORCHESTRATE_AGENT:
            # 编排智能体
            return "我将帮你编排智能体完成任务"
        elif intent == IntentType.MANAGE_AGENT:
            # 管理智能体
            return "我将帮你管理智能体"
        elif intent == IntentType.CONTROL_DEVICE:
            # 控制设备
            return "我将帮你控制指定的设备"
        elif intent == IntentType.DISCOVER_DEVICE:
            # 发现设备
            return "我将帮你发现网络中的设备"
        elif intent == IntentType.MONITOR_DEVICE:
            # 监控设备
            return "我将帮你监控设备状态"
        elif intent == IntentType.ANALYZE_HABITS:
            # 分析习惯
            return "我将帮你分析使用习惯"
        elif intent == IntentType.PREDICT_BEHAVIOR:
            # 预测行为
            return "我将帮你预测可能的行为"
        elif intent == IntentType.GET_RECOMMENDATIONS:
            # 获取推荐
            return "我将根据你的习惯为你提供推荐"
        elif intent == IntentType.CHECK_UPDATE:
            # 检查更新
            return "我将帮你检查应用程序更新"
        elif intent == IntentType.UPDATE_APPLICATION:
            # 更新应用
            return "我将帮你更新应用程序"
        elif intent == IntentType.GENERATE_INSTALL_PACKAGE:
            # 生成安装包
            return "我将帮你生成本地安装包"
        elif intent == IntentType.GET_INSTALL_GUIDE:
            # 获取安装指南
            return "我将为你提供安装指南"
        # 生成式AI相关意图响应
        elif intent == IntentType.GENERATE_CREATIVE_IDEA:
            # 生成创造性想法
            ideas = creativity_service.generate_creative_ideas(text, num_ideas=3)
            if ideas:
                response = "创意想法:\n"
                for i, idea in enumerate(ideas, 1):
                    response += f"{i}. {idea['idea']} (评分: {idea['overall_score']:.2f})\n"
                return response
            return "抱歉，我无法生成创意想法"
        elif intent == IntentType.GENERATE_STORY:
            # 生成故事
            story_result = creativity_service.generate_creative_story(text, genre="科幻")
            if story_result["story"]:
                return f"生成的故事:\n{story_result['story']}"
            return "抱歉，我无法生成故事"
        elif intent == IntentType.GENERATE_POEM:
            # 生成诗歌
            return "抱歉，诗歌生成功能正在开发中"
        elif intent == IntentType.GENERATE_CODE:
            # 生成代码
            return "抱歉，代码生成功能正在开发中"
        elif intent == IntentType.GENERATE_IMAGE:
            # 生成图像
            return "抱歉，图像生成功能正在开发中"
        elif intent == IntentType.GENERATE_DESIGN:
            # 生成设计
            return "抱歉，设计生成功能正在开发中"
        elif intent == IntentType.GENERATE_REPORT:
            # 生成报告
            return "抱歉，报告生成功能正在开发中"
        elif intent == IntentType.GENERATE_PRESENTATION:
            # 生成演示文稿
            return "抱歉，演示文稿生成功能正在开发中"
        else:
            # 未知意图
            return f"抱歉，我还不理解你的需求: {text}"
    
    def _generate_chat_response(self, text: str, context: ConversationContext) -> str:
        """生成聊天响应
        
        Args:
            text: 用户输入文本
            context: 对话上下文
            
        Returns:
            聊天响应
        """
        # 如果OpenAI可用，使用OpenAI API
        if self.use_openai and self.openai_client:
            try:
                # 构建消息历史
                messages = [
                    {"role": "system", "content": "你是一个智能AI助手，可以帮助用户执行各种任务，如打开文件、拍照、执行命令等。请用中文回答。"}
                ]
                
                # 添加历史对话
                for msg in context.get_history():
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # 调用OpenAI API，使用GPT-4o支持多模态
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API调用失败: {str(e)}，回退到本地规则匹配")
                # 回退到本地规则匹配
                pass
        
        # 本地规则匹配
        text_lower = text.lower()
        
        if any(greeting in text_lower for greeting in ["你好", "hello", "hi", "在吗"]):
            return "你好！我是你的AI助手，有什么可以帮助你的吗？"
        elif any(question in text_lower for question in ["你是谁", "你是什么"]):
            return "我是一个基于开源技术开发的AI助手，可以帮助你执行各种任务，如打开文件、拍照、执行命令等。"
        elif any(question in text_lower for question in ["你能做什么", "你的功能", "帮助"]):
            return "我可以帮助你执行以下任务：\n- 打开文件\n- 调用摄像头拍照\n- 执行系统命令\n- 列出目录中的文件\n- 获取系统信息\n- 导航到不同页面\n- 进行日常聊天"
        elif any(question in text_lower for question in ["谢谢", "感谢"]):
            return "不客气，很高兴能帮到你！"
        elif any(question in text_lower for question in ["再见", "拜拜", "再见了"]):
            return "再见！有需要随时找我。"
        else:
            return f"我理解你说的是: {text}。我正在不断学习中，会努力提高我的理解能力。"
    
    def clear_context(self):
        """清空对话上下文"""
        self.context.clear()
    
    def get_context(self) -> ConversationContext:
        """获取对话上下文
        
        Returns:
            对话上下文
        """
        return self.context
    
    # 聊天扩展管理方法
    def register_chat_extension(self, extension_name: str, extension_func: Callable):
        """注册聊天扩展
        
        Args:
            extension_name: 扩展名称
            extension_func: 扩展函数
        """
        return self.chat_extension_manager.register_extension(extension_name, extension_func)
    
    def unregister_chat_extension(self, extension_name: str):
        """注销聊天扩展
        
        Args:
            extension_name: 扩展名称
        """
        return self.chat_extension_manager.unregister_extension(extension_name)
    
    def get_chat_extensions(self) -> List[str]:
        """获取所有已注册的聊天扩展
        
        Returns:
            扩展名称列表
        """
        return self.chat_extension_manager.get_extensions()
    
    def set_chat_extensions_enabled(self, enabled: bool):
        """设置聊天扩展是否启用
        
        Args:
            enabled: 是否启用
        """
        return self.chat_extension_manager.set_enabled(enabled)
    
    def is_chat_extensions_enabled(self) -> bool:
        """检查聊天扩展是否启用
        
        Returns:
            是否启用
        """
        return self.chat_extension_manager.is_enabled()


# 单例模式
aimodel_service = AIModelService()
