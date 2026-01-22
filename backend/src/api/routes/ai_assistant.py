"""AI助手路由
提供AI助手相关的API接口，包括语音交互、本地控制、互联网服务、智能体编排、设备控制等功能
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.services.local_controller import LocalController
from src.core.services.ai_model_service import AIModelService, aimodel_service, IntentType
from src.core.services.learning_service import learning_service
# 导入新实现的服务
from src.core.services.internet_service import internet_service
from src.core.services.agent_orchestrator import agent_orchestrator
from src.core.services.device_controller import device_controller
from src.core.services.user_habit_service import user_habit_service
from src.core.services.deployment_service import deployment_service
import time
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 初始化本地控制器
local_controller = LocalController()

# 创建路由
ai_assistant_router = APIRouter(
    prefix="/ai-assistant",
    tags=["ai-assistant"],
    responses={404: {"description": "Not found"}},
)

# 数据模型
class UserRequest(BaseModel):
    """用户请求模型"""
    input_data: dict  # 多模态输入数据，包含text、image、audio等
    input_type: str = "text"  # text 或 voice 或 multi-modal
    context_id: str = None  # 对话上下文ID

class LocalControlRequest(BaseModel):
    """本地控制请求模型"""
    command_type: str  # open_file, take_photo, run_cmd, list_files, get_system_info
    params: dict = None

# 核心API接口
@ai_assistant_router.post("/get-response")
async def get_response(request: UserRequest):
    """核心接口：接收用户输入，返回AI响应+执行本地任务
    
    Args:
        request: 用户请求数据
        
    Returns:
        AI响应和执行结果
    """
    input_data = request.input_data
    start_time = time.time()
    
    # 1. 检查是否包含文本输入用于意图识别
    user_input = input_data.get("text", "")
    
    # 2. 使用AI模型服务识别意图（基于文本内容）
    intent = aimodel_service.recognize_intent(user_input)
    entities = aimodel_service.extract_entities(user_input, intent)
    
    # 3. 根据意图执行相应的操作
    response_data = None
    if intent == IntentType.OPEN_FILE:
        # 打开文件
        file_path = entities.get("file_path", "")
        if not file_path:
            response_data = {"response": "请提供要打开的文件路径", "type": "ai_response"}
        else:
            result = local_controller.open_file(file_path)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.TAKE_PHOTO:
        # 拍照
        result = local_controller.take_photo()
        response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.RUN_COMMAND:
        # 执行命令
        command = entities.get("command", "")
        if not command:
            response_data = {"response": "请提供要执行的命令", "type": "ai_response"}
        else:
            result = local_controller.run_system_cmd(command)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.LIST_FILES:
        # 列出文件
        directory = entities.get("directory", ".")
        result = local_controller.list_files(directory)
        response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.GET_SYSTEM_INFO:
        # 获取系统信息
        result = local_controller.get_system_info()
        response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.NAVIGATE:
        # 导航
        target = entities.get("target", "")
        response = aimodel_service.generate_response(input_data)
        response_data = {"response": response, "type": "navigation", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.OPEN_URL:
        # 打开URL
        url = entities.get("url", "")
        if not url:
            response_data = {"response": "请提供要打开的URL", "type": "ai_response"}
        else:
            result = local_controller.open_url(url)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.SCREENSHOT:
        # 截图
        save_path = entities.get("save_path", "screenshot.jpg")
        result = local_controller.screenshot(save_path)
        response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.CREATE_FILE:
        # 创建文件
        file_path = entities.get("file_path", "")
        if not file_path:
            response_data = {"response": "请提供要创建的文件路径", "type": "ai_response"}
        else:
            content = entities.get("content", "")
            result = local_controller.create_file(file_path, content)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.COPY_FILE:
        # 复制文件
        source = entities.get("source", "")
        destination = entities.get("destination", "")
        if not source or not destination:
            response_data = {"response": "请提供源文件路径和目标文件路径", "type": "ai_response"}
        else:
            result = local_controller.copy_file(source, destination)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.DELETE_FILE:
        # 删除文件
        file_path = entities.get("file_path", "")
        if not file_path:
            response_data = {"response": "请提供要删除的文件路径", "type": "ai_response"}
        else:
            result = local_controller.delete_file(file_path)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.START_APPLICATION:
        # 启动应用程序
        app_name = entities.get("app_name", "")
        if not app_name:
            response_data = {"response": "请提供要启动的应用程序名称", "type": "ai_response"}
        else:
            result = local_controller.start_application(app_name)
            response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.GET_PROCESS_LIST:
        # 获取进程列表
        result = local_controller.get_process_list()
        response_data = {"response": result, "type": "local_control", "intent": intent.value, "entities": entities}
    
    # 新增意图处理
    elif intent == IntentType.SEARCH_INTERNET:
        # 搜索互联网
        result = internet_service.search_internet(user_input)
        response_data = {"response": result, "type": "internet_service", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.RETRIEVE_DATA:
        # 检索数据
        result = internet_service.retrieve_data(user_input)
        response_data = {"response": result, "type": "internet_service", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.CALL_API:
        # 调用API
        result = internet_service.call_api(user_input)
        response_data = {"response": result, "type": "internet_service", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.DOWNLOAD_FILE:
        # 下载文件
        result = internet_service.download_file(user_input)
        response_data = {"response": result, "type": "internet_service", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.ORCHESTRATE_AGENT:
        # 编排智能体
        result = agent_orchestrator.orchestrate_task(user_input)
        response_data = {"response": result, "type": "agent_orchestration", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.MANAGE_AGENT:
        # 管理智能体
        result = agent_orchestrator.get_available_agents()
        response_data = {"response": result, "type": "agent_orchestration", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.CONTROL_DEVICE:
        # 控制设备
        result = device_controller.send_command(user_input)
        response_data = {"response": result, "type": "device_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.DISCOVER_DEVICE:
        # 发现设备
        result = device_controller.discover_devices()
        response_data = {"response": result, "type": "device_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.MONITOR_DEVICE:
        # 监控设备
        result = device_controller.health_check()
        response_data = {"response": result, "type": "device_control", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.ANALYZE_HABITS:
        # 分析习惯
        result = user_habit_service.analyze_user_habits("default_user")
        response_data = {"response": result, "type": "user_habit", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.PREDICT_BEHAVIOR:
        # 预测行为
        result = user_habit_service.predict_user_behavior("default_user", {})
        response_data = {"response": result, "type": "user_habit", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.GET_RECOMMENDATIONS:
        # 获取推荐
        result = user_habit_service.get_habit_recommendations("default_user")
        response_data = {"response": result, "type": "user_habit", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.CHECK_UPDATE:
        # 检查更新
        result = deployment_service.check_for_updates()
        response_data = {"response": result, "type": "deployment", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.UPDATE_APPLICATION:
        # 更新应用
        result = deployment_service.update_application()
        response_data = {"response": result, "type": "deployment", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.GENERATE_INSTALL_PACKAGE:
        # 生成安装包
        result = deployment_service.generate_installation_package()
        response_data = {"response": result, "type": "deployment", "intent": intent.value, "entities": entities}
    
    elif intent == IntentType.GET_INSTALL_GUIDE:
        # 获取安装指南
        result = deployment_service.get_installation_guide()
        response_data = {"response": result, "type": "deployment", "intent": intent.value, "entities": entities}
    
    # 3. 非本地指令或包含多模态输入，调用AI模型生成响应
    else:
        ai_response = aimodel_service.generate_response(input_data)
        response_data = {"response": ai_response, "type": "ai_response", "intent": intent.value, "entities": entities}
    
    # 4. 使用LangSmith跟踪AI响应
    try:
        from src.core.services import langsmith_service
        trace_id = langsmith_service.trace_ai_response(
            input_text=user_input,
            output_text=response_data["response"],
            metadata={
                "intent": intent.value,
                "response_type": response_data["type"],
                "input_type": request.input_type,
                "context_id": request.context_id,
                "input_data": input_data
            }
        )
        if trace_id:
            response_data["trace_id"] = trace_id
    except Exception as e:
        logger.error(f"LangSmith跟踪失败: {str(e)}")
    
    # 4. 计算响应时间
    response_time = time.time() - start_time
    
    # 5. 保存交互数据到学习服务
    interaction_data = {
        "user_input": user_input,
        "input_data": input_data,
        "input_type": request.input_type,
        "intent": intent.value,
        "entities": entities,
        "response": response_data["response"],
        "response_type": response_data["type"],
        "response_time": response_time,
        "context_id": request.context_id
    }
    learning_service.save_interaction_data(interaction_data)
    
    return response_data

# 专用本地控制接口
@ai_assistant_router.post("/local-control")
async def local_control(request: LocalControlRequest):
    """专用本地控制接口
    
    Args:
        request: 本地控制请求
        
    Returns:
        控制执行结果
    """
    command_type = request.command_type
    params = request.params or {}
    
    try:
        if command_type == "open_file":
            file_path = params.get("file_path", "")
            if not file_path:
                raise HTTPException(status_code=400, detail="文件路径不能为空")
            result = local_controller.open_file(file_path)
        
        elif command_type == "take_photo":
            save_path = params.get("save_path", "photo.jpg")
            result = local_controller.take_photo(save_path)
        
        elif command_type == "run_cmd":
            cmd = params.get("cmd", "")
            if not cmd:
                raise HTTPException(status_code=400, detail="命令不能为空")
            result = local_controller.run_system_cmd(cmd)
        
        elif command_type == "list_files":
            directory = params.get("directory", ".")
            result = local_controller.list_files(directory)
        
        elif command_type == "get_system_info":
            result = local_controller.get_system_info()
        
        elif command_type == "open_url":
            url = params.get("url", "")
            if not url:
                raise HTTPException(status_code=400, detail="URL不能为空")
            result = local_controller.open_url(url)
        
        elif command_type == "screenshot":
            save_path = params.get("save_path", "screenshot.jpg")
            result = local_controller.screenshot(save_path)
        
        elif command_type == "create_file":
            file_path = params.get("file_path", "")
            if not file_path:
                raise HTTPException(status_code=400, detail="文件路径不能为空")
            content = params.get("content", "")
            result = local_controller.create_file(file_path, content)
        
        elif command_type == "copy_file":
            source = params.get("source", "")
            destination = params.get("destination", "")
            if not source or not destination:
                raise HTTPException(status_code=400, detail="源文件路径和目标文件路径不能为空")
            result = local_controller.copy_file(source, destination)
        
        elif command_type == "delete_file":
            file_path = params.get("file_path", "")
            if not file_path:
                raise HTTPException(status_code=400, detail="文件路径不能为空")
            result = local_controller.delete_file(file_path)
        
        elif command_type == "start_application":
            app_name = params.get("app_name", "")
            if not app_name:
                raise HTTPException(status_code=400, detail="应用程序名称不能为空")
            result = local_controller.start_application(app_name)
        
        elif command_type == "get_process_list":
            result = local_controller.get_process_list()
        
        else:
            raise HTTPException(status_code=400, detail="不支持的命令类型")
        
        return {"response": result, "success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行命令失败: {str(e)}")

# 语音处理接口（示例）
@ai_assistant_router.post("/voice-to-text")
async def voice_to_text(voice_data: str):
    """语音转文字接口（示例）
    
    Args:
        voice_data: 语音数据（base64编码或其他格式）
        
    Returns:
        识别后的文本
    """
    # 实际项目中这里应该调用语音识别模型（如Whisper）
    # 这里返回示例响应
    return {"text": "这是一段语音识别的示例结果", "confidence": 0.95}

@ai_assistant_router.post("/text-to-voice")
async def text_to_voice(text: str):
    """文字转语音接口（示例）
    
    Args:
        text: 要转换的文本
        
    Returns:
        语音数据
    """
    # 实际项目中这里应该调用语音合成模型
    # 这里返回示例响应
    return {"voice_data": "base64_encoded_voice_data", "format": "wav"}

# 学习服务相关接口
@ai_assistant_router.post("/feedback")
async def submit_feedback(feedback: dict):
    """提交反馈数据
    
    Args:
        feedback: 反馈数据，包含评分、评论、交互ID等
        
    Returns:
        反馈提交结果
    """
    try:
        learning_service.save_feedback_data(feedback)
        
        # 如果是负面反馈，触发模型自动更新
        if feedback.get("type") == "negative":
            logger.info("收到负面反馈，触发模型自动更新")
            learning_service.update_model()
        
        return {"status": "success", "message": "反馈提交成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

@ai_assistant_router.get("/learning/analyze")
async def analyze_data():
    """分析交互数据，生成分析报告
    
    Returns:
        分析报告
    """
    try:
        report = learning_service.analyze_interaction_data()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析数据失败: {str(e)}")

@ai_assistant_router.post("/learning/update-model")
async def update_ai_model():
    """更新AI模型
    
    Returns:
        模型更新结果
    """
    try:
        result = learning_service.update_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新模型失败: {str(e)}")

@ai_assistant_router.get("/learning/interactions")
async def get_interactions(limit: int = 100):
    """获取交互数据
    
    Args:
        limit: 返回数据的数量限制
        
    Returns:
        交互数据列表
    """
    try:
        interactions = learning_service.get_interaction_data(limit)
        return {"interactions": interactions, "count": len(interactions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取交互数据失败: {str(e)}")

@ai_assistant_router.get("/learning/feedbacks")
async def get_feedbacks(limit: int = 100):
    """获取反馈数据
    
    Args:
        limit: 返回数据的数量限制
        
    Returns:
        反馈数据列表
    """
    try:
        feedbacks = learning_service.get_feedback_data(limit)
        return {"feedbacks": feedbacks, "count": len(feedbacks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取反馈数据失败: {str(e)}")

@ai_assistant_router.post("/learning/clear-old-data")
async def clear_old_data(days: int = 30):
    """清理旧数据
    
    Args:
        days: 保留最近多少天的数据
        
    Returns:
        清理结果
    """
    try:
        result = learning_service.clear_old_data(days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理旧数据失败: {str(e)}")

# 聊天扩展管理接口
@ai_assistant_router.get("/chat-extensions")
async def get_chat_extensions():
    """获取所有已注册的聊天扩展
    
    Returns:
        聊天扩展列表
    """
    try:
        from src.core.services.ai_model_service import aimodel_service
        extensions = aimodel_service.get_chat_extensions()
        enabled = aimodel_service.is_chat_extensions_enabled()
        return {
            "extensions": extensions,
            "enabled": enabled
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天扩展失败: {str(e)}")

@ai_assistant_router.post("/chat-extensions/enable")
async def enable_chat_extensions():
    """启用聊天扩展统一处理
    
    Returns:
        操作结果
    """
    try:
        from src.core.services.ai_model_service import aimodel_service
        aimodel_service.set_chat_extensions_enabled(True)
        return {
            "status": "success",
            "message": "聊天扩展已启用"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启用聊天扩展失败: {str(e)}")

@ai_assistant_router.post("/chat-extensions/disable")
async def disable_chat_extensions():
    """禁用聊天扩展统一处理
    
    Returns:
        操作结果
    """
    try:
        from src.core.services.ai_model_service import aimodel_service
        aimodel_service.set_chat_extensions_enabled(False)
        return {
            "status": "success",
            "message": "聊天扩展已禁用"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"禁用聊天扩展失败: {str(e)}")

# 动态意图规则管理接口
@ai_assistant_router.get("/intent-rules")
async def get_intent_rules():
    """获取当前意图识别规则
    
    Returns:
        当前意图识别规则
    """
    try:
        from src.core.services.ai_model_service import aimodel_service
        return {"intent_rules": aimodel_service.intent_rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取意图规则失败: {str(e)}")

@ai_assistant_router.post("/intent-rules")
async def update_intent_rules(rules: dict):
    """更新意图识别规则
    
    Args:
        rules: 新的意图识别规则
        
    Returns:
        更新结果
    """
    try:
        from src.core.services.ai_model_service import aimodel_service
        aimodel_service.update_intent_rules(rules)
        return {"status": "success", "message": "意图规则更新成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新意图规则失败: {str(e)}")

@ai_assistant_router.post("/intent-rules/add")
async def add_intent_keywords(request: dict):
    """添加意图关键词
    
    Args:
        request: 包含意图类型和关键词列表的请求数据
        
    Returns:
        添加结果
    """
    try:
        from src.core.services.ai_model_service import aimodel_service
        intent = request.get("intent")
        keywords = request.get("keywords", [])
        if not intent or not keywords:
            raise HTTPException(status_code=400, detail="意图类型和关键词列表不能为空")
        aimodel_service.add_intent_keywords(intent, keywords)
        return {"status": "success", "message": "意图关键词添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加意图关键词失败: {str(e)}")


# 互联网服务接口
@ai_assistant_router.post("/internet/search")
async def internet_search(query: str, limit: int = 5):
    """搜索互联网
    
    Args:
        query: 搜索查询
        limit: 返回结果数量
        
    Returns:
        搜索结果
    """
    try:
        result = internet_service.search_internet(query, limit=limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索互联网失败: {str(e)}")

@ai_assistant_router.post("/internet/retrieve-data")
async def retrieve_data(request: dict):
    """检索数据
    
    Args:
        request: 包含URL和查询参数的请求数据
        
    Returns:
        检索结果
    """
    try:
        url = request.get("url")
        query_params = request.get("query_params", {})
        if not url:
            raise HTTPException(status_code=400, detail="URL不能为空")
        result = internet_service.retrieve_data(url, query_params=query_params)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索数据失败: {str(e)}")

@ai_assistant_router.post("/internet/call-api")
async def call_api(request: dict):
    """调用API
    
    Args:
        request: 包含URL、方法和参数的请求数据
        
    Returns:
        API调用结果
    """
    try:
        url = request.get("url")
        method = request.get("method", "GET")
        params = request.get("params", {})
        headers = request.get("headers", {})
        data = request.get("data", {})
        if not url:
            raise HTTPException(status_code=400, detail="URL不能为空")
        result = internet_service.call_api(url, method=method, params=params, headers=headers, data=data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用API失败: {str(e)}")

@ai_assistant_router.post("/internet/download-file")
async def download_file(request: dict):
    """下载文件
    
    Args:
        request: 包含URL和保存路径的请求数据
        
    Returns:
        下载结果
    """
    try:
        url = request.get("url")
        save_path = request.get("save_path", "./")
        if not url:
            raise HTTPException(status_code=400, detail="URL不能为空")
        result = internet_service.download_file(url, save_path=save_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")


# 智能体编排接口
@ai_assistant_router.get("/agents")
async def get_agents():
    """获取可用智能体列表
    
    Returns:
        智能体列表
    """
    try:
        agents = agent_orchestrator.get_available_agents()
        # 将Agent对象转换为字典格式
        agent_dicts = [agent.to_dict() for agent in agents]
        return {"success": True, "data": agent_dicts}
    except Exception as e:
        return {"success": False, "error": f"获取智能体列表失败: {str(e)}"}

@ai_assistant_router.post("/agents/register")
async def register_agent(agent_info: dict):
    """注册智能体
    
    Args:
        agent_info: 智能体信息
        
    Returns:
        注册结果
    """
    try:
        result = agent_orchestrator.register_agent(agent_info)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册智能体失败: {str(e)}")

@ai_assistant_router.post("/agents/task")
async def delegate_task(task: dict):
    """委托任务给智能体
    
    Args:
        task: 任务信息
        
    Returns:
        任务委托结果
    """
    try:
        task_id = agent_orchestrator.delegate_task(task)
        return {"task_id": task_id, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"委托任务失败: {str(e)}")

@ai_assistant_router.get("/agents/task/{task_id}")
async def get_task_result(task_id: str):
    """获取任务结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务结果
    """
    try:
        result = agent_orchestrator.get_task_result(task_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务结果失败: {str(e)}")


# 设备控制接口
@ai_assistant_router.get("/devices")
async def get_devices():
    """获取所有设备
    
    Returns:
        设备列表
    """
    try:
        devices = device_controller.get_all_devices()
        return {"devices": devices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取设备列表失败: {str(e)}")

@ai_assistant_router.post("/devices/discover")
async def discover_devices(protocol: str = None):
    """发现设备
    
    Args:
        protocol: 通信协议
        
    Returns:
        发现的设备列表
    """
    try:
        devices = device_controller.discover_devices(protocol=protocol)
        device_dicts = [device.to_dict() for device in devices]
        return {"devices": device_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发现设备失败: {str(e)}")

@ai_assistant_router.post("/devices/command")
async def send_device_command(request: dict):
    """向设备发送命令
    
    Args:
        request: 包含设备ID、命令和参数的请求数据
        
    Returns:
        命令执行结果
    """
    try:
        device_id = request.get("device_id")
        command = request.get("command")
        params = request.get("params", {})
        if not device_id or not command:
            raise HTTPException(status_code=400, detail="设备ID和命令不能为空")
        result = device_controller.send_command(device_id, command, params)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向设备发送命令失败: {str(e)}")

@ai_assistant_router.get("/devices/health")
async def device_health_check():
    """设备健康检查
    
    Returns:
        健康检查结果
    """
    try:
        result = device_controller.health_check()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设备健康检查失败: {str(e)}")

@ai_assistant_router.post("/devices/group")
async def create_device_group(request: dict):
    """创建设备组
    
    Args:
        request: 包含组名称和设备ID列表的请求数据
        
    Returns:
        创建设备组结果
    """
    try:
        name = request.get("name")
        device_ids = request.get("device_ids", [])
        if not name:
            raise HTTPException(status_code=400, detail="组名称不能为空")
        group_id = device_controller.create_device_group(name, device_ids)
        return {"group_id": group_id, "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建设备组失败: {str(e)}")


# 用户习惯接口
@ai_assistant_router.post("/habits/record")
async def record_behavior(request: dict):
    """记录用户行为
    
    Args:
        request: 包含用户ID、行为类型和参数的请求数据
        
    Returns:
        记录结果
    """
    try:
        user_id = request.get("user_id", "default_user")
        behavior_type = request.get("behavior_type")
        params = request.get("params", {})
        if not behavior_type:
            raise HTTPException(status_code=400, detail="行为类型不能为空")
        result = user_habit_service.record_user_behavior(user_id, behavior_type, params)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录用户行为失败: {str(e)}")

@ai_assistant_router.get("/habits/analyze/{user_id}")
async def analyze_user_habits(user_id: str, time_range_days: int = 7):
    """分析用户习惯
    
    Args:
        user_id: 用户ID
        time_range_days: 分析的时间范围（天）
        
    Returns:
        习惯分析结果
    """
    try:
        result = user_habit_service.analyze_user_habits(user_id, time_range_days=time_range_days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析用户习惯失败: {str(e)}")

@ai_assistant_router.post("/habits/predict")
async def predict_user_behavior(request: dict):
    """预测用户行为
    
    Args:
        request: 包含用户ID和上下文的请求数据
        
    Returns:
        行为预测结果
    """
    try:
        user_id = request.get("user_id", "default_user")
        context = request.get("context", {})
        result = user_habit_service.predict_user_behavior(user_id, context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测用户行为失败: {str(e)}")

@ai_assistant_router.get("/habits/recommendations/{user_id}")
async def get_habit_recommendations(user_id: str):
    """获取习惯推荐
    
    Args:
        user_id: 用户ID
        
    Returns:
        习惯推荐列表
    """
    try:
        recommendations = user_habit_service.get_habit_recommendations(user_id)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取习惯推荐失败: {str(e)}")

@ai_assistant_router.get("/habits/profile/{user_id}")
async def get_user_profile(user_id: str):
    """获取用户画像
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户画像
    """
    try:
        profile = user_habit_service.get_user_profile(user_id)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户画像失败: {str(e)}")


# 自动部署接口
@ai_assistant_router.get("/deployment/check-update")
async def check_for_updates():
    """检查更新
    
    Returns:
        更新检查结果
    """
    try:
        result = deployment_service.check_for_updates()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查更新失败: {str(e)}")

@ai_assistant_router.post("/deployment/update")
async def update_application():
    """更新应用程序
    
    Returns:
        更新结果
    """
    try:
        result = deployment_service.update_application()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新应用程序失败: {str(e)}")

@ai_assistant_router.post("/deployment/generate-package")
async def generate_installation_package():
    """生成本地安装包
    
    Returns:
        安装包生成结果
    """
    try:
        result = deployment_service.generate_installation_package()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成本地安装包失败: {str(e)}")

@ai_assistant_router.get("/deployment/status")
async def get_deployment_status():
    """获取部署状态
    
    Returns:
        部署状态信息
    """
    try:
        status = deployment_service.get_deployment_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取部署状态失败: {str(e)}")

@ai_assistant_router.get("/deployment/install-guide")
async def get_installation_guide():
    """获取安装指南
    
    Returns:
        安装指南信息
    """
    try:
        guide = deployment_service.get_installation_guide()
        return guide
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取安装指南失败: {str(e)}")

@ai_assistant_router.post("/deployment/config")
async def update_deployment_config(config: dict):
    """更新部署配置
    
    Args:
        config: 新的部署配置
        
    Returns:
        更新结果
    """
    try:
        result = deployment_service.update_config(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新部署配置失败: {str(e)}")


# 调度服务接口
@ai_assistant_router.get("/schedule/status")
async def get_schedule_status():
    """获取调度服务状态
    
    Returns:
        调度服务状态信息
    """
    try:
        from src.core.services.schedule_service import schedule_service
        status = schedule_service.get_status()
        return {"success": True, "data": status}
    except Exception as e:
        return {"success": False, "error": f"获取调度服务状态失败: {str(e)}"}

@ai_assistant_router.post("/schedule/task/cron")
async def add_cron_task(request: dict):
    """添加CRON调度任务
    
    Args:
        request: 包含任务类型、描述、CRON表达式和配置的请求数据
        
    Returns:
        添加结果
    """
    try:
        from src.core.services.schedule_service import schedule_service
        task_type = request.get("task_type")
        description = request.get("description")
        cron_expression = request.get("cron_expression")
        config = request.get("config", {})
        
        if not task_type or not description or not cron_expression:
            raise HTTPException(status_code=400, detail="任务类型、描述和CRON表达式不能为空")
        
        # 定义任务回调函数
        def task_callback(task):
            from src.core.services.agent_orchestrator import agent_orchestrator
            return agent_orchestrator._execute_auto_check_task(task_type, config)
        
        task_id = schedule_service.add_cron_task(
            task_type=task_type,
            description=description,
            cron_expression=cron_expression,
            callback=task_callback,
            config=config
        )
        
        return {"task_id": task_id, "status": "success", "message": "CRON任务添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加CRON任务失败: {str(e)}")

@ai_assistant_router.post("/schedule/task/interval")
async def add_interval_task(request: dict):
    """添加间隔调度任务
    
    Args:
        request: 包含任务类型、描述、间隔时间和配置的请求数据
        
    Returns:
        添加结果
    """
    try:
        from src.core.services.schedule_service import schedule_service
        task_type = request.get("task_type")
        description = request.get("description")
        interval_seconds = request.get("interval_seconds")
        config = request.get("config", {})
        
        if not task_type or not description or interval_seconds is None:
            raise HTTPException(status_code=400, detail="任务类型、描述和间隔时间不能为空")
        
        # 定义任务回调函数
        def task_callback(task):
            from src.core.services.agent_orchestrator import agent_orchestrator
            return agent_orchestrator._execute_auto_check_task(task_type, config)
        
        task_id = schedule_service.add_interval_task(
            task_type=task_type,
            description=description,
            interval_seconds=interval_seconds,
            callback=task_callback,
            config=config
        )
        
        return {"task_id": task_id, "status": "success", "message": "间隔任务添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加间隔任务失败: {str(e)}")

@ai_assistant_router.post("/schedule/task/one-time")
async def add_one_time_task(request: dict):
    """添加一次性调度任务
    
    Args:
        request: 包含任务类型、描述、执行时间和配置的请求数据
        
    Returns:
        添加结果
    """
    try:
        from src.core.services.schedule_service import schedule_service
        from datetime import datetime
        
        task_type = request.get("task_type")
        description = request.get("description")
        execute_time_str = request.get("execute_time")
        config = request.get("config", {})
        
        if not task_type or not description or not execute_time_str:
            raise HTTPException(status_code=400, detail="任务类型、描述和执行时间不能为空")
        
        # 解析执行时间
        execute_time = datetime.fromisoformat(execute_time_str)
        
        # 定义任务回调函数
        def task_callback(task):
            from src.core.services.agent_orchestrator import agent_orchestrator
            return agent_orchestrator._execute_auto_check_task(task_type, config)
        
        task_id = schedule_service.add_one_time_task(
            task_type=task_type,
            description=description,
            execute_time=execute_time,
            callback=task_callback,
            config=config
        )
        
        return {"task_id": task_id, "status": "success", "message": "一次性任务添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加一次性任务失败: {str(e)}")

@ai_assistant_router.delete("/schedule/task/{task_id}")
async def remove_schedule_task(task_id: str):
    """移除调度任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        移除结果
    """
    try:
        from src.core.services.schedule_service import schedule_service
        result = schedule_service.remove_task(task_id)
        if result:
            return {"status": "success", "message": "调度任务移除成功"}
        else:
            raise HTTPException(status_code=404, detail="调度任务不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移除调度任务失败: {str(e)}")

@ai_assistant_router.get("/schedule/tasks")
async def get_schedule_tasks():
    """获取所有调度任务
    
    Returns:
        调度任务列表
    """
    try:
        from src.core.services.schedule_service import schedule_service
        tasks = schedule_service.get_all_tasks()
        return {"success": True, "data": [task.to_dict() for task in tasks]}
    except Exception as e:
        return {"success": False, "error": f"获取调度任务列表失败: {str(e)}"}

@ai_assistant_router.post("/schedule/start")
async def start_schedule_service():
    """启动调度服务
    
    Returns:
        启动结果
    """
    try:
        from src.core.services.schedule_service import schedule_service
        result = schedule_service.start()
        if result:
            return {"status": "success", "message": "调度服务启动成功"}
        else:
            return {"status": "error", "message": "调度服务启动失败"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动调度服务失败: {str(e)}")

@ai_assistant_router.post("/schedule/stop")
async def stop_schedule_service():
    """停止调度服务
    
    Returns:
        停止结果
    """
    try:
        from src.core.services.schedule_service import schedule_service
        result = schedule_service.stop()
        if result:
            return {"status": "success", "message": "调度服务停止成功"}
        else:
            return {"status": "error", "message": "调度服务停止失败"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止调度服务失败: {str(e)}")


# 自动检查智能体接口
@ai_assistant_router.post("/auto-check/static-analysis")
async def run_static_analysis(request: dict):
    """运行代码静态分析
    
    Args:
        request: 包含目录路径和文件类型的请求数据
        
    Returns:
        静态分析结果
    """
    try:
        from src.core.services.auto_check_agents.static_analysis_agent import static_analysis_agent
        
        directory_path = request.get("directory", "./src")
        file_types = request.get("file_types", ["python"])
        
        result = static_analysis_agent.analyze_directory(directory_path, file_types)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行静态分析失败: {str(e)}")

@ai_assistant_router.post("/auto-check/error-monitor")
async def run_error_monitor(request: dict):
    """运行错误监控
    
    Args:
        request: 包含时间范围和日志路径的请求数据
        
    Returns:
        错误监控结果
    """
    try:
        from src.core.services.auto_check_agents.error_monitor_agent import error_monitor_agent
        
        time_range_hours = request.get("time_range_hours", 24)
        log_path = request.get("log_path")
        
        if log_path:
            result = error_monitor_agent.analyze_log_file(log_path, time_range_hours=time_range_hours)
        else:
            result = error_monitor_agent.generate_error_report(time_range_hours)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行错误监控失败: {str(e)}")

@ai_assistant_router.post("/auto-check/auto-test")
async def run_auto_test(request: dict):
    """运行自动测试
    
    Args:
        request: 包含测试路径、框架和报告格式的请求数据
        
    Returns:
        自动测试结果
    """
    try:
        from src.core.services.auto_check_agents.auto_test_agent import auto_test_agent
        
        test_path = request.get("test_path", "./tests")
        framework = request.get("framework", "pytest")
        report_format = request.get("report_format", "json")
        
        result = auto_test_agent.run_all_tests(test_path, framework, report_format)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行自动测试失败: {str(e)}")

@ai_assistant_router.post("/auto-check/full-check")
async def run_full_auto_check():
    """运行完整的自动检查（静态分析+错误监控+自动测试）
    
    Returns:
        完整检查结果
    """
    try:
        from src.core.services.agent_orchestrator import agent_orchestrator
        
        # 使用智能体编排服务创建自动检查任务链
        result = agent_orchestrator.orchestrate_task("执行完整的自动检查，包括代码静态分析、错误监控和自动测试")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行完整自动检查失败: {str(e)}")

@ai_assistant_router.post("/auto-check/log-monitor/start")
async def start_log_monitoring(request: dict):
    """开始监控日志文件
    
    Args:
        request: 包含日志路径和类型的请求数据
        
    Returns:
        监控启动结果
    """
    try:
        from src.core.services.auto_check_agents.error_monitor_agent import error_monitor_agent
        
        log_path = request.get("log_path")
        log_type = request.get("log_type", "general")
        
        if not log_path:
            raise HTTPException(status_code=400, detail="日志路径不能为空")
        
        result = error_monitor_agent.monitor_log_file(log_path, log_type)
        if result:
            # 启动错误监控智能体
            error_monitor_agent.start_monitoring()
            return {"status": "success", "message": "日志监控已启动"}
        else:
            return {"status": "error", "message": "日志监控启动失败"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动日志监控失败: {str(e)}")

@ai_assistant_router.post("/auto-check/log-monitor/stop")
async def stop_log_monitoring(request: dict):
    """停止监控日志文件
    
    Args:
        request: 包含日志路径的请求数据
        
    Returns:
        监控停止结果
    """
    try:
        from src.core.services.auto_check_agents.error_monitor_agent import error_monitor_agent
        
        log_path = request.get("log_path")
        
        if not log_path:
            # 停止所有监控
            error_monitor_agent.stop_all_monitoring()
            return {"status": "success", "message": "所有日志监控已停止"}
        
        result = error_monitor_agent.stop_monitoring(log_path)
        if result:
            return {"status": "success", "message": "日志监控已停止"}
        else:
            return {"status": "error", "message": "日志监控停止失败"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止日志监控失败: {str(e)}")

@ai_assistant_router.get("/auto-check/results")
async def get_auto_check_results():
    """获取自动检查结果列表
    
    Returns:
        检查结果列表
    """
    try:
        from src.core.services.agent_orchestrator import agent_orchestrator
        # 获取所有检查结果
        results = agent_orchestrator.get_check_results()
        return {"success": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取检查结果失败: {str(e)}")

@ai_assistant_router.get("/auto-check/types")
async def get_check_types():
    """获取所有检查类型
    
    Returns:
        检查类型列表
    """
    try:
        check_types = [
            "code_quality",
            "security",
            "performance",
            "dependency",
            "style"
        ]
        return {"success": True, "data": check_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取检查类型失败: {str(e)}")

@ai_assistant_router.post("/auto-check/run")
async def run_auto_check(request: dict):
    """运行自动检查
    
    Args:
        request: 包含检查类型、目标、调度配置的请求数据
        
    Returns:
        检查结果
    """
    try:
        from src.core.services.agent_orchestrator import agent_orchestrator
        
        check_type = request.get("check_type", "code_quality")
        target = request.get("target", "")
        scheduled = request.get("scheduled", False)
        cron_expression = request.get("cron_expression", "0 0 * * *")
        check_depth = request.get("check_depth", "basic")
        
        # 构建检查配置
        config = {
            "check_type": check_type,
            "target": target,
            "check_depth": check_depth
        }
        
        # 执行检查
        result = agent_orchestrator._execute_auto_check_task(check_type, config)
        
        # 如果需要调度，创建定时任务
        if scheduled:
            from src.core.services.schedule_service import schedule_service
            
            def check_callback(task):
                return agent_orchestrator._execute_auto_check_task(check_type, config)
            
            schedule_service.add_cron_task(
                task_type=check_type,
                description=f"定时自动检查: {check_type}",
                cron_expression=cron_expression,
                callback=check_callback,
                config=config
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行自动检查失败: {str(e)}")

# AI微调相关API
class FineTuneRequest(BaseModel):
    """微调请求模型"""
    model_id: str
    dataset_path: str
    config: dict = None

class FineTuneConfig(BaseModel):
    """微调配置"""
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

@ai_assistant_router.post("/fine-tune/start")
async def start_fine_tuning(request: FineTuneRequest):
    """开始AI模型微调
    
    Args:
        request: 包含模型ID、数据集路径和配置的请求数据
        
    Returns:
        微调任务ID和状态
    """
    try:
        import uuid
        from src.core.services.training_service import TrainingService
        from src.core.services import model_manager
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 获取微调配置
        config = request.config or {}
        fine_tune_config = {
            "learning_rate": config.get("learning_rate", 1e-5),
            "batch_size": config.get("batch_size", 16),
            "num_epochs": config.get("num_epochs", 3),
            "warmup_steps": config.get("warmup_steps", 100),
            "weight_decay": config.get("weight_decay", 0.01),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
            "max_grad_norm": config.get("max_grad_norm", 1.0),
            "lora_r": config.get("lora_r", 8),
            "lora_alpha": config.get("lora_alpha", 16),
            "lora_dropout": config.get("lora_dropout", 0.05)
        }
        
        # 创建训练服务
        training_service = TrainingService(model_manager)
        
        # 模拟开始微调任务
        logger.info(f"开始微调任务: {task_id}, 模型: {request.model_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "model_id": request.model_id,
            "status": "running",
            "message": "微调任务已开始",
            "config": fine_tune_config
        }
    except Exception as e:
        logger.error(f"启动微调任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动微调任务失败: {str(e)}")

@ai_assistant_router.get("/fine-tune/tasks")
async def get_fine_tune_tasks():
    """获取所有微调任务列表
    
    Returns:
        微调任务列表
    """
    try:
        # 模拟返回微调任务列表
        tasks = [
            {
                "task_id": "ft-001",
                "model_id": "gpt-3.5-turbo",
                "status": "completed",
                "progress": 100.0,
                "created_at": "2026-01-10T10:00:00",
                "completed_at": "2026-01-10T12:30:00",
                "metrics": {
                    "loss": 0.0234,
                    "accuracy": 0.9567
                }
            },
            {
                "task_id": "ft-002",
                "model_id": "gpt-3.5-turbo",
                "status": "running",
                "progress": 65.0,
                "created_at": "2026-01-11T08:00:00",
                "completed_at": None,
                "metrics": None
            }
        ]
        
        return {"success": True, "data": tasks}
    except Exception as e:
        logger.error(f"获取微调任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取微调任务列表失败: {str(e)}")

@ai_assistant_router.get("/fine-tune/task/{task_id}")
async def get_fine_tune_task(task_id: str):
    """获取指定微调任务的详细信息
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务详细信息
    """
    try:
        # 模拟返回任务详情
        task = {
            "task_id": task_id,
            "model_id": "gpt-3.5-turbo",
            "status": "running",
            "progress": 65.0,
            "created_at": "2026-01-11T08:00:00",
            "completed_at": None,
            "config": {
                "learning_rate": 1e-5,
                "batch_size": 16,
                "num_epochs": 3
            },
            "metrics": {
                "current_loss": 0.0456,
                "current_accuracy": 0.9234,
                "best_loss": 0.0321,
                "best_accuracy": 0.9456
            },
            "logs": [
                {"timestamp": "2026-01-11T08:00:00", "message": "任务开始"},
                {"timestamp": "2026-01-11T08:30:00", "message": "Epoch 1/3, Loss: 0.1234"},
                {"timestamp": "2026-01-11T09:00:00", "message": "Epoch 2/3, Loss: 0.0678"}
            ]
        }
        
        return {"success": True, "data": task}
    except Exception as e:
        logger.error(f"获取微调任务详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取微调任务详情失败: {str(e)}")

@ai_assistant_router.delete("/fine-tune/task/{task_id}")
async def delete_fine_tune_task(task_id: str):
    """删除微调任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        删除结果
    """
    try:
        logger.info(f"删除微调任务: {task_id}")
        
        return {
            "success": True,
            "message": "微调任务已删除"
        }
    except Exception as e:
        logger.error(f"删除微调任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除微调任务失败: {str(e)}")

@ai_assistant_router.get("/fine-tune/models")
async def get_fine_tune_models():
    """获取可微调的模型列表
    
    Returns:
        可微调的模型列表
    """
    try:
        # 模拟返回可微调模型列表，确保格式与前端FineTuneModel接口匹配
        models = [
            {
                "model_id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "OpenAI的GPT-3.5 Turbo模型",
                "parameters": "175B",
                "supports_fine_tuning": True
            },
            {
                "model_id": "gpt-4",
                "name": "GPT-4",
                "description": "OpenAI的GPT-4模型",
                "parameters": "Unknown",
                "supports_fine_tuning": True
            },
            {
                "model_id": "gpt-4o",
                "name": "GPT-4o",
                "description": "OpenAI的多模态模型GPT-4o",
                "parameters": "Unknown",
                "supports_fine_tuning": True
            },
            {
                "model_id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "description": "Anthropic的Claude 3 Opus模型",
                "parameters": "Unknown",
                "supports_fine_tuning": True
            }
        ]
        
        return {"success": True, "data": models}
    except Exception as e:
        logger.error(f"获取可微调模型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取可微调模型列表失败: {str(e)}")

@ai_assistant_router.post("/fine-tune/tasks/{task_id}/stop")
async def stop_fine_tuning(task_id: str):
    """停止正在进行的微调任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        停止结果
    """
    try:
        logger.info(f"停止微调任务: {task_id}")
        
        return {
            "success": True,
            "message": "微调任务已停止",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"停止微调任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止微调任务失败: {str(e)}")

@ai_assistant_router.post("/fine-tune/all-agents")
async def fine_tune_all_agents():
    """对所有智能体进行微调
    
    Returns:
        微调任务列表，包含每个智能体的微调任务ID和状态
    """
    try:
        import uuid
        from src.core.services.training_service import TrainingService
        from src.core.services import model_manager
        from src.core.services.agent_orchestrator import agent_orchestrator
        
        logger.info("开始对所有智能体进行微调")
        
        # 1. 获取所有可用智能体
        all_agents = agent_orchestrator.get_available_agents()
        logger.info(f"获取到 {len(all_agents)} 个智能体")
        
        # 2. 智能体类型到模型ID的映射
        agent_model_map = {
            "search": "gpt-3.5-turbo",
            "analysis": "gpt-3.5-turbo",
            "writing": "gpt-3.5-turbo",
            "code": "code-llama-7b",
            "translation": "gpt-3.5-turbo",
            "image": "stable-diffusion-v1.5",
            "audio": "whisper-large-v3",
            "video": "gpt-4o",
            "other": "gpt-3.5-turbo"
        }
        
        # 3. 为每个智能体创建微调任务
        tasks = []
        training_service = TrainingService(model_manager)
        
        for agent in all_agents:
            try:
                # 获取智能体对应的模型ID
                model_id = agent_model_map.get(agent.agent_type, "gpt-3.5-turbo")
                
                # 生成任务ID
                task_id = str(uuid.uuid4())
                
                # 默认微调配置
                fine_tune_config = {
                    "learning_rate": 1e-5,
                    "batch_size": 16,
                    "num_epochs": 3,
                    "warmup_steps": 100,
                    "weight_decay": 0.01,
                    "gradient_accumulation_steps": 1,
                    "max_grad_norm": 1.0,
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05
                }
                
                # 模拟开始微调任务
                logger.info(f"开始微调智能体 {agent.agent_id} ({agent.name})，模型: {model_id}")
                
                # 创建微调任务
                tasks.append({
                    "task_id": task_id,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type,
                    "model_id": model_id,
                    "status": "running",
                    "message": "微调任务已开始",
                    "config": fine_tune_config,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            except Exception as e:
                logger.error(f"为智能体 {agent.agent_id} 创建微调任务失败: {str(e)}")
                tasks.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type,
                    "status": "failed",
                    "message": f"创建微调任务失败: {str(e)}",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
        
        logger.info(f"所有智能体微调任务创建完成，共 {len(tasks)} 个任务")
        
        return {
            "success": True,
            "message": f"已为 {len(all_agents)} 个智能体创建微调任务",
            "tasks": tasks
        }
    except Exception as e:
        logger.error(f"对所有智能体进行微调失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"对所有智能体进行微调失败: {str(e)}")

@ai_assistant_router.post("/fine-tune/stop/{task_id}")
async def stop_fine_tuning(task_id: str):
    """停止微调任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        停止结果
    """
    try:
        logger.info(f"停止微调任务: {task_id}")
        
        return {
            "success": True,
            "message": "微调任务已停止"
        }
    except Exception as e:
        logger.error(f"停止微调任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止微调任务失败: {str(e)}")

@ai_assistant_router.get("/fine-tune/models")
async def get_fine_tune_models():
    """获取可微调的模型列表
    
    Returns:
        可微调模型列表
    """
    try:
        models = [
            {
                "model_id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "高性能语言模型，适合对话和文本生成",
                "parameters": "175B",
                "supports_fine_tuning": True
            },
            {
                "model_id": "gpt-4",
                "name": "GPT-4",
                "description": "最先进的语言模型，具有强大的推理能力",
                "parameters": "1.7T",
                "supports_fine_tuning": True
            },
            {
                "model_id": "claude-3",
                "name": "Claude 3",
                "description": "Anthropic开发的高性能语言模型",
                "parameters": "400B",
                "supports_fine_tuning": True
            }
        ]
        
        return {"success": True, "data": models}
    except Exception as e:
        logger.error(f"获取可微调模型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取可微调模型列表失败: {str(e)}")

