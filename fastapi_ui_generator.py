from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import shutil
import subprocess
import json
from typing import List, Optional, Literal, Dict, Any
from dotenv import load_dotenv
import openai
import uuid

# 加载环境变量（用于存储API密钥）
load_dotenv()

# 配置OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建FastAPI应用
app = FastAPI(
    title=os.getenv("APP_NAME", "应用开发与设备控制助手"),
    description="自动生成应用代码和控制设备的AI助手",
    version="1.0.0"
)

# 历史记录存储
class HistoryItem:
    def __init__(self, requirement: str, generated_code: str, ui_framework: str, timestamp: float = None):
        self.requirement = requirement
        self.generated_code = generated_code
        self.ui_framework = ui_framework
        self.timestamp = timestamp or time.time()

# 历史记录列表（实际项目中可替换为数据库）
history: List[HistoryItem] = []

# 支持的UI框架
UI_FRAMEWORKS = ["element-plus", "ant-design-vue", "vuetify", "quasar", "tailwindcss"]

# 支持的应用类型
APP_TYPES = ["web", "mobile", "desktop", "api", "crud", "admin"]

# 支持的后端框架
BACKEND_FRAMEWORKS = ["fastapi", "flask", "django"]

# 支持的前端框架
FRONTEND_FRAMEWORKS = ["vue3", "react", "svelte", "react-native", "flutter", "electron-vue3", "tauri-react"]

# 应用模板配置
APP_TEMPLATES = {
    # Web应用模板
    "fastapi-vue3": {
        "name": "FastAPI + Vue 3 + TypeScript",
        "description": "现代化的全栈应用模板，使用FastAPI作为后端，Vue 3 + TypeScript作为前端",
        "backend": "fastapi",
        "frontend": "vue3",
        "features": ["REST API", "TypeScript", "Vue 3 Composition API", "Tailwind CSS", "Docker支持"]
    },
    "flask-react": {
        "name": "Flask + React + TypeScript",
        "description": "经典的全栈应用模板，使用Flask作为后端，React + TypeScript作为前端",
        "backend": "flask",
        "frontend": "react",
        "features": ["REST API", "TypeScript", "React Hooks", "Material UI", "Docker支持"]
    },
    "django-vue3": {
        "name": "Django + Vue 3",
        "description": "功能丰富的全栈应用模板，使用Django作为后端，Vue 3作为前端",
        "backend": "django",
        "frontend": "vue3",
        "features": ["Django REST Framework", "Vue 3", "Element Plus", "Docker支持"]
    },
    
    # 专用模板
    "api-only": {
        "name": "纯API服务",
        "description": "仅包含后端API服务，适合微服务架构",
        "backend": "fastapi",
        "frontend": None,
        "features": ["REST API", "TypeScript", "Docker支持", "API文档"]
    },
    "web-admin": {
        "name": "管理后台",
        "description": "适合构建管理后台的应用模板",
        "backend": "fastapi",
        "frontend": "vue3",
        "features": ["权限管理", "数据可视化", "表单生成器", "日志系统"]
    },
    
    # 移动端应用模板
    "react-native": {
        "name": "React Native 移动应用",
        "description": "跨平台移动端应用模板，使用React Native开发",
        "backend": "fastapi",
        "frontend": "react-native",
        "features": ["跨平台", "TypeScript", "REST API集成", "推送通知", "设备访问"]
    },
    "flutter-fastapi": {
        "name": "Flutter + FastAPI",
        "description": "高性能跨平台移动应用模板，使用Flutter开发前端，FastAPI作为后端",
        "backend": "fastapi",
        "frontend": "flutter",
        "features": ["跨平台", "Dart语言", "REST API集成", "高性能UI", "设备访问"]
    },
    
    # 桌面端应用模板
    "electron-vue3": {
        "name": "Electron + Vue 3",
        "description": "跨平台桌面应用模板，使用Electron和Vue 3开发",
        "backend": "fastapi",
        "frontend": "electron-vue3",
        "features": ["跨平台", "TypeScript", "桌面API访问", "REST API集成", "自动更新"]
    },
    "tauri-react": {
        "name": "Tauri + React",
        "description": "轻量级跨平台桌面应用模板，使用Tauri和React开发",
        "backend": "fastapi",
        "frontend": "tauri-react",
        "features": ["轻量级", "TypeScript", "原生API访问", "REST API集成"]
    },
    
    # IoT设备控制模板
    "iot-device-control": {
        "name": "IoT设备控制平台",
        "description": "用于控制和监控IoT设备的应用模板",
        "backend": "fastapi",
        "frontend": "vue3",
        "features": ["MQTT支持", "WebSocket实时通信", "设备管理", "数据可视化", "告警系统"]
    },
    "edge-computing": {
        "name": "边缘计算应用",
        "description": "适合边缘计算场景的应用模板",
        "backend": "fastapi",
        "frontend": "vue3",
        "features": ["边缘设备管理", "本地计算", "云端同步", "数据缓存", "低延迟通信"]
    }
}

# 项目规范提示词模板（可根据实际项目修改）
PROJECT_PROMPT_TEMPLATE = """
你是我的专属Vue UI代码生成助手，必须严格遵守以下规范：

# 1. 基础规范
- 框架版本：Vue 3 + TypeScript
- 组件风格：单文件组件（SFC）
- 脚本语法：setup语法糖，script标签必须加lang="ts"
- 命名规范：组件名大驼峰，函数名小驼峰，变量名小驼峰
- 注释要求：关键逻辑加单行注释，组件加功能描述

# 2. UI框架规范 - {ui_framework}
{framework_specific_rules}

# 3. 布局规范
- 主布局：使用flex布局，支持响应式设计
- 移动端适配：必须使用媒体查询（@media (max-width:768px)）
- 间距规范：统一使用20px作为基础间距，使用倍数关系（10px, 20px, 40px）

# 4. 接口规范
- 请求返回格式：所有请求返回格式为 {{code: number, data: any, msg: string}}
- 数据处理：使用try-catch处理API请求异常
- 加载状态：所有异步操作必须显示加载状态

# 5. 响应式设计
- 使用Vue 3的ref和reactive，避免使用Options API
- 计算属性：复杂逻辑使用computed，避免模板中复杂表达式
- 监听属性：必要时使用watch，添加deep选项

# 6. 表单规范
- 表单组件：使用框架自带的表单组件
- 表单验证：使用框架自带的表单验证机制
- 提交处理：添加表单提交防抖，时长300ms

# 7. 列表规范
- 表格组件：使用框架自带的表格组件
- 分页支持：数据量大于10条时必须支持分页
- 搜索功能：必须包含搜索框，支持多条件搜索

# 8. 样式规范
- 样式范围：必须使用scoped样式
- CSS预处理器：使用SCSS
- 主题颜色：主色调#6C63FF，辅助色#4ECDC4，警告色#FF6B6B
- 字体规范：使用系统默认字体，大小14px，行高1.5

# 9. 代码质量要求
- 避免console.log调试代码
- 避免未使用的变量和导入
- 函数参数不超过3个，复杂参数使用对象
- 组件props使用TypeScript接口定义

# 10. 性能优化
- 列表渲染：使用v-for时必须添加key
- 组件懒加载：大型组件使用defineAsyncComponent
- 避免不必要的重新渲染：使用memo包裹纯展示组件

# 需求
{requirement}

# 额外上下文
{additional_context}
"""

# 框架特定规则
FRAMEWORK_SPECIFIC_RULES = {
    "element-plus": """
- 主题色：使用官方Element Plus主题色#6C63FF
- 按钮样式：默认按钮使用el-button，主按钮使用type="primary"
- 表单组件：el-form + el-form-item + el-input/el-select等
- 表格组件：el-table + el-table-column
- 对话框：el-dialog，标题使用title属性
- 消息提示：使用ElMessage，加载使用ElLoading
    """,
    "ant-design-vue": """
- 主题色：使用官方Ant Design Vue主题色#1890ff
- 按钮样式：默认按钮使用a-button，主按钮使用type="primary"
- 表单组件：a-form + a-form-item + a-input/a-select等
- 表格组件：a-table + columns配置
- 对话框：a-modal，标题使用title属性
- 消息提示：使用message，加载使用notification
    """,
    "vuetify": """
- 主题色：使用官方Vuetify主题色#1976D2
- 按钮样式：默认按钮使用v-btn，主按钮使用color="primary"
- 表单组件：v-form + v-text-field/v-select等
- 表格组件：v-data-table + items配置
- 对话框：v-dialog，标题使用v-card-title
- 消息提示：使用v-snackbar，加载使用v-progress-circular
    """,
    "quasar": """
- 主题色：使用官方Quasar主题色#1976D2
- 按钮样式：默认按钮使用q-btn，主按钮使用color="primary"
- 表单组件：q-form + q-input/q-select等
- 表格组件：q-table + columns配置
- 对话框：q-dialog，标题使用q-card-section
- 消息提示：使用useQuasar().notify，加载使用q-spinner
    """,
    "tailwindcss": """
- 主题色：根据需求自定义，默认使用tailwind内置颜色
- 按钮样式：使用btn类，主按钮使用bg-blue-600 hover:bg-blue-700等
- 表单组件：使用原生input + tailwind类
- 表格组件：使用div + table + tailwind类或自定义组件
- 对话框：使用div + tailwind类实现
- 消息提示：使用第三方库如toast或自定义实现
    """
}

# 定义请求模型
class UIRequest(BaseModel):
    requirement: str
    additional_context: str = ""  # 额外上下文，可选
    ui_framework: Literal["element-plus", "ant-design-vue", "vuetify", "quasar", "tailwindcss"] = "element-plus"  # UI框架选择

# 定义应用脚手架生成请求模型
class AppScaffoldRequest(BaseModel):
    app_name: str
    description: str = ""
    app_type: Literal["web", "mobile", "desktop", "api", "crud", "admin", "iot"] = "web"
    template: str = "fastapi-vue3"
    backend_framework: Literal["fastapi", "flask", "django"] = "fastapi"
    frontend_framework: Literal["vue3", "react", "svelte", "react-native", "flutter", "electron-vue3", "tauri-react"] = "vue3"
    features: List[str] = []
    database: Optional[str] = "sqlite"
    auth_required: bool = True
    docker_support: bool = True
    git_initialized: bool = True
    mqtt_support: bool = False
    websocket_support: bool = False
    device_management: bool = False

# 定义响应模型
class UIResponse(BaseModel):
    code: int
    data: dict
    msg: str

# 定义应用脚手架生成响应模型
class AppScaffoldResponse(BaseModel):
    code: int
    data: dict
    msg: str

# 定义历史记录响应模型
class HistoryResponse(BaseModel):
    code: int
    data: List[dict]
    msg: str
    total: int

# 定义导出请求模型
class ExportRequest(BaseModel):
    code: str
    filename: str = "generated-component.vue"

# 定义导出响应模型
class ExportResponse(BaseModel):
    code: int
    data: dict
    msg: str

# 定义代码质量检查响应模型
class CodeQualityResponse(BaseModel):
    code: int
    data: dict
    msg: str

# 实现真实的OpenAI API调用
async def call_ai_model(prompt: str, system_prompt: str = "你是一个专业的全栈开发助手，生成符合要求的代码，只返回代码，不包含任何解释。") -> str:
    """调用真实的OpenAI API生成代码"""
    try:
        # 使用真实的OpenAI API调用
        response = await openai.ChatCompletion.acreate(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API调用失败: {str(e)}")
        # 调用失败时返回模拟数据
        return generate_mock_ui_code(prompt)

# 生成应用脚手架
async def generate_app_scaffold(request: AppScaffoldRequest) -> dict:
    """生成完整的应用脚手架"""
    # 创建项目目录
    project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join("generated_apps", f"{request.app_name}-{project_id}")
    
    # 创建目录结构
    os.makedirs(project_dir, exist_ok=True)
    
    # 选择模板
    template_config = APP_TEMPLATES.get(request.template, APP_TEMPLATES["fastapi-vue3"])
    
    # 生成项目结构
    project_structure = {
        "root": project_dir,
        "backend": os.path.join(project_dir, "backend"),
        "frontend": os.path.join(project_dir, "frontend"),
        "docs": os.path.join(project_dir, "docs"),
        "tests": os.path.join(project_dir, "tests")
    }
    
    # 创建子目录
    for dir_path in project_structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 生成后端代码
    backend_result = await generate_backend_code(request, project_structure["backend"])
    
    # 生成前端代码
    frontend_result = {}  # 默认空
    if template_config["frontend"]:
        frontend_result = await generate_frontend_code(request, project_structure["frontend"])
    
    # 生成根目录文件
    generate_root_files(request, project_structure["root"])
    
    # 生成README
    generate_readme(request, project_structure["root"])
    
    # 初始化Git
    if request.git_initialized:
        initialize_git(project_structure["root"])
    
    # 生成Docker配置
    if request.docker_support:
        generate_docker_config(request, project_structure["root"])
    
    return {
        "project_id": project_id,
        "project_path": project_dir,
        "app_name": request.app_name,
        "template": request.template,
        "backend": backend_result,
        "frontend": frontend_result,
        "features": request.features,
        "generated_at": time.time()
    }

# 生成后端代码
async def generate_backend_code(request: AppScaffoldRequest, backend_dir: str) -> dict:
    """生成后端代码"""
    result = {
        "framework": request.backend_framework,
        "files": []
    }
    
    # 根据后端框架生成不同的代码
    if request.backend_framework == "fastapi":
        result["files"] = await generate_fastapi_backend(request, backend_dir)
    elif request.backend_framework == "flask":
        result["files"] = await generate_flask_backend(request, backend_dir)
    elif request.backend_framework == "django":
        result["files"] = await generate_django_backend(request, backend_dir)
    
    return result

# 生成FastAPI后端代码
async def generate_fastapi_backend(request: AppScaffoldRequest, backend_dir: str) -> List[str]:
    """生成FastAPI后端代码"""
    files = []
    
    # 生成main.py
    main_content = f'''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
'''    
    # 添加WebSocket支持
    if request.websocket_support:
        main_content += '''from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
'''
    
    main_content += f'''
# 创建FastAPI应用
app = FastAPI(
    title="{request.app_name}",
    description="{request.description}",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 根路径
@app.get("/")
def root():
    return {
        "message": "Welcome to {request.app_name}",
        "version": "1.0.0",
        "description": "{request.description}"
    }

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "ok"}

# API路由
@app.get("/api/v1/items")
def get_items():
    return {{"items": [], "total": 0}}
'''
    
    # 添加设备管理路由
    if request.device_management:
        main_content += '''
# 设备管理路由
@app.get("/api/v1/devices")
def get_devices():
    return {"devices": [], "total": 0}

@app.post("/api/v1/devices/{device_id}/control")
def control_device(device_id: str, command: dict):
    return {"device_id": device_id, "command": command, "status": "success"}
'''
    
    # 添加WebSocket支持
    if request.websocket_support:
        main_content += '''
# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# WebSocket端点
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You said: {{data}}", websocket)
            await manager.broadcast(f"Client {{client_id}} says: {{data}}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {{client_id}} left the chat")
'''
    
    main_content += '''
# 启动命令: uvicorn main:app --reload
'''
    
    main_path = os.path.join(backend_dir, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)
    files.append(main_path)
    
    # 生成requirements.txt
    requirements_content = '''fastapi
uvicorn
python-dotenv
pydantic
pydantic-settings
'''
    
    if request.database:
        requirements_content += f'{request.database}\n'
    
    if request.auth_required:
        requirements_content += '''python-jose[cryptography]\npasslib[bcrypt]\n'''
    
    # 添加MQTT支持
    if request.mqtt_support:
        requirements_content += '''paho-mqtt\n'''
    
    requirements_path = os.path.join(backend_dir, "requirements.txt")
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write(requirements_content)
    files.append(requirements_path)
    
    # 生成.env文件
    env_content = f'''# 应用配置
APP_NAME={request.app_name}

# 数据库配置
DATABASE_URL={request.database}:///app.db

# 认证配置
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
'''
    
    # 添加MQTT配置
    if request.mqtt_support:
        env_content += '''
# MQTT配置
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=
MQTT_TOPIC_PREFIX=devices/\n'''
    
    env_path = os.path.join(backend_dir, ".env")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    files.append(env_path)
    
    # 添加MQTT客户端代码（如果需要）
    if request.mqtt_support:
        mqtt_content = '''import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# MQTT配置
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "devices/")

# MQTT客户端
class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        if MQTT_USERNAME and MQTT_PASSWORD:
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        # 设置回调函数
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {{rc}}")
        # 订阅设备主题
        self.client.subscribe(f"{{MQTT_TOPIC_PREFIX}}#")
    
    def on_message(self, client, userdata, msg):
        print(f"{{msg.topic}} {{msg.payload}}")
    
    def connect(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    def loop_start(self):
        self.client.loop_start()
    
    def publish(self, topic, message):
        self.client.publish(f"{{MQTT_TOPIC_PREFIX}}{{topic}}", message)
    
    def disconnect(self):
        self.client.disconnect()

# 创建MQTT客户端实例
mqtt_client = MQTTClient()
'''
        
        mqtt_path = os.path.join(backend_dir, "mqtt_client.py")
        with open(mqtt_path, "w", encoding="utf-8") as f:
            f.write(mqtt_content)
        files.append(mqtt_path)
    
    return files

# 生成Flask后端代码
async def generate_flask_backend(request: AppScaffoldRequest, backend_dir: str) -> List[str]:
    """生成Flask后端代码"""
    files = []
    
    # 生成app.py
    app_content = f'''from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 根路径
@app.route("/")
def root():
    return jsonify({
        "message": "Welcome to {request.app_name}",
        "version": "1.0.0",
        "description": "{request.description}"
    })

# 健康检查
@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

# API路由
@app.route("/api/v1/items")
def get_items():
    return jsonify({"items": [], "total": 0})

# 启动命令: flask run --reload
'''
    
    app_path = os.path.join(backend_dir, "app.py")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_content)
    files.append(app_path)
    
    # 生成requirements.txt
    requirements_content = '''flask
flask-cors
python-dotenv
'''
    
    if request.database:
        requirements_content += f'{request.database}\n'
    
    if request.auth_required:
        requirements_content += '''flask-jwt-extended\npasslib[bcrypt]\n'''
    
    requirements_path = os.path.join(backend_dir, "requirements.txt")
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write(requirements_content)
    files.append(requirements_path)
    
    return files

# 生成Django后端代码
async def generate_django_backend(request: AppScaffoldRequest, backend_dir: str) -> List[str]:
    """生成Django后端代码"""
    files = []
    
    # 生成基本的Django项目结构（简化版）
    django_content = f'''# Django项目生成提示
# 要创建完整的Django项目，请运行以下命令：
# django-admin startproject {request.app_name} .
# python manage.py migrate
# python manage.py createsuperuser
# python manage.py runserver

# 项目结构：
# {request.app_name}/
# ├── __init__.py
# ├── settings.py
# ├── urls.py
# └── wsgi.py
'''
    
    django_path = os.path.join(backend_dir, "django_setup.txt")
    with open(django_path, "w", encoding="utf-8") as f:
        f.write(django_content)
    files.append(django_path)
    
    return files

# 生成前端代码
async def generate_frontend_code(request: AppScaffoldRequest, frontend_dir: str) -> dict:
    """生成前端代码"""
    result = {
        "framework": request.frontend_framework,
        "files": []
    }
    
    # 根据前端框架生成不同的代码
    if request.frontend_framework == "vue3":
        result["files"] = generate_vue3_frontend(request, frontend_dir)
    elif request.frontend_framework == "react":
        result["files"] = generate_react_frontend(request, frontend_dir)
    elif request.frontend_framework == "svelte":
        result["files"] = generate_svelte_frontend(request, frontend_dir)
    elif request.frontend_framework == "react-native":
        result["files"] = generate_react_native_frontend(request, frontend_dir)
    elif request.frontend_framework == "flutter":
        result["files"] = generate_flutter_frontend(request, frontend_dir)
    elif request.frontend_framework == "electron-vue3":
        result["files"] = generate_electron_vue3_frontend(request, frontend_dir)
    elif request.frontend_framework == "tauri-react":
        result["files"] = generate_tauri_react_frontend(request, frontend_dir)
    
    return result

# 生成Vue 3前端代码
def generate_vue3_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成Vue 3前端代码"""
    files = []
    
    # 生成package.json
    package_content = {
        "name": f"{request.app_name}-frontend",
        "version": "0.1.0",
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vue-tsc && vite build",
            "preview": "vite preview"
        },
        "dependencies": {
            "vue": "^3.4.0",
            "vue-router": "^4.2.5",
            "pinia": "^2.1.7"
        },
        "devDependencies": {
            "@vitejs/plugin-vue": "^5.0.0",
            "typescript": "^5.2.2",
            "vite": "^5.0.0",
            "vue-tsc": "^1.8.27"
        }
    }
    
    package_path = os.path.join(frontend_dir, "package.json")
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package_content, f, indent=2)
    files.append(package_path)
    
    # 生成index.html
    index_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{request.app_name}</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
'''
    
    index_path = os.path.join(frontend_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_content)
    files.append(index_path)
    
    # 创建src目录
    src_dir = os.path.join(frontend_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    # 生成main.ts
    main_content = '''import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

import './style.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')
'''
    
    main_path = os.path.join(src_dir, "main.ts")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)
    files.append(main_path)
    
    # 生成App.vue
    app_content = '''<template>
  <div>
    <header>
      <h1>Welcome to {{ appName }}</h1>
    </header>
    <main>
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
const appName = import.meta.env.VITE_APP_NAME || 'My App'
</script>

<style scoped>
header {
  background-color: #6366f1;
  color: white;
  padding: 1rem;
  text-align: center;
}

main {
  padding: 1rem;
}
</style>
'''
    
    app_path = os.path.join(src_dir, "App.vue")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_content)
    files.append(app_path)
    
    return files

# 生成React前端代码
def generate_react_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成React前端代码"""
    files = []
    
    # 生成package.json
    package_content = {
        "name": f"{request.app_name}-frontend",
        "version": "0.1.0",
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "tsc && vite build",
            "preview": "vite preview"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-router-dom": "^6.18.0"
        },
        "devDependencies": {
            "@types/react": "^18.2.37",
            "@types/react-dom": "^18.2.15",
            "@vitejs/plugin-react": "^4.2.0",
            "typescript": "^5.2.2",
            "vite": "^5.0.0"
        }
    }
    
    package_path = os.path.join(frontend_dir, "package.json")
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package_content, f, indent=2)
    files.append(package_path)
    
    return files

# 生成Svelte前端代码
def generate_svelte_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成Svelte前端代码"""
    files = []
    
    # 生成package.json
    package_content = {
        "name": f"{request.app_name}-frontend",
        "version": "0.1.0",
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "preview": "vite preview"
        },
        "dependencies": {
            "@sveltejs/kit": "^1.27.4"
        },
        "devDependencies": {
            "@sveltejs/adapter-auto": "^2.1.1",
            "svelte": "^4.2.8",
            "vite": "^5.0.0"
        }
    }
    
    package_path = os.path.join(frontend_dir, "package.json")
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package_content, f, indent=2)
    files.append(package_path)
    
    return files

# 生成React Native前端代码
def generate_react_native_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成React Native前端代码"""
    files = []
    
    # 生成package.json
    package_content = {
        "name": f"{request.app_name}",
        "version": "0.0.1",
        "private": True,
        "type": "module",
        "scripts": {
            "android": "react-native run-android",
            "ios": "react-native run-ios",
            "lint": "eslint . --ext .js,.jsx,.ts,.tsx",
            "start": "react-native start",
            "test": "jest"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-native": "^0.73.0"
        },
        "devDependencies": {
            "@babel/core": "^7.20.0",
            "@babel/preset-env": "^7.20.0",
            "@babel/runtime": "^7.20.0",
            "@react-native/babel-preset": "^0.73.0",
            "@react-native/eslint-config": "^0.73.0",
            "@react-native/metro-config": "^0.73.0",
            "@react-native/typescript-config": "^0.73.0",
            "@types/react": "^18.2.6",
            "@types/react-test-renderer": "^18.0.0",
            "babel-jest": "^29.6.3",
            "eslint": "^8.19.0",
            "jest": "^29.6.3",
            "prettier": "^2.8.8",
            "react-test-renderer": "^18.2.0",
            "typescript": "^5.0.4"
        }
    }
    
    package_path = os.path.join(frontend_dir, "package.json")
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package_content, f, indent=2)
    files.append(package_path)
    
    # 生成App.tsx
    app_content = f'''import React from 'react';
import {{
  SafeAreaView,
  StyleSheet,
  Text,
  View
}} from 'react-native';

function App(): JSX.Element {{
  return (
    <SafeAreaView style={styles.container}>
      <View>
        <Text style={styles.title}>Welcome to {{request.app_name}}</Text>
        <Text style={styles.subtitle}>{{request.description || 'Your React Native App'}}</Text>
      </View>
    </SafeAreaView>
  );
}}

const styles = StyleSheet.create({{
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {{
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  }},
  subtitle: {{
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  }},
}});

export default App;
'''
    
    app_path = os.path.join(frontend_dir, "App.tsx")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_content)
    files.append(app_path)
    
    return files

# 生成Flutter前端代码
def generate_flutter_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成Flutter前端代码"""
    files = []
    
    # 生成pubspec.yaml
    pubspec_content = f'''name: {request.app_name.lower().replace(' ', '_')}
description: {request.description or 'A new Flutter project.'}
publish_to: 'none'

version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.2
  http: ^1.1.0
  provider: ^6.1.1
  shared_preferences: ^2.2.2

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.0

flutter:
  uses-material-design: true
'''
    
    pubspec_path = os.path.join(frontend_dir, "pubspec.yaml")
    with open(pubspec_path, "w", encoding="utf-8") as f:
        f.write(pubspec_content)
    files.append(pubspec_path)
    
    # 创建lib目录
    lib_dir = os.path.join(frontend_dir, "lib")
    os.makedirs(lib_dir, exist_ok=True)
    
    # 生成main.dart
    main_content = f'''import 'package:flutter/material.dart';
import 'package:{request.app_name.lower().replace(' ', '_')}/home_page.dart';

void main() {{
  runApp(const MyApp());
}}

class MyApp extends StatelessWidget {{
  const MyApp({{super.key}});

  @override
  Widget build(BuildContext context) {{
    return MaterialApp(
      title: '{request.app_name}',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }}
}}
'''
    
    main_path = os.path.join(lib_dir, "main.dart")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)
    files.append(main_path)
    
    # 生成home_page.dart
    home_page_content = f'''import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {{
  const HomePage({{super.key}});

  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: AppBar(
        title: const Text('{request.app_name}'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'Welcome to',
              style: TextStyle(fontSize: 18),
            ),
            Text(
              '{request.app_name}',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.blue,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              '{request.description}',
              style: TextStyle(fontSize: 16, color: Colors.grey[600]),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }}
}}
'''
    
    home_page_path = os.path.join(lib_dir, "home_page.dart")
    with open(home_page_path, "w", encoding="utf-8") as f:
        f.write(home_page_content)
    files.append(home_page_path)
    
    return files

# 生成Electron + Vue 3前端代码
def generate_electron_vue3_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成Electron + Vue 3前端代码"""
    files = []
    
    # 生成package.json
    package_content = {
        "name": f"{request.app_name}",
        "version": "0.0.1",
        "private": true,
        "type": "module",
        "main": "dist-electron/main.js",
        "scripts": {
            "dev": "vite",
            "build": "vue-tsc && vite build && electron-builder",
            "preview": "vite preview",
            "electron:build": "electron-builder",
            "electron:serve": "vite preview"
        },
        "dependencies": {
            "vue": "^3.4.0",
            "vue-router": "^4.2.5",
            "pinia": "^2.1.7"
        },
        "devDependencies": {
            "@vitejs/plugin-vue": "^5.0.0",
            "@vue/cli-plugin-babel": "~5.0.0",
            "@vue/cli-plugin-router": "~5.0.0",
            "@vue/cli-service": "~5.0.0",
            "@vue/tsconfig": "^0.5.0",
            "electron": "^28.0.0",
            "electron-builder": "^24.13.3",
            "typescript": "~5.2.0",
            "vite": "^5.0.0",
            "vite-plugin-electron": "^0.28.0",
            "vue-tsc": "^1.8.27"
        }
    }
    
    package_path = os.path.join(frontend_dir, "package.json")
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package_content, f, indent=2)
    files.append(package_path)
    
    return files

# 生成Tauri + React前端代码
def generate_tauri_react_frontend(request: AppScaffoldRequest, frontend_dir: str) -> List[str]:
    """生成Tauri + React前端代码"""
    files = []
    
    # 生成package.json
    package_content = {
        "name": f"{request.app_name}",
        "private": true,
        "version": "0.0.1",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "tsc && vite build",
            "preview": "vite preview",
            "tauri": "tauri"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0"
        },
        "devDependencies": {
            "@tauri-apps/cli": "^1.5.0",
            "@types/react": "^18.2.37",
            "@types/react-dom": "^18.2.15",
            "@vitejs/plugin-react": "^4.2.0",
            "typescript": "^5.2.2",
            "vite": "^5.0.0"
        }
    }
    
    package_path = os.path.join(frontend_dir, "package.json")
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package_content, f, indent=2)
    files.append(package_path)
    
    # 生成src/App.tsx
    src_dir = os.path.join(frontend_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    app_content = f'''import React from 'react';
import './App.css';

function App() {{
  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to {{request.app_name}}</h1>
        <p>{{request.description || 'Your Tauri + React App'}}</p>
      </header>
    </div>
  );
}}

export default App;
'''
    
    app_path = os.path.join(src_dir, "App.tsx")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_content)
    files.append(app_path)
    
    return files

# 生成根目录文件
def generate_root_files(request: AppScaffoldRequest, root_dir: str) -> None:
    """生成项目根目录文件"""
    # 生成.gitignore
    gitignore_content = '''# Dependencies
node_modules/
*.pyc
__pycache__/
*.swp
*.swo

# Environment files
.env
.env.local
.env.*.local

# Build outputs
dist/
build/
generated_apps/

# IDE
.vscode/
.idea/
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

# OS files
.DS_Store
Thumbs.db

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

# Testing
coverage/
*.lcov

# Temporary files
*.tmp
*.temp
.cache/
'''
    
    gitignore_path = os.path.join(root_dir, ".gitignore")
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(gitignore_content)

# 生成README
def generate_readme(request: AppScaffoldRequest, root_dir: str) -> None:
    """生成README文件"""
    readme_content = f'''# {request.app_name}

{request.description}

## 项目信息

- **模板**: {request.template}
- **应用类型**: {request.app_type}
- **后端框架**: {request.backend_framework}
- **前端框架**: {request.frontend_framework}
- **数据库**: {request.database}
- **认证**: {'已启用' if request.auth_required else '未启用'}
- **Docker支持**: {'已启用' if request.docker_support else '未启用'}

## 快速开始

### 后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

## 项目结构

```
.
├── backend/          # 后端代码
├── frontend/         # 前端代码
├── docs/             # 文档
├── tests/            # 测试代码
├── .gitignore        # Git忽略文件
└── README.md         # 项目说明
```

## 功能特性

{chr(10).join([f'- {feature}' for feature in request.features])}

## 开发命令

### 后端

| 命令 | 描述 |
|------|------|
| `pip install -r requirements.txt` | 安装依赖 |
| `python main.py` | 启动开发服务器 |
| `python -m pytest` | 运行测试 |

### 前端

| 命令 | 描述 |
|------|------|
| `npm install` | 安装依赖 |
| `npm run dev` | 启动开发服务器 |
| `npm run build` | 构建生产版本 |
| `npm run preview` | 预览构建结果 |
'''
    
    readme_path = os.path.join(root_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

# 初始化Git
def initialize_git(root_dir: str) -> None:
    """初始化Git仓库"""
    try:
        # 运行git init命令
        subprocess.run(["git", "init"], cwd=root_dir, capture_output=True, text=True)
        subprocess.run(["git", "add", "."], cwd=root_dir, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=root_dir, capture_output=True, text=True)
    except Exception as e:
        print(f"Git初始化失败: {str(e)}")

# 生成Docker配置
def generate_docker_config(request: AppScaffoldRequest, root_dir: str) -> None:
    """生成Docker配置文件"""
    # 生成docker-compose.yml
    # 构建db服务配置
    db_service = ""
    if request.database == "postgresql":
        db_service = f'''  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB={request.app_name}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

'''
    
    db_depends = "- db" if request.database not in ["sqlite", None] else ""
    db_volumes = "postgres_data:" if request.database == "postgresql" else ""
    
    docker_compose_content = f'''version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env
    depends_on:
      {db_depends}

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    depends_on:
      - backend

{db_service}volumes:
  {db_volumes}
'''
    
    docker_compose_path = os.path.join(root_dir, "docker-compose.yml")
    with open(docker_compose_path, "w", encoding="utf-8") as f:
        f.write(docker_compose_content)
    
    # 生成后端Dockerfile
    backend_dockerfile = '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    backend_dockerfile_path = os.path.join(root_dir, "backend", "Dockerfile")
    with open(backend_dockerfile_path, "w", encoding="utf-8") as f:
        f.write(backend_dockerfile)
    
    # 生成前端Dockerfile
    frontend_dockerfile = '''FROM node:18-slim AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:stable-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
'''
    
    frontend_dockerfile_path = os.path.join(root_dir, "frontend", "Dockerfile")
    with open(frontend_dockerfile_path, "w", encoding="utf-8") as f:
        f.write(frontend_dockerfile)

# 生成模拟UI代码（API调用失败时使用）
def generate_mock_ui_code(prompt: str) -> str:
    """生成模拟的UI代码"""
    prompt_preview = prompt[:100] + "..."
    
    mock_code = '''<!-- 生成的UI代码 -->
<template>
  <div class="ui-component">
    <h3>根据需求生成的组件</h3>
    <p>提示词：''' + prompt_preview + '''</p>
    <!-- 实际项目中这里会是真实生成的UI代码 -->
    <button class="primary-button">主按钮</button>
    <button class="secondary-button" disabled>禁用按钮</button>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue';

// 响应式数据
const isLoading = ref(false);
const formData = reactive({
  name: '',
  value: ''
});

// 示例函数
const handleSubmit = () => {
  isLoading.value = true;
  // 实际项目中这里会是API调用
  setTimeout(() => {
    isLoading.value = false;
    console.log('表单提交:', formData);
  }, 1000);
};
</script>

<style scoped>
.ui-component {
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.primary-button {
  background-color: #6C63FF;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  margin-right: 10px;
  cursor: pointer;
}

.secondary-button {
  background-color: #e0e0e0;
  color: #666;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: not-allowed;
}
</style>'''
    
    return mock_code

# 代码质量检查函数
def check_code_quality(code: str) -> dict:
    """检查生成的代码质量"""
    issues = {
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    # 检查代码质量规则
    lines = code.split('\n')
    
    # 1. 基础语法检查
    # 1.1 检查script标签是否加了lang="ts"
    if '<script' in code and 'lang="ts"' not in code and 'lang=ts' not in code:
        issues["errors"].append("script标签必须添加lang=\"ts\"属性")
    
    # 1.2 检查是否使用了scoped样式
    if '<style' in code and 'scoped' not in code:
        issues["warnings"].append("样式标签建议添加scoped属性")
    
    # 2. ESLint相关规则
    # 2.1 检查是否有console.log
    if 'console.log' in code:
        issues["warnings"].append("应避免使用console.log调试代码")
    
    # 2.2 检查是否有debugger
    if 'debugger' in code:
        issues["errors"].append("代码中不应包含debugger语句")
    
    # 2.3 检查是否有未使用的变量（简单检查）
    unused_vars = []
    for line in lines:
        if ('const ' in line or 'let ' in line or 'var ' in line) and '=' in line:
            var_name = line.split('=')[0].strip().split(' ')[-1]
            # 跳过函数定义和箭头函数
            if var_name and '=>' not in var_name and '(' not in var_name and 'function' not in line:
                # 检查变量是否在其他地方使用
                if var_name not in code.replace(line, ''):
                    unused_vars.append(var_name)
    if unused_vars:
        issues["suggestions"].append(f"可能存在未使用的变量：{', '.join(unused_vars)}")
    
    # 3. Vue最佳实践
    # 3.1 检查v-for是否有key
    if 'v-for' in code and ':key' not in code and 'v-bind:key' not in code:
        issues["errors"].append("v-for必须添加key属性")
    
    # 3.2 检查是否使用了setup语法糖
    if '<script' in code and 'setup' not in code:
        issues["suggestions"].append("建议使用setup语法糖")
    
    # 3.3 检查是否使用了ref/reactive而不是直接赋值
    if 'this.' in code and 'setup' in code:
        issues["errors"].append("setup语法糖中不应使用this关键字")
    
    # 3.4 检查是否使用了v-if和v-for在同一元素上
    for line in lines:
        if 'v-if' in line and 'v-for' in line:
            issues["warnings"].append("避免在同一元素上同时使用v-if和v-for")
            break
    
    # 4. 类型检查相关
    # 4.1 检查是否有明确的类型注解
    if '<script' in code and 'ts' in code:
        has_type_annotation = False
        for line in lines:
            if (':' in line and '=' in line and ('const ' in line or 'let ' in line or 'var ' in line)) or 'interface' in line or 'type' in line:
                has_type_annotation = True
                break
        if not has_type_annotation:
            issues["suggestions"].append("建议为变量添加明确的类型注解")
    
    # 5. 性能相关
    # 5.1 检查是否使用了large-list等性能优化组件
    if 'v-for' in code and 'large-list' not in code and 'virtual-list' not in code:
        issues["suggestions"].append("对于长列表，建议使用虚拟滚动组件")
    
    # 5.2 检查是否使用了memo或shallowRef
    if 'computed(' in code or 'watch(' in code:
        issues["suggestions"].append("对于复杂计算，建议使用memo优化性能")
    
    # 6. 代码风格
    # 6.1 检查是否有多余的空行
    empty_lines = 0
    for line in lines:
        if line.strip() == '':
            empty_lines += 1
        else:
            if empty_lines > 2:
                issues["suggestions"].append("代码中不应有连续超过2个的空行")
                break
            empty_lines = 0
    
    # 6.2 检查缩进（简单检查，假设使用2或4个空格）
    indent_issues = False
    for line in lines:
        if line.startswith(' ') and len(line) - len(line.lstrip()) not in [2, 4, 0]:
            indent_issues = True
            break
    if indent_issues:
        issues["suggestions"].append("建议使用2或4个空格进行缩进，保持一致")
    
    # 7. 安全性检查
    # 7.1 检查是否直接使用innerHTML
    if 'innerHTML' in code:
        issues["warnings"].append("避免使用innerHTML，存在XSS风险")
    
    # 7.2 检查是否使用了eval
    if 'eval(' in code:
        issues["errors"].append("代码中不应使用eval函数，存在安全风险")
    
    return issues

# 定义UI生成接口
@app.post("/api/generate-ui", response_model=UIResponse)
async def generate_ui(request: UIRequest):
    """
    生成符合项目规范的UI代码
    
    - **requirement**: 详细的UI需求描述
    - **additional_context**: 额外的上下文信息（可选）
    - **ui_framework**: UI框架选择（element-plus, ant-design-vue, vuetify）
    """
    try:
        # 获取框架特定规则
        framework_specific_rules = FRAMEWORK_SPECIFIC_RULES.get(request.ui_framework, "")
        
        # 构建完整的提示词
        full_prompt = PROJECT_PROMPT_TEMPLATE.format(
            ui_framework=request.ui_framework,
            framework_specific_rules=framework_specific_rules,
            requirement=request.requirement,
            additional_context=request.additional_context
        )
        
        # 调用AI模型生成代码
        generated_code = await call_ai_model(full_prompt)
        
        # 检查代码质量
        quality_issues = check_code_quality(generated_code)
        
        # 保存到历史记录
        history_item = HistoryItem(
            requirement=request.requirement,
            generated_code=generated_code,
            ui_framework=request.ui_framework
        )
        history.append(history_item)
        
        # 限制历史记录数量（最多保存100条）
        if len(history) > 100:
            history.pop(0)
        
        # 返回生成的代码
        return {
            "code": 200,
            "data": {
                "prompt": full_prompt,
                "ui_code": generated_code,
                "requirement": request.requirement,
                "ui_framework": request.ui_framework,
                "quality_issues": quality_issues,
                "timestamp": history_item.timestamp
            },
            "msg": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成UI代码失败：{str(e)}")

# 定义获取历史记录接口
@app.get("/api/history", response_model=HistoryResponse)
def get_history(limit: int = 20, offset: int = 0, search: str = "", framework: str = ""):
    """
    获取生成记录历史
    
    - **limit**: 返回的记录数量（默认20）
    - **offset**: 偏移量（默认0）
    - **search**: 搜索关键词（可选）
    - **framework**: 按框架过滤（可选）
    """
    try:
        # 按时间倒序排序
        sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
        
        # 搜索和过滤
        filtered_history = []
        for item in sorted_history:
            # 按关键词搜索
            if search and search not in item.requirement:
                continue
            # 按框架过滤
            if framework and item.ui_framework != framework:
                continue
            filtered_history.append(item)
        
        # 分页
        paginated_history = filtered_history[offset:offset + limit]
        
        # 转换为字典格式
        history_dicts = []
        for i, item in enumerate(paginated_history):
            history_dicts.append({
                "id": i + 1 + offset,
                "requirement": item.requirement,
                "ui_code": item.generated_code,
                "ui_framework": item.ui_framework,
                "timestamp": item.timestamp
            })
        
        return {
            "code": 200,
            "data": history_dicts,
            "msg": "success",
            "total": len(filtered_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败：{str(e)}")

# 定义删除历史记录接口
@app.delete("/api/history/{history_id}", response_model=UIResponse)
def delete_history(history_id: int):
    """
    删除指定的历史记录
    
    - **history_id**: 要删除的历史记录ID
    """
    try:
        global history
        # 按时间倒序排序
        sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
        
        if history_id < 1 or history_id > len(sorted_history):
            raise HTTPException(status_code=404, detail="历史记录不存在")
        
        # 删除对应的历史记录
        del sorted_history[history_id - 1]
        
        # 更新历史列表
        history = sorted_history
        
        return {
            "code": 200,
            "data": {
                "id": history_id
            },
            "msg": "历史记录删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除历史记录失败：{str(e)}")

# 定义清空历史记录接口
@app.delete("/api/history", response_model=UIResponse)
def clear_history():
    """
    清空所有历史记录
    """
    try:
        global history
        history = []
        
        return {
            "code": 200,
            "data": {},
            "msg": "历史记录清空成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空历史记录失败：{str(e)}")

# 定义代码质量检查接口
@app.post("/api/check-code-quality", response_model=CodeQualityResponse)
def check_code_endpoint(code: str):
    """
    检查代码质量
    
    - **code**: 要检查的代码内容
    """
    try:
        quality_issues = check_code_quality(code)
        return {
            "code": 200,
            "data": {
                "issues": quality_issues,
                "code_length": len(code),
                "line_count": len(code.split('\n'))
            },
            "msg": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码质量检查失败：{str(e)}")

# 定义导出代码接口
@app.post("/api/export-code", response_model=ExportResponse)
def export_code(request: ExportRequest):
    """
    导出生成的代码为文件
    
    - **code**: 要导出的代码内容
    - **filename**: 导出的文件名（默认：generated-component.vue）
    """
    try:
        # 创建exports目录（如果不存在）
        exports_dir = "exports"
        if not os.path.exists(exports_dir):
            os.makedirs(exports_dir)
        
        # 构建文件路径
        file_path = os.path.join(exports_dir, request.filename)
        
        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(request.code)
        
        return {
            "code": 200,
            "data": {
                "file_path": file_path,
                "filename": request.filename,
                "file_size": len(request.code),
                "export_time": time.time()
            },
            "msg": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出代码失败：{str(e)}")

# 定义获取项目规范接口
@app.get("/api/project-specs", response_model=UIResponse)
def get_project_specs():
    """
    获取项目的UI设计规范
    """
    return {
        "code": 200,
        "data": {
            "specs": PROJECT_PROMPT_TEMPLATE,
            "ui_frameworks": UI_FRAMEWORKS,
            "theme_color": "#6C63FF"
        },
        "msg": "success"
    }

# 定义获取支持的UI框架接口
@app.get("/api/ui-frameworks", response_model=UIResponse)
def get_ui_frameworks():
    """
    获取支持的UI框架列表
    """
    return {
        "code": 200,
        "data": {
            "frameworks": UI_FRAMEWORKS,
            "total": len(UI_FRAMEWORKS)
        },
        "msg": "success"
    }

# 根路径，返回API信息
@app.get("/")
def root():
    return {
        "message": "UI代码生成API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "version": "1.0.0",
        "endpoints": [
            {"method": "POST", "path": "/api/generate-ui", "description": "生成UI代码"},
            {"method": "GET", "path": "/api/project-specs", "description": "获取项目规范"},
            {"method": "GET", "path": "/api/history", "description": "获取历史记录"},
            {"method": "DELETE", "path": "/api/history/{history_id}", "description": "删除指定历史记录"},
            {"method": "DELETE", "path": "/api/history", "description": "清空所有历史记录"},
            {"method": "POST", "path": "/api/check-code-quality", "description": "检查代码质量"},
            {"method": "POST", "path": "/api/export-code", "description": "导出代码为文件"},
            {"method": "GET", "path": "/api/ui-frameworks", "description": "获取支持的UI框架"}
        ]
    }

# 运行FastAPI应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_ui_generator:app", host="0.0.0.0", port=8000, reload=True)
