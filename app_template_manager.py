import os
import json
import uuid
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel

# 支持的应用类型
APP_TYPES = ["web", "mobile", "desktop", "api", "crud", "admin", "iot"]

# 支持的后端框架
BACKEND_FRAMEWORKS = ["fastapi", "flask", "django"]

# 支持的前端框架
FRONTEND_FRAMEWORKS = ["vue3", "react", "svelte", "react-native", "flutter", "electron-vue3", "tauri-react"]

# 应用模板配置
def get_app_templates() -> Dict[str, Dict[str, Any]]:
    """获取所有应用模板配置"""
    return {
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

# 应用模板请求模型
class AppTemplateRequest(BaseModel):
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

# 应用模板响应模型
class AppTemplateResponse(BaseModel):
    code: int
    data: dict
    msg: str

class AppTemplateManager:
    """应用模板管理器"""
    
    def __init__(self):
        self.templates = get_app_templates()
        self.generated_apps_dir = "generated_apps"
        os.makedirs(self.generated_apps_dir, exist_ok=True)
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模板"""
        return self.templates
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取指定模板"""
        return self.templates.get(template_id)
    
    def validate_template(self, template_id: str) -> bool:
        """验证模板是否存在"""
        return template_id in self.templates
    
    def generate_app(self, request: AppTemplateRequest) -> Dict[str, Any]:
        """生成应用"""
        # 验证模板
        if not self.validate_template(request.template):
            raise ValueError(f"无效的模板ID: {request.template}")
        
        # 创建项目目录
        project_id = str(uuid.uuid4())[:8]
        project_dir = os.path.join(self.generated_apps_dir, f"{request.app_name}-{project_id}")
        
        # 创建目录结构
        os.makedirs(project_dir, exist_ok=True)
        
        # 获取模板配置
        template_config = self.templates[request.template]
        
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
        backend_result = self.generate_backend_code(request, project_structure["backend"])
        
        # 生成前端代码
        frontend_result = {}
        if template_config["frontend"]:
            frontend_result = self.generate_frontend_code(request, project_structure["frontend"])
        
        # 生成根目录文件
        self.generate_root_files(request, project_structure["root"])
        
        # 生成README
        self.generate_readme(request, project_structure["root"])
        
        # 初始化Git
        if request.git_initialized:
            self.initialize_git(project_structure["root"])
        
        # 生成Docker配置
        if request.docker_support:
            self.generate_docker_config(request, project_structure["root"])
        
        return {
            "project_id": project_id,
            "project_path": project_dir,
            "app_name": request.app_name,
            "template": request.template,
            "backend": backend_result,
            "frontend": frontend_result,
            "features": request.features,
            "generated_at": os.path.getctime(project_dir)
        }
    
    def generate_backend_code(self, request: AppTemplateRequest, backend_dir: str) -> Dict[str, Any]:
        """生成后端代码"""
        result = {
            "framework": request.backend_framework,
            "files": []
        }
        
        # 根据后端框架生成不同的代码
        if request.backend_framework == "fastapi":
            result["files"] = self.generate_fastapi_backend(request, backend_dir)
        elif request.backend_framework == "flask":
            result["files"] = self.generate_flask_backend(request, backend_dir)
        elif request.backend_framework == "django":
            result["files"] = self.generate_django_backend(request, backend_dir)
        
        return result
    
    def generate_fastapi_backend(self, request: AppTemplateRequest, backend_dir: str) -> List[str]:
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
    return {{
        "message": "Welcome to {request.app_name}",
        "version": "1.0.0",
        "description": "{request.description}"
    }}

# 健康检查
@app.get("/health")
def health_check():
    return {{"status": "ok"}}

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
    
    def generate_flask_backend(self, request: AppTemplateRequest, backend_dir: str) -> List[str]:
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
    return jsonify({{
        "message": "Welcome to {request.app_name}",
        "version": "1.0.0",
        "description": "{request.description}"
    }})

# 健康检查
@app.route("/health")
def health_check():
    return jsonify({{"status": "ok"}})

# API路由
@app.route("/api/v1/items")
def get_items():
    return jsonify({{"items": [], "total": 0}})

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
    
    def generate_django_backend(self, request: AppTemplateRequest, backend_dir: str) -> List[str]:
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
    
    def generate_frontend_code(self, request: AppTemplateRequest, frontend_dir: str) -> Dict[str, Any]:
        """生成前端代码"""
        result = {
            "framework": request.frontend_framework,
            "files": []
        }
        
        # 根据前端框架生成不同的代码
        if request.frontend_framework == "vue3":
            result["files"] = self.generate_vue3_frontend(request, frontend_dir)
        elif request.frontend_framework == "react":
            result["files"] = self.generate_react_frontend(request, frontend_dir)
        elif request.frontend_framework == "svelte":
            result["files"] = self.generate_svelte_frontend(request, frontend_dir)
        elif request.frontend_framework == "react-native":
            result["files"] = self.generate_react_native_frontend(request, frontend_dir)
        elif request.frontend_framework == "flutter":
            result["files"] = self.generate_flutter_frontend(request, frontend_dir)
        elif request.frontend_framework == "electron-vue3":
            result["files"] = self.generate_electron_vue3_frontend(request, frontend_dir)
        elif request.frontend_framework == "tauri-react":
            result["files"] = self.generate_tauri_react_frontend(request, frontend_dir)
        
        return result
    
    def generate_vue3_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
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
    
    def generate_react_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
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
    
    def generate_svelte_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
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
    
    def generate_react_native_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
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
  container: {{
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  }},
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
    
    def generate_flutter_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
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
        title: const Text('{{request.app_name}}'),
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
              '{{request.app_name}}',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.blue,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              '{{request.description}}',
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
    
    def generate_electron_vue3_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
        """生成Electron + Vue 3前端代码"""
        files = []
        
        # 生成package.json
        package_content = {
            "name": f"{request.app_name}",
            "version": "0.0.1",
            "private": True,
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
    
    def generate_tauri_react_frontend(self, request: AppTemplateRequest, frontend_dir: str) -> List[str]:
        """生成Tauri + React前端代码"""
        files = []
        
        # 生成package.json
        package_content = {
            "name": f"{request.app_name}",
            "private": True,
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
    
    def generate_root_files(self, request: AppTemplateRequest, root_dir: str) -> None:
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
    
    def generate_readme(self, request: AppTemplateRequest, root_dir: str) -> None:
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
- **MQTT支持**: {'已启用' if request.mqtt_support else '未启用'}
- **WebSocket支持**: {'已启用' if request.websocket_support else '未启用'}
- **设备管理**: {'已启用' if request.device_management else '未启用'}

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
    
    def initialize_git(self, root_dir: str) -> None:
        """初始化Git仓库"""
        try:
            # 运行git init命令
            subprocess.run(["git", "init"], cwd=root_dir, capture_output=True, text=True)
            subprocess.run(["git", "add", "."], cwd=root_dir, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=root_dir, capture_output=True, text=True)
        except Exception as e:
            print(f"Git初始化失败: {str(e)}")
    
    def generate_docker_config(self, request: AppTemplateRequest, root_dir: str) -> None:
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

# 初始化应用模板管理器
template_manager = AppTemplateManager()

if __name__ == "__main__":
    # 示例用法
    request = AppTemplateRequest(
        app_name="test-app",
        description="测试应用",
        app_type="web",
        template="fastapi-vue3",
        backend_framework="fastapi",
        frontend_framework="vue3",
        features=["测试功能1", "测试功能2"],
        database="sqlite",
        auth_required=True,
        docker_support=True,
        git_initialized=True,
        mqtt_support=True,
        websocket_support=True,
        device_management=True
    )
    
    result = template_manager.generate_app(request)
    print(f"应用生成成功: {result['project_path']}")
