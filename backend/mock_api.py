#!/usr/bin/env python3
"""
完整的模拟API服务
用于支持前端页面正常加载
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI项目API服务",
    description="基于JAX+Flax的最先进AI项目API服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟数据
mock_data = {
    "system": {
        "metrics": {
            "cpu_usage": 35.2,
            "memory_usage": 45.8,
            "disk_usage": 62.5,
            "active_connections": 128,
            "status": "healthy"
        },
        "health": {
            "status": "healthy",
            "timestamp": "2026-01-13T08:00:00Z"
        }
    },
    "edge": {
        "devices": [
            {
                "id": "device-1",
                "name": "边缘设备1",
                "status": "online",
                "cpu_usage": 25.5,
                "memory_usage": 38.2,
                "last_seen": "2026-01-13T07:59:00Z"
            },
            {
                "id": "device-2",
                "name": "边缘设备2",
                "status": "online",
                "cpu_usage": 42.3,
                "memory_usage": 55.7,
                "last_seen": "2026-01-13T07:58:00Z"
            }
        ],
        "status": {
            "total_devices": 2,
            "online_devices": 2,
            "offline_devices": 0
        }
    },
    "blockchain": {
        "status": {
            "is_running": True,
            "block_height": 12345,
            "peer_count": 8,
            "last_block_time": "2026-01-13T07:57:00Z"
        }
    },
    "models": {
        "list": [
            {
                "id": "model-1",
                "name": "基础模型",
                "version": "1.0.0",
                "status": "active",
                "created_at": "2026-01-10T10:00:00Z"
            },
            {
                "id": "model-2",
                "name": "高级模型",
                "version": "2.0.0",
                "status": "active",
                "created_at": "2026-01-12T14:30:00Z"
            }
        ]
    },
    "ai_control": {
        "devices": [
            {
                "id": "control-device-1",
                "name": "AI控制设备1",
                "status": "online",
                "type": "camera",
                "location": "仓库1"
            },
            {
                "id": "control-device-2",
                "name": "AI控制设备2",
                "status": "online",
                "type": "sensor",
                "location": "仓库2"
            }
        ]
    },
    "decision": {
        "tasks": [
            {
                "id": "decision-1",
                "name": "决策任务1",
                "status": "completed",
                "created_at": "2026-01-13T07:00:00Z",
                "completed_at": "2026-01-13T07:30:00Z"
            },
            {
                "id": "decision-2",
                "name": "决策任务2",
                "status": "running",
                "created_at": "2026-01-13T07:45:00Z"
            }
        ]
    },
    "monitoring": {
        "alerts": [
            {
                "id": "alert-1",
                "level": "info",
                "message": "系统正常运行",
                "timestamp": "2026-01-13T07:55:00Z"
            }
        ]
    },
    "camera": {
        "status": {
            "is_online": True,
            "recording": False,
            "stream_url": "http://example.com/stream"
        }
    },
    "performance": {
        "metrics": {
            "inference_time": 0.123,
            "training_speed": 10.5,
            "accuracy": 0.987
        }
    },
    "community": {
        "posts": [
            {
                "id": "post-1",
                "title": "欢迎加入社区",
                "content": "这是一个AI项目社区",
                "author": "admin",
                "created_at": "2026-01-10T09:00:00Z"
            }
        ]
    },
    "auto_dev": {
        "scenes": [
            {
                "id": "scene-1",
                "name": "默认场景",
                "description": "默认自动化场景",
                "status": "active"
            }
        ],
        "automation_rules": [
            {
                "id": "rule-1",
                "name": "自动化规则1",
                "status": "active"
            }
        ]
    },
    "ai_assistant": {
        "schedule": {
            "tasks": [],
            "status": "idle"
        },
        "fine_tune": {
            "tasks": [],
            "models": []
        }
    },
    "federated": {
        "status": {
            "is_running": False,
            "clients": 0
        }
    }
}

# 根路径
@app.get("/")
async def root():
    return {
        "message": "AI项目API服务",
        "version": "1.0.0",
        "status": "ok"
    }

# 健康检查
@app.get("/api/health")
async def health_check():
    return mock_data["system"]["health"]

# 系统指标
@app.get("/api/system/metrics")
async def get_system_metrics():
    return mock_data["system"]["metrics"]

# 边缘设备列表
@app.get("/api/edge/devices")
async def get_edge_devices():
    return mock_data["edge"]["devices"]

# 边缘状态
@app.get("/api/edge/status")
async def get_edge_status():
    return mock_data["edge"]["status"]

# 区块链状态
@app.get("/api/blockchain/status")
async def get_blockchain_status():
    return mock_data["blockchain"]["status"]

# 模型列表
@app.get("/api/models/list")
async def get_models_list():
    return mock_data["models"]["list"]

# AI控制设备
@app.get("/api/ai-control/devices")
async def get_ai_control_devices():
    return mock_data["ai_control"]["devices"]

# 决策任务
@app.get("/api/decision/tasks")
async def get_decision_tasks():
    return mock_data["decision"]["tasks"]

# 监控警报
@app.get("/api/monitoring/alerts")
async def get_monitoring_alerts():
    return mock_data["monitoring"]["alerts"]

# 摄像头状态
@app.get("/api/camera/status")
async def get_camera_status():
    return mock_data["camera"]["status"]

# 性能指标
@app.get("/api/performance/metrics")
async def get_performance_metrics():
    return mock_data["performance"]["metrics"]

# 社区帖子
@app.get("/api/community/posts")
async def get_community_posts():
    return mock_data["community"]["posts"]

# 自动化场景
@app.get("/api/auto-dev/scenes")
async def get_auto_dev_scenes():
    return mock_data["auto_dev"]["scenes"]

# 自动化规则
@app.get("/api/auto-dev/automation-rules")
async def get_auto_dev_rules():
    return mock_data["auto_dev"]["automation_rules"]

# AI助手调度任务
@app.get("/api/ai-assistant/schedule/tasks")
async def get_ai_assistant_tasks():
    return mock_data["ai_assistant"]["schedule"]["tasks"]

# AI助手调度状态
@app.get("/api/ai-assistant/schedule/status")
async def get_ai_assistant_status():
    return mock_data["ai_assistant"]["schedule"]["status"]

# AI助手微调任务
@app.get("/api/ai-assistant/fine-tune/tasks")
async def get_ai_assistant_fine_tune_tasks():
    return mock_data["ai_assistant"]["fine_tune"]["tasks"]

# AI助手微调模型
@app.get("/api/ai-assistant/fine-tune/models")
async def get_ai_assistant_fine_tune_models():
    return mock_data["ai_assistant"]["fine_tune"]["models"]

# 联邦学习状态
@app.get("/api/federated/status")
async def get_federated_status():
    return mock_data["federated"]["status"]

# 自适应路由
@app.get("/api/adaptive/status")
async def get_adaptive_status():
    return {"status": "ok"}

# 农业NLP
@app.get("/api/agriculture-nlp/analyze")
async def get_agriculture_nlp_analyze():
    return {"result": "分析结果"}

# AI核心主控状态查询
@app.get("/api/ai-control/master-control/status")
async def get_master_control_status():
    return {"master_control_active": False}

# AI核心主控激活/关闭
@app.post("/api/ai-control/master-control/activate")
async def activate_master_control(is_active: bool):
    return {"success": True, "message": f"主控已{'激活' if is_active else '关闭'}"}

# JEPA-DT-MPC状态查询
@app.get("/api/ai-control/jepa-dtmpc/status")
async def get_jepa_dtmpc_status():
    return {
        "success": True,
        "controller_status": {
            "jepa_enabled": False
        },
        "message": "JEPA-DT-MPC状态查询成功"
    }

# JEPA-DT-MPC激活/关闭
@app.post("/api/ai-control/jepa-dtmpc/activate")
async def activate_jepa_dtmpc(params: dict):
    return {
        "success": True,
        "message": f"JEPA-DT-MPC已{'激活' if params.get('controller_params', {}).get('control_switch', False) else '关闭'}",
        "controller_status": {
            "jepa_enabled": params.get('controller_params', {}).get('control_switch', False)
        }
    }

# 设备列表
@app.get("/api/devices")
async def get_devices():
    return {
        "success": True,
        "data": [
            {
                "id": 1,
                "name": "边缘计算设备1",
                "type": "edge",
                "status": "online",
                "connected": True,
                "last_seen": "2026-01-13T08:00:00Z"
            },
            {
                "id": 2,
                "name": "AI摄像头1",
                "type": "camera",
                "status": "online",
                "connected": True,
                "last_seen": "2026-01-13T08:00:00Z"
            },
            {
                "id": 3,
                "name": "传感器节点1",
                "type": "sensor",
                "status": "offline",
                "connected": False,
                "last_seen": "2026-01-13T07:55:00Z"
            }
        ],
        "message": "设备列表获取成功"
    }

# 设备扫描
@app.post("/api/devices/scan")
async def scan_devices():
    return {
        "success": True,
        "data": [
            {
                "id": 1,
                "name": "边缘计算设备1",
                "type": "edge",
                "status": "online",
                "connected": True,
                "last_seen": "2026-01-13T08:01:00Z"
            },
            {
                "id": 2,
                "name": "AI摄像头1",
                "type": "camera",
                "status": "online",
                "connected": True,
                "last_seen": "2026-01-13T08:01:00Z"
            },
            {
                "id": 3,
                "name": "传感器节点1",
                "type": "sensor",
                "status": "offline",
                "connected": False,
                "last_seen": "2026-01-13T07:55:00Z"
            }
        ],
        "message": "设备扫描成功"
    }

# 摄像头状态
@app.get("/api/camera/status")
async def get_camera_status():
    return {
        "success": True,
        "data": {
            "is_open": False,
            "camera_index": 0
        },
        "message": "摄像头状态查询成功"
    }

# 摄像头列表
@app.get("/api/camera/list")
async def get_camera_list():
    return {
        "success": True,
        "data": {
            "cameras": [
                {
                    "index": 0,
                    "name": "主摄像头",
                    "is_available": True,
                    "backend": "USB"
                },
                {
                    "index": 1,
                    "name": "备用摄像头",
                    "is_available": False,
                    "backend": "IP"
                }
            ]
        },
        "message": "摄像头列表获取成功"
    }

# 摄像头打开
@app.post("/api/camera/open/{camera_idx}")
async def open_camera(camera_idx: int):
    return {
        "success": True,
        "message": f"摄像头 {camera_idx} 已打开"
    }

# 摄像头关闭
@app.post("/api/camera/close/{camera_idx}")
async def close_camera(camera_idx: int):
    return {
        "success": True,
        "message": f"摄像头 {camera_idx} 已关闭"
    }

# 跟踪状态
@app.get("/api/camera/tracking/status")
async def get_tracking_status():
    return {
        "success": True,
        "data": {
            "tracking_enabled": False
        },
        "message": "跟踪状态查询成功"
    }

# 识别状态
@app.get("/api/camera/recognition/status")
async def get_recognition_status():
    return {
        "success": True,
        "data": {
            "recognizing_enabled": False
        },
        "message": "识别状态查询成功"
    }

# 智能体相关API

# 获取智能体列表
@app.get("/api/agents")
async def get_agents():
    return {
        "success": True,
        "data": [
            {
                "agent_id": "agent_001",
                "name": "代码智能体001",
                "agent_type": "code",
                "endpoint": "http://localhost:8003/agents/code-001",
                "status": "available",
                "capabilities": ["代码分析", "代码生成", "代码重构"],
                "last_heartbeat": int(time.time()),
                "current_task_id": None
            },
            {
                "agent_id": "agent_002",
                "name": "分析智能体002",
                "agent_type": "analysis",
                "endpoint": "http://localhost:8003/agents/analysis-002",
                "status": "busy",
                "capabilities": ["数据分析", "异常检测", "预测分析"],
                "last_heartbeat": int(time.time()),
                "current_task_id": "task_789012"
            },
            {
                "agent_id": "agent_003",
                "name": "搜索智能体003",
                "agent_type": "search",
                "endpoint": "http://localhost:8003/agents/search-003",
                "status": "error",
                "capabilities": ["信息检索", "文档搜索", "知识图谱构建"],
                "last_heartbeat": int(time.time()) - 600,
                "current_task_id": None
            }
        ],
        "message": "智能体列表获取成功"
    }

# 注册智能体
@app.post("/api/agents/register")
async def register_agent(agent_info: dict):
    return {
        "success": True,
        "data": {
            "agent_id": f"agent_{str(int(time.time()))[-6:]}",
            "name": agent_info.get("name", "新智能体"),
            "agent_type": agent_info.get("agent_type", "other"),
            "endpoint": agent_info.get("endpoint", ""),
            "status": agent_info.get("status", "available"),
            "capabilities": agent_info.get("capabilities", []),
            "last_heartbeat": int(time.time()),
            "current_task_id": None
        },
        "message": "智能体注册成功"
    }

# 获取任务列表
@app.get("/api/tasks")
async def get_tasks():
    return {
        "success": True,
        "data": [
            {
                "task_id": "task_123456",
                "task_type": "code",
                "description": "执行代码静态分析，检查代码质量和潜在问题",
                "priority": 8,
                "agent_type": "code",
                "user_id": "default_user",
                "status": "success",
                "result": {"issues": 5, "passed": 120, "failed": 5},
                "created_at": int(time.time()) - 7200,
                "started_at": int(time.time()) - 5400,
                "completed_at": int(time.time()) - 3600,
                "assigned_agent_id": "agent_001",
                "agentName": "代码智能体001",
                "subtasks": [],
                "predecessors": [],
                "successors": [],
                "dependencies": [],
                "chain_id": "chain_001"
            },
            {
                "task_id": "task_789012",
                "task_type": "analysis",
                "description": "监控系统运行时错误，生成错误报告，包括详细的错误堆栈和影响范围分析",
                "priority": 5,
                "agent_type": "analysis",
                "user_id": "default_user",
                "status": "running",
                "created_at": int(time.time()) - 1800,
                "started_at": int(time.time()) - 1500,
                "assigned_agent_id": "agent_002",
                "agentName": "分析智能体002",
                "subtasks": [],
                "predecessors": [],
                "successors": [],
                "dependencies": [],
                "chain_id": "chain_002"
            },
            {
                "task_id": "task_345678",
                "task_type": "search",
                "description": "搜索相关技术文档和最佳实践，整理成报告",
                "priority": 2,
                "agent_type": "search",
                "user_id": "default_user",
                "status": "failed",
                "error": "网络连接超时",
                "created_at": int(time.time()) - 86400,
                "started_at": int(time.time()) - 85800,
                "completed_at": int(time.time()) - 85200,
                "assigned_agent_id": "agent_003",
                "agentName": "搜索智能体003",
                "subtasks": [],
                "predecessors": [],
                "successors": [],
                "dependencies": [],
                "chain_id": "chain_003"
            }
        ],
        "message": "任务列表获取成功"
    }

# 委托任务
@app.post("/api/tasks/delegate")
async def delegate_task(task_info: dict):
    return {
        "success": True,
        "data": f"task_{str(int(time.time()))[-6:]}",
        "message": "任务委托成功"
    }

# 捕获所有未实现的API请求
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str):
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": f"Mock API: {path} 端点已响应",
            "status": "ok"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)