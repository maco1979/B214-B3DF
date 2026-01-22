"""自动应用开发和设备控制API
提供应用模板管理、场景管理和设备自动化规则管理的API接口
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time

# 配置日志
logger = logging.getLogger(__name__)

# 导入服务组件
from src.core.services.scene_manager import scene_manager, Scene, SceneAction, SceneCondition, SceneStatus
# 暂时注释掉app_template_manager导入，因为该模块不存在
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# from app_template_manager import template_manager, AppTemplateRequest

# 临时创建一个mock的template_manager，以便服务器能够启动
class MockTemplateManager:
    def get_all_templates(self):
        return []
    def get_template(self, template_id):
        return None
    def generate_app(self, request):
        return {"status": "success", "app_path": "mock_path"}

template_manager = MockTemplateManager()

class AppTemplateRequest:
    pass
from src.core.services.device_automation_engine import device_automation_engine, AutomationRule, Condition, Action, EventType, ConditionOperator, ActionType

# 创建路由
auto_dev_router = APIRouter(
    prefix="/auto-dev",
    tags=["auto-dev"],
    responses={404: {"description": "Not found"}},
)

# 数据模型
class GenerateAppRequest(BaseModel):
    """生成应用请求模型"""
    app_name: str
    description: str = ""
    app_type: str = "web"
    template: str = "fastapi-vue3"
    backend_framework: str = "fastapi"
    frontend_framework: str = "vue3"
    features: List[str] = []
    database: Optional[str] = "sqlite"
    auth_required: bool = True
    docker_support: bool = True
    git_initialized: bool = True
    mqtt_support: bool = False
    websocket_support: bool = False
    device_management: bool = False

class CreateSceneRequest(BaseModel):
    """创建场景请求模型"""
    name: str
    description: Optional[str] = None
    actions: List[SceneAction]
    conditions: List[SceneCondition] = []
    enabled: bool = True
    is_recurring: bool = False

class CreateAutomationRuleRequest(BaseModel):
    """创建自动化规则请求模型"""
    name: str
    description: Optional[str] = None
    enabled: bool = True
    conditions: List[Condition]
    actions: List[Action]

# 应用模板管理API
@auto_dev_router.get("/templates")
async def get_all_templates():
    """获取所有应用模板
    
    Returns:
        应用模板列表
    """
    try:
        templates = template_manager.get_all_templates()
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取应用模板失败: {str(e)}")

@auto_dev_router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """获取指定应用模板
    
    Args:
        template_id: 模板ID
        
    Returns:
        应用模板详情
    """
    try:
        template = template_manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"模板 {template_id} 不存在")
        return {"template": template}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取应用模板失败: {str(e)}")

@auto_dev_router.post("/generate-app")
async def generate_app(request: GenerateAppRequest):
    """生成应用
    
    Args:
        request: 生成应用请求数据
        
    Returns:
        应用生成结果
    """
    try:
        # 转换为AppTemplateRequest
        template_request = AppTemplateRequest(
            app_name=request.app_name,
            description=request.description,
            app_type=request.app_type,
            template=request.template,
            backend_framework=request.backend_framework,
            frontend_framework=request.frontend_framework,
            features=request.features,
            database=request.database,
            auth_required=request.auth_required,
            docker_support=request.docker_support,
            git_initialized=request.git_initialized,
            mqtt_support=request.mqtt_support,
            websocket_support=request.websocket_support,
            device_management=request.device_management
        )
        
        result = template_manager.generate_app(template_request)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成应用失败: {str(e)}")

# 场景管理API
@auto_dev_router.post("/scenes")
async def create_scene(request: CreateSceneRequest):
    """创建新场景
    
    Args:
        request: 创建场景请求数据
        
    Returns:
        创建的场景
    """
    try:
        import uuid
        scene_id = f"scene_{uuid.uuid4()[:8]}"
        
        scene = Scene(
            id=scene_id,
            name=request.name,
            description=request.description,
            actions=request.actions,
            conditions=request.conditions,
            enabled=request.enabled,
            is_recurring=request.is_recurring
        )
        
        created_scene = scene_manager.create_scene(scene)
        return {"success": True, "data": created_scene}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建场景失败: {str(e)}")

@auto_dev_router.get("/scenes")
async def get_all_scenes():
    """获取所有场景
    
    Returns:
        场景列表
    """
    try:
        scenes = scene_manager.get_all_scenes()
        return {"success": True, "data": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取场景列表失败: {str(e)}")

@auto_dev_router.get("/scenes/{scene_id}")
async def get_scene(scene_id: str):
    """获取指定场景
    
    Args:
        scene_id: 场景ID
        
    Returns:
        场景详情
    """
    try:
        scene = scene_manager.get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail=f"场景 {scene_id} 不存在")
        return {"success": True, "data": scene}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取场景失败: {str(e)}")

@auto_dev_router.post("/scenes/{scene_id}/trigger")
async def trigger_scene(scene_id: str):
    """触发场景执行
    
    Args:
        scene_id: 场景ID
        
    Returns:
        触发结果
    """
    try:
        result = await scene_manager.trigger_scene(scene_id)
        if result:
            return {"success": True, "message": f"场景 {scene_id} 触发成功"}
        else:
            raise HTTPException(status_code=404, detail=f"场景 {scene_id} 触发失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"触发场景失败: {str(e)}")

@auto_dev_router.put("/scenes/{scene_id}")
async def update_scene(scene_id: str, scene: Scene):
    """更新场景
    
    Args:
        scene_id: 场景ID
        scene: 更新后的场景数据
        
    Returns:
        更新后的场景
    """
    try:
        updated_scene = scene_manager.update_scene(scene_id, scene)
        if not updated_scene:
            raise HTTPException(status_code=404, detail=f"场景 {scene_id} 不存在")
        return {"success": True, "data": updated_scene}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新场景失败: {str(e)}")

@auto_dev_router.delete("/scenes/{scene_id}")
async def delete_scene(scene_id: str):
    """删除场景
    
    Args:
        scene_id: 场景ID
        
    Returns:
        删除结果
    """
    try:
        result = scene_manager.delete_scene(scene_id)
        if result:
            return {"success": True, "message": f"场景 {scene_id} 删除成功"}
        else:
            raise HTTPException(status_code=404, detail=f"场景 {scene_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除场景失败: {str(e)}")

@auto_dev_router.post("/scenes/{scene_id}/enable")
async def enable_scene(scene_id: str):
    """启用场景
    
    Args:
        scene_id: 场景ID
        
    Returns:
        启用结果
    """
    try:
        result = scene_manager.enable_scene(scene_id)
        if result:
            return {"success": True, "message": f"场景 {scene_id} 启用成功"}
        else:
            raise HTTPException(status_code=404, detail=f"场景 {scene_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启用场景失败: {str(e)}")

@auto_dev_router.post("/scenes/{scene_id}/disable")
async def disable_scene(scene_id: str):
    """禁用场景
    
    Args:
        scene_id: 场景ID
        
    Returns:
        禁用结果
    """
    try:
        result = scene_manager.disable_scene(scene_id)
        if result:
            return {"success": True, "message": f"场景 {scene_id} 禁用成功"}
        else:
            raise HTTPException(status_code=404, detail=f"场景 {scene_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"禁用场景失败: {str(e)}")

# 设备自动化规则管理API
@auto_dev_router.post("/automation-rules")
async def create_automation_rule(request: CreateAutomationRuleRequest):
    """创建自动化规则
    
    Args:
        request: 创建自动化规则请求数据
        
    Returns:
        创建的自动化规则
    """
    try:
        import uuid
        rule_id = f"rule_{uuid.uuid4()[:8]}"
        
        rule = AutomationRule(
            id=rule_id,
            name=request.name,
            description=request.description,
            enabled=request.enabled,
            conditions=request.conditions,
            actions=request.actions
        )
        
        created_rule = device_automation_engine.create_rule(rule)
        return {"success": True, "data": created_rule}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建自动化规则失败: {str(e)}")

@auto_dev_router.get("/automation-rules")
async def get_all_automation_rules():
    """获取所有自动化规则
    
    Returns:
        自动化规则列表
    """
    try:
        rules = device_automation_engine.get_all_rules()
        return {"success": True, "data": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取自动化规则失败: {str(e)}")

@auto_dev_router.get("/automation-rules/{rule_id}")
async def get_automation_rule(rule_id: str):
    """获取指定自动化规则
    
    Args:
        rule_id: 规则ID
        
    Returns:
        自动化规则详情
    """
    try:
        rule = device_automation_engine.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail=f"自动化规则 {rule_id} 不存在")
        return {"success": True, "data": rule}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取自动化规则失败: {str(e)}")

@auto_dev_router.put("/automation-rules/{rule_id}")
async def update_automation_rule(rule_id: str, rule: AutomationRule):
    """更新自动化规则
    
    Args:
        rule_id: 规则ID
        rule: 更新后的规则数据
        
    Returns:
        更新后的规则
    """
    try:
        updated_rule = device_automation_engine.update_rule(rule_id, rule)
        if not updated_rule:
            raise HTTPException(status_code=404, detail=f"自动化规则 {rule_id} 不存在")
        return {"success": True, "data": updated_rule}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新自动化规则失败: {str(e)}")

@auto_dev_router.delete("/automation-rules/{rule_id}")
async def delete_automation_rule(rule_id: str):
    """删除自动化规则
    
    Args:
        rule_id: 规则ID
        
    Returns:
        删除结果
    """
    try:
        result = device_automation_engine.delete_rule(rule_id)
        if result:
            return {"success": True, "message": f"自动化规则 {rule_id} 删除成功"}
        else:
            raise HTTPException(status_code=404, detail=f"自动化规则 {rule_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除自动化规则失败: {str(e)}")

@auto_dev_router.post("/automation-rules/{rule_id}/enable")
async def enable_automation_rule(rule_id: str):
    """启用自动化规则
    
    Args:
        rule_id: 规则ID
        
    Returns:
        启用结果
    """
    try:
        result = device_automation_engine.enable_rule(rule_id)
        if result:
            return {"success": True, "message": f"自动化规则 {rule_id} 启用成功"}
        else:
            raise HTTPException(status_code=404, detail=f"自动化规则 {rule_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启用自动化规则失败: {str(e)}")

@auto_dev_router.post("/automation-rules/{rule_id}/disable")
async def disable_automation_rule(rule_id: str):
    """禁用自动化规则
    
    Args:
        rule_id: 规则ID
        
    Returns:
        禁用结果
    """
    try:
        result = device_automation_engine.disable_rule(rule_id)
        if result:
            return {"success": True, "message": f"自动化规则 {rule_id} 禁用成功"}
        else:
            raise HTTPException(status_code=404, detail=f"自动化规则 {rule_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"禁用自动化规则失败: {str(e)}")

@auto_dev_router.get("/automation-logs")
async def get_automation_logs(rule_id: Optional[str] = None, limit: int = 100):
    """获取自动化规则执行日志
    
    Args:
        rule_id: 规则ID（可选，指定则只返回该规则的日志）
        limit: 返回日志数量限制
        
    Returns:
        自动化规则执行日志列表
    """
    try:
        logs = device_automation_engine.get_rule_execution_logs(rule_id=rule_id, limit=limit)
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取自动化规则执行日志失败: {str(e)}")

# 引擎管理API
@auto_dev_router.post("/engine/start")
async def start_automation_engine():
    """启动设备自动化引擎
    
    Returns:
        启动结果
    """
    try:
        device_automation_engine.start()
        return {"status": "success", "message": "设备自动化引擎启动成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动设备自动化引擎失败: {str(e)}")

@auto_dev_router.post("/engine/stop")
async def stop_automation_engine():
    """停止设备自动化引擎
    
    Returns:
        停止结果
    """
    try:
        device_automation_engine.stop()
        return {"status": "success", "message": "设备自动化引擎停止成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止设备自动化引擎失败: {str(e)}")

@auto_dev_router.post("/scene-manager/start")
async def start_scene_manager():
    """启动场景管理器
    
    Returns:
        启动结果
    """
    try:
        from src.core.services.scene_manager import start_scene_manager as start_scene_mgr
        start_scene_mgr()
        return {"status": "success", "message": "场景管理器启动成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动场景管理器失败: {str(e)}")

@auto_dev_router.post("/scene-manager/stop")
async def stop_scene_manager():
    """停止场景管理器
    
    Returns:
        停止结果
    """
    try:
        from src.core.services.scene_manager import stop_scene_manager as stop_scene_mgr
        stop_scene_mgr()
        return {"status": "success", "message": "场景管理器停止成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止场景管理器失败: {str(e)}")

# 集成测试API
@auto_dev_router.post("/test/integration")
async def run_integration_test():
    """运行集成测试
    
    Returns:
        测试结果
    """
    try:
        start_time = time.time()
        
        # 1. 测试应用模板管理
        templates = template_manager.get_all_templates()
        
        # 2. 测试场景管理
        scene = Scene(
            id="test_scene",
            name="测试场景",
            description="用于集成测试的场景",
            actions=[],
            conditions=[]
        )
        scene_manager.create_scene(scene)
        
        # 3. 测试自动化规则管理
        import uuid
        rule_id = f"test_rule_{uuid.uuid4()[:8]}"
        condition = Condition(
            type="device_state",
            device_id="test_device",
            property="status",
            operator=ConditionOperator.EQUAL,
            value="online"
        )
        action = Action(
            type=ActionType.DEVICE_COMMAND,
            device_id="test_device",
            command="test_command"
        )
        rule = AutomationRule(
            id=rule_id,
            name="测试规则",
            description="用于集成测试的自动化规则",
            conditions=[condition],
            actions=[action]
        )
        device_automation_engine.create_rule(rule)
        
        end_time = time.time()
        
        # 清理测试数据
        scene_manager.delete_scene("test_scene")
        device_automation_engine.delete_rule(rule_id)
        
        return {
            "status": "success",
            "message": "集成测试通过",
            "templates_count": len(templates),
            "execution_time": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行集成测试失败: {str(e)}")
