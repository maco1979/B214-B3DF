import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from .device_controller import device_controller

logger = logging.getLogger(__name__)


class SceneStatus(str, Enum):
    """场景状态枚举"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    SCHEDULED = "scheduled"
    ERROR = "error"


class SceneAction(BaseModel):
    """场景动作模型"""
    device_id: str = Field(..., description="设备ID")
    command: str = Field(..., description="命令名称")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="命令参数")
    delay: int = Field(default=0, description="执行延迟（毫秒）")


class SceneCondition(BaseModel):
    """场景触发条件模型"""
    type: str = Field(..., description="条件类型")
    device_id: Optional[str] = Field(None, description="设备ID")
    property: Optional[str] = Field(None, description="设备属性")
    operator: Optional[str] = Field(None, description="比较操作符")
    value: Optional[Any] = Field(None, description="比较值")
    time_expression: Optional[str] = Field(None, description="时间表达式")


class Scene(BaseModel):
    """场景模型"""
    id: str = Field(..., description="场景ID")
    name: str = Field(..., description="场景名称")
    description: Optional[str] = Field(None, description="场景描述")
    status: SceneStatus = Field(default=SceneStatus.INACTIVE, description="场景状态")
    actions: List[SceneAction] = Field(default_factory=list, description="场景动作列表")
    conditions: List[SceneCondition] = Field(default_factory=list, description="场景触发条件")
    enabled: bool = Field(default=True, description="是否启用")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    last_executed: Optional[float] = Field(None, description="最后执行时间")
    execution_count: int = Field(default=0, description="执行次数")
    is_recurring: bool = Field(default=False, description="是否重复执行")


class SceneManager:
    """场景管理器"""
    
    def __init__(self, device_controller=None):
        self.scenes: Dict[str, Scene] = {}
        self.device_controller = device_controller
        self.is_running = False
        self.execution_tasks: List[asyncio.Task] = []
        logger.info("场景管理器已初始化")
    
    def start(self):
        """启动场景管理器"""
        if not self.is_running:
            self.is_running = True
            logger.info("场景管理器已启动")
    
    def stop(self):
        """停止场景管理器"""
        if self.is_running:
            self.is_running = False
            # 取消所有执行任务
            for task in self.execution_tasks:
                task.cancel()
            self.execution_tasks.clear()
            logger.info("场景管理器已停止")
    
    def create_scene(self, scene: Scene) -> Scene:
        """创建新场景"""
        self.scenes[scene.id] = scene
        logger.info(f"创建场景: {scene.name} ({scene.id})")
        return scene
    
    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """获取指定场景"""
        return self.scenes.get(scene_id)
    
    def get_all_scenes(self) -> List[Scene]:
        """获取所有场景"""
        return list(self.scenes.values())
    
    def update_scene(self, scene_id: str, scene: Scene) -> Optional[Scene]:
        """更新场景"""
        if scene_id in self.scenes:
            scene.updated_at = time.time()
            self.scenes[scene_id] = scene
            logger.info(f"更新场景: {scene.name} ({scene_id})")
            return scene
        return None
    
    def delete_scene(self, scene_id: str) -> bool:
        """删除场景"""
        if scene_id in self.scenes:
            del self.scenes[scene_id]
            logger.info(f"删除场景: {scene_id}")
            return True
        return False
    
    def enable_scene(self, scene_id: str) -> bool:
        """启用场景"""
        if scene_id in self.scenes:
            self.scenes[scene_id].enabled = True
            self.scenes[scene_id].updated_at = time.time()
            logger.info(f"启用场景: {scene_id}")
            return True
        return False
    
    def disable_scene(self, scene_id: str) -> bool:
        """禁用场景"""
        if scene_id in self.scenes:
            self.scenes[scene_id].enabled = False
            self.scenes[scene_id].updated_at = time.time()
            logger.info(f"禁用场景: {scene_id}")
            return True
        return False
    
    async def trigger_scene(self, scene_id: str) -> bool:
        """触发场景执行"""
        scene = self.get_scene(scene_id)
        if not scene or not scene.enabled:
            logger.warning(f"场景不存在或已禁用: {scene_id}")
            return False
        
        logger.info(f"触发场景: {scene.name} ({scene_id})")
        
        # 更新场景状态
        scene.status = SceneStatus.ACTIVE
        scene.last_executed = time.time()
        scene.execution_count += 1
        scene.updated_at = time.time()
        
        # 异步执行场景动作
        task = asyncio.create_task(self._execute_scene_actions(scene))
        self.execution_tasks.append(task)
        task.add_done_callback(lambda t: self.execution_tasks.remove(t) if t in self.execution_tasks else None)
        
        return True
    
    async def _execute_scene_actions(self, scene: Scene):
        """执行场景动作"""
        try:
            for action in scene.actions:
                # 执行延迟
                if action.delay > 0:
                    await asyncio.sleep(action.delay / 1000)
                
                # 调用设备控制器执行命令
                if self.device_controller:
                    logger.info(f"执行场景动作: 设备 {action.device_id}, 命令 {action.command}, 参数 {action.parameters}")
                    # 实际调用设备控制器的控制方法
                    await self.device_controller.control_device(action.device_id, action.command, action.parameters)
                else:
                    logger.warning(f"设备控制器未初始化，无法执行场景动作: {action}")
            
            # 更新场景状态
            scene.status = SceneStatus.INACTIVE
            logger.info(f"场景执行完成: {scene.name} ({scene.id})")
        except Exception as e:
            scene.status = SceneStatus.ERROR
            logger.error(f"执行场景动作时出错: {str(e)}", exc_info=True)
    
    def evaluate_scene_conditions(self, scene: Scene, device_states: Dict[str, Any]) -> bool:
        """评估场景触发条件"""
        if not scene.conditions:
            # 无条件场景，直接触发
            return True
        
        all_conditions_met = True
        
        for condition in scene.conditions:
            condition_met = self._evaluate_condition(condition, device_states)
            if not condition_met:
                all_conditions_met = False
                break
        
        return all_conditions_met
    
    def _evaluate_condition(self, condition: SceneCondition, device_states: Dict[str, Any]) -> bool:
        """评估单个条件"""
        try:
            if condition.type == "device_state":
                # 设备状态条件
                if not condition.device_id or not condition.property or not condition.operator:
                    return False
                
                device_state = device_states.get(condition.device_id)
                if not device_state:
                    return False
                
                property_value = device_state.get(condition.property)
                if property_value is None:
                    return False
                
                # 根据操作符进行比较
                expected_value = condition.value
                operator = condition.operator
                
                if operator == "==":
                    return property_value == expected_value
                elif operator == "!=":
                    return property_value != expected_value
                elif operator == ">":
                    return float(property_value) > float(expected_value)
                elif operator == "<":
                    return float(property_value) < float(expected_value)
                elif operator == ">=":
                    return float(property_value) >= float(expected_value)
                elif operator == "<=":
                    return float(property_value) <= float(expected_value)
                elif operator == "contains":
                    return str(expected_value) in str(property_value)
                elif operator == "not_contains":
                    return str(expected_value) not in str(property_value)
                
            elif condition.type == "time":
                # 时间条件，简单实现
                # 实际应使用cron表达式或其他时间表达式解析
                return True
            
            return False
        except Exception as e:
            logger.error(f"评估条件时出错: {str(e)}", exc_info=True)
            return False
    
    def handle_device_state_change(self, device_id: str, state: Dict[str, Any]):
        """处理设备状态变化，检查是否触发场景"""
        if not self.is_running:
            return
        
        # 获取所有启用的场景
        enabled_scenes = [scene for scene in self.scenes.values() if scene.enabled]
        
        # 评估每个场景的条件
        for scene in enabled_scenes:
            # 构建设备状态字典
            device_states = {}
            # 这里需要从设备管理器获取所有设备状态
            # 简化实现，仅使用当前变化的设备状态
            device_states[device_id] = state
            
            # 评估条件
            if self.evaluate_scene_conditions(scene, device_states):
                # 触发场景
                asyncio.create_task(self.trigger_scene(scene.id))


# 全局场景管理器实例
scene_manager = SceneManager(device_controller=device_controller)

# 引擎启动函数
def start_scene_manager():
    """启动场景管理器"""
    scene_manager.start()

# 引擎停止函数
def stop_scene_manager():
    """停止场景管理器"""
    scene_manager.stop()

if __name__ == "__main__":
    # 示例用法
    import asyncio
    
    # 创建场景
    scene = Scene(
        id="scene_001",
        name="回家模式",
        description="回家时自动执行的场景",
        actions=[
            SceneAction(
                device_id="device_light_001",
                command="turn_on",
                parameters={"brightness": 80}
            ),
            SceneAction(
                device_id="device_ac_001",
                command="set_temperature",
                parameters={"temperature": 26},
                delay=1000
            )
        ],
        conditions=[
            SceneCondition(
                type="device_state",
                device_id="device_door_001",
                property="status",
                operator="==",
                value="open"
            )
        ]
    )
    
    # 添加场景
    scene_manager.create_scene(scene)
    
    # 启动场景管理器
    scene_manager.start()
    
    # 触发场景
    asyncio.run(scene_manager.trigger_scene("scene_001"))
    
    # 等待场景执行
    time.sleep(3)
    
    # 停止场景管理器
    scene_manager.stop()
    print("场景管理器示例运行完成")