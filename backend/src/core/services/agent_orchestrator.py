"""智能体编排服务
提供智能体注册、任务分解、任务委派、结果聚合等功能
"""

from typing import Optional, Dict, List, Any, Callable
import logging
import uuid
import time
from enum import Enum
from datetime import datetime

# 导入自动检查智能体
from .auto_check_agents.static_analysis_agent import static_analysis_agent
from .auto_check_agents.error_monitor_agent import error_monitor_agent
from .auto_check_agents.auto_test_agent import auto_test_agent

# 导入调度服务
from .schedule_service import schedule_service

# 导入用户习惯服务
from .user_habit_service import user_habit_service

# 配置日志
logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 待处理
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消


class AgentStatus(str, Enum):
    """智能体状态枚举"""
    AVAILABLE = "available"    # 可用
    BUSY = "busy"              # 忙碌
    OFFLINE = "offline"         # 离线
    ERROR = "error"             # 错误


class AgentType(str, Enum):
    """智能体类型枚举"""
    SEARCH = "search"           # 搜索智能体
    ANALYSIS = "analysis"       # 分析智能体
    WRITING = "writing"         # 写作智能体
    CODE = "code"               # 代码智能体
    TRANSLATION = "translation" # 翻译智能体
    IMAGE = "image"             # 图像智能体
    AUDIO = "audio"             # 音频智能体
    VIDEO = "video"             # 视频智能体
    OTHER = "other"             # 其他类型


class Task:
    """任务类"""
    
    def __init__(self, task_id: Optional[str] = None, task_type: str = "general", 
                 description: str = "", priority: int = 0, 
                 agent_type: Optional[str] = None, user_id: Optional[str] = None):
        """初始化任务
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            description: 任务描述
            priority: 任务优先级
            agent_type: 适用的智能体类型
            user_id: 用户ID
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.description = description
        self.priority = priority
        self.agent_type = agent_type
        self.user_id = user_id
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.assigned_agent_id = None
        self.subtasks = []
        # 任务链支持
        self.predecessors = []  # 前置任务ID列表
        self.successors = []  # 后置任务ID列表
        self.dependencies = []  # 依赖任务ID列表
        self.chain_id = None  # 任务链ID
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "priority": self.priority,
            "agent_type": self.agent_type,
            "user_id": self.user_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "assigned_agent_id": self.assigned_agent_id,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            # 任务链相关字段
            "predecessors": self.predecessors,
            "successors": self.successors,
            "dependencies": self.dependencies,
            "chain_id": self.chain_id
        }
    
    def add_predecessor(self, task_id: str):
        """添加前置任务
        
        Args:
            task_id: 前置任务ID
        """
        if task_id not in self.predecessors:
            self.predecessors.append(task_id)
    
    def add_successor(self, task_id: str):
        """添加后置任务
        
        Args:
            task_id: 后置任务ID
        """
        if task_id not in self.successors:
            self.successors.append(task_id)
    
    def add_dependency(self, task_id: str):
        """添加依赖任务
        
        Args:
            task_id: 依赖任务ID
        """
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)


class Agent:
    """智能体类"""
    
    def __init__(self, agent_id: str, name: str, agent_type: str, 
                 endpoint: str, status: str = AgentStatus.AVAILABLE):
        """初始化智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            agent_type: 智能体类型
            endpoint: 智能体端点
            status: 智能体状态
        """
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.endpoint = endpoint
        self.status = status
        self.current_task_id = None
        self.capabilities = []
        self.last_heartbeat = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "endpoint": self.endpoint,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "capabilities": self.capabilities,
            "last_heartbeat": self.last_heartbeat
        }


class AgentOrchestrator:
    """智能体编排服务类"""
    
    def __init__(self):
        """初始化智能体编排服务"""
        self.agents = {}  # 智能体注册表
        self.tasks = {}   # 任务注册表
        self.agent_callbacks = {}  # 智能体回调注册表
        # 任务链管理
        self.task_chains = {}  # 任务链注册表
        # 自动任务触发器
        self.auto_triggers = {}  # 自动任务触发器注册表
        self.check_results = []  # 自动检查结果存储
        
        # 启动调度服务
        schedule_service.start()
        
        # 注册内置的自动检查智能体
        self._register_auto_check_agents()
        
        # 注册自动检查任务的自动触发器
        self._register_auto_check_triggers()
        
        logger.info("智能体编排服务初始化完成")
    
    def register_agent(self, agent: Agent) -> bool:
        """注册智能体
        
        Args:
            agent: 智能体对象
            
        Returns:
            注册是否成功
        """
        try:
            self.agents[agent.agent_id] = agent
            logger.info(f"智能体注册成功: {agent.name} ({agent.agent_id})")
            return True
        except Exception as e:
            logger.error(f"智能体注册失败: {str(e)}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            注销是否成功
        """
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"智能体注销成功: {agent_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"智能体注销失败: {str(e)}")
            return False
    
    def get_available_agents(self, agent_type: Optional[str] = None) -> List[Agent]:
        """获取可用智能体列表
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            可用智能体列表
        """
        available_agents = []
        for agent in self.agents.values():
            if agent.status == AgentStatus.AVAILABLE:
                if agent_type is None or agent.agent_type == agent_type:
                    available_agents.append(agent)
        return available_agents
    
    def create_task(self, task_type: str, description: str, priority: int = 0, 
                    agent_type: Optional[str] = None, user_id: Optional[str] = None) -> Task:
        """创建任务
        
        Args:
            task_type: 任务类型
            description: 任务描述
            priority: 任务优先级
            agent_type: 适用的智能体类型
            user_id: 用户ID
            
        Returns:
            任务对象
        """
        task = Task(
            task_type=task_type,
            description=description,
            priority=priority,
            agent_type=agent_type,
            user_id=user_id
        )
        self.tasks[task.task_id] = task
        logger.info(f"任务创建成功: {task.task_id} - {description}")
        return task
    
    def assign_task(self, task_id: str, agent_id: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        """分配任务
        
        Args:
            task_id: 任务ID
            agent_id: 智能体ID
            user_id: 用户ID，用于基于用户习惯选择智能体
            
        Returns:
            分配是否成功
        """
        try:
            if task_id not in self.tasks:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            task = self.tasks[task_id]
            
            # 如果没有指定智能体，则自动选择
            if not agent_id:
                available_agents = self.get_available_agents(task.agent_type)
                if not available_agents:
                    logger.error(f"没有可用的智能体: {task.agent_type}")
                    return False
                
                # 基于用户习惯选择智能体
                if user_id:
                    agent = self._select_agent_based_on_habit(user_id, available_agents, task)
                else:
                    # 简单的负载均衡：选择第一个可用智能体
                    agent = available_agents[0]
            else:
                if agent_id not in self.agents:
                    logger.error(f"智能体不存在: {agent_id}")
                    return False
                agent = self.agents[agent_id]
            
            # 检查智能体状态
            if agent.status != AgentStatus.AVAILABLE:
                logger.error(f"智能体不可用: {agent.agent_id} - {agent.status}")
                return False
            
            # 分配任务
            task.status = TaskStatus.RUNNING
            task.assigned_agent_id = agent.agent_id
            task.started_at = time.time()
            
            # 更新智能体状态
            agent.status = AgentStatus.BUSY
            agent.current_task_id = task_id
            
            logger.info(f"任务分配成功: {task_id} -> {agent.agent_id} ({agent.name})")
            return True
            
        except Exception as e:
            logger.error(f"任务分配失败: {str(e)}")
            return False
    
    def _select_agent_based_on_habit(self, user_id: str, available_agents: List[Agent], task: Task) -> Agent:
        """基于用户习惯选择最优智能体
        
        Args:
            user_id: 用户ID
            available_agents: 可用智能体列表
            task: 任务对象
            
        Returns:
            选择的智能体
        """
        logger.info(f"基于用户习惯选择智能体: 用户ID: {user_id}，任务类型: {task.agent_type}")
        
        # 获取用户的智能体偏好
        preferences = user_habit_service.predict_agent_preference(user_id, task.task_type)
        
        # 如果没有足够的历史数据，返回第一个可用智能体
        if not preferences:
            return available_agents[0]
        
        # 将可用智能体按照用户偏好排序
        preferred_agents = []
        other_agents = []
        
        # 构建可用智能体ID集合
        available_agent_ids = {agent.agent_id for agent in available_agents}
        
        # 优先选择用户偏好的智能体
        for pref in preferences:
            if pref["agent_id"] in available_agent_ids:
                # 找到对应的智能体对象
                for agent in available_agents:
                    if agent.agent_id == pref["agent_id"]:
                        preferred_agents.append(agent)
                        break
        
        # 剩余的智能体
        for agent in available_agents:
            if agent not in preferred_agents:
                other_agents.append(agent)
        
        # 如果有偏好的智能体，返回第一个
        if preferred_agents:
            selected_agent = preferred_agents[0]
            logger.info(f"选择用户偏好的智能体: {selected_agent.agent_id} ({selected_agent.name})")
            return selected_agent
        else:
            # 否则返回第一个可用智能体
            logger.info(f"没有用户偏好的智能体，选择第一个可用智能体: {available_agents[0].agent_id} ({available_agents[0].name})")
            return available_agents[0]
    
    def complete_task(self, task_id: str, result: Any) -> bool:
        """完成任务
        
        Args:
            task_id: 任务ID
            result: 任务结果
            
        Returns:
            完成是否成功
        """
        try:
            if task_id not in self.tasks:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            # 更新智能体状态
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                agent = self.agents[task.assigned_agent_id]
                agent.status = AgentStatus.AVAILABLE
                agent.current_task_id = None
            
            logger.info(f"任务完成: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"任务完成处理失败: {str(e)}")
            return False
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """标记任务失败
        
        Args:
            task_id: 任务ID
            error: 失败原因
            
        Returns:
            处理是否成功
        """
        try:
            if task_id not in self.tasks:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()
            
            # 更新智能体状态
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                agent = self.agents[task.assigned_agent_id]
                agent.status = AgentStatus.AVAILABLE
                agent.current_task_id = None
            
            logger.error(f"任务失败: {task_id} - {error}")
            return True
            
        except Exception as e:
            logger.error(f"任务失败处理失败: {str(e)}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            取消是否成功
        """
        try:
            if task_id not in self.tasks:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            # 更新智能体状态
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                agent = self.agents[task.assigned_agent_id]
                agent.status = AgentStatus.AVAILABLE
                agent.current_task_id = None
            
            logger.info(f"任务取消: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"任务取消处理失败: {str(e)}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象或None
        """
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """获取指定状态的任务列表
        
        Args:
            status: 任务状态
            
        Returns:
            任务列表
        """
        return [task for task in self.tasks.values() if task.status == status]
    
    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """更新智能体状态
        
        Args:
            agent_id: 智能体ID
            status: 智能体状态
            
        Returns:
            更新是否成功
        """
        try:
            if agent_id in self.agents:
                self.agents[agent_id].status = status
                self.agents[agent_id].last_heartbeat = time.time()
                logger.info(f"智能体状态更新: {agent_id} -> {status}")
                return True
            return False
        except Exception as e:
            logger.error(f"智能体状态更新失败: {str(e)}")
            return False
    
    def heartbeat(self, agent_id: str) -> bool:
        """智能体心跳
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            心跳处理是否成功
        """
        try:
            if agent_id in self.agents:
                self.agents[agent_id].last_heartbeat = time.time()
                if self.agents[agent_id].status == AgentStatus.OFFLINE:
                    self.agents[agent_id].status = AgentStatus.AVAILABLE
                return True
            return False
        except Exception as e:
            logger.error(f"心跳处理失败: {str(e)}")
            return False
    
    def register_agent_callback(self, agent_id: str, callback: Callable) -> bool:
        """注册智能体回调
        
        Args:
            agent_id: 智能体ID
            callback: 回调函数
            
        Returns:
            注册是否成功
        """
        try:
            self.agent_callbacks[agent_id] = callback
            logger.info(f"智能体回调注册成功: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"智能体回调注册失败: {str(e)}")
            return False
    
    def execute_task(self, task_type: str, description: str, priority: int = 0, 
                     agent_type: Optional[str] = None) -> Any:
        """执行任务（同步方式）
        
        Args:
            task_type: 任务类型
            description: 任务描述
            priority: 任务优先级
            agent_type: 适用的智能体类型
            
        Returns:
            任务结果
        """
        # 创建任务
        task = self.create_task(task_type, description, priority, agent_type)
        
        # 分配任务
        if not self.assign_task(task.task_id):
            return {
                "error": "无法分配任务",
                "task_id": task.task_id
            }
        
        # 等待任务完成（简单的轮询机制）
        max_wait_time = 60  # 最大等待时间（秒）
        wait_interval = 0.5  # 轮询间隔（秒）
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            task = self.get_task(task.task_id)
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
            time.sleep(wait_interval)
        
        # 返回结果
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            return {
                "error": task.error,
                "task_id": task.task_id
            }
        else:
            return {
                "error": "任务超时",
                "task_id": task.task_id
            }
    
    # 任务链管理方法
    def create_task_chain(self, name: str, description: str = "") -> str:
        """创建任务链
        
        Args:
            name: 任务链名称
            description: 任务链描述
            
        Returns:
            任务链ID
        """
        chain_id = str(uuid.uuid4())
        self.task_chains[chain_id] = {
            "chain_id": chain_id,
            "name": name,
            "description": description,
            "tasks": [],
            "status": "pending",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None
        }
        logger.info(f"任务链创建成功: {chain_id} - {name}")
        return chain_id
    
    def add_task_to_chain(self, chain_id: str, task: Task, predecessor_id: Optional[str] = None) -> bool:
        """将任务添加到任务链
        
        Args:
            chain_id: 任务链ID
            task: 任务对象
            predecessor_id: 前置任务ID（可选）
            
        Returns:
            添加是否成功
        """
        if chain_id not in self.task_chains:
            logger.error(f"任务链不存在: {chain_id}")
            return False
        
        # 设置任务的链ID
        task.chain_id = chain_id
        
        # 添加到任务注册表
        self.tasks[task.task_id] = task
        
        # 添加到任务链
        self.task_chains[chain_id]["tasks"].append(task.task_id)
        
        # 设置依赖关系
        if predecessor_id:
            if predecessor_id not in self.tasks:
                logger.error(f"前置任务不存在: {predecessor_id}")
                return False
            
            # 更新前置任务的后置任务列表
            predecessor = self.tasks[predecessor_id]
            predecessor.add_successor(task.task_id)
            
            # 更新当前任务的前置任务列表
            task.add_predecessor(predecessor_id)
            
            logger.info(f"任务链 {chain_id}: 任务 {task.task_id} 添加成功，前置任务: {predecessor_id}")
        else:
            logger.info(f"任务链 {chain_id}: 任务 {task.task_id} 添加成功")
        
        return True
    
    def start_task_chain(self, chain_id: str, user_id: Optional[str] = None) -> bool:
        """启动任务链
        
        Args:
            chain_id: 任务链ID
            user_id: 用户ID，用于基于用户习惯选择智能体
            
        Returns:
            启动是否成功
        """
        if chain_id not in self.task_chains:
            logger.error(f"任务链不存在: {chain_id}")
            return False
        
        chain = self.task_chains[chain_id]
        chain["status"] = "running"
        chain["started_at"] = time.time()
        
        # 启动所有没有前置任务的任务
        for task_id in chain["tasks"]:
            task = self.tasks.get(task_id)
            if task and not task.predecessors:
                self.assign_task(task_id, user_id=user_id)
        
        logger.info(f"任务链启动成功: {chain_id}")
        return True
    
    def complete_task(self, task_id: str, result: Any) -> bool:
        """完成任务
        
        Args:
            task_id: 任务ID
            result: 任务结果
            
        Returns:
            完成是否成功
        """
        try:
            if task_id not in self.tasks:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            # 更新智能体状态
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                agent = self.agents[task.assigned_agent_id]
                agent.status = AgentStatus.AVAILABLE
                agent.current_task_id = None
            
            logger.info(f"任务完成: {task_id}")
            
            # 自动触发后续任务
            self._trigger_successor_tasks(task_id)
            
            # 检查任务链是否完成
            if task.chain_id and task.chain_id in self.task_chains:
                self._check_chain_completion(task.chain_id)
            
            return True
            
        except Exception as e:
            logger.error(f"任务完成处理失败: {str(e)}")
            return False
    
    def _trigger_successor_tasks(self, task_id: str) -> None:
        """触发后续任务
        
        Args:
            task_id: 已完成任务的ID
        """
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # 遍历所有后续任务
        for successor_id in task.successors:
            successor = self.tasks.get(successor_id)
            if not successor:
                continue
            
            # 检查所有前置任务是否都已完成
            all_predecessors_completed = True
            for predecessor_id in successor.predecessors:
                predecessor = self.tasks.get(predecessor_id)
                if not predecessor or predecessor.status != TaskStatus.COMPLETED:
                    all_predecessors_completed = False
                    break
            
            # 如果所有前置任务都已完成，则启动该任务
            if all_predecessors_completed and successor.status == TaskStatus.PENDING:
                # 传递相同的user_id
                self.assign_task(successor_id, user_id=task.user_id)
                logger.info(f"自动触发后续任务: {successor_id} (由任务 {task_id} 触发)")
    
    def _check_chain_completion(self, chain_id: str) -> None:
        """检查任务链是否完成
        
        Args:
            chain_id: 任务链ID
        """
        chain = self.task_chains.get(chain_id)
        if not chain:
            return
        
        # 检查所有任务是否都已完成
        all_tasks_completed = True
        for task_id in chain["tasks"]:
            task = self.tasks.get(task_id)
            if not task or task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                all_tasks_completed = False
                break
        
        if all_tasks_completed:
            chain["status"] = "completed"
            chain["completed_at"] = time.time()
            logger.info(f"任务链完成: {chain_id}")
    
    def orchestrate_task(self, task_description: str) -> Dict[str, Any]:
        """编排任务，根据任务描述自动分解和委派
        
        Args:
            task_description: 任务描述
            
        Returns:
            任务编排结果
        """
        logger.info(f"开始编排任务: {task_description}")
        
        # 创建任务链
        chain_id = self.create_task_chain(
            name="自动编排任务链",
            description=task_description
        )
        
        # 示例：根据任务描述分解为多个子任务
        # 这里是简单的示例，实际项目中可以使用更复杂的任务分解算法
        if "检查" in task_description or "bug" in task_description or "错误" in task_description:
            # 错误检查任务链
            # 1. 代码静态分析
            static_analysis_task = self.create_task(
                task_type="static_analysis",
                description="代码静态分析",
                agent_type="code"
            )
            self.add_task_to_chain(chain_id, static_analysis_task)
            
            # 2. 运行时错误监控
            error_monitor_task = self.create_task(
                task_type="error_monitor",
                description="运行时错误监控",
                agent_type="analysis"
            )
            self.add_task_to_chain(chain_id, error_monitor_task, predecessor_id=static_analysis_task.task_id)
            
            # 3. 自动测试
            auto_test_task = self.create_task(
                task_type="auto_test",
                description="自动测试",
                agent_type="code"
            )
            self.add_task_to_chain(chain_id, auto_test_task, predecessor_id=error_monitor_task.task_id)
        else:
            # 其他类型任务
            main_task = self.create_task(
                task_type="general",
                description=task_description,
                agent_type="other"
            )
            self.add_task_to_chain(chain_id, main_task)
        
        # 启动任务链
        self.start_task_chain(chain_id)
        
        return {
            "chain_id": chain_id,
            "status": "orchestrated",
            "message": f"任务已编排，任务链ID: {chain_id}"
        }
    
    # 自动任务触发相关方法
    def register_auto_trigger(self, trigger_id: str, trigger_type: str, config: Dict[str, Any], 
                             task_description: str) -> bool:
        """注册自动触发器
        
        Args:
            trigger_id: 触发器ID
            trigger_type: 触发器类型（cron、event等）
            config: 触发器配置
            task_description: 要触发的任务描述
            
        Returns:
            注册是否成功
        """
        self.auto_triggers[trigger_id] = {
            "trigger_id": trigger_id,
            "trigger_type": trigger_type,
            "config": config,
            "task_description": task_description,
            "status": "active",
            "last_triggered": None,
            "created_at": time.time()
        }
        logger.info(f"自动触发器注册成功: {trigger_id} - {task_description}")
        return True
    
    def unregister_auto_trigger(self, trigger_id: str) -> bool:
        """注销自动触发器
        
        Args:
            trigger_id: 触发器ID
            
        Returns:
            注销是否成功
        """
        if trigger_id in self.auto_triggers:
            del self.auto_triggers[trigger_id]
            logger.info(f"自动触发器注销成功: {trigger_id}")
            return True
        return False
    
    def _register_auto_check_agents(self) -> None:
        """注册内置的自动检查智能体"""
        try:
            # 注册静态分析智能体
            static_agent = Agent(
                agent_id="static_analysis_agent",
                name="静态分析智能体",
                agent_type=AgentType.CODE,
                endpoint="local",
                status=AgentStatus.AVAILABLE
            )
            static_agent.capabilities = static_analysis_agent.capabilities
            self.register_agent(static_agent)
            
            # 注册错误监控智能体
            error_agent = Agent(
                agent_id="error_monitor_agent",
                name="错误监控智能体",
                agent_type=AgentType.ANALYSIS,
                endpoint="local",
                status=AgentStatus.AVAILABLE
            )
            error_agent.capabilities = error_monitor_agent.capabilities
            self.register_agent(error_agent)
            
            # 注册自动测试智能体
            test_agent = Agent(
                agent_id="auto_test_agent",
                name="自动测试智能体",
                agent_type=AgentType.CODE,
                endpoint="local",
                status=AgentStatus.AVAILABLE
            )
            test_agent.capabilities = auto_test_agent.capabilities
            self.register_agent(test_agent)
            
            logger.info("自动检查智能体注册完成")
        except Exception as e:
            logger.error(f"注册自动检查智能体失败: {str(e)}")
    
    def _register_auto_check_triggers(self) -> None:
        """注册自动检查任务的自动触发器"""
        try:
            # 注册每日代码静态分析触发器
            self.register_auto_trigger(
                trigger_id="daily_static_analysis",
                trigger_type="cron",
                config={"cron": "0 2 * * *"},  # 每天凌晨2点
                task_description="执行代码静态分析，检查代码质量和潜在问题"
            )
            
            # 注册每小时错误监控触发器
            self.register_auto_trigger(
                trigger_id="hourly_error_monitor",
                trigger_type="cron",
                config={"cron": "0 * * * *"},  # 每小时
                task_description="监控系统运行时错误，生成错误报告"
            )
            
            # 注册每日自动测试触发器
            self.register_auto_trigger(
                trigger_id="daily_auto_test",
                trigger_type="cron",
                config={"cron": "0 3 * * *"},  # 每天凌晨3点
                task_description="运行自动化测试，确保系统功能正常"
            )
            
            logger.info("自动检查任务触发器注册完成")
        except Exception as e:
            logger.error(f"注册自动检查触发器失败: {str(e)}")
    
    def trigger_auto_task(self, trigger_id: str) -> Dict[str, Any]:
        """触发自动任务
        
        Args:
            trigger_id: 触发器ID
            
        Returns:
            触发结果
        """
        if trigger_id not in self.auto_triggers:
            return {
                "error": f"触发器不存在: {trigger_id}",
                "status": "failed"
            }
        
        trigger = self.auto_triggers[trigger_id]
        
        # 执行任务编排
        result = self.orchestrate_task(trigger["task_description"])
        
        # 更新触发器状态
        trigger["last_triggered"] = time.time()
        
        logger.info(f"自动任务触发成功: {trigger_id} - {trigger['task_description']}")
        return {
            "status": "success",
            "chain_id": result["chain_id"],
            "trigger_id": trigger_id
        }
    
    def _execute_auto_check_task(self, task_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行自动检查任务
        
        Args:
            task_type: 任务类型
            config: 任务配置
            
        Returns:
            任务执行结果
        """
        try:
            result = {}
            
            if task_type == "static_analysis":
                # 执行代码静态分析
                analysis_result = static_analysis_agent.analyze_directory("./src")
                logger.info(f"代码静态分析完成: {analysis_result.get('summary', {})}")
                result = {
                    "check_type": "代码质量",
                    "target": config.get("target", "./src"),
                    "status": "completed",
                    "passed": analysis_result.get("passed", False),
                    "issues_count": analysis_result.get("issues_count", 0),
                    "start_time": time.time(),
                    "end_time": time.time(),
                    "results": analysis_result
                }
            elif task_type == "error_monitor":
                # 执行错误监控
                monitor_result = error_monitor_agent.generate_error_report(24)
                logger.info(f"错误监控完成: {monitor_result.get('summary', {})}")
                result = {
                    "check_type": "错误监控",
                    "target": config.get("target", "系统日志"),
                    "status": "completed",
                    "passed": monitor_result.get("passed", False),
                    "issues_count": monitor_result.get("issues_count", 0),
                    "start_time": time.time(),
                    "end_time": time.time(),
                    "results": monitor_result
                }
            elif task_type == "auto_test":
                # 执行自动测试
                test_result = auto_test_agent.run_all_tests("./tests")
                logger.info(f"自动测试完成: {test_result.get('summary', {})}")
                result = {
                    "check_type": "自动测试",
                    "target": config.get("target", "./tests"),
                    "status": "completed",
                    "passed": test_result.get("passed", False),
                    "issues_count": test_result.get("failed_count", 0),
                    "start_time": time.time(),
                    "end_time": time.time(),
                    "results": test_result
                }
            else:
                logger.error(f"不支持的自动检查任务类型: {task_type}")
                return {"error": f"不支持的自动检查任务类型: {task_type}", "status": "failed"}
            
            # 保存检查结果
            self.save_check_result(result)
            
            return result
        except Exception as e:
            logger.error(f"执行自动检查任务失败: {str(e)}")
            error_result = {
                "check_type": task_type,
                "target": config.get("target", ""),
                "status": "failed",
                "passed": False,
                "issues_count": 0,
                "start_time": time.time(),
                "end_time": time.time(),
                "results": {"error": str(e)}
            }
            self.save_check_result(error_result)
            return error_result
    
    def get_check_results(self) -> List[Dict[str, Any]]:
        """获取所有自动检查结果
        
        Returns:
            检查结果列表
        """
        return self.check_results
    
    def save_check_result(self, result: Dict[str, Any]):
        """保存自动检查结果
        
        Args:
            result: 检查结果
        """
        result["check_id"] = str(uuid.uuid4())
        result["timestamp"] = datetime.now().isoformat()
        self.check_results.append(result)
        
        # 只保留最近100条记录
        if len(self.check_results) > 100:
            self.check_results = self.check_results[-100:]
        
        logger.info(f"检查结果已保存: {result.get('check_id')}")


# 单例模式
agent_orchestrator = AgentOrchestrator()