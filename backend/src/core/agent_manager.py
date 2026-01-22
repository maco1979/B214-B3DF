"""
智能体管理器
用于管理多个智能体的生命周期、通信和协调
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """智能体状态枚举"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AgentType(Enum):
    """智能体类型枚举"""
    DECISION = "decision"
    CONTROL = "control"
    LEARNING = "learning"
    RISK = "risk"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"


@dataclass
class AgentMessage:
    """智能体消息数据结构"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str  # request, response, notification, event
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None


class BaseAgent:
    """智能体基类"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.status = AgentStatus.CREATED
        self.created_at = datetime.now()
        self.last_update = datetime.now()
        self.message_queue = asyncio.Queue()
        self.subscribers = set()  # 订阅此智能体消息的其他智能体
        
    async def initialize(self):
        """初始化智能体"""
        self.status = AgentStatus.INITIALIZING
        logger.info(f"智能体 {self.agent_id} 开始初始化")
        # 子类可以重写此方法以实现特定初始化逻辑
        self.status = AgentStatus.RUNNING
        self.last_update = datetime.now()
        logger.info(f"智能体 {self.agent_id} 初始化完成")
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体主逻辑"""
        raise NotImplementedError("子类必须实现execute方法")
    
    async def pause(self):
        """暂停智能体"""
        self.status = AgentStatus.PAUSED
        self.last_update = datetime.now()
        logger.info(f"智能体 {self.agent_id} 已暂停")
    
    async def resume(self):
        """恢复智能体"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            self.last_update = datetime.now()
            logger.info(f"智能体 {self.agent_id} 已恢复")
    
    async def stop(self):
        """停止智能体"""
        self.status = AgentStatus.STOPPED
        self.last_update = datetime.now()
        logger.info(f"智能体 {self.agent_id} 已停止")
    
    def is_active(self) -> bool:
        """检查智能体是否活跃"""
        return self.status in [AgentStatus.RUNNING, AgentStatus.INITIALIZING]
    
    async def send_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], 
                          correlation_id: Optional[str] = None):
        """发送消息到指定智能体"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        # 通过代理发送消息（避免循环引用）
        await AgentBus.send_message(message)
    
    async def broadcast_message(self, message_type: str, content: Dict[str, Any]):
        """广播消息到所有订阅者"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id="*",
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        await AgentBus.broadcast_message(message, self.subscribers)
    
    def subscribe_to_agent(self, agent_id: str):
        """订阅其他智能体的消息"""
        self.subscribers.add(agent_id)
    
    def unsubscribe_from_agent(self, agent_id: str):
        """取消订阅其他智能体的消息"""
        self.subscribers.discard(agent_id)


class DecisionAgent(BaseAgent):
    """决策智能体"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.DECISION, config)
        self.decision_history = []
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策逻辑"""
        if self.status != AgentStatus.RUNNING:
            return {"error": "智能体未运行", "agent_id": self.agent_id}
        
        try:
            # 模拟决策过程
            decision_input = context.get("decision_input", {})
            decision_result = {
                "action": "adjust_temperature",
                "parameters": {"target": 25.5, "range": 2.0},
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            }
            
            # 记录决策历史
            self.decision_history.append(decision_result)
            
            # 限制历史记录大小
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
            
            logger.debug(f"决策智能体 {self.agent_id} 生成决策: {decision_result['action']}")
            
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result": decision_result,
                "context": context
            }
        except Exception as e:
            logger.error(f"决策智能体 {self.agent_id} 执行错误: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e)
            }


class ControlAgent(BaseAgent):
    """控制智能体"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CONTROL, config)
        self.control_history = []
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行控制逻辑"""
        if self.status != AgentStatus.RUNNING:
            return {"error": "智能体未运行", "agent_id": self.agent_id}
        
        try:
            # 模拟控制过程
            control_input = context.get("control_input", {})
            control_result = {
                "command": "activate_device",
                "device_id": control_input.get("device_id", "unknown"),
                "status": "executed",
                "timestamp": datetime.now().isoformat()
            }
            
            # 记录控制历史
            self.control_history.append(control_result)
            
            # 限制历史记录大小
            if len(self.control_history) > 1000:
                self.control_history = self.control_history[-500:]
            
            logger.debug(f"控制智能体 {self.agent_id} 执行控制: {control_result['command']}")
            
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result": control_result,
                "context": context
            }
        except Exception as e:
            logger.error(f"控制智能体 {self.agent_id} 执行错误: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e)
            }


class LearningAgent(BaseAgent):
    """学习智能体"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.LEARNING, config)
        self.learning_memory = []
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行学习逻辑"""
        if self.status != AgentStatus.RUNNING:
            return {"error": "智能体未运行", "agent_id": self.agent_id}
        
        try:
            # 模拟学习过程
            learning_input = context.get("learning_input", {})
            learning_result = {
                "knowledge_updated": True,
                "new_patterns": ["pattern_1", "pattern_2"],
                "confidence_improved": 0.1,
                "timestamp": datetime.now().isoformat()
            }
            
            # 更新学习记忆
            self.learning_memory.append(learning_input)
            
            # 限制记忆大小
            if len(self.learning_memory) > 10000:
                self.learning_memory = self.learning_memory[-5000:]
            
            logger.debug(f"学习智能体 {self.agent_id} 完成学习任务")
            
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result": learning_result,
                "context": context
            }
        except Exception as e:
            logger.error(f"学习智能体 {self.agent_id} 执行错误: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e)
            }


class AgentBus:
    """智能体通信总线"""
    
    _agents: Dict[str, BaseAgent] = {}
    _message_queues: Dict[str, asyncio.Queue] = {}
    
    @classmethod
    def register_agent(cls, agent: BaseAgent):
        """注册智能体到总线"""
        cls._agents[agent.agent_id] = agent
        cls._message_queues[agent.agent_id] = asyncio.Queue()
        logger.info(f"智能体 {agent.agent_id} 已注册到通信总线")
    
    @classmethod
    def unregister_agent(cls, agent_id: str):
        """从总线注销智能体"""
        cls._agents.pop(agent_id, None)
        cls._message_queues.pop(agent_id, None)
        logger.info(f"智能体 {agent_id} 已从通信总线注销")
    
    @classmethod
    async def send_message(cls, message: AgentMessage):
        """发送消息到指定智能体"""
        if message.receiver_id in cls._message_queues:
            await cls._message_queues[message.receiver_id].put(message)
            logger.debug(f"消息从 {message.sender_id} 发送到 {message.receiver_id}")
        else:
            logger.warning(f"接收智能体 {message.receiver_id} 不存在")
    
    @classmethod
    async def broadcast_message(cls, message: AgentMessage, target_agents: set = None):
        """广播消息到多个智能体"""
        target_list = target_agents if target_agents else cls._agents.keys()
        for agent_id in target_list:
            if agent_id in cls._message_queues and agent_id != message.sender_id:
                new_message = AgentMessage(
                    message_id=message.message_id,
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    message_type=message.message_type,
                    content=message.content,
                    timestamp=message.timestamp,
                    correlation_id=message.correlation_id
                )
                await cls._message_queues[agent_id].put(new_message)
        logger.debug(f"广播消息从 {message.sender_id} 发送到 {len(target_list)} 个智能体")
    
    @classmethod
    async def get_message(cls, agent_id: str) -> Optional[AgentMessage]:
        """获取智能体的消息"""
        if agent_id in cls._message_queues:
            try:
                # 非阻塞获取消息
                return cls._message_queues[agent_id].get_nowait()
            except asyncio.QueueEmpty:
                return None
        return None


class AgentManager:
    """智能体管理器"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_id_counter = 0
        self.task_registry = {}
    
    def create_agent(self, agent_type: AgentType, config: Dict[str, Any] = None) -> str:
        """创建新智能体"""
        if config is None:
            config = {}
        
        agent_id = f"agent_{agent_type.value}_{self.agent_id_counter}"
        self.agent_id_counter += 1
        
        # 根据类型创建智能体
        if agent_type == AgentType.DECISION:
            agent = DecisionAgent(agent_id, config)
        elif agent_type == AgentType.CONTROL:
            agent = ControlAgent(agent_id, config)
        elif agent_type == AgentType.LEARNING:
            agent = LearningAgent(agent_id, config)
        elif agent_type == AgentType.RISK:
            # 风险智能体，暂时使用基类
            agent = BaseAgent(agent_id, agent_type, config)
        elif agent_type == AgentType.MONITORING:
            # 监控智能体，暂时使用基类
            agent = BaseAgent(agent_id, agent_type, config)
        elif agent_type == AgentType.OPTIMIZATION:
            # 优化智能体，暂时使用基类
            agent = BaseAgent(agent_id, agent_type, config)
        else:
            raise ValueError(f"未知的智能体类型: {agent_type}")
        
        # 注册到总线
        AgentBus.register_agent(agent)
        
        # 保存到管理器
        self.agents[agent_id] = agent
        
        logger.info(f"创建智能体: {agent_id} (类型: {agent_type.value})")
        
        return agent_id
    
    async def initialize_agent(self, agent_id: str):
        """初始化智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"智能体不存在: {agent_id}")
        
        agent = self.agents[agent_id]
        await agent.initialize()
    
    async def execute_agent(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"智能体不存在: {agent_id}")
        
        agent = self.agents[agent_id]
        return await agent.execute(context)
    
    async def execute_all_active_agents(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """执行所有活跃的智能体"""
        results = {}
        for agent_id, agent in self.agents.items():
            if agent.is_active():
                try:
                    result = await agent.execute(context)
                    results[agent_id] = result
                except Exception as e:
                    logger.error(f"执行智能体 {agent_id} 时出错: {e}")
                    results[agent_id] = {
                        "success": False,
                        "agent_id": agent_id,
                        "error": str(e)
                    }
        return results
    
    async def start_agent(self, agent_id: str):
        """启动智能体（如果已停止）"""
        if agent_id not in self.agents:
            raise ValueError(f"智能体不存在: {agent_id}")
        
        agent = self.agents[agent_id]
        if agent.status == AgentStatus.STOPPED:
            await agent.initialize()
    
    async def pause_agent(self, agent_id: str):
        """暂停智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"智能体不存在: {agent_id}")
        
        agent = self.agents[agent_id]
        await agent.pause()
    
    async def resume_agent(self, agent_id: str):
        """恢复智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"智能体不存在: {agent_id}")
        
        agent = self.agents[agent_id]
        await agent.resume()
    
    async def stop_agent(self, agent_id: str):
        """停止智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"智能体不存在: {agent_id}")
        
        agent = self.agents[agent_id]
        await agent.stop()
        
        # 从总线注销
        AgentBus.unregister_agent(agent_id)
        
        # 从管理器移除
        del self.agents[agent_id]
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体状态"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type.value,
            "status": agent.status.value,
            "created_at": agent.created_at.isoformat(),
            "last_update": agent.last_update.isoformat()
        }
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有智能体"""
        agent_list = []
        for agent_id, agent in self.agents.items():
            agent_list.append({
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "status": agent.status.value,
                "created_at": agent.created_at.isoformat(),
                "last_update": agent.last_update.isoformat()
            })
        return agent_list
    
    async def cleanup(self):
        """清理所有智能体"""
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            try:
                await self.stop_agent(agent_id)
            except Exception as e:
                logger.error(f"清理智能体 {agent_id} 时出错: {e}")


# 全局智能体管理器实例
agent_manager = AgentManager()


async def get_agent_manager():
    """获取智能体管理器实例"""
    return agent_manager