"""调度服务
提供CRON调度和事件触发功能，用于自动执行各种任务
"""

from typing import Dict, List, Any, Callable, Optional
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
import schedule

# 配置日志
logger = logging.getLogger(__name__)


class ScheduleStatus(str):
    """调度状态枚举"""
    IDLE = "idle"           # 空闲
    RUNNING = "running"      # 运行中
    PAUSED = "paused"        # 已暂停
    ERROR = "error"           # 错误


class ScheduleType(str):
    """调度类型枚举"""
    CRON = "cron"           # CRON调度
    INTERVAL = "interval"     # 间隔调度
    ONE_TIME = "one_time"     # 一次性调度
    EVENT = "event"         # 事件触发


class ScheduleTask:
    """调度任务类"""
    
    def __init__(self, task_id: str, task_type: str, schedule_type: ScheduleType, 
                 description: str, config: Dict[str, Any], callback: Callable):
        """初始化调度任务
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            schedule_type: 调度类型
            description: 任务描述
            config: 调度配置
            callback: 任务回调函数
        """
        self.task_id = task_id
        self.task_type = task_type
        self.schedule_type = schedule_type
        self.description = description
        self.config = config
        self.callback = callback
        self.status = ScheduleStatus.IDLE
        self.last_executed = None
        self.next_execution = None
        self.execution_count = 0
        self.error_count = 0
        self.created_at = datetime.now().isoformat()
        
        # 内部调度任务引用
        self._schedule_job = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "schedule_type": self.schedule_type,
            "description": self.description,
            "config": self.config,
            "status": self.status,
            "last_executed": self.last_executed,
            "next_execution": self.next_execution,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "created_at": self.created_at
        }


class ScheduleService:
    """调度服务类"""
    
    def __init__(self):
        """初始化调度服务"""
        self.tasks = {}  # 调度任务注册表
        self.event_handlers = {}  # 事件处理器注册表
        self.is_running = False
        self.schedule_thread = None
        self.status = ScheduleStatus.IDLE
        
        logger.info("调度服务初始化完成")
    
    def start(self) -> bool:
        """启动调度服务
        
        Returns:
            启动是否成功
        """
        if self.is_running:
            logger.warning("调度服务已经在运行中")
            return True
        
        try:
            self.is_running = True
            self.status = ScheduleStatus.RUNNING
            
            # 创建调度线程
            self.schedule_thread = threading.Thread(target=self._schedule_loop, daemon=True)
            self.schedule_thread.start()
            
            logger.info("调度服务启动成功")
            return True
        except Exception as e:
            logger.error(f"调度服务启动失败: {str(e)}")
            self.is_running = False
            self.status = ScheduleStatus.ERROR
            return False
    
    def stop(self) -> bool:
        """停止调度服务
        
        Returns:
            停止是否成功
        """
        if not self.is_running:
            logger.warning("调度服务已经停止")
            return True
        
        try:
            self.is_running = False
            self.status = ScheduleStatus.IDLE
            
            if self.schedule_thread:
                self.schedule_thread.join(timeout=5)
            
            logger.info("调度服务停止成功")
            return True
        except Exception as e:
            logger.error(f"调度服务停止失败: {str(e)}")
            self.status = ScheduleStatus.ERROR
            return False
    
    def _schedule_loop(self) -> None:
        """调度循环
        定期运行调度器的pending任务
        """
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                logger.error(f"调度循环错误: {str(e)}")
                time.sleep(5)  # 出错后暂停5秒
    
    def add_cron_task(self, task_type: str, description: str, 
                     cron_expression: str, callback: Callable, 
                     config: Optional[Dict[str, Any]] = None) -> str:
        """添加CRON调度任务
        
        Args:
            task_type: 任务类型
            description: 任务描述
            cron_expression: CRON表达式
            callback: 任务回调函数
            config: 任务配置
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        # 解析CRON表达式
        # CRON格式: 分 时 日 月 周
        cron_parts = cron_expression.split()
        if len(cron_parts) != 5:
            raise ValueError(f"无效的CRON表达式: {cron_expression}，格式应为: 分 时 日 月 周")
        
        minute, hour, day, month, weekday = cron_parts
        
        # 创建调度任务
        task = ScheduleTask(
            task_id=task_id,
            task_type=task_type,
            schedule_type=ScheduleType.CRON,
            description=description,
            config=config or {},
            callback=callback
        )
        
        # 添加到schedule
        job = schedule.every()
        
        # 格式化时间，确保两位数格式
        def format_time(h, m):
            h_str = h.zfill(2) if isinstance(h, str) else f"{h:02d}"
            m_str = m.zfill(2) if isinstance(m, str) else f"{m:02d}"
            return f"{h_str}:{m_str}"
        
        # 设置CRON参数
        if weekday != '*':
            # 转换为schedule支持的星期格式（0-6，周一-周日）
            weekday_map = {
                '0': 'monday', '1': 'tuesday', '2': 'wednesday',
                '3': 'thursday', '4': 'friday', '5': 'saturday', '6': 'sunday'
            }
            if weekday in weekday_map:
                job = getattr(job, weekday_map[weekday])
                # 每周特定时间执行
                job = job.at(format_time(hour, minute))
            else:
                raise ValueError(f"无效的星期值: {weekday}")
        elif hour != '*':
            # 每天特定时间执行
            job = job.day.at(format_time(hour, minute))
        elif minute != '*':
            # 每小时特定分钟执行
            # 确保分钟是两位数格式
            minute_str = minute.zfill(2) if isinstance(minute, str) else f"{minute:02d}"
            job = job.hour.at(f":{minute_str}")
        else:
            # 每分钟执行
            job = job.minute
        
        # 设置任务函数
        def job_func():
            self._execute_task(task_id)
        
        job.do(job_func)
        
        # 保存任务
        task._schedule_job = job
        self.tasks[task_id] = task
        
        logger.info(f"CRON调度任务添加成功: {task_id} - {description}，表达式: {cron_expression}")
        return task_id
    
    def add_interval_task(self, task_type: str, description: str, 
                         interval_seconds: int, callback: Callable, 
                         config: Optional[Dict[str, Any]] = None) -> str:
        """添加间隔调度任务
        
        Args:
            task_type: 任务类型
            description: 任务描述
            interval_seconds: 执行间隔（秒）
            callback: 任务回调函数
            config: 任务配置
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        # 创建调度任务
        task = ScheduleTask(
            task_id=task_id,
            task_type=task_type,
            schedule_type=ScheduleType.INTERVAL,
            description=description,
            config=config or {},
            callback=callback
        )
        
        # 添加到schedule
        def job_func():
            self._execute_task(task_id)
        
        job = schedule.every(interval_seconds).seconds.do(job_func)
        
        # 保存任务
        task._schedule_job = job
        self.tasks[task_id] = task
        
        logger.info(f"间隔调度任务添加成功: {task_id} - {description}，间隔: {interval_seconds}秒")
        return task_id
    
    def add_one_time_task(self, task_type: str, description: str, 
                         execute_time: datetime, callback: Callable, 
                         config: Optional[Dict[str, Any]] = None) -> str:
        """添加一次性调度任务
        
        Args:
            task_type: 任务类型
            description: 任务描述
            execute_time: 执行时间
            callback: 任务回调函数
            config: 任务配置
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        # 创建调度任务
        task = ScheduleTask(
            task_id=task_id,
            task_type=task_type,
            schedule_type=ScheduleType.ONE_TIME,
            description=description,
            config=config or {},
            callback=callback
        )
        
        # 计算延迟时间
        delay_seconds = (execute_time - datetime.now()).total_seconds()
        
        if delay_seconds <= 0:
            # 如果执行时间已过，立即执行
            threading.Thread(target=lambda: self._execute_task(task_id), daemon=True).start()
        else:
            # 创建延迟执行线程
            def delayed_execution():
                time.sleep(delay_seconds)
                self._execute_task(task_id)
                # 执行后移除任务
                self.remove_task(task_id)
            
            threading.Thread(target=delayed_execution, daemon=True).start()
        
        # 保存任务
        self.tasks[task_id] = task
        
        logger.info(f"一次性调度任务添加成功: {task_id} - {description}，执行时间: {execute_time}")
        return task_id
    
    def remove_task(self, task_id: str) -> bool:
        """移除调度任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            移除是否成功
        """
        if task_id not in self.tasks:
            logger.warning(f"调度任务不存在: {task_id}")
            return False
        
        try:
            task = self.tasks[task_id]
            
            # 取消schedule任务
            if task._schedule_job:
                schedule.cancel_job(task._schedule_job)
            
            # 从注册表中移除
            del self.tasks[task_id]
            
            logger.info(f"调度任务移除成功: {task_id} - {task.description}")
            return True
        except Exception as e:
            logger.error(f"移除调度任务失败: {str(e)}")
            return False
    
    def _execute_task(self, task_id: str) -> None:
        """执行调度任务
        
        Args:
            task_id: 任务ID
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"调度任务不存在: {task_id}")
            return
        
        logger.info(f"开始执行调度任务: {task_id} - {task.description}")
        task.status = ScheduleStatus.RUNNING
        
        try:
            # 执行任务回调
            result = task.callback(task)
            
            task.last_executed = datetime.now().isoformat()
            task.execution_count += 1
            task.status = ScheduleStatus.IDLE
            
            logger.info(f"调度任务执行成功: {task_id} - {task.description}")
            return result
        except Exception as e:
            logger.error(f"调度任务执行失败: {task_id} - {task.description}，错误: {str(e)}")
            task.status = ScheduleStatus.ERROR
            task.error_count += 1
            return None
    
    def get_task(self, task_id: str) -> Optional[ScheduleTask]:
        """获取调度任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            调度任务对象或None
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ScheduleTask]:
        """获取所有调度任务
        
        Returns:
            调度任务列表
        """
        return list(self.tasks.values())
    
    def get_tasks_by_type(self, task_type: str) -> List[ScheduleTask]:
        """按类型获取调度任务
        
        Args:
            task_type: 任务类型
            
        Returns:
            调度任务列表
        """
        return [task for task in self.tasks.values() if task.task_type == task_type]
    
    # 事件触发相关方法
    def register_event_handler(self, event_type: str, handler: Callable) -> bool:
        """注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            
        Returns:
            注册是否成功
        """
        try:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            
            self.event_handlers[event_type].append(handler)
            logger.info(f"事件处理器注册成功: {event_type}")
            return True
        except Exception as e:
            logger.error(f"注册事件处理器失败: {str(e)}")
            return False
    
    def unregister_event_handler(self, event_type: str, handler: Callable) -> bool:
        """注销事件处理器
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            
        Returns:
            注销是否成功
        """
        try:
            if event_type not in self.event_handlers:
                return False
            
            if handler in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(handler)
                logger.info(f"事件处理器注销成功: {event_type}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"注销事件处理器失败: {str(e)}")
            return False
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """触发事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
            
        Returns:
            触发是否成功
        """
        try:
            if event_type not in self.event_handlers:
                logger.warning(f"没有注册的事件处理器: {event_type}")
                return False
            
            logger.info(f"触发事件: {event_type}，数据: {event_data}")
            
            # 执行所有注册的处理器
            for handler in self.event_handlers[event_type]:
                # 使用线程异步执行，避免阻塞
                threading.Thread(target=lambda: handler(event_type, event_data), daemon=True).start()
            
            return True
        except Exception as e:
            logger.error(f"触发事件失败: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度服务状态
        
        Returns:
            服务状态信息
        """
        return {
            "status": self.status,
            "is_running": self.is_running,
            "task_count": len(self.tasks),
            "event_handler_count": sum(len(handlers) for handlers in self.event_handlers.values()),
            "tasks": [task.to_dict() for task in self.tasks.values()]
        }
    
    # 快捷方法
    def schedule_daily(self, task_type: str, description: str, 
                      hour: int, minute: int, callback: Callable, 
                      config: Optional[Dict[str, Any]] = None) -> str:
        """添加每日调度任务
        
        Args:
            task_type: 任务类型
            description: 任务描述
            hour: 小时（0-23）
            minute: 分钟（0-59）
            callback: 任务回调函数
            config: 任务配置
            
        Returns:
            任务ID
        """
        cron_expression = f"{minute} {hour} * * *"
        return self.add_cron_task(task_type, description, cron_expression, callback, config)
    
    def schedule_weekly(self, task_type: str, description: str, 
                       weekday: str, hour: int, minute: int, callback: Callable, 
                       config: Optional[Dict[str, Any]] = None) -> str:
        """添加每周调度任务
        
        Args:
            task_type: 任务类型
            description: 任务描述
            weekday: 星期（monday, tuesday, ..., sunday）
            hour: 小时（0-23）
            minute: 分钟（0-59）
            callback: 任务回调函数
            config: 任务配置
            
        Returns:
            任务ID
        """
        # 转换星期为数字格式（0-6，周一-周日）
        weekday_map = {
            'monday': '0', 'tuesday': '1', 'wednesday': '2',
            'thursday': '3', 'friday': '4', 'saturday': '5', 'sunday': '6'
        }
        weekday_num = weekday_map.get(weekday.lower(), '*')
        
        cron_expression = f"{minute} {hour} * * {weekday_num}"
        return self.add_cron_task(task_type, description, cron_expression, callback, config)


# 单例模式
schedule_service = ScheduleService()