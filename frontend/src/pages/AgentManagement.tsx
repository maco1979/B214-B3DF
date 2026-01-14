import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { Agent, Task, ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import { Link } from 'react-router-dom';

// 添加CSS样式 - 使用内联样式而非style jsx，避免TypeScript错误

// 配置dayjs插件
dayjs.extend(relativeTime);

// 自定义Tooltip组件
const Tooltip: React.FC<{
  children: React.ReactNode;
  content: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}> = ({ children, content, position = 'top' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const targetRef = useRef<HTMLDivElement>(null);

  const positionClasses = {
    top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 transform -translate-y-1/2 ml-2',
  };

  return (
    <div
      ref={targetRef}
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div
          ref={tooltipRef}
          className={`absolute z-50 px-3 py-2 bg-gray-800 text-white text-sm rounded-md shadow-lg ${positionClasses[position]}`}
        >
          {content}
        </div>
      )}
    </div>
  );
};

// 移除所有不存在的UI组件导入

const AgentManagement: React.FC = () => {
  const queryClient = useQueryClient();

  // 移除toast相关代码

  // 状态管理
  const [showRegisterForm, setShowRegisterForm] = useState(false);
  const [showTaskForm, setShowTaskForm] = useState(false);

  // P3-2 自动刷新 + 异常告警通知
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true); // 默认开启自动刷新
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const [errorNotifications, setErrorNotifications] = useState<Array<{id: string, message: string}>>([]);

  // 核心：异常告警通知（全局调用，在接口请求失败/状态异常时触发）
  const openErrorNotification = (title: string, desc: string, id?: string) => {
    const notificationId = `${id || Date.now()}`;
    setErrorNotifications(prev => [...prev, { id: notificationId, message: `${title}: ${desc}` }]);

    // 10秒后自动关闭通知
    setTimeout(() => {
      setErrorNotifications(prev => prev.filter(n => n.id !== notificationId));
    }, 10000);
  };

  // 获取智能体列表
  const { data: agents, isLoading: isAgentsLoading, error: agentsError, refetch: refetchAgents } = useQuery<ApiResponse<Agent[]>>({
    queryKey: ['agents'],
    queryFn: async () => {
      /*
       * 由于当前API可能没有返回完整的智能体数据，这里暂时返回模拟数据
       * 实际实现时应替换为真实API调用：apiClient.getAgents()
       */
      const response = await apiClient.getAgents();
      if (response.success && response.data) {
        // 为模拟数据添加资源占用字段
        const enhancedAgents = response.data.map(agent => ({
          ...agent,
          resource_usage: {
            cpu: Math.floor(Math.random() * 50) + 5,
            memory: Math.floor(Math.random() * 4096) + 1024,
            disk: Math.floor(Math.random() * 200) + 50,
            network: Math.floor(Math.random() * 100) + 10,
          },
          error_message: agent.status === 'error' ? '连接超时，心跳中断' : undefined,
        }));

        // 智能体离线时触发告警
        enhancedAgents.forEach(agent => {
          if (agent.status === 'offline') {
            openErrorNotification(`智能体【${agent.name}】离线`, '心跳中断超过10分钟，请及时处理', agent.agent_id);
          }
        });

        return {
          ...response,
          data: enhancedAgents,
        };
      }
      return response;
    },
  });

  // 获取任务列表
  const { data: tasks, isLoading: isTasksLoading, error: tasksError, refetch: refetchTasks } = useQuery<ApiResponse<Task[]>>({
    queryKey: ['tasks'],
    queryFn: async () =>

      /*
       * 由于当前API可能没有获取任务列表的端点，这里暂时返回模拟数据
       * 实际实现时应替换为真实API调用：apiClient.getTasks()
       */
       ({
        success: true,
        data: [
          {
            task_id: 'task_123456',
            task_type: 'code',
            description: '执行代码静态分析，检查代码质量和潜在问题',
            priority: 8,
            agent_type: 'code',
            user_id: 'default_user',
            status: 'success',
            result: { issues: 5, passed: 120, failed: 5 },
            created_at: dayjs().subtract(2, 'hour').unix(),
            started_at: dayjs().subtract(1.5, 'hour').unix(),
            completed_at: dayjs().subtract(1, 'hour').unix(),
            assigned_agent_id: 'agent_001',
            agentName: '代码智能体001', // 添加智能体名称用于P3-1跳转
            subtasks: [],
            predecessors: [],
            successors: [],
            dependencies: [],
            chain_id: 'chain_001',
          },
          {
            task_id: 'task_789012',
            task_type: 'analysis',
            description: '监控系统运行时错误，生成错误报告，包括详细的错误堆栈和影响范围分析',
            priority: 5,
            agent_type: 'analysis',
            user_id: 'default_user',
            status: 'running',
            created_at: dayjs().subtract(30, 'minute').unix(),
            started_at: dayjs().subtract(25, 'minute').unix(),
            assigned_agent_id: 'agent_002',
            agentName: '分析智能体002', // 添加智能体名称用于P3-1跳转
            subtasks: [],
            predecessors: [],
            successors: [],
            dependencies: [],
            chain_id: 'chain_002',
          },
          {
            task_id: 'task_345678',
            task_type: 'search',
            description: '搜索相关技术文档和最佳实践，整理成报告',
            priority: 2,
            agent_type: 'search',
            user_id: 'default_user',
            status: 'failed',
            error: '网络连接超时',
            created_at: dayjs().subtract(1, 'day').unix(),
            started_at: dayjs().subtract(1, 'day').add(10, 'minute')
.unix(),
            completed_at: dayjs().subtract(1, 'day').add(20, 'minute')
.unix(),
            assigned_agent_id: 'agent_003',
            agentName: '搜索智能体003', // 添加智能体名称用于P3-1跳转
            subtasks: [],
            predecessors: [],
            successors: [],
            dependencies: [],
            chain_id: 'chain_003',
          },
        ],
      }),

  });

  // 核心：自动刷新逻辑（5分钟刷新一次，可自定义）
  useEffect(() => {
    if (autoRefresh) {
      timerRef.current = setInterval(() => {
        refetchAgents();
        refetchTasks();
      }, 5 * 60 * 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [autoRefresh, refetchAgents, refetchTasks]);

  const [newAgent, setNewAgent] = useState<Partial<Agent>>({
    name: '',
    agent_type: 'code',
    endpoint: '',
    status: 'available',
    capabilities: [],
    last_heartbeat: Date.now(),
  });
  const [newTask, setNewTask] = useState<Partial<Task>>({
    task_type: 'general',
    description: '',
    priority: 0,
    agent_type: 'code',
    user_id: 'default_user',
    subtasks: [],
    predecessors: [],
    successors: [],
    dependencies: [],
  });

  // 状态详情模态框
  const [showStatusModal, setShowStatusModal] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);

  // 打开状态详情模态框
  const openStatusModal = (agent: Agent) => {
    setSelectedAgent(agent);
    setShowStatusModal(true);
  };

  // 关闭状态详情模态框
  const closeStatusModal = () => {
    setShowStatusModal(false);
    setSelectedAgent(null);
  };

  // 加载倒计时状态
  const [loadingCountdown, setLoadingCountdown] = useState(30);

  // 加载中倒计时逻辑
  useEffect(() => {
    if (isAgentsLoading) {
      const interval = setInterval(() => {
        setLoadingCountdown(prev => {
          if (prev <= 1) {
            clearInterval(interval);
            return 30;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(interval);
    }
      setLoadingCountdown(30);
  }, [isAgentsLoading]);

  // 注册智能体
  const registerAgentMutation = useMutation<ApiResponse<Agent>, Error, Partial<Agent>>({
    mutationFn: async agentInfo => apiClient.registerAgent(agentInfo),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] });
      setShowRegisterForm(false);
      setNewAgent({
        name: '',
        agent_type: 'code',
        endpoint: '',
        status: 'available',
        capabilities: [],
        last_heartbeat: Date.now(),
      });
      // 移除toast通知
      alert('智能体注册成功');
    },
    onError: error => {
      // 移除toast通知
      alert(`智能体注册失败: ${error.message}`);
    },
  });

  // 委托任务
  const delegateTaskMutation = useMutation<ApiResponse<string>, Error, Partial<Task>>({
    mutationFn: task => apiClient.delegateTask(task),
    onSuccess: data => {
      // 移除toast通知
      alert(`任务委托成功，任务ID: ${data.data}`);
      setShowTaskForm(false);
      setNewTask({
        task_type: 'general',
        description: '',
        priority: 0,
        agent_type: 'code',
        user_id: 'default_user',
        subtasks: [],
        predecessors: [],
        successors: [],
        dependencies: [],
      });
      // 刷新任务列表
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
    onError: error => {
      // 移除toast通知
      alert(`任务委托失败: ${error.message}`);
    },
  });

  // 根据优先级获取颜色类
  const getPriorityColorClass = (priority: number) => {
    if (priority >= 7) {
      return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300';
    }
    if (priority >= 4) {
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300';
    }
    return 'bg-muted text-muted-foreground';
  };

  // 根据优先级获取文本
  const getPriorityText = (priority: number) => {
    if (priority >= 7) {
      return '高';
    }
    if (priority >= 4) {
      return '中';
    }
    return '低';
  };

  // 日志/报告模态框状态
  const [showTaskLog, setShowTaskLog] = useState(false);
  const [currentTask, setCurrentTask] = useState<Task | null>(null);

  // 打开日志/报告模态框
  const openTaskLogModal = (task: Task) => {
    setCurrentTask(task);
    setShowTaskLog(true);
  };

  // 关闭日志/报告模态框
  const closeTaskLogModal = () => {
    setShowTaskLog(false);
    setCurrentTask(null);
  };

  // 任务操作方法
  const handlePauseTask = async (taskId: string) => {
    try {
      /*
       * 实际实现时应调用真实API
       * await apiClient.pauseTask(taskId);
       */
      alert(`任务 ${taskId} 已暂停`);
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (error) {
      alert(`暂停任务失败: ${(error as Error).message}`);
    }
  };

  const handleTerminateTask = async (taskId: string) => {
    try {
      /*
       * 实际实现时应调用真实API
       * await apiClient.terminateTask(taskId);
       */
      alert(`任务 ${taskId} 已终止`);
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (error) {
      alert(`终止任务失败: ${(error as Error).message}`);
    }
  };

  const handleRetryTask = async (taskId: string) => {
    try {
      /*
       * 实际实现时应调用真实API
       * await apiClient.retryTask(taskId);
       */
      alert(`任务 ${taskId} 已重试`);
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (error) {
      alert(`重试任务失败: ${(error as Error).message}`);
    }
  };

  const handleReExecuteTask = async (taskId: string) => {
    try {
      /*
       * 实际实现时应调用真实API
       * await apiClient.reExecuteTask(taskId);
       */
      alert(`任务 ${taskId} 已重新执行`);
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (error) {
      alert(`重新执行任务失败: ${(error as Error).message}`);
    }
  };

  // 渲染任务操作按钮组
  const renderTaskActions = (task: Task) => {
    const { task_id, status } = task;

    // 运行中任务：查看日志 + 暂停 + 终止
    if (status === 'running') {
      return (
        <div className="flex space-x-2">
          <button
            className="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={() => openTaskLogModal(task)}
          >
            <span className="mr-1">📋</span>
            日志
          </button>
          <button
            className="px-2 py-1 bg-yellow-500 hover:bg-yellow-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={async () => handlePauseTask(task_id)}
          >
            <span className="mr-1">⏸️</span>
            暂停
          </button>
          <button
            className="px-2 py-1 bg-red-500 hover:bg-red-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={async () => handleTerminateTask(task_id)}
          >
            <span className="mr-1">⏹️</span>
            终止
          </button>
        </div>
      );
    }

    // 失败任务：查看原因 + 一键重试
    if (status === 'failed') {
      return (
        <div className="flex space-x-2">
          <button
            className="px-2 py-1 bg-red-500 hover:bg-red-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={() => openTaskLogModal(task)}
          >
            <span className="mr-1">❌</span>
            原因
          </button>
          <button
            className="px-2 py-1 bg-green-500 hover:bg-green-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={async () => handleRetryTask(task_id)}
          >
            <span className="mr-1">🔄</span>
            重试
          </button>
        </div>
      );
    }

    // 已完成任务：查看报告 + 重新执行
    if (status === 'success') {
      return (
        <div className="flex space-x-2">
          <button
            className="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={() => openTaskLogModal(task)}
          >
            <span className="mr-1">📊</span>
            报告
          </button>
          <button
            className="px-2 py-1 bg-gray-500 hover:bg-gray-600 text-white rounded-md text-xs transition-colors flex items-center"
            onClick={async () => handleReExecuteTask(task_id)}
          >
            <span className="mr-1">▶️</span>
            重执行
          </button>
        </div>
      );
    }

    // 其他状态：只显示查看日志
    return (
      <div className="flex space-x-2">
        <button
          className="px-2 py-1 bg-gray-500 hover:bg-gray-600 text-white rounded-md text-xs transition-colors flex items-center"
          onClick={() => openTaskLogModal(task)}
        >
          <span className="mr-1">📋</span>
          详情
        </button>
      </div>
    );
  };

  // 处理注册智能体
  const handleRegisterAgent = () => {
    if (!newAgent.name || !newAgent.endpoint) {
      // 移除toast通知
      alert('智能体名称和端点不能为空');
      return;
    }

    registerAgentMutation.mutate(newAgent);
  };

  // 处理委托任务
  const handleDelegateTask = () => {
    if (!newTask.description) {
      // 移除toast通知
      alert('任务描述不能为空');
      return;
    }

    delegateTaskMutation.mutate(newTask);
  };

  return (
    <div className="space-y-6 p-4">

      {/* 顶部标题和操作栏 */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-foreground">智能体管理</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setShowRegisterForm(!showRegisterForm)}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md transition-colors"
          >
            {showRegisterForm ? '取消' : '注册智能体'}
          </button>
          <button
            onClick={() => setShowTaskForm(!showTaskForm)}
            className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-colors"
          >
            {showTaskForm ? '取消' : '创建任务'}
          </button>
        </div>
      </div>

      {/* 自动刷新 + 异常告警通知 */}
      <div className="flex justify-between items-center">
        {/* 自动刷新开关 + 手动刷新按钮 */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm">自动刷新</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={e => setAutoRefresh(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          <button
            className="px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded-md text-sm transition-colors flex items-center"
            onClick={() => {
              refetchAgents(); refetchTasks();
            }}
          >
            <span className="mr-1">🔄</span>
            手动刷新
          </button>
        </div>

        {/* 异常告警通知列表 */}
        <div className="flex flex-col items-end space-y-2">
          {errorNotifications.map(notification => (
            <div
              key={notification.id}
              className="bg-red-100 text-red-800 px-3 py-2 rounded-md text-sm shadow-md flex items-center space-x-2 animate-fade-in"
            >
              <span className="text-lg">⚠️</span>
              <span>{notification.message}</span>
              <button
                className="text-red-500 hover:text-red-700"
                onClick={() => setErrorNotifications(prev => prev.filter(n => n.id !== notification.id))}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* 注册智能体表单 */}
      {showRegisterForm && (
        <div className="border rounded-lg p-6 bg-card shadow-sm">
          <h2 className="text-xl font-semibold mb-4 text-foreground">注册新智能体</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="agent-name" className="text-sm font-medium text-foreground">智能体名称</label>
              <input
                id="agent-name"
                placeholder="输入智能体名称"
                value={newAgent.name || ''}
                onChange={e => setNewAgent({ ...newAgent, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="agent-type" className="text-sm font-medium text-foreground">智能体类型</label>
              <select
                value={newAgent.agent_type}
                onChange={e => setNewAgent({ ...newAgent, agent_type: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="code">代码智能体</option>
                <option value="analysis">分析智能体</option>
                <option value="search">搜索智能体</option>
                <option value="writing">写作智能体</option>
                <option value="translation">翻译智能体</option>
                <option value="image">图像智能体</option>
                <option value="audio">音频智能体</option>
                <option value="video">视频智能体</option>
                <option value="other">其他类型</option>
              </select>
            </div>
            <div className="space-y-2 md:col-span-2">
              <label htmlFor="agent-endpoint" className="text-sm font-medium text-foreground">智能体端点</label>
              <input
                id="agent-endpoint"
                placeholder="输入智能体端点URL"
                value={newAgent.endpoint || ''}
                onChange={e => setNewAgent({ ...newAgent, endpoint: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="agent-status" className="text-sm font-medium text-foreground">智能体状态</label>
              <select
                value={newAgent.status}
                onChange={e => setNewAgent({ ...newAgent, status: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="available">可用</option>
                <option value="busy">忙碌</option>
                <option value="offline">离线</option>
                <option value="error">错误</option>
              </select>
            </div>
            <div className="flex items-center justify-end space-x-2">
              <button
                onClick={handleRegisterAgent}
                disabled={registerAgentMutation.isPending}
                className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {registerAgentMutation.isPending ?
(
                  <>
                    <span className="mr-2">⏳</span>
                    注册中...
                  </>
                ) :
(
                  '注册智能体'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 创建任务表单 */}
      {showTaskForm && (
        <div className="border rounded-lg p-6 bg-card shadow-sm">
          <h2 className="text-xl font-semibold mb-4 text-foreground">创建新任务</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="task-type" className="text-sm font-medium text-foreground">任务类型</label>
              <select
                value={newTask.task_type}
                onChange={e => setNewTask({ ...newTask, task_type: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="general">通用任务</option>
                <option value="code">代码任务</option>
                <option value="analysis">分析任务</option>
                <option value="search">搜索任务</option>
                <option value="writing">写作任务</option>
                <option value="translation">翻译任务</option>
              </select>
            </div>
            <div className="space-y-2">
              <label htmlFor="task-priority" className="text-sm font-medium text-foreground">任务优先级</label>
              <div className="flex items-center space-x-2">
                <input
                  id="task-priority"
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={newTask.priority || 0}
                  onChange={e => setNewTask({ ...newTask, priority: parseInt(e.target.value) || 0 })}
                  className="flex-1"
                />
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={newTask.priority || 0}
                  onChange={e => setNewTask({ ...newTask, priority: parseInt(e.target.value) || 0 })}
                  className="w-20 px-3 py-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
            <div className="space-y-2 md:col-span-2">
              <label htmlFor="task-description" className="text-sm font-medium text-foreground">任务描述</label>
              <input
                id="task-description"
                placeholder="输入任务描述"
                value={newTask.description || ''}
                onChange={e => setNewTask({ ...newTask, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="task-agent-type" className="text-sm font-medium text-foreground">适用智能体类型</label>
              <select
                value={newTask.agent_type}
                onChange={e => setNewTask({ ...newTask, agent_type: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="code">代码智能体</option>
                <option value="analysis">分析智能体</option>
                <option value="search">搜索智能体</option>
                <option value="writing">写作智能体</option>
                <option value="translation">翻译智能体</option>
                <option value="image">图像智能体</option>
                <option value="audio">音频智能体</option>
                <option value="video">视频智能体</option>
                <option value="other">其他类型</option>
              </select>
            </div>
            <div className="space-y-2">
              <label htmlFor="user-id" className="text-sm font-medium text-foreground">用户ID</label>
              <input
                id="user-id"
                placeholder="输入用户ID"
                value={newTask.user_id || ''}
                onChange={e => setNewTask({ ...newTask, user_id: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex items-center justify-end space-x-2 md:col-span-2">
              <button
                onClick={handleDelegateTask}
                disabled={delegateTaskMutation.isPending}
                className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {delegateTaskMutation.isPending ?
(
                  <>
                    <span className="mr-2">⏳</span>
                    创建任务中...
                  </>
                ) :
(
                  '创建任务'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 智能体列表 */}
      <div className="border rounded-lg p-6 bg-card shadow-sm">
        <h2 className="text-xl font-semibold mb-4 text-foreground">智能体列表</h2>
        {isAgentsLoading ? (
          <div className="flex flex-col justify-center items-center py-8">
            <div className="flex items-center mb-2">
              <span className="text-xl">⏳</span>
              <span className="ml-2">正在加载智能体列表</span>
            </div>
            <div className="text-sm text-gray-500">
              {`(${loadingCountdown}/30s)`}
            </div>
            <div className="mt-2 w-64 h-1 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-1000 ease-out"
                style={{ width: `${((30 - loadingCountdown) / 30) * 100}%` }}
              />
            </div>
          </div>
        ) : agentsError || !agents?.success ? (
          <div className="p-4 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-md text-red-700 dark:text-red-300">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-semibold text-red-800 dark:text-red-200">加载失败</h3>
              <button
                  className="px-3 py-1 bg-red-200 hover:bg-red-300 text-red-800 rounded-md text-sm transition-colors"
                  onClick={async () => refetchAgents()}
                >
                  一键重试
                </button>
            </div>
            <p className="text-red-700 dark:text-red-300 mb-2">
              {agentsError?.message || agents?.error || '未知错误'}
            </p>
            <div className="text-sm bg-red-50 dark:bg-red-800/30 p-3 rounded-md">
              <h4 className="font-medium mb-1">排查建议：</h4>
              <ul className="list-disc list-inside space-y-1">
                <li>请检查智能体服务是否正常运行</li>
                <li>请检查网络连接是否稳定</li>
                <li>请确认API配置是否正确</li>
                <li>稍后重试，可能是临时故障</li>
              </ul>
            </div>
          </div>
        ) : agents.data?.length === 0 ? (
          <div className="flex flex-col justify-center items-center py-12 bg-gray-50 rounded-md">
            <div className="text-6xl mb-4">🤖</div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">暂无已注册的智能体</h3>
            <p className="text-gray-500 mb-6">开始注册您的第一个智能体，开启AI自动化之旅</p>
            <button
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors flex items-center"
              onClick={() => setShowRegisterForm(true)}
            >
              <span className="mr-2">➕</span>
              注册智能体
            </button>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-muted border">
                    <th className="px-4 py-2 text-left border-b text-foreground">智能体ID</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">名称</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">类型</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">状态</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">资源占用</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">当前任务</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">能力</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">最后心跳</th>
                    <th className="px-4 py-2 text-left border-b text-foreground">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {agents.data?.map(agent => (
                    <tr key={agent.agent_id} className="hover:bg-gray-50 border-gray-200">
                      <td className="px-4 py-2 border-b font-mono text-sm text-foreground">{agent.agent_id}</td>
                      <td className="px-4 py-2 border-b font-medium">{agent.name}</td>
                      <td className="px-4 py-2 border-b">
                        <span className="px-2 py-1 bg-muted rounded-full text-xs">{agent.agent_type}</span>
                      </td>
                      <td className="px-4 py-2 border-b">
                        <span
                          className={`px-2 py-1 rounded-full text-xs cursor-pointer ${agent.status === 'available' ? 'bg-green-100 text-green-800 hover:bg-green-200' : agent.status === 'busy' ? 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200' : agent.status === 'error' ? 'bg-red-100 text-red-800 hover:bg-red-200' : 'bg-muted text-muted-foreground hover:bg-gray-200'}`}
                          onClick={() => openStatusModal(agent)}
                        >
                          {agent.status === 'available' ? '在线' : agent.status === 'busy' ? '忙碌' : agent.status === 'error' ? '异常' : '离线'}
                        </span>
                      </td>
                      <td className="px-4 py-2 border-b">
                        <div className="space-y-1">
                          <div className="flex items-center">
                            <span className="text-xs text-gray-500 w-12">CPU:</span>
                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 transition-all duration-500 ease-out"
                                style={{ width: `${agent.resource_usage?.cpu || 0}%` }}
                              />
                            </div>
                            <span className="text-xs ml-2">{agent.resource_usage?.cpu || 0}%</span>
                          </div>
                          <div className="flex items-center">
                            <span className="text-xs text-gray-500 w-12">内存:</span>
                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-green-500 transition-all duration-500 ease-out"
                                style={{ width: `${Math.min((agent.resource_usage?.memory || 0) / 8192 * 100, 100)}%` }}
                              />
                            </div>
                            <span className="text-xs ml-2">
                              {((agent.resource_usage?.memory || 0) / 1024).toFixed(1)}G
                            </span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-2 border-b">
                        {agent.current_task_id || <span className="text-gray-500">无</span>}
                      </td>
                      <td className="px-4 py-2 border-b">
                        <div className="flex flex-wrap gap-1">
                          {agent.capabilities.slice(0, 3).map((capability, index) => (
                            <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                              {capability}
                            </span>
                          ))}
                          {agent.capabilities.length > 3 && (
                            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                              +{agent.capabilities.length - 3}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-2 border-b text-sm text-gray-500">
                        <Tooltip content={new Date(agent.last_heartbeat * 1000).toLocaleString()}>
                          <span className="hover:underline cursor-help">
                            {dayjs(agent.last_heartbeat * 1000).fromNow()}
                          </span>
                        </Tooltip>
                      </td>
                      <td className="px-4 py-2 border-b">
                        <button className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded-md text-sm transition-colors">
                          详情
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* 状态详情模态框 */}
            {showStatusModal && selectedAgent && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
                <div className="bg-card border border-border rounded-lg shadow-xl max-w-md w-full">
                  <div className="p-4 border-b border-border">
                    <div className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold text-foreground">智能体状态详情</h3>
                      <button
                        className="text-muted-foreground hover:text-foreground"
                        onClick={closeStatusModal}
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                  <div className="p-4 space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">智能体ID</h4>
                        <p className="font-mono text-sm text-foreground">{selectedAgent.agent_id}</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">名称</h4>
                        <p className="text-foreground">{selectedAgent.name}</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">类型</h4>
                        <p className="text-foreground">{selectedAgent.agent_type}</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground">状态</h4>
                        <span className={`px-2 py-1 rounded-full text-xs ${selectedAgent.status === 'available' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : selectedAgent.status === 'busy' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : selectedAgent.status === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 'bg-muted text-muted-foreground'}`}>
                          {selectedAgent.status === 'available' ? '在线' : selectedAgent.status === 'busy' ? '忙碌' : selectedAgent.status === 'error' ? '异常' : '离线'}
                        </span>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">资源占用</h4>
                      <div className="space-y-2">
                        <div className="flex items-center">
                          <span className="text-xs text-muted-foreground w-12">CPU:</span>
                          <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden mx-2">
                            <div
                              className="h-full bg-primary transition-all duration-500 ease-out"
                              style={{ width: `${selectedAgent.resource_usage?.cpu || 0}%` }}
                            />
                          </div>
                          <span className="text-sm text-foreground">{selectedAgent.resource_usage?.cpu || 0}%</span>
                        </div>
                        <div className="flex items-center">
                          <span className="text-xs text-muted-foreground w-12">内存:</span>
                          <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden mx-2">
                            <div
                              className="h-full bg-green-500 transition-all duration-500 ease-out"
                              style={{ width: `${Math.min((selectedAgent.resource_usage?.memory || 0) / 8192 * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-sm text-foreground">
                              {((selectedAgent.resource_usage?.memory || 0) / 1024).toFixed(1)}G
                            </span>
                        </div>
                        {selectedAgent.resource_usage?.disk !== undefined && (
                          <div className="flex items-center">
                            <span className="text-xs text-muted-foreground w-12">磁盘:</span>
                            <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden mx-2">
                              <div
                                className="h-full bg-yellow-500 transition-all duration-500 ease-out"
                                style={{ width: `${selectedAgent.resource_usage.disk || 0}%` }}
                              />
                            </div>
                            <span className="text-sm text-foreground">{selectedAgent.resource_usage.disk || 0}%</span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">最后心跳</h4>
                      <p className="text-foreground">{new Date(selectedAgent.last_heartbeat * 1000).toLocaleString()}</p>
                      <p className="text-sm text-muted-foreground mt-1">{dayjs(selectedAgent.last_heartbeat * 1000).fromNow()}</p>
                    </div>

                    {selectedAgent.status === 'error' && selectedAgent.error_message && (
                      <div className="bg-destructive/30 border border-destructive/50 p-3 rounded-md">
                        <h4 className="text-sm font-medium text-destructive mb-1">异常原因</h4>
                        <p className="text-sm text-destructive">{selectedAgent.error_message}</p>
                      </div>
                    )}
                  </div>
                  <div className="p-4 border-t border-border flex justify-end">
                    <button
                      className="px-4 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-md text-sm transition-colors"
                      onClick={closeStatusModal}
                    >
                      关闭
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* 任务列表 */}
      <div className="border rounded-lg p-6 bg-card shadow-sm">
        <h2 className="text-xl font-semibold mb-4 text-foreground">任务列表</h2>
        <div className="flex justify-between items-center mb-4">
          <div>
            {isTasksLoading && (
              <span className="text-sm text-gray-500 flex items-center">
                <span className="mr-1">⏳</span>
                加载任务列表中...
              </span>
            )}
            {tasksError && (
              <span className="text-sm text-red-500">
                加载失败: {tasksError.message}
              </span>
            )}
          </div>
          <button
            onClick={() => setShowTaskForm(true)}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md transition-colors"
          >
            创建新任务
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-muted border">
                <th className="px-4 py-2 text-left border-b text-foreground">任务ID</th>
                <th className="px-4 py-2 text-left border-b text-foreground">类型</th>
                <th className="px-4 py-2 text-left border-b text-foreground">描述</th>
                <th className="px-4 py-2 text-left border-b text-foreground">优先级</th>
                <th className="px-4 py-2 text-left border-b text-foreground">状态</th>
                <th className="px-4 py-2 text-left border-b text-foreground">分配的智能体</th>
                <th className="px-4 py-2 text-left border-b text-foreground">创建时间</th>
                <th className="px-4 py-2 text-left border-b text-foreground">操作</th>
              </tr>
            </thead>
            <tbody>
              {!isTasksLoading && !tasksError && tasks?.success && tasks.data?.length === 0 ?
(
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center">
                    <div className="flex flex-col justify-center items-center">
                      <div className="text-6xl mb-4">📋</div>
                      <h3 className="text-xl font-semibold text-gray-800 mb-2">暂无创建的任务</h3>
                      <p className="text-gray-500 mb-6">开始创建您的第一个任务，让智能体为您工作</p>
                      <button
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors flex items-center"
                        onClick={() => setShowTaskForm(true)}
                      >
                        <span className="mr-2">➕</span>
                        创建新任务
                      </button>
                    </div>
                  </td>
                </tr>
              ) :
(
                (tasks?.data || []).map(task => (
                  <tr key={task.task_id} className="hover:bg-gray-50 border-gray-200">
                    <td className="px-4 py-2 border-b font-mono text-sm text-foreground">{task.task_id}</td>
                    <td className="px-4 py-2 border-b">
                      <span className="px-2 py-1 bg-muted rounded-full text-xs">{task.task_type}</span>
                    </td>
                    <td className="px-4 py-2 border-b max-w-[200px] truncate">
                      <Tooltip content={task.description}>
                        <span className="cursor-help border-b border-dashed border-gray-400 hover:border-gray-600">
                          {task.description}
                        </span>
                      </Tooltip>
                    </td>
                    <td className="px-4 py-2 border-b">
                      <Tooltip content={`优先级值: ${task.priority}`}>
                        <span className={`px-2 py-1 rounded-full text-xs ${getPriorityColorClass(task.priority)}`}>
                          {getPriorityText(task.priority)}
                        </span>
                      </Tooltip>
                    </td>
                    <td className="px-4 py-2 border-b">
                      <span className={`px-2 py-1 rounded-full text-xs ${task.status === 'success' ? 'bg-green-100 text-green-800' : task.status === 'running' ? 'bg-yellow-100 text-yellow-800' : task.status === 'failed' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}`}>
                        {task.status === 'success' ? '已完成' : task.status === 'running' ? '执行中' : task.status === 'failed' ? '失败' : task.status}
                      </span>
                    </td>
                    <td className="px-4 py-2 border-b">
                      <Link
                        to={`/agent-management/${task.assigned_agent_id}`}
                        className="text-blue-600 hover:underline flex items-center"
                      >
                        <span>{task.agentName || task.assigned_agent_id}</span>
                        <span className="ml-1 text-xs">🔗</span>
                      </Link>
                    </td>
                    <td className="px-4 py-2 border-b text-sm text-gray-500">
                      <Tooltip content={new Date(task.created_at * 1000).toLocaleString()}>
                        <span className="hover:underline cursor-help">
                          {dayjs(task.created_at * 1000).fromNow()}
                        </span>
                      </Tooltip>
                    </td>
                    <td className="px-4 py-2 border-b">
                      {renderTaskActions(task)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* 任务日志/报告模态框 */}
        {showTaskLog && currentTask && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
            <div className="bg-card border border-border rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-4 border-b border-border sticky top-0 bg-card z-10">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold text-foreground">
                    {currentTask.status === 'success' ? '任务报告' : currentTask.status === 'failed' ? '错误详情' : '任务日志'}
                  </h3>
                  <button
                    className="text-muted-foreground hover:text-foreground"
                    onClick={closeTaskLogModal}
                  >
                    ✕
                  </button>
                </div>
              </div>
              <div className="p-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">任务ID</h4>
                    <p className="font-mono text-sm text-foreground">{currentTask.task_id}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">任务类型</h4>
                    <p className="text-foreground">{currentTask.task_type}</p>
                  </div>
                  <div className="md:col-span-2">
                    <h4 className="text-sm font-medium text-muted-foreground">任务描述</h4>
                    <p className="text-foreground">{currentTask.description}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">优先级</h4>
                    <span className={`px-2 py-1 rounded-full text-xs ${getPriorityColorClass(currentTask.priority)}`}>
                      {getPriorityText(currentTask.priority)} ({currentTask.priority})
                    </span>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">状态</h4>
                    <span className={`px-2 py-1 rounded-full text-xs ${currentTask.status === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300' : currentTask.status === 'running' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300' : currentTask.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300' : 'bg-muted text-muted-foreground'}`}>
                      {currentTask.status === 'success' ? '已完成' : currentTask.status === 'running' ? '执行中' : currentTask.status === 'failed' ? '失败' : currentTask.status}
                    </span>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">分配的智能体</h4>
                    <p className="text-foreground">{currentTask.agentName || currentTask.assigned_agent_id}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">创建时间</h4>
                    <p className="text-sm text-foreground">{new Date(currentTask.created_at * 1000).toLocaleString()}</p>
                  </div>
                  {currentTask.started_at && (
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">开始时间</h4>
                      <p className="text-sm text-foreground">{new Date(currentTask.started_at * 1000).toLocaleString()}</p>
                    </div>
                  )}
                  {currentTask.completed_at && (
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">结束时间</h4>
                      <p className="text-sm text-foreground">{new Date(currentTask.completed_at * 1000).toLocaleString()}</p>
                    </div>
                  )}
                </div>

                {/* 任务结果或错误信息 */}
                {currentTask.status === 'success' && currentTask.result && (
                  <div className="mt-4">
                    <h4 className="text-lg font-semibold mb-2">任务结果</h4>
                    <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                      <pre className="whitespace-pre-wrap text-sm">
                        {JSON.stringify(currentTask.result, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}

                {currentTask.status === 'failed' && currentTask.error && (
                  <div className="mt-4">
                    <h4 className="text-lg font-semibold mb-2 text-red-600 dark:text-red-400">错误信息</h4>
                    <div className="bg-red-50 dark:bg-red-900/30 p-4 rounded-md">
                      <pre className="whitespace-pre-wrap text-sm text-red-800 dark:text-red-300">
                        {currentTask.error}
                      </pre>
                    </div>
                  </div>
                )}

                {currentTask.status === 'running' && (
                  <div className="mt-4">
                    <h4 className="text-lg font-semibold mb-2">执行状态</h4>
                    <div className="bg-yellow-50 dark:bg-yellow-900/30 p-4 rounded-md">
                      <p className="text-sm text-yellow-800 dark:text-yellow-300">
                        任务正在执行中...
                      </p>
                      <div className="mt-2 w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-yellow-500 transition-all duration-1000 ease-out animate-pulse"
                          style={{ width: '50%' }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div className="p-4 border-t flex justify-end">
                <button
                  className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md text-sm transition-colors"
                  onClick={closeTaskLogModal}
                >
                  关闭
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentManagement;
