import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { ScheduleTask, ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

// 任务模板定义
const taskTemplates = {
  general: [
    { id: 'default', name: '默认通用模板', config: { param1: 'value1', param2: 123 } },
    { id: 'custom', name: '自定义通用任务', config: {} },
  ],
  code_quality: [
    { id: 'basic', name: '基础代码质量检查', config: { check_depth: 'basic', include_tests: false } },
    { id: 'detailed', name: '详细代码质量检查', config: { check_depth: 'detailed', include_tests: true, check_dependencies: true } },
    { id: 'custom', name: '自定义代码质量检查', config: {} },
  ],
  security: [
    { id: 'basic', name: '基础安全检查', config: { scan_dependencies: true, check_secrets: false } },
    { id: 'comprehensive', name: '全面安全检查', config: { scan_dependencies: true, check_secrets: true, check_vulnerabilities: true, severity_level: 'medium' } },
    { id: 'custom', name: '自定义安全检查', config: {} },
  ],
  performance: [
    { id: 'basic', name: '基础性能检查', config: { check_memory: true, check_cpu: false, duration: 30 } },
    { id: 'detailed', name: '详细性能检查', config: { check_memory: true, check_cpu: true, check_network: true, duration: 60 } },
    { id: 'custom', name: '自定义性能检查', config: {} },
  ],
  data_backup: [
    { id: 'daily', name: '每日备份', config: { backup_path: './backups', retention_days: 7, compression: 'gzip' } },
    { id: 'weekly', name: '每周备份', config: { backup_path: './backups', retention_days: 30, compression: 'gzip', include_logs: true } },
    { id: 'custom', name: '自定义备份', config: {} },
  ],
  report_generation: [
    { id: 'daily', name: '每日报告', config: { report_type: 'summary', format: 'pdf', recipients: [] } },
    { id: 'weekly', name: '每周报告', config: { report_type: 'detailed', format: 'html', recipients: [], include_charts: true } },
    { id: 'custom', name: '自定义报告', config: {} },
  ],
};

// 动态表单字段定义
const configFields = {
  general: [
    { name: 'param1', label: '参数1', type: 'text', defaultValue: 'value1', required: false },
    { name: 'param2', label: '参数2', type: 'number', defaultValue: 123, required: false, min: 1 },
    { name: 'enabled', label: '启用', type: 'checkbox', defaultValue: true, required: false },
  ],
  code_quality: [
    { name: 'check_depth',
label: '检查深度',
type: 'select',
defaultValue: 'basic',
required: true,
options: [
      { value: 'basic', label: '基础' },
      { value: 'standard', label: '标准' },
      { value: 'detailed', label: '详细' },
      { value: 'comprehensive', label: '全面' },
    ] },
    { name: 'include_tests', label: '包含测试文件', type: 'checkbox', defaultValue: false, required: false },
    { name: 'check_dependencies', label: '检查依赖', type: 'checkbox', defaultValue: false, required: false },
    { name: 'max_file_size', label: '最大文件大小(KB)', type: 'number', defaultValue: 1000, required: false, min: 100 },
  ],
  security: [
    { name: 'scan_dependencies', label: '扫描依赖', type: 'checkbox', defaultValue: true, required: false },
    { name: 'check_secrets', label: '检查敏感信息', type: 'checkbox', defaultValue: false, required: false },
    { name: 'check_vulnerabilities', label: '检查漏洞', type: 'checkbox', defaultValue: false, required: false },
    { name: 'severity_level',
label: '严重级别',
type: 'select',
defaultValue: 'medium',
required: false,
options: [
      { value: 'low', label: '低' },
      { value: 'medium', label: '中' },
      { value: 'high', label: '高' },
      { value: 'critical', label: '严重' },
    ] },
  ],
  performance: [
    { name: 'check_memory', label: '检查内存使用', type: 'checkbox', defaultValue: true, required: false },
    { name: 'check_cpu', label: '检查CPU使用', type: 'checkbox', defaultValue: false, required: false },
    { name: 'check_network', label: '检查网络性能', type: 'checkbox', defaultValue: false, required: false },
    { name: 'duration', label: '测试时长(秒)', type: 'number', defaultValue: 30, required: false, min: 10, max: 300 },
  ],
  data_backup: [
    { name: 'backup_path', label: '备份路径', type: 'text', defaultValue: './backups', required: true },
    { name: 'retention_days', label: '保留天数', type: 'number', defaultValue: 7, required: false, min: 1, max: 365 },
    { name: 'compression',
label: '压缩格式',
type: 'select',
defaultValue: 'gzip',
required: false,
options: [
      { value: 'none', label: '无压缩' },
      { value: 'gzip', label: 'GZIP' },
      { value: 'zip', label: 'ZIP' },
    ] },
    { name: 'include_logs', label: '包含日志', type: 'checkbox', defaultValue: false, required: false },
  ],
  report_generation: [
    { name: 'report_type',
label: '报告类型',
type: 'select',
defaultValue: 'summary',
required: true,
options: [
      { value: 'summary', label: '摘要' },
      { value: 'detailed', label: '详细' },
    ] },
    { name: 'format',
label: '报告格式',
type: 'select',
defaultValue: 'pdf',
required: true,
options: [
      { value: 'pdf', label: 'PDF' },
      { value: 'html', label: 'HTML' },
      { value: 'csv', label: 'CSV' },
    ] },
    { name: 'include_charts', label: '包含图表', type: 'checkbox', defaultValue: false, required: false },
    { name: 'recipients', label: '收件人(逗号分隔)', type: 'text', defaultValue: '', required: false },
  ],
};

const ScheduleService: React.FC = () => {
  const queryClient = useQueryClient();

  // 状态管理
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [taskType, setTaskType] = useState<'cron' | 'interval' | 'one-time'>('cron');
  const [configMode, setConfigMode] = useState<'template' | 'visual' | 'json'>('template');
  const [selectedTemplate, setSelectedTemplate] = useState('default');
  const [visualConfig, setVisualConfig] = useState<any>({});
  const [scheduleTask, setScheduleTask] = useState({
    task_type: 'general',
    description: '',
    config: '{}',
    cron_expression: '0 0 * * *',
    customCron: '',
    interval_seconds: 3600,
    customInterval: 0,
    execute_time: new Date(Date.now() + 3600000).toISOString().slice(0, 19),
    customExecuteTime: '',
  });

  // 当任务类型改变时，重置模板和配置
  useEffect(() => {
    const defaultTemplate = taskTemplates[scheduleTask.task_type as keyof typeof taskTemplates][0].id;
    setSelectedTemplate(defaultTemplate);
    const templateConfig = taskTemplates[scheduleTask.task_type as keyof typeof taskTemplates][0].config;
    setVisualConfig(templateConfig);
    setScheduleTask(prev => ({
      ...prev,
      config: JSON.stringify(templateConfig, null, 2),
    }));
  }, [scheduleTask.task_type]);

  // 当模板改变时，更新配置
  useEffect(() => {
    const template = taskTemplates[scheduleTask.task_type as keyof typeof taskTemplates].find(t => t.id === selectedTemplate);
    if (template) {
      setVisualConfig(template.config);
      setScheduleTask(prev => ({
        ...prev,
        config: JSON.stringify(template.config, null, 2),
      }));
    }
  }, [selectedTemplate, scheduleTask.task_type]);

  // 当可视化配置改变时，更新JSON配置
  useEffect(() => {
    if (configMode !== 'json') {
      setScheduleTask(prev => ({
        ...prev,
        config: JSON.stringify(visualConfig, null, 2),
      }));
    }
  }, [visualConfig, configMode]);

  // 处理可视化配置字段变化
  const handleVisualConfigChange = (fieldName: string, value: any) => {
    setVisualConfig(prev => {
      const updated = { ...prev, [fieldName]: value };
      return updated;
    });
  };

  // 处理JSON配置手动输入
  const handleJsonConfigChange = (value: string) => {
    setScheduleTask(prev => ({
      ...prev,
      config: value,
    }));
    // 尝试解析JSON更新可视化配置
    try {
      const parsed = JSON.parse(value);
      setVisualConfig(parsed);
    } catch (error) {
      // JSON格式错误，不更新可视化配置
    }
  };

  // 获取调度任务列表
  const {
    data: scheduleTasks,
    isLoading: isTasksLoading,
    error: tasksError,
    refetch: refetchTasks,
  } = useQuery<ApiResponse<ScheduleTask[]>>({
    queryKey: ['scheduleTasks'],
    queryFn: async () => apiClient.getScheduleTasks(),
  });

  // 获取调度服务状态
  const {
    data: scheduleStatus,
    isLoading: isStatusLoading,
  } = useQuery<ApiResponse<any>>({
    queryKey: ['scheduleStatus'],
    queryFn: async () => apiClient.getScheduleStatus(),
  });

  // 创建Cron调度任务
  const createCronTaskMutation = useMutation<ApiResponse<{ task_id: string }>, Error, typeof scheduleTask>({
    mutationFn: async task => apiClient.createCronTask({
      task_type: task.task_type,
      description: task.description,
      config: JSON.parse(task.config),
      cron_expression: task.cron_expression,
    }),
    onSuccess: () => {
      handleTaskCreated();
    },
    onError: error => {
      handleTaskError(error);
    },
  });

  // 创建间隔调度任务
  const createIntervalTaskMutation = useMutation<ApiResponse<{ task_id: string }>, Error, typeof scheduleTask>({
    mutationFn: async task => apiClient.createIntervalTask({
      task_type: task.task_type,
      description: task.description,
      config: JSON.parse(task.config),
      interval_seconds: task.interval_seconds,
    }),
    onSuccess: () => {
      handleTaskCreated();
    },
    onError: error => {
      handleTaskError(error);
    },
  });

  // 创建一次性调度任务
  const createOneTimeTaskMutation = useMutation<ApiResponse<{ task_id: string }>, Error, typeof scheduleTask>({
    mutationFn: async task => apiClient.createOneTimeTask({
      task_type: task.task_type,
      description: task.description,
      config: JSON.parse(task.config),
      execute_time: task.execute_time,
    }),
    onSuccess: () => {
      handleTaskCreated();
    },
    onError: error => {
      handleTaskError(error);
    },
  });

  // 移除调度任务
  const removeTaskMutation = useMutation<ApiResponse<any>, Error, string>({
    mutationFn: async task_id => apiClient.removeScheduleTask(task_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scheduleTasks'] });
      alert('调度任务已移除');
    },
    onError: error => {
      alert(`移除调度任务失败: ${error.message}`);
    },
  });

  // 处理任务创建成功
  const handleTaskCreated = () => {
    queryClient.invalidateQueries({ queryKey: ['scheduleTasks'] });
    setShowCreateForm(false);
    resetForm();
    alert('调度任务已创建');
  };

  // 处理任务创建错误
  const handleTaskError = (error: Error) => {
    alert(`创建调度任务失败: ${error.message}`);
  };

  // 重置表单
  const resetForm = () => {
    setScheduleTask({
      task_type: 'general',
      description: '',
      config: '{}',
      cron_expression: '0 0 * * *',
      customCron: '',
      interval_seconds: 3600,
      customInterval: 0,
      execute_time: new Date(Date.now() + 3600000).toISOString().slice(0, 19),
      customExecuteTime: '',
    });
    setTaskType('cron');
    setConfigMode('template');
    setSelectedTemplate('default');
    const defaultTemplate = taskTemplates.general[0].config;
    setVisualConfig(defaultTemplate);
  };

  // 提交调度任务
  const handleSubmitTask = () => {
    // 验证表单
    if (!scheduleTask.task_type || !scheduleTask.description) {
      alert('任务类型和描述不能为空');
      return;
    }

    try {
      JSON.parse(scheduleTask.config);
    } catch (error) {
      alert('配置必须是有效的JSON格式');
      return;
    }

    // 根据任务类型调用不同的创建方法
    switch (taskType) {
      case 'cron':
        createCronTaskMutation.mutate(scheduleTask);
        break;
      case 'interval':
        createIntervalTaskMutation.mutate(scheduleTask);
        break;
      case 'one-time':
        createOneTimeTaskMutation.mutate(scheduleTask);
        break;
    }
  };

  // 获取任务状态的样式
  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-green-900/50 text-green-300 border border-green-800';
      case 'paused':
        return 'bg-yellow-900/50 text-yellow-300 border border-yellow-800';
      case 'failed':
        return 'bg-red-900/50 text-red-300 border border-red-800';
      case 'completed':
        return 'bg-gray-900/50 text-gray-300 border border-gray-800';
      default:
        return 'bg-gray-900/50 text-gray-300 border border-gray-800';
    }
  };

  return (
    <div className="space-y-6 p-4 md:p-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center bg-gradient-to-r from-blue-900/30 to-indigo-900/30 p-4 rounded-xl shadow-sm border">
        <div>
          <h1 className="text-3xl font-bold text-foreground">调度服务管理</h1>
          <p className="text-sm text-muted-foreground mt-1">管理和监控系统调度任务</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={async () => refetchTasks()}
            className="px-4 py-2 bg-muted text-muted-foreground border hover:bg-muted/80 rounded-md transition-all hover:shadow-sm"
          >
            🔄 刷新任务
          </button>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="px-5 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-all hover:shadow-md transform hover:-translate-y-0.5"
          >
            {showCreateForm ?
(
              <>
                <span className="mr-2">✕</span>
                取消
              </>
            ) :
(
              <>
                <span className="mr-2">+</span>
                创建任务
              </>
            )}
          </button>
        </div>
      </div>

      {/* 调度服务状态卡片 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 服务状态卡片 */}
        <div className="bg-card rounded-xl shadow-sm border p-6 transition-all hover:shadow-md">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-foreground">调度服务状态</h2>
            <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse"></div>
          </div>

          {isStatusLoading ? (
            <div className="flex justify-center items-center py-8">
              <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            </div>
          ) : scheduleStatus?.success ? (
            <div className="space-y-6">
              {/* 服务状态 */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">服务状态</p>
                  <p className="text-2xl font-bold mt-1">
                    {scheduleStatus.data?.running ?
(
                      <span className="flex items-center text-green-600">
                        <span className="mr-2 w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                        运行中
                      </span>
                    ) :
(
                      <span className="flex items-center text-red-600">
                        <span className="mr-2 w-2 h-2 rounded-full bg-red-500"></span>
                        已停止
                      </span>
                    )}
                  </p>
                </div>
              </div>

              {/* 任务统计 */}
              <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                <div className="bg-muted p-4 rounded-lg border">
                  <p className="text-sm text-muted-foreground">活跃任务数</p>
                  <p className="text-2xl font-bold mt-1 text-foreground">{scheduleStatus.data?.active_tasks || 0}</p>
                </div>
                <div className="bg-muted p-4 rounded-lg border">
                  <p className="text-sm text-muted-foreground">总任务数</p>
                  <p className="text-2xl font-bold mt-1 text-foreground">{scheduleStatus.data?.total_tasks || 0}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-4 bg-red-900/30 border border-red-800 rounded-lg text-red-300">
              <h3 className="font-semibold text-red-800 flex items-center">
                <span className="mr-2">⚠️</span>
                加载失败
              </h3>
              <p className="text-red-700 text-sm mt-1">{scheduleStatus?.error || '无法获取调度服务状态'}</p>
            </div>
          )}
        </div>

        {/* 任务类型分布卡片 */}
        <div className="bg-card rounded-xl shadow-sm border p-6 transition-all hover:shadow-md">
          <h2 className="text-xl font-semibold text-foreground mb-4">任务类型分布</h2>
          {isTasksLoading ?
(
            <div className="flex justify-center items-center py-8">
              <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            </div>
          ) :
scheduleTasks?.success && scheduleTasks.data?.length > 0 ?
(
            <div className="space-y-4">
              {[
                  { type: 'general', label: '通用任务', color: 'bg-blue-900/50 text-blue-300 border border-blue-800' },
                  { type: 'code_quality', label: '代码质量检查', color: 'bg-green-900/50 text-green-300 border border-green-800' },
                  { type: 'security', label: '安全性检查', color: 'bg-red-900/50 text-red-300 border border-red-800' },
                  { type: 'performance', label: '性能检查', color: 'bg-purple-900/50 text-purple-300 border border-purple-800' },
                  { type: 'data_backup', label: '数据备份', color: 'bg-yellow-900/50 text-yellow-300 border border-yellow-800' },
                  { type: 'report_generation', label: '报告生成', color: 'bg-indigo-900/50 text-indigo-300 border border-indigo-800' },
                ].map(item => {
                const count = scheduleTasks.data?.filter(task => task.task_type === item.type).length || 0;
                const total = scheduleTasks.data?.length || 1;
                const percentage = Math.round((count / total) * 100);

                return count > 0 && (
                  <div key={item.type} className="space-y-1">
                    <div className="flex justify-between text-sm">
                  <span className="font-medium text-foreground">{item.label}</span>
                  <span className="text-muted-foreground">{count} 个 ({percentage}%)</span>
                </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 bg-gradient-to-r ${item.color.replace('bg-', 'from-').replace(' text-', ' to-')}`}
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
                  </div>
                );
              })}
            </div>
          ) :
(
            <div className="p-4 bg-muted rounded-lg border">
              <p className="text-muted-foreground text-center">暂无任务数据</p>
            </div>
          )}
        </div>

        {/* 任务状态概览卡片 */}
        <div className="bg-card rounded-xl shadow-sm border p-6 transition-all hover:shadow-md">
          <h2 className="text-xl font-semibold text-foreground mb-4">任务状态概览</h2>
          {isTasksLoading ?
(
            <div className="flex justify-center items-center py-8">
              <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            </div>
          ) :
scheduleTasks?.success && scheduleTasks.data?.length > 0 ?
(
            <div className="grid grid-cols-2 gap-3">
              {[
                { status: 'running', label: '运行中', color: 'bg-green-900/50 text-green-300 border border-green-800', icon: '▶️' },
                { status: 'paused', label: '已暂停', color: 'bg-yellow-900/50 text-yellow-300 border border-yellow-800', icon: '⏸️' },
                { status: 'failed', label: '执行失败', color: 'bg-red-900/50 text-red-300 border border-red-800', icon: '❌' },
                { status: 'completed', label: '已完成', color: 'bg-gray-900/50 text-gray-300 border border-gray-800', icon: '✅' },
              ].map(item => {
                const count = scheduleTasks.data?.filter(task => task.status === item.status).length || 0;

                return (
                  <div key={item.status} className={`${item.color} p-4 rounded-lg text-center`}>
                    <div className="text-3xl mb-1">{item.icon}</div>
                    <p className="text-sm font-medium">{item.label}</p>
                    <p className="text-2xl font-bold mt-1">{count}</p>
                  </div>
                );
              })}
            </div>
          ) :
(
            <div className="p-4 bg-muted rounded-lg border">
              <p className="text-muted-foreground text-center">暂无任务数据</p>
            </div>
          )}
        </div>
      </div>

      {/* 创建调度任务表单 */}
      {showCreateForm && (
        <div className="bg-card rounded-xl shadow-sm border p-6 transition-all hover:shadow-md">
          <h2 className="text-2xl font-bold text-foreground mb-6 flex items-center">
            <span className="mr-3 p-2 bg-blue-900/30 text-blue-300 rounded-full border border-blue-800">
              ⚙️
            </span>
            创建调度任务
          </h2>

          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 基本信息 */}
              <div className="space-y-4">
                <div>
                  <label htmlFor="task-type" className="block text-sm font-medium text-muted-foreground mb-1">
                    任务类型 <span className="text-red-500">*</span>
                  </label>
                  <select
                    id="task-type"
                    value={scheduleTask.task_type}
                    onChange={e => setScheduleTask({ ...scheduleTask, task_type: e.target.value })}
                    className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground"
                  >
                    <option value="">选择任务类型</option>
                    <option value="general">通用任务</option>
                    <option value="code_quality">代码质量检查</option>
                    <option value="security">安全性检查</option>
                    <option value="performance">性能检查</option>
                    <option value="data_backup">数据备份</option>
                    <option value="report_generation">报告生成</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="schedule-type" className="block text-sm font-medium text-muted-foreground mb-1">
                    调度类型 <span className="text-red-500">*</span>
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { value: 'cron', label: 'Cron表达式', icon: '⏰' },
                      { value: 'interval', label: '固定间隔', icon: '🔄' },
                      { value: 'one-time', label: '一次性执行', icon: '📅' },
                    ].map(item => (
                      <button
                        key={item.value}
                        type="button"
                        onClick={() => setTaskType(item.value as any)}
                        className={`px-4 py-3 border rounded-lg transition-all ${taskType === item.value ? 'bg-cyber-cyan/20 border-cyber-cyan text-cyber-cyan' : 'bg-muted border text-muted-foreground hover:border-cyber-cyan/30'}`}
                      >
                        <div className="flex flex-col items-center">
                          <span className="text-xl mb-1">{item.icon}</span>
                          <span className="text-sm font-medium">{item.label}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label htmlFor="task-description" className="block text-sm font-medium text-muted-foreground mb-1">
                    任务描述 <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="task-description"
                    placeholder="输入任务描述，清晰说明任务的目的"
                    value={scheduleTask.description}
                    onChange={e => setScheduleTask({ ...scheduleTask, description: e.target.value })}
                    className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground placeholder-muted-foreground"
                  />
                </div>
              </div>

              {/* 调度配置 */}
              <div className="space-y-4">
                {/* 不同调度类型的配置 */}
                {taskType === 'cron' && (
                  <div>
                    <label htmlFor="cron-expression" className="block text-sm font-medium text-muted-foreground mb-1">
                      Cron表达式 <span className="text-red-500">*</span>
                    </label>
                    <select
                      id="cron-expression"
                      value={scheduleTask.cron_expression}
                      onChange={e => setScheduleTask({ ...scheduleTask, cron_expression: e.target.value })}
                      className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all font-mono text-sm bg-muted text-foreground"
                    >
                      <option value="">选择执行时间</option>
                      <option value="0 0 * * *">每天午夜 (00:00)</option>
                      <option value="0 6 * * *">每天早上6点 (06:00)</option>
                      <option value="0 12 * * *">每天中午12点 (12:00)</option>
                      <option value="0 18 * * *">每天晚上6点 (18:00)</option>
                      <option value="0 */4 * * *">每4小时一次</option>
                      <option value="0 */2 * * *">每2小时一次</option>
                      <option value="0 0 * * 1">每周一午夜</option>
                      <option value="0 0 1 * *">每月1号午夜</option>
                      <option value="custom">自定义...</option>
                    </select>
                    {scheduleTask.cron_expression === 'custom' && (
                      <input
                        className="mt-2 w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all font-mono text-sm bg-muted text-foreground placeholder-muted-foreground"
                        placeholder="输入自定义Cron表达式"
                        value={scheduleTask.customCron || ''}
                        onChange={e => setScheduleTask({ ...scheduleTask, customCron: e.target.value })}
                      />
                    )}
                    <p className="text-xs text-muted-foreground mt-1">格式: 分 时 日 月 周</p>
                  </div>
                )}

                {taskType === 'interval' && (
                  <div>
                    <label htmlFor="interval-seconds" className="block text-sm font-medium text-muted-foreground mb-1">
                      间隔时间 <span className="text-red-500">*</span>
                    </label>
                    <select
                      id="interval-seconds"
                      value={scheduleTask.interval_seconds}
                      onChange={e => setScheduleTask({ ...scheduleTask, interval_seconds: parseInt(e.target.value) || 0 })}
                      className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground"
                    >
                      <option value="">选择间隔时间</option>
                      <option value="60">1分钟</option>
                      <option value="300">5分钟</option>
                      <option value="600">10分钟</option>
                      <option value="1800">30分钟</option>
                      <option value="3600">1小时</option>
                      <option value="7200">2小时</option>
                      <option value="21600">6小时</option>
                      <option value="43200">12小时</option>
                      <option value="86400">24小时</option>
                      <option value="604800">7天</option>
                      <option value="custom">自定义...</option>
                    </select>
                    {scheduleTask.interval_seconds === 'custom' && (
                      <div className="mt-2 flex gap-3">
                        <input
                          type="number"
                          placeholder="输入秒数"
                          min="1"
                          value={scheduleTask.customInterval || ''}
                          onChange={e => setScheduleTask({ ...scheduleTask, customInterval: parseInt(e.target.value) || 0 })}
                          className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground placeholder-muted-foreground"
                        />
                        <span className="flex items-center text-muted-foreground">秒</span>
                      </div>
                    )}
                  </div>
                )}

                {taskType === 'one-time' && (
                  <div>
                    <label htmlFor="execute-time" className="block text-sm font-medium text-muted-foreground mb-1">
                      执行时间 <span className="text-red-500">*</span>
                    </label>
                    <select
                      id="execute-time"
                      value={scheduleTask.execute_time}
                      onChange={e => setScheduleTask({ ...scheduleTask, execute_time: e.target.value })}
                      className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground"
                    >
                      <option value="">选择执行时间</option>
                      <option value={new Date(Date.now() + 60000).toISOString().slice(0, 19)}>1分钟后</option>
                      <option value={new Date(Date.now() + 300000).toISOString().slice(0, 19)}>5分钟后</option>
                      <option value={new Date(Date.now() + 600000).toISOString().slice(0, 19)}>10分钟后</option>
                      <option value={new Date(Date.now() + 1800000).toISOString().slice(0, 19)}>30分钟后</option>
                      <option value={new Date(Date.now() + 3600000).toISOString().slice(0, 19)}>1小时后</option>
                      <option value={new Date(Date.now() + 7200000).toISOString().slice(0, 19)}>2小时后</option>
                      <option value={new Date(Date.now() + 21600000).toISOString().slice(0, 19)}>6小时后</option>
                      <option value={new Date(Date.now() + 86400000).toISOString().slice(0, 19)}>24小时后</option>
                      <option value="custom">自定义时间...</option>
                    </select>
                    {scheduleTask.execute_time === 'custom' && (
                      <input
                        className="mt-2 w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground placeholder-muted-foreground"
                        type="datetime-local"
                        value={scheduleTask.customExecuteTime || ''}
                        onChange={e => setScheduleTask({ ...scheduleTask, customExecuteTime: e.target.value })}
                      />
                    )}
                  </div>
                )}

                <div>
                  {/* 配置模式切换 */}
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                      配置方式
                    </label>
                    <div className="flex gap-3">
                      {[
                        { id: 'template', name: '📋 模板', label: '模板' },
                        { id: 'visual', name: '🎨 可视化', label: '可视化' },
                        { id: 'json', name: '📝 JSON', label: 'JSON' },
                      ].map(mode => (
                        <button
                          key={mode.id}
                          type="button"
                          onClick={() => setConfigMode(mode.id as any)}
                          className={`px-4 py-2 border rounded-lg transition-all ${configMode === mode.id ? 'bg-blue-900/30 text-blue-300 border-blue-800' : 'bg-muted border text-muted-foreground hover:border-cyber-cyan/30'}`}
                        >
                          <div className="flex items-center gap-1">
                            <span>{mode.name}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* 模板选择 */}
                  {configMode === 'template' && (
                    <div className="space-y-3">
                      <label htmlFor="template-select" className="block text-sm font-medium text-muted-foreground">
                        选择任务模板
                      </label>
                      <select
                        id="template-select"
                        value={selectedTemplate}
                        onChange={e => setSelectedTemplate(e.target.value)}
                        className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground"
                      >
                        {taskTemplates[scheduleTask.task_type as keyof typeof taskTemplates].map(template => (
                          <option key={template.id} value={template.id}>
                            {template.name}
                          </option>
                        ))}
                      </select>
                      <div className="mt-4 p-4 bg-blue-900/20 rounded-lg border border-blue-800">
                        <h4 className="text-sm font-semibold text-blue-300 mb-2">模板配置预览：</h4>
                        <pre className="text-xs font-mono text-foreground bg-muted/50 p-3 rounded border">
                          {JSON.stringify(visualConfig, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* 可视化配置表单 */}
                  {configMode === 'visual' && (
                    <div className="space-y-4">
                      {configFields[scheduleTask.task_type as keyof typeof configFields].map(field => {
                        // 获取当前值或默认值
                        const currentValue = visualConfig[field.name] !== undefined ?
                          visualConfig[field.name] :
                          field.defaultValue;

                        return (
                          <div key={field.name} className="space-y-1">
                            <label className="block text-sm font-medium text-muted-foreground">
                              {field.label} {field.required && <span className="text-red-500">*</span>}
                            </label>

                            {/* 文本输入 */}
                            {field.type === 'text' && (
                              <input
                                type="text"
                                value={currentValue || ''}
                                onChange={e => handleVisualConfigChange(field.name, e.target.value)}
                                placeholder={field.defaultValue.toString() || ''}
                                className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground placeholder-muted-foreground"
                              />
                            )}

                            {/* 数字输入 */}
                            {field.type === 'number' && (
                              <input
                                type="number"
                                value={currentValue || ''}
                                onChange={e => handleVisualConfigChange(field.name, e.target.value ? parseInt(e.target.value) : undefined)}
                                min={field.min}
                                max={field.max}
                                placeholder={field.defaultValue.toString() || ''}
                                className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground placeholder-muted-foreground"
                              />
                            )}

                            {/* 复选框 */}
                            {field.type === 'checkbox' && (
                              <div className="flex items-center">
                                <input
                                  type="checkbox"
                                  checked={currentValue || false}
                                  onChange={e => handleVisualConfigChange(field.name, e.target.checked)}
                                  className="mr-3 h-4 w-4 rounded border bg-muted text-blue-600 focus:ring-blue-500"
                                />
                                <span className="text-sm text-foreground">{field.label}</span>
                              </div>
                            )}

                            {/* 下拉选择 */}
                            {field.type === 'select' && (
                              <select
                                value={currentValue || ''}
                                onChange={e => handleVisualConfigChange(field.name, e.target.value)}
                                className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-muted text-foreground"
                              >
                                {field.options?.map(option => (
                                  <option key={option.value} value={option.value}>
                                    {option.label}
                                  </option>
                                ))}
                              </select>
                            )}
                          </div>
                        );
                      })}

                      {/* 配置预览 */}
                      <div className="mt-4 p-4 bg-blue-900/20 rounded-lg border border-blue-800">
                        <h4 className="text-sm font-semibold text-blue-300 mb-2">配置预览：</h4>
                        <pre className="text-xs font-mono text-foreground bg-muted/50 p-3 rounded border">
                          {JSON.stringify(visualConfig, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* JSON配置输入 */}
                  {configMode === 'json' && (
                    <div className="space-y-3">
                      <label htmlFor="task-config" className="block text-sm font-medium text-muted-foreground">
                        JSON配置
                      </label>
                      <textarea
                        id="task-config"
                        placeholder='例如: {"param1": "value1", "param2": 123}'
                        value={scheduleTask.config}
                        onChange={e => handleJsonConfigChange(e.target.value)}
                        className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all min-h-[150px] font-mono text-sm bg-muted text-foreground placeholder-muted-foreground"
                      />
                      <div className="flex items-center gap-2">
                        <p className="text-xs text-muted-foreground">请输入有效的JSON格式配置</p>
                        {/* JSON格式验证 */}
                        {(() => {
                          try {
                            JSON.parse(scheduleTask.config);
                            return <span className="text-xs text-green-500">✓ 格式正确</span>;
                          } catch (error) {
                            return <span className="text-xs text-red-500">✗ 格式错误</span>;
                          }
                        })()}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* 提交按钮 */}
            <div className="flex items-center justify-end gap-4 pt-6 border-t">
              <button
                onClick={() => {
                  setShowCreateForm(false);
                  resetForm();
                }}
                className="px-5 py-3 bg-muted text-muted-foreground hover:bg-muted/80 rounded-lg transition-colors border"
              >
                取消
              </button>
              <button
                onClick={handleSubmitTask}
                disabled={
                  createCronTaskMutation.isPending ||
                  createIntervalTaskMutation.isPending ||
                  createOneTimeTaskMutation.isPending
                }
                className="px-6 py-3 bg-blue-900 text-white hover:bg-blue-800 rounded-lg transition-all hover:shadow-md transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none border border-blue-800"
              >
                {createCronTaskMutation.isPending ||
                 createIntervalTaskMutation.isPending ||
                 createOneTimeTaskMutation.isPending ?
(
                  <div className="flex items-center">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                    创建中...
                  </div>
                ) :
(
                  <>
                    <span className="mr-2">🚀</span>
                    立即创建
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 调度任务列表 */}
      <div className="bg-card rounded-xl shadow-sm border p-6 transition-all hover:shadow-md">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-foreground flex items-center">
            <span className="mr-3 p-2 bg-purple-900/30 text-purple-300 rounded-full border border-purple-800">
              📋
            </span>
            调度任务列表
          </h2>
          <div className="text-sm text-muted-foreground">
            {scheduleTasks?.success && scheduleTasks.data?.length > 0 && (
              <span>共 {scheduleTasks.data?.length} 个任务</span>
            )}
          </div>
        </div>

        {isTasksLoading ?
(
          <div className="flex justify-center items-center py-12">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
              <p className="text-muted-foreground">加载调度任务列表中...</p>
            </div>
          </div>
        ) :
tasksError ?
(
          <div className="p-8 bg-red-900/30 border border-red-800 rounded-lg text-center text-red-300">
            <div className="text-4xl mb-4">😢</div>
            <h3 className="font-semibold text-red-800 text-lg mb-2">加载失败</h3>
            <p className="text-red-700">{tasksError.message}</p>
          </div>
        ) :
!scheduleTasks?.success ?
(
          <div className="p-8 bg-red-900/30 border border-red-800 rounded-lg text-center text-red-300">
            <div className="text-4xl mb-4">😢</div>
            <h3 className="font-semibold text-red-800 text-lg mb-2">加载失败</h3>
            <p className="text-red-700">{scheduleTasks?.error}</p>
          </div>
        ) :
scheduleTasks.data?.length === 0 ?
(
          <div className="p-8 bg-yellow-900/30 border border-yellow-800 rounded-lg text-center text-yellow-300">
            <div className="text-4xl mb-4">📝</div>
            <h3 className="font-semibold text-yellow-800 text-lg mb-2">无调度任务</h3>
            <p className="text-yellow-700 mb-4">当前没有调度任务</p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-colors"
            >
              <span className="mr-2">+</span>
              创建第一个任务
            </button>
          </div>
        ) :
(
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-card text-foreground">
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">
                    <div className="flex items-center">
                      <input type="checkbox" className="mr-2 h-4 w-4 rounded border bg-muted text-blue-600 focus:ring-blue-500" />
                      任务ID
                    </div>
                  </th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">任务类型</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">描述</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">状态</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">调度类型</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">调度规则</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">上次执行</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">下次执行</th>
                  <th className="px-6 py-4 text-sm font-semibold text-muted-foreground border-b">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y bg-card">
                {scheduleTasks.data?.map(task => (
                  <tr key={task.task_id} className="hover:bg-muted/50 transition-colors text-foreground">
                    <td className="px-6 py-4 font-mono text-sm text-muted-foreground">
                      <div className="flex items-center">
                        <input type="checkbox" className="mr-3 h-4 w-4 rounded border bg-muted text-blue-600 focus:ring-blue-500" />
                        <span className="truncate max-w-[120px]">{task.task_id}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${task.task_type === 'general' ?
'bg-blue-900/50 text-blue-300 border border-blue-800' :
                                                                          task.task_type === 'code_quality' ?
'bg-green-900/50 text-green-300 border border-green-800' :
                                                                          task.task_type === 'security' ?
'bg-red-900/50 text-red-300 border border-red-800' :
                                                                          task.task_type === 'performance' ?
'bg-purple-900/50 text-purple-300 border border-purple-800' :
                                                                          task.task_type === 'data_backup' ?
'bg-yellow-900/50 text-yellow-300 border border-yellow-800' :
                                                                          'bg-indigo-900/50 text-indigo-300 border border-indigo-800'}`}>
                        {task.task_type === 'general' && '通用任务'}
                        {task.task_type === 'code_quality' && '代码质量'}
                        {task.task_type === 'security' && '安全性检查'}
                        {task.task_type === 'performance' && '性能检查'}
                        {task.task_type === 'data_backup' && '数据备份'}
                        {task.task_type === 'report_generation' && '报告生成'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="max-w-[250px]">
                        <p className="font-medium text-foreground truncate">{task.description}</p>
                        <p className="text-xs text-muted-foreground mt-1">回调: {task.callback.split('://')[1] || task.callback}</p>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center">
                        <span className={`w-2 h-2 rounded-full mr-2 ${task.status === 'running' ?
'bg-green-500 animate-pulse' :
                                                                      task.status === 'paused' ?
'bg-yellow-500' :
                                                                      task.status === 'failed' ?
'bg-red-500' :
                                                                      'bg-gray-500'}`}></span>
                        <span className={`text-sm font-medium ${task.status === 'running' ?
'text-green-400' :
                                                              task.status === 'paused' ?
'text-yellow-400' :
                                                              task.status === 'failed' ?
'text-red-400' :
                                                              'text-gray-400'}`}>
                          {task.status === 'running' && '运行中'}
                          {task.status === 'paused' && '已暂停'}
                          {task.status === 'failed' && '执行失败'}
                          {task.status === 'completed' && '已完成'}
                          {task.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 bg-muted text-muted-foreground border rounded-full text-xs">
                        {task.cron_expression && 'Cron'}
                        {task.interval_seconds && '间隔'}
                        {task.execute_time && !task.cron_expression && !task.interval_seconds && '一次性'}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-xs text-muted-foreground max-w-[180px] truncate">
                      {task.cron_expression ||
                       (task.interval_seconds && `${task.interval_seconds}秒`) ||
                       (task.execute_time && new Date(task.execute_time).toLocaleString())}
                    </td>
                    <td className="px-6 py-4 text-sm text-muted-foreground">
                      {task.last_executed ?
(
                        <div>
                          <p>{new Date(task.last_executed).toLocaleString()}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {Math.round((Date.now() - new Date(task.last_executed).getTime()) / 60000)} 分钟前
                          </p>
                        </div>
                      ) :
(
                        <span className="text-muted-foreground">从未执行</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm text-muted-foreground">
                      {task.next_execution ?
(
                        <div>
                          <p>{new Date(task.next_execution).toLocaleString()}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {Math.round((new Date(task.next_execution).getTime() - Date.now()) / 60000)} 分钟后
                          </p>
                        </div>
                      ) :
(
                        <span className="text-muted-foreground">无</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button
                          onClick={() => removeTaskMutation.mutate(task.task_id)}
                          disabled={removeTaskMutation.isPending}
                          className="px-3 py-1 bg-red-900/50 text-red-300 hover:bg-red-800/50 rounded-lg text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-red-800"
                        >
                          {removeTaskMutation.isPending ?
(
                            <div className="w-3 h-3 border-2 border-red-500 border-t-transparent rounded-full animate-spin"></div>
                          ) :
(
                            <span>🗑️ 移除</span>
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default ScheduleService;
