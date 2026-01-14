import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

interface FineTuneModel {
  model_id: string
  name: string
  description: string
  parameters: string
  supports_fine_tuning: boolean
}

interface FineTuneTask {
  task_id: string
  model_id: string
  status: string
  progress: number
  created_at: string
  completed_at: string | null
  metrics: {
    loss?: number
    accuracy?: number
    current_loss?: number
    current_accuracy?: number
    best_loss?: number
    best_accuracy?: number
  } | null
  config?: {
    learning_rate: number
    batch_size: number
    num_epochs: number
  }
  logs?: Array<{
    timestamp: string
    message: string
  }>
}

const FineTuning: React.FC = () => {
  const queryClient = useQueryClient();

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);

  const [fineTuneConfig, setFineTuneConfig] = useState({
    model_id: 'gpt-3.5-turbo',
    dataset_path: '',
    customDatasetPath: '',
    learning_rate: 1e-5,
    batch_size: 16,
    num_epochs: 3,
    warmup_steps: 100,
    weight_decay: 0.01,
    gradient_accumulation_steps: 1,
    max_grad_norm: 1.0,
    lora_r: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
  });

  const {
    data: models,
    isLoading: isModelsLoading,
  } = useQuery<ApiResponse<FineTuneModel[]>>({
    queryKey: ['fineTuneModels'],
    queryFn: async () => apiClient.getFineTuneModels(),
  });

  const {
    data: tasks,
    isLoading: isTasksLoading,
    refetch: refetchTasks,
  } = useQuery<ApiResponse<FineTuneTask[]>>({
    queryKey: ['fineTuneTasks'],
    queryFn: async () => apiClient.getFineTuneTasks(),
  });

  const {
    data: taskDetail,
    isLoading: isDetailLoading,
    refetch: refetchTaskDetail,
  } = useQuery<ApiResponse<FineTuneTask>>({
    queryKey: ['fineTuneTask', selectedTaskId],
    queryFn: async () => apiClient.getFineTuneTask(selectedTaskId!),
    enabled: Boolean(selectedTaskId),
  });

  const startFineTuneMutation = useMutation<ApiResponse<any>, Error, typeof fineTuneConfig>({
    mutationFn: async config => apiClient.startFineTune({
      model_id: config.model_id,
      dataset_path: config.dataset_path,
      config: {
        learning_rate: config.learning_rate,
        batch_size: config.batch_size,
        num_epochs: config.num_epochs,
        warmup_steps: config.warmup_steps,
        weight_decay: config.weight_decay,
        gradient_accumulation_steps: config.gradient_accumulation_steps,
        max_grad_norm: config.max_grad_norm,
        lora_r: config.lora_r,
        lora_alpha: config.lora_alpha,
        lora_dropout: config.lora_dropout,
      },
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fineTuneTasks'] });
      setShowCreateForm(false);
      resetForm();
      alert('微调任务已开始');
    },
    onError: error => {
      alert(`启动微调任务失败: ${error.message}`);
    },
  });

  const deleteTaskMutation = useMutation<ApiResponse<any>, Error, string>({
    mutationFn: async taskId => apiClient.deleteFineTuneTask(taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fineTuneTasks'] });
      setSelectedTaskId(null);
      alert('微调任务已删除');
    },
    onError: error => {
      alert(`删除微调任务失败: ${error.message}`);
    },
  });

  const stopTaskMutation = useMutation<ApiResponse<any>, Error, string>({
    mutationFn: async taskId => apiClient.stopFineTuning(taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fineTuneTasks'] });
      if (selectedTaskId) {
        refetchTaskDetail();
      }
      alert('微调任务已停止');
    },
    onError: error => {
      alert(`停止微调任务失败: ${error.message}`);
    },
  });

  // 微调所有智能体的mutation
  const fineTuneAllAgentsMutation = useMutation<ApiResponse<any>, Error>({
    mutationFn: async () => apiClient.fineTuneAllAgents(),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['fineTuneTasks'] });
      if (data.success && data.data && data.data.tasks) {
        alert(`已为 ${data.data.tasks.length} 个智能体创建微调任务`);
      } else {
        alert('已为智能体创建微调任务');
      }
    },
    onError: error => {
      alert(`对所有智能体进行微调失败: ${error.message || '未知错误'}`);
    },
  });

  const handleFineTuneAllAgents = () => {
    if (confirm('确定要对所有智能体进行微调吗？这将为每个智能体创建一个微调任务。')) {
      fineTuneAllAgentsMutation.mutate();
    }
  };

  const handleStartFineTune = () => {
    if (!fineTuneConfig.dataset_path) {
      alert('请选择数据集路径');
      return;
    }

    startFineTuneMutation.mutate(fineTuneConfig);
  };

  const handleDeleteTask = (taskId: string) => {
    if (confirm('确定要删除这个微调任务吗？')) {
      deleteTaskMutation.mutate(taskId);
    }
  };

  const handleStopTask = (taskId: string) => {
    if (confirm('确定要停止这个微调任务吗？')) {
      stopTaskMutation.mutate(taskId);
    }
  };

  const resetForm = () => {
    setFineTuneConfig({
      model_id: 'gpt-3.5-turbo',
      dataset_path: '',
      customDatasetPath: '',
      learning_rate: 1e-5,
      batch_size: 16,
      num_epochs: 3,
      warmup_steps: 100,
      weight_decay: 0.01,
      gradient_accumulation_steps: 1,
      max_grad_norm: 1.0,
      lora_r: 8,
      lora_alpha: 16,
      lora_dropout: 0.05,
    });
  };

  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running':
        return '运行中';
      case 'completed':
        return '已完成';
      case 'failed':
        return '失败';
      case 'pending':
        return '等待中';
      default:
        return status;
    }
  };

  return (
    <div className="space-y-6 p-4 md:p-6 bg-background text-foreground">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-foreground">AI模型微调</h1>
        <div className="flex gap-4">
          <button
            onClick={handleFineTuneAllAgents}
            disabled={fineTuneAllAgentsMutation.isPending}
            className="px-4 py-2 bg-green-600 text-white hover:bg-green-700 rounded-md transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {fineTuneAllAgentsMutation.isPending ? '正在创建...' : '微调所有智能体'}
          </button>
          <button
              onClick={() => setShowCreateForm(!showCreateForm)}
              className="px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90 rounded-md transition-colors"
            >
              {showCreateForm ? '取消' : '创建微调任务'}
            </button>
        </div>
      </div>

      {showCreateForm && (
        <div className="bg-card p-6 rounded-lg shadow-md border">
          <h2 className="text-xl font-semibold mb-4 text-foreground">创建微调任务</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="model-select" className="block text-sm font-medium text-foreground mb-1">
                选择模型 <span className="text-red-500">*</span>
              </label>
              <select
                id="model-select"
                value={fineTuneConfig.model_id}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, model_id: e.target.value })}
                className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {isModelsLoading ?
(
                  <option value="">加载中...</option>
                ) :
models?.success && models.data ?
(
                  models.data.map(model => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.name} ({model.parameters})
                    </option>
                  ))
                ) :
null}
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="dataset-path" className="block text-sm font-medium text-foreground mb-1">
                数据集路径 <span className="text-red-500">*</span>
              </label>
              <select
                id="dataset-path"
                value={fineTuneConfig.dataset_path}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, dataset_path: e.target.value })}
                className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">选择数据集</option>
                <option value="./data/training">训练数据集 (./data/training)</option>
                <option value="./data/custom">自定义数据集 (./data/custom)</option>
                <option value="./data/qa">问答数据集 (./data/qa)</option>
                <option value="custom">自定义路径...</option>
              </select>
              {fineTuneConfig.dataset_path === 'custom' && (
                <input
                  className="mt-2 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="输入自定义数据集路径"
                  value={fineTuneConfig.customDatasetPath || ''}
                  onChange={e => setFineTuneConfig({ ...fineTuneConfig, customDatasetPath: e.target.value })}
                />
              )}
            </div>

            <div className="space-y-2">
              <label htmlFor="learning-rate" className="block text-sm font-medium text-foreground mb-1">
                学习率
              </label>
              <select
                id="learning-rate"
                value={fineTuneConfig.learning_rate}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, learning_rate: parseFloat(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1e-6">1e-6 (极小)</option>
                <option value="1e-5">1e-5 (小)</option>
                <option value="5e-5">5e-5 (较小)</option>
                <option value="1e-4">1e-4 (中等)</option>
                <option value="5e-4">5e-4 (较大)</option>
                <option value="1e-3">1e-3 (大)</option>
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="batch-size" className="block text-sm font-medium text-foreground mb-1">
                批次大小
              </label>
              <select
                id="batch-size"
                value={fineTuneConfig.batch_size}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, batch_size: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="4">4 (小)</option>
                <option value="8">8 (较小)</option>
                <option value="16">16 (中等)</option>
                <option value="32">32 (较大)</option>
                <option value="64">64 (大)</option>
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="num-epochs" className="block text-sm font-medium text-foreground mb-1">
                训练轮数
              </label>
              <select
                id="num-epochs"
                value={fineTuneConfig.num_epochs}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, num_epochs: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1">1 轮</option>
                <option value="2">2 轮</option>
                <option value="3">3 轮</option>
                <option value="5">5 轮</option>
                <option value="10">10 轮</option>
                <option value="20">20 轮</option>
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="lora-r" className="block text-sm font-medium text-foreground mb-1">
                LoRA Rank (r)
              </label>
              <select
                id="lora-r"
                value={fineTuneConfig.lora_r}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, lora_r: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="4">4 (低)</option>
                <option value="8">8 (中等)</option>
                <option value="16">16 (高)</option>
                <option value="32">32 (极高)</option>
                <option value="64">64 (超高)</option>
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="lora-alpha" className="block text-sm font-medium text-foreground mb-1">
                LoRA Alpha
              </label>
              <select
                id="lora-alpha"
                value={fineTuneConfig.lora_alpha}
                onChange={e => setFineTuneConfig({ ...fineTuneConfig, lora_alpha: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="8">8 (低)</option>
                <option value="16">16 (中等)</option>
                <option value="32">32 (高)</option>
                <option value="64">64 (极高)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center justify-end space-x-2 mt-4">
            <button
              onClick={resetForm}
              className="px-4 py-2 bg-muted text-foreground hover:bg-muted/80 rounded-md transition-colors"
            >
              重置
            </button>
            <button
              onClick={handleStartFineTune}
              disabled={startFineTuneMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {startFineTuneMutation.isPending ?
(
                <>
                  <span className="mr-2">⏳</span>
                  启动中...
                </>
              ) :
(
                '开始微调'
              )}
            </button>
          </div>
        </div>
      )}

      <div className="bg-card p-6 rounded-lg shadow-md border">
        <h2 className="text-xl font-semibold mb-4">微调任务列表</h2>
        {isTasksLoading ?
(
          <div className="flex justify-center items-center py-8">
            <span className="text-xl">⏳</span>
            <span className="ml-2">加载任务列表中...</span>
          </div>
        ) :
!tasks?.success ?
(
          <div className="p-4 bg-red-100 border border-red-300 rounded-md text-red-800">
            <h3 className="font-semibold">加载失败</h3>
            <p>{tasks?.error}</p>
          </div>
        ) :
tasks.data?.length === 0 ?
(
          <div className="p-4 bg-yellow-100 border border-yellow-300 rounded-md text-yellow-800">
            <h3 className="font-semibold">无微调任务</h3>
            <p>当前没有微调任务，请先创建一个。</p>
          </div>
        ) :
(
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted border-b">
                  <th className="px-4 py-2 text-left">任务ID</th>
                  <th className="px-4 py-2 text-left">模型</th>
                  <th className="px-4 py-2 text-left">状态</th>
                  <th className="px-4 py-2 text-left">进度</th>
                  <th className="px-4 py-2 text-left">创建时间</th>
                  <th className="px-4 py-2 text-left">完成时间</th>
                  <th className="px-4 py-2 text-left">指标</th>
                  <th className="px-4 py-2 text-left">操作</th>
                </tr>
              </thead>
              <tbody>
                {tasks.data?.map(task => (
                  <tr key={task.task_id} className="hover:bg-gray-50 border-b">
                    <td className="px-4 py-2 font-mono text-sm">{task.task_id}</td>
                    <td className="px-4 py-2">{task.model_id}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded-full text-xs ${getStatusStyle(task.status)}`}>
                        {getStatusText(task.status)}
                      </span>
                    </td>
                    <td className="px-4 py-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all"
                          style={{ width: `${task.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground">{task.progress.toFixed(1)}%</span>
                    </td>
                    <td className="px-4 py-2 text-sm">
                      {new Date(task.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-2 text-sm">
                      {task.completed_at ? new Date(task.completed_at).toLocaleString() : '-'}
                    </td>
                    <td className="px-4 py-2">
                      {task.metrics ?
(
                        <div className="text-xs">
                          <div>Loss: {task.metrics.loss?.toFixed(4) || '-'}</div>
                          <div>Accuracy: {task.metrics.accuracy?.toFixed(4) || '-'}</div>
                        </div>
                      ) :
'-'}
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex gap-2">
                        <button
                          onClick={() => setSelectedTaskId(task.task_id)}
                          className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded-md text-sm transition-colors"
                        >
                          详情
                        </button>
                        {task.status === 'running' && (
                          <button
                            onClick={() => handleStopTask(task.task_id)}
                            className="px-3 py-1 bg-yellow-200 hover:bg-yellow-300 rounded-md text-sm transition-colors"
                          >
                            停止
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteTask(task.task_id)}
                          className="px-3 py-1 bg-red-200 hover:bg-red-300 rounded-md text-sm transition-colors"
                        >
                          删除
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

      {selectedTaskId && taskDetail?.data && (
        <div className="bg-card p-6 rounded-lg shadow-md border">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">任务详情</h2>
            <button
              onClick={() => setSelectedTaskId(null)}
              className="px-4 py-2 bg-muted text-foreground hover:bg-muted/80 rounded-md transition-colors"
            >
              关闭
            </button>
          </div>

          {isDetailLoading ?
(
            <div className="flex justify-center items-center py-8">
              <span className="text-xl">⏳</span>
              <span className="ml-2">加载详情中...</span>
            </div>
          ) :
(
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold text-foreground mb-2">基本信息</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="text-muted-foreground">任务ID:</span> {taskDetail.data.task_id}</div>
                    <div><span className="text-muted-foreground">模型ID:</span> {taskDetail.data.model_id}</div>
                    <div><span className="text-muted-foreground">状态:</span>
                      <span className={`px-2 py-1 rounded-full text-xs ${getStatusStyle(taskDetail.data.status)}`}>
                        {getStatusText(taskDetail.data.status)}
                      </span>
                    </div>
                    <div><span className="text-muted-foreground">进度:</span> {taskDetail.data.progress.toFixed(1)}%</div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold text-foreground mb-2">训练配置</h3>
                  {taskDetail.data.config && (
                    <div className="space-y-1 text-sm">
                      <div><span className="text-muted-foreground">学习率:</span> {taskDetail.data.config.learning_rate}</div>
                      <div><span className="text-muted-foreground">批次大小:</span> {taskDetail.data.config.batch_size}</div>
                      <div><span className="text-muted-foreground">训练轮数:</span> {taskDetail.data.config.num_epochs}</div>
                    </div>
                  )}
                </div>
              </div>

              {taskDetail.data.metrics && (
                <div>
                  <h3 className="font-semibold text-foreground mb-2">训练指标</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">当前Loss</div>
                      <div className="text-lg font-bold text-blue-900">{taskDetail.data.metrics.current_loss?.toFixed(4) || '-'}</div>
                    </div>
                    <div className="bg-green-50 p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">当前准确率</div>
                      <div className="text-lg font-bold text-green-900">{taskDetail.data.metrics.current_accuracy?.toFixed(4) || '-'}</div>
                    </div>
                    <div className="bg-yellow-50 p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">最佳Loss</div>
                      <div className="text-lg font-bold text-yellow-900">{taskDetail.data.metrics.best_loss?.toFixed(4) || '-'}</div>
                    </div>
                    <div className="bg-purple-50 p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">最佳准确率</div>
                      <div className="text-lg font-bold text-purple-900">{taskDetail.data.metrics.best_accuracy?.toFixed(4) || '-'}</div>
                    </div>
                  </div>
                </div>
              )}

              {taskDetail.data.logs && taskDetail.data.logs.length > 0 && (
                <div>
                  <h3 className="font-semibold text-foreground mb-2">训练日志</h3>
                  <div className="bg-gray-50 p-4 rounded-md max-h-60 overflow-y-auto">
                    {taskDetail.data.logs.map((log, index) => (
                      <div key={index} className="text-sm py-1 border-b border-gray-200">
                        <span className="text-muted-foreground mr-2">{new Date(log.timestamp).toLocaleTimeString()}</span>
                        <span>{log.message}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FineTuning;
