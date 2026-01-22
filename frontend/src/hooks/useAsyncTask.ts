/**
 * 异步任务管理Hook
 * 用于处理前后端协同的异步任务，实现异步任务提交、状态查询和结果获取
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '../services/api';
import { useDebounceFn } from './useDebounce';

// 定义任务状态
export type AsyncTaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

// 定义任务类型
export type AsyncTaskType =
  | 'attention_visualization'
  | 'transfer_matrix_generation'
  | 'attribution_analysis'
  | 'feature_clustering'
  | 'model_training'
  | 'other'

// 定义任务配置
export interface AsyncTaskConfig {

  /** 任务类型 */
  type: AsyncTaskType

  /** 任务参数 */
  params: any

  /** 轮询间隔，单位毫秒 */
  pollingInterval?: number

  /** 最大轮询次数 */
  maxPollingAttempts?: number
}

// 定义任务信息
export interface AsyncTaskInfo {

  /** 任务ID */
  taskId: string

  /** 任务类型 */
  type: AsyncTaskType

  /** 任务状态 */
  status: AsyncTaskStatus

  /** 任务进度，0-100 */
  progress: number

  /** 任务创建时间 */
  createdAt: string

  /** 任务更新时间 */
  updatedAt: string

  /** 错误信息 */
  error?: string

  /** 结果数据 */
  result?: any
}

// 定义后端负载信息
export interface BackendLoadInfo {

  /** CPU使用率，0-100 */
  cpuUsage: number

  /** 内存使用率，0-100 */
  memoryUsage: number

  /** GPU使用率，0-100 */
  gpuUsage?: number

  /** 任务队列长度 */
  taskQueueLength: number

  /** 活跃连接数 */
  activeConnections: number

  /** 服务状态 */
  status: 'healthy' | 'degraded' | 'critical'
}

/**
 * 异步任务管理Hook
 */
export function useAsyncTask() {
  const [activeTask, setActiveTask] = useState<AsyncTaskInfo | null>(null);
  const [backendLoad, setBackendLoad] = useState<BackendLoadInfo | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [pollingAttempts, setPollingAttempts] = useState(0);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const maxPollingAttemptsRef = useRef<number>(50); // 默认最大轮询50次
  const pollingIntervalRef = useRef<number>(2000); // 默认2秒轮询一次

  // 清理轮询
  const clearPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
      setIsPolling(false);
      setPollingAttempts(0);
    }
  }, []);

  // 组件卸载时清理轮询
  useEffect(() => () => {
      clearPolling();
    }, [clearPolling]);

  // 查询后端负载状态
  const checkBackendLoad = useCallback(async () => {
    try {
      const response = await apiClient.getBackendLoad();
      if (response.success && response.data) {
        setBackendLoad(response.data);
        return response.data;
      }
    } catch (error) {
      console.error('Failed to check backend load:', error);
    }
    return null;
  }, []);

  // 定期检查后端负载
  useEffect(() => {
    // 初始检查
    checkBackendLoad();

    // 每30秒检查一次后端负载
    const loadCheckInterval = setInterval(() => {
      checkBackendLoad();
    }, 30000);

    return () => {
      clearInterval(loadCheckInterval);
    };
  }, [checkBackendLoad]);

  // 提交异步任务
  const submitTask = useCallback(async (
    config: AsyncTaskConfig,
  ): Promise<string | null> => {
    try {
      // 检查后端负载
      const loadInfo = await checkBackendLoad();

      // 如果后端负载过高，拒绝提交新任务
      if (loadInfo && loadInfo.status === 'critical') {
        throw new Error('Backend is under critical load, please try again later');
      }

      // 提交任务
      const response = await apiClient.post<{ task_id: string }>('/api/tasks', {
        task_type: config.type,
        task_params: config.params,
      });

      if (response.success && response.data?.task_id) {
        // 保存任务信息
        const taskInfo: AsyncTaskInfo = {
          taskId: response.data.task_id,
          type: config.type,
          status: 'pending',
          progress: 0,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        setActiveTask(taskInfo);

        // 保存配置
        maxPollingAttemptsRef.current = config.maxPollingAttempts || 50;
        pollingIntervalRef.current = config.pollingInterval || 2000;

        return response.data.task_id;
      }
    } catch (error) {
      console.error('Failed to submit task:', error);
      throw error;
    }
    return null;
  }, [checkBackendLoad]);

  // 查询任务状态
  const queryTaskStatus = useCallback(async (
    taskId: string,
  ): Promise<AsyncTaskInfo | null> => {
    try {
      const response = await apiClient.get<AsyncTaskInfo>(`/api/tasks/${taskId}`);
      if (response.success && response.data) {
        setActiveTask(response.data);
        return response.data;
      }
    } catch (error) {
      console.error(`Failed to query task status for ${taskId}:`, error);
    }
    return null;
  }, []);

  // 获取任务结果
  const getTaskResult = useCallback(async (
    taskId: string,
  ): Promise<any | null> => {
    try {
      const response = await apiClient.get<any>(`/api/tasks/${taskId}/result`);
      if (response.success && response.data) {
        return response.data;
      }
    } catch (error) {
      console.error(`Failed to get task result for ${taskId}:`, error);
    }
    return null;
  }, []);

  // 取消任务
  const cancelTask = useCallback(async (
    taskId: string,
  ): Promise<boolean> => {
    try {
      const response = await apiClient.post<any>(`/api/tasks/${taskId}/cancel`);
      if (response.success) {
        // 更新任务状态
        setActiveTask(prev => prev && {
          ...prev,
          status: 'cancelled',
          updatedAt: new Date().toISOString(),
        });
        clearPolling();
        return true;
      }
    } catch (error) {
      console.error(`Failed to cancel task ${taskId}:`, error);
    }
    return false;
  }, [clearPolling]);

  // 轮询任务状态
  const startPolling = useCallback((taskId: string) => {
    // 清理之前的轮询
    clearPolling();
    setIsPolling(true);
    setPollingAttempts(0);

    // 定义轮询函数
    const pollTaskStatus = async () => {
      setPollingAttempts(prev => {
        const newAttempts = prev + 1;

        // 检查是否超过最大轮询次数
        if (newAttempts > maxPollingAttemptsRef.current) {
          clearPolling();
          return newAttempts;
        }

        // 查询任务状态
        queryTaskStatus(taskId).then(taskInfo => {
          if (taskInfo) {
            // 如果任务已完成或失败，停止轮询
            if (taskInfo.status === 'completed' || taskInfo.status === 'failed' || taskInfo.status === 'cancelled') {
              clearPolling();
            }
          }
        });

        return newAttempts;
      });
    };

    // 立即执行一次轮询
    pollTaskStatus();

    // 设置轮询定时器
    pollingRef.current = setInterval(pollTaskStatus, pollingIntervalRef.current);
  }, [queryTaskStatus, clearPolling]);

  // 提交并跟踪任务
  const submitAndTrackTask = useCallback(async (
    config: AsyncTaskConfig,
  ): Promise<string | null> => {
    const taskId = await submitTask(config);
    if (taskId) {
      startPolling(taskId);
    }
    return taskId;
  }, [submitTask, startPolling]);

  // 防抖的任务提交，避免频繁提交相同任务
  const debouncedSubmitTask = useDebounceFn(
    submitAndTrackTask,
    300, // 300ms防抖
  );

  return {
    // 任务信息
    activeTask,
    backendLoad,
    isPolling,
    pollingAttempts,

    // 任务操作
    submitTask,
    submitAndTrackTask,
    debouncedSubmitTask,
    queryTaskStatus,
    getTaskResult,
    cancelTask,
    startPolling,
    stopPolling: clearPolling,

    // 负载相关
    checkBackendLoad,
    isBackendHealthy: backendLoad?.status !== 'critical',
    isBackendDegraded: backendLoad?.status === 'degraded',
    isBackendCritical: backendLoad?.status === 'critical',
  };
}

export default useAsyncTask;
