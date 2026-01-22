/**
 * 注意力计算Web Worker Hook
 * 用于在React组件中方便地使用注意力计算Worker
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// 定义消息类型
export type AttentionWorkerMessageType =
  | 'normalize'
  | 'standardize'
  | 'clip'
  | 'similarity'
  | 'gradient_attribution'
  | 'weight_statistics'
  | 'transfer_matrix_processing'
  | 'pca_reduction'
  | 'tsne_reduction'
  | 'clustering'
  | 'correlation_analysis'

// 定义Worker状态
type WorkerStatus = 'idle' | 'running' | 'error' | 'terminated'

// 定义Worker请求配置
export interface AttentionWorkerRequest {
  type: AttentionWorkerMessageType
  data: any
}

// 定义Worker响应
export interface AttentionWorkerResponse {
  result: any
  error?: string
}

/**
 * 注意力计算Worker Hook
 */
export function useAttentionWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState<WorkerStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const callbacksRef = useRef<Map<string,(response: AttentionWorkerResponse) => void>>(new Map());
  const messageIdRef = useRef<number>(0);

  // 初始化Worker
  useEffect(() => {
    // 创建Worker实例
    const worker = new Worker(new URL('../workers/attention-calculator.worker.ts', import.meta.url), {
      type: 'module',
    });

    workerRef.current = worker;
    setStatus('idle');
    setError(null);

    // 处理Worker消息
    worker.onmessage = (event: MessageEvent<any>) => {
      const { id, result, error } = event.data;
      const callback = callbacksRef.current.get(id);

      if (callback) {
        callback({ result, error });
        callbacksRef.current.delete(id);
      }

      // 如果没有待处理的回调，设置为idle状态
      if (callbacksRef.current.size === 0) {
        setStatus('idle');
      }
    };

    // 处理Worker错误
    worker.onerror = event => {
      console.error('Attention Worker error:', event);
      setStatus('error');
      setError(`Worker error: ${event.message}`);
    };

    // 清理函数
    return () => {
      worker.terminate();
      workerRef.current = null;
      setStatus('terminated');
      callbacksRef.current.clear();
    };
  }, []);

  // 发送消息到Worker
  const postMessage = useCallback(async (
    request: AttentionWorkerRequest,
  ): Promise<AttentionWorkerResponse> => {
    if (!workerRef.current || status === 'terminated') {
      throw new Error('Worker is not available');
    }

    return new Promise<AttentionWorkerResponse>(resolve => {
      // 生成唯一消息ID
      const messageId = `msg_${messageIdRef.current++}`;

      // 保存回调
      callbacksRef.current.set(messageId, resolve);

      // 设置状态为running
      setStatus('running');

      // 发送消息
      workerRef.current!.postMessage({
        id: messageId,
        type: request.type,
        data: request.data,
      });
    });
  }, [status]);

  // 便捷方法：归一化数据
  const normalize = useCallback(async (data: number[] | number[][] | number[][][]) => postMessage({ type: 'normalize', data }), [postMessage]);

  // 便捷方法：标准化数据
  const standardize = useCallback(async (data: number[] | number[][] | number[][][]) => postMessage({ type: 'standardize', data }), [postMessage]);

  // 便捷方法：裁剪数据精度
  const clip = useCallback(async (data: number[] | number[][] | number[][][], precision = 3) => postMessage({ type: 'clip', data: { data, precision } }), [postMessage]);

  // 便捷方法：计算余弦相似度
  const calculateSimilarity = useCallback(async (vec1: number[], vec2: number[]) => postMessage({ type: 'similarity', data: { vec1, vec2 } }), [postMessage]);

  // 便捷方法：计算注意力归因权重
  const calculateGradientAttribution = useCallback(async (attentionWeights: number[][], gradients: number[][]) => postMessage({ type: 'gradient_attribution', data: { attentionWeights, gradients } }), [postMessage]);

  // 便捷方法：计算注意力权重统计信息
  const calculateWeightStatistics = useCallback(async (attentionWeights: number[][]) => postMessage({ type: 'weight_statistics', data: attentionWeights }), [postMessage]);

  // 便捷方法：处理跨域迁移矩阵
  const processTransferMatrix = useCallback(async (
    matrix: number[][],
    operation: 'normalize' | 'clip' | 'stat',
  ) => postMessage({ type: 'transfer_matrix_processing', data: { matrix, operation } }), [postMessage]);

  // 便捷方法：PCA降维
  const pcaReduction = useCallback(async (
    data: number[][],
    dimensions = 2,
  ) => postMessage({ type: 'pca_reduction', data: { data, dimensions } }), [postMessage]);

  // 便捷方法：t-SNE降维
  const tsneReduction = useCallback(async (
    data: number[][],
    dimensions = 2,
    perplexity = 30,
  ) => postMessage({ type: 'tsne_reduction', data: { data, dimensions, perplexity } }), [postMessage]);

  // 便捷方法：K-means聚类
  const clustering = useCallback(async (
    data: number[][],
    k = 5,
  ) => postMessage({ type: 'clustering', data: { data, k } }), [postMessage]);

  // 便捷方法：相关性分析
  const correlationAnalysis = useCallback(async (
    data: number[][],
  ) => postMessage({ type: 'correlation_analysis', data: { data } }), [postMessage]);

  // 重置Worker（创建新的Worker实例）
  const resetWorker = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }

    // 创建新的Worker实例
    const worker = new Worker(new URL('../workers/attention-calculator.worker.ts', import.meta.url), {
      type: 'module',
    });

    workerRef.current = worker;
    setStatus('idle');
    setError(null);

    // 重新设置消息处理
    worker.onmessage = (event: MessageEvent<any>) => {
      const { id, result, error } = event.data;
      const callback = callbacksRef.current.get(id);

      if (callback) {
        callback({ result, error });
        callbacksRef.current.delete(id);
      }

      if (callbacksRef.current.size === 0) {
        setStatus('idle');
      }
    };

    worker.onerror = event => {
      console.error('Attention Worker error:', event);
      setStatus('error');
      setError(`Worker error: ${event.message}`);
    };
  }, []);

  return {
    // 状态
    status,
    error,
    isRunning: status === 'running',
    isIdle: status === 'idle',
    isError: status === 'error',
    isTerminated: status === 'terminated',

    // 基础方法
    postMessage,
    resetWorker,

    // 便捷方法
    normalize,
    standardize,
    clip,
    calculateSimilarity,
    calculateGradientAttribution,
    calculateWeightStatistics,
    processTransferMatrix,
    pcaReduction,
    tsneReduction,
    clustering,
    correlationAnalysis,
  };
}

export default useAttentionWorker;
