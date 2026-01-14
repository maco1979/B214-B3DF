import { useQuery } from '@tanstack/react-query';
import { http } from '@/lib/api-client';
import type { ApiResponse, SystemMetrics, BlockchainStatus } from '@/services/api';
import { Device } from '@/services/api';
import { useEventSource } from './useEventSource';

interface QueryTimingOptions {
  staleTime?: number
  refetchInterval?: number | false
}

export function useSystemMetricsQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['system-metrics'],
    queryFn: async () => {
      const res = await http.get('/api/system/metrics') as unknown as ApiResponse<SystemMetrics>;
      if (!res.success) {
 throw new Error(res.error || '获取系统指标失败');
}
      return res.data;
    },
    staleTime: options.staleTime ?? 30_000,
    refetchInterval: options.refetchInterval ?? 60_000,
  });
}

export function usePerformanceSummaryQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['performance-summary'],
    queryFn: async () => {
      const res = await http.get('/api/performance/summary') as unknown as ApiResponse<any>;
      if (!res.success) {
 throw new Error(res.error || '获取性能概览失败');
}
      return res.data;
    },
    staleTime: options.staleTime ?? 30_000,
    refetchInterval: options.refetchInterval ?? 60_000,
  });
}

export function usePerformanceMetricsQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['performance-metrics'],
    queryFn: async () => {
      const res = await http.get('/api/performance/summary') as unknown as ApiResponse<any>;
      if (!res.success) {
 throw new Error(res.error || '获取性能指标失败');
}
      return res.data;
    },
    staleTime: options.staleTime ?? 10_000,
    refetchInterval: options.refetchInterval ?? 20_000,
  });
}

export function useOptimizationStatusQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['optimization-status'],
    queryFn: async () => {
      const res = await http.get('/api/performance/optimization/status') as unknown as ApiResponse<any>;
      if (!res.success) {
 throw new Error(res.error || '获取优化状态失败');
}
      return res.data;
    },
    staleTime: options.staleTime ?? 30_000,
    refetchInterval: options.refetchInterval ?? 60_000,
  });
}

export function useBlockchainStatusQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['blockchain-status'],
    queryFn: async () => {
      const res = await http.get('/api/blockchain/status') as unknown as ApiResponse<BlockchainStatus>;
      if (!res.success) {
 throw new Error(res.error || '获取区块链状态失败');
}
      return res.data;
    },
    staleTime: options.staleTime ?? 60_000,
    refetchInterval: options.refetchInterval,
  });
}

export function useEdgeDevicesQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['edge-devices'],
    queryFn: async () => {
      const res = await http.get('/api/edge/devices') as unknown as ApiResponse<any[]>;
      if (!res.success) {
 throw new Error(res.error || '获取设备列表失败');
}
      return res.data || [];
    },
    staleTime: options.staleTime ?? 5_000,
    refetchInterval: options.refetchInterval ?? 10_000,
  });
}

export function useInferenceHistoryQuery(options: QueryTimingOptions = {}) {
  return useQuery({
    queryKey: ['inference-history'],
    queryFn: async () =>
      // 后端不存在该API，直接返回空数据
       [],
    staleTime: options.staleTime ?? 30_000,
    refetchInterval: options.refetchInterval ?? false,
  });
}

export function useSystemMetricsRealtime(options?: { enabled?: boolean }) {
  // 移除SSE流请求，直接使用普通查询
  const query = useSystemMetricsQuery({
    refetchInterval: options?.enabled ? 10_000 : false,
  });

  return query;
}

export function useInferenceHistoryRealtime(options?: { enabled?: boolean; keepLast?: number }) {
  // 移除SSE流请求，直接使用普通查询
  const query = useInferenceHistoryQuery({
    refetchInterval: options?.enabled ? 15_000 : false,
  });

  return query;
}
