import { useQuery } from '@tanstack/react-query';
import { fetchModels, fetchModel, fetchModelVersions } from '@/services/modelService';
import type { Model } from '@/services/api';

export function useModelsQuery() {
  return useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 5 * 60 * 1000,
    // 确保返回默认空数组，避免undefined问题
    select: data => data || [],
  });
}

export function useModelQuery(id: string) {
  return useQuery<Model>({
    queryKey: ['model', id],
    queryFn: async () => fetchModel(id),
    enabled: Boolean(id),
    staleTime: 5 * 60 * 1000,
  });
}

export function useModelVersionsQuery(id: string) {
  return useQuery<Model[]>({
    queryKey: ['model', id, 'versions'],
    queryFn: async () => fetchModelVersions(id),
    enabled: Boolean(id),
    staleTime: 5 * 60 * 1000,
    // 确保返回默认空数组，避免undefined问题
    select: data => data || [],
  });
}
