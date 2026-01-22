import { http } from '@/lib/api-client';
import type { Model } from '@/services/api';
import { ApiResponse } from '@/services/api';
import { apiClient } from './api';

export async function fetchModels(): Promise<Model[]> {
  const res = await apiClient.getModels();
  if (!res.success) {
 throw new Error(res.error || '获取模型列表失败');
}
  return res.data || [];
}

export async function fetchModel(id: string): Promise<Model> {
  // 使用原始ID，不做任何修改
  const res = await apiClient.getModel(id);
  if (!res.success || !res.data) {
 throw new Error(res.error || '获取模型详情失败');
}
  return res.data;
}

export async function fetchModelVersions(id: string): Promise<Model[]> {
  const res = await apiClient.getModelVersions(id);
  if (!res.success) {
 throw new Error(res.error || '获取模型版本失败');
}
  return res.data || [];
}
