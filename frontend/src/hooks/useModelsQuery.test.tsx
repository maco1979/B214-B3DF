import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useModelsQuery } from './useModelsQuery';

const { fetchModelsMock } = vi.hoisted(() => ({
  fetchModelsMock: vi.fn(),
}));

vi.mock('@/services/modelService', async () => {
  const actual = await vi.importActual<any>('@/services/modelService');
  return {
    ...actual,
    fetchModels: fetchModelsMock,
  };
});


function createWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useModelsQuery', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchModelsMock.mockResolvedValue([{ id: '1', name: 'm1', version: 'v1', description: '', status: 'ready', created_at: '', updated_at: '' }]);
  });

  it('应返回模型列表并成功状态', async () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useModelsQuery(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(fetchModelsMock).toHaveBeenCalledTimes(1);
    expect(result.current.data?.[0].name).toBe('m1');
  });
});
