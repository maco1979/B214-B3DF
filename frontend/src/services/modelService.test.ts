import { describe, it, expect, vi, afterEach } from 'vitest';
import { fetchModels, fetchModel, fetchModelVersions } from './modelService';
import { apiClient } from './api';

// Mock the apiClient to avoid real API calls
vi.mock('./api', () => ({
  apiClient: {
    getModels: vi.fn(),
    getModel: vi.fn(),
    getModelVersions: vi.fn(),
  },
}));

const mockedApiClient = vi.mocked(apiClient, true);

describe('modelService', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  const mockModels = [
    { id: '1', name: 'Model 1', version: '1.0', description: 'Test model', status: 'ready', created_at: '2023-01-01', updated_at: '2023-01-01' },
    { id: '2', name: 'Model 2', version: '2.0', description: 'Another test model', status: 'deployed', created_at: '2023-02-01', updated_at: '2023-02-01' },
  ];

  const mockModel = mockModels[0];

  describe('fetchModels', () => {
    it('should fetch models successfully', async () => {
      mockedApiClient.getModels.mockResolvedValue({
        success: true,
        data: mockModels,
      });

      const result = await fetchModels();

      expect(mockedApiClient.getModels).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockModels);
    });

    it('should throw error when fetching models fails', async () => {
      mockedApiClient.getModels.mockResolvedValue({
        success: false,
        error: 'Failed to fetch models',
      });

      await expect(fetchModels()).rejects.toThrow('Failed to fetch models');
      expect(mockedApiClient.getModels).toHaveBeenCalledTimes(1);
    });

    it('should return empty array when models data is null', async () => {
      mockedApiClient.getModels.mockResolvedValue({
        success: true,
        data: null,
      });

      const result = await fetchModels();

      expect(result).toEqual([]);
      expect(mockedApiClient.getModels).toHaveBeenCalledTimes(1);
    });

    it('should handle empty models array', async () => {
      mockedApiClient.getModels.mockResolvedValue({
        success: true,
        data: [],
      });

      const result = await fetchModels();

      expect(result).toEqual([]);
      expect(mockedApiClient.getModels).toHaveBeenCalledTimes(1);
    });
  });

  describe('fetchModel', () => {
    it('should fetch model by id successfully', async () => {
      mockedApiClient.getModel.mockResolvedValue({
        success: true,
        data: mockModel,
      });

      const result = await fetchModel('1');

      expect(mockedApiClient.getModel).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getModel).toHaveBeenCalledWith('1');
      expect(result).toEqual(mockModel);
    });

    it('should throw error when fetching model fails', async () => {
      mockedApiClient.getModel.mockResolvedValue({
        success: false,
        error: 'Model not found',
      });

      await expect(fetchModel('999')).rejects.toThrow('Model not found');
      expect(mockedApiClient.getModel).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getModel).toHaveBeenCalledWith('999');
    });

    it('should throw error when model data is null', async () => {
      mockedApiClient.getModel.mockResolvedValue({
        success: true,
        data: null,
      });

      await expect(fetchModel('1')).rejects.toThrow('获取模型详情失败');
      expect(mockedApiClient.getModel).toHaveBeenCalledTimes(1);
    });

    it('should handle invalid model id', async () => {
      mockedApiClient.getModel.mockResolvedValue({
        success: false,
        error: 'Invalid model id',
      });

      await expect(fetchModel('')).rejects.toThrow('Invalid model id');
      expect(mockedApiClient.getModel).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getModel).toHaveBeenCalledWith('');
    });
  });

  describe('fetchModelVersions', () => {
    it('should fetch model versions successfully', async () => {
      const modelVersions = [
        { ...mockModel, version: '1.0', id: '1-1.0' },
        { ...mockModel, version: '2.0', id: '1-2.0' },
      ];

      mockedApiClient.getModelVersions.mockResolvedValue({
        success: true,
        data: modelVersions,
      });

      const result = await fetchModelVersions('1');

      expect(mockedApiClient.getModelVersions).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getModelVersions).toHaveBeenCalledWith('1');
      expect(result).toEqual(modelVersions);
    });

    it('should throw error when fetching model versions fails', async () => {
      mockedApiClient.getModelVersions.mockResolvedValue({
        success: false,
        error: 'Failed to fetch model versions',
      });

      await expect(fetchModelVersions('1')).rejects.toThrow('Failed to fetch model versions');
      expect(mockedApiClient.getModelVersions).toHaveBeenCalledTimes(1);
    });

    it('should return empty array when model versions data is null', async () => {
      mockedApiClient.getModelVersions.mockResolvedValue({
        success: true,
        data: null,
      });

      const result = await fetchModelVersions('1');

      expect(result).toEqual([]);
      expect(mockedApiClient.getModelVersions).toHaveBeenCalledTimes(1);
    });

    it('should handle model with no versions', async () => {
      mockedApiClient.getModelVersions.mockResolvedValue({
        success: true,
        data: [],
      });

      const result = await fetchModelVersions('1');

      expect(result).toEqual([]);
      expect(mockedApiClient.getModelVersions).toHaveBeenCalledTimes(1);
    });
  });
});

