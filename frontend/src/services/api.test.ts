import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { ApiResponse, Model } from './api';
import { apiClient } from './api';
import { http } from '@/lib/api-client';

// Mock the http client
vi.mock('@/lib/api-client', () => ({
  http: {
    request: vi.fn(),
  },
}));

describe('ApiClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('get', () => {
    it('should make a GET request and return the response', async () => {
      const mockResponse: ApiResponse<{ test: string }> = {
        success: true,
        data: { test: 'test data' },
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      const response = await apiClient.get('/test-endpoint');

      expect(http.request).toHaveBeenCalledWith({
        url: '/test-endpoint',
        method: 'GET',
        data: undefined,
        headers: { 'Content-Type': 'application/json' },
      });
      expect(response).toEqual(mockResponse);
    });

    it('should handle errors correctly', async () => {
      const mockError = new Error('Network error');
      (http.request as vi.Mock).mockRejectedValue(mockError);

      const response = await apiClient.get('/test-endpoint');

      expect(response).toEqual({
        success: false,
        error: 'Network error',
      });
    });
  });

  describe('model management', () => {
    it('should get models list', async () => {
      const mockModels: Model[] = [
        { id: '1', name: 'Model 1', version: '1.0', description: 'Test model', status: 'ready', created_at: '2023-01-01', updated_at: '2023-01-01' },
      ];
      const mockResponse: ApiResponse<Model[]> = {
        success: true,
        data: mockModels,
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      const response = await apiClient.getModels();

      expect(http.request).toHaveBeenCalledWith({
        url: '/api/v1/models',
        method: 'GET',
        data: undefined,
        headers: { 'Content-Type': 'application/json' },
      });
      expect(response).toEqual(mockResponse);
    });

    it('should get a single model', async () => {
      const mockModel: Model = {
        id: '1', name: 'Model 1', version: '1.0', description: 'Test model', status: 'ready', created_at: '2023-01-01', updated_at: '2023-01-01',
      };
      const mockResponse: ApiResponse<Model> = {
        success: true,
        data: mockModel,
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      const response = await apiClient.getModel('1');

      expect(http.request).toHaveBeenCalledWith({
        url: '/api/v1/models/1',
        method: 'GET',
        data: undefined,
        headers: { 'Content-Type': 'application/json' },
      });
      expect(response).toEqual(mockResponse);
    });

    it('should create a model', async () => {
      const mockModel: Model = {
        id: '1', name: 'Model 1', version: '1.0', description: 'Test model', status: 'ready', created_at: '2023-01-01', updated_at: '2023-01-01',
      };
      const mockResponse: ApiResponse<Model> = {
        success: true,
        data: mockModel,
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      const response = await apiClient.createModel({
        name: 'Model 1',
        version: '1.0',
        description: 'Test model',
      });

      expect(http.request).toHaveBeenCalledWith({
        url: '/api/v1/models/',
        method: 'POST',
        data: JSON.stringify({
          name: 'Model 1',
          version: '1.0',
          description: 'Test model',
        }),
        headers: { 'Content-Type': 'application/json' },
      });
      expect(response).toEqual(mockResponse);
    });

    it('should get model versions', async () => {
      const mockModels: Model[] = [
        { id: '1-1.0', name: 'Model 1', version: '1.0', description: 'Test model', status: 'ready', created_at: '2023-01-01', updated_at: '2023-01-01' },
        { id: '1-2.0', name: 'Model 1', version: '2.0', description: 'Test model', status: 'ready', created_at: '2023-02-01', updated_at: '2023-02-01' },
      ];
      const mockResponse: ApiResponse<Model[]> = {
        success: true,
        data: mockModels,
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      const response = await apiClient.getModelVersions('1');

      expect(http.request).toHaveBeenCalledWith({
        url: '/api/v1/models/1/versions',
        method: 'GET',
        data: undefined,
        headers: { 'Content-Type': 'application/json' },
      });
      expect(response).toEqual(mockResponse);
    });

    it('should create a model version', async () => {
      const mockModel: Model = {
        id: '1-2.0', name: 'Model 1', version: '2.0', description: 'Test model', status: 'ready', created_at: '2023-02-01', updated_at: '2023-02-01',
      };
      const mockResponse: ApiResponse<Model> = {
        success: true,
        data: mockModel,
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      const response = await apiClient.createModelVersion('1', {
        version: '2.0',
        description: 'Updated model',
      });

      expect(http.request).toHaveBeenCalledWith({
        url: '/api/v1/models/1/versions',
        method: 'POST',
        data: JSON.stringify({
          version: '2.0',
          description: 'Updated model',
        }),
        headers: { 'Content-Type': 'application/json' },
      });
      expect(response).toEqual(mockResponse);
    });
  });

  describe('request method', () => {
    it('should handle array headers correctly', async () => {
      const mockResponse: ApiResponse<{ test: string }> = {
        success: true,
        data: { test: 'test data' },
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      /*
       * We need to test the private request method indirectly
       * by calling a public method that uses it with array headers
       */
      await apiClient.get('/test-endpoint', {
        headers: [['Authorization', 'Bearer token'], ['X-Custom-Header', 'custom-value']],
      });

      expect(http.request).toHaveBeenCalledWith({
        url: '/test-endpoint',
        method: 'GET',
        data: undefined,
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer token',
          'X-Custom-Header': 'custom-value',
        },
      });
    });

    it('should handle object headers correctly', async () => {
      const mockResponse: ApiResponse<{ test: string }> = {
        success: true,
        data: { test: 'test data' },
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      await apiClient.get('/test-endpoint', {
        headers: { Authorization: 'Bearer token', 'X-Custom-Header': 'custom-value' },
      });

      expect(http.request).toHaveBeenCalledWith({
        url: '/test-endpoint',
        method: 'GET',
        data: undefined,
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer token',
          'X-Custom-Header': 'custom-value',
        },
      });
    });
  });

  describe('importModel', () => {
    it('should send form data correctly', async () => {
      const mockModel: Model = {
        id: 'imported-1', name: 'Imported Model', version: '1.0', description: 'Imported model', status: 'ready', created_at: '2023-01-01', updated_at: '2023-01-01',
      };
      const mockResponse: ApiResponse<Model> = {
        success: true,
        data: mockModel,
      };

      (http.request as vi.Mock).mockResolvedValue(mockResponse);

      // Create a mock File object
      const mockFile = new File(['model data'], 'model.zip', { type: 'application/zip' });

      await apiClient.importModel({ name: 'Imported Model' }, mockFile);

      // Check that http.request was called with form data
      expect(http.request).toHaveBeenCalled();
      const callArgs = (http.request as vi.Mock).mock.calls[0][0];
      expect(callArgs.url).toBe('/api/v1/models/import');
      expect(callArgs.method).toBe('POST');
      // Check that FormData was sent as data
      expect(callArgs.data instanceof FormData).toBe(true);
      // Note: We can't easily check the contents of FormData in mocks
    });
  });
});

