import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { ApiRequest, ApiResponse, ApiError } from '@/types';
import { AuthManager } from '../auth-manager/AuthManager';

/**
 * API请求构建器
 * 负责根据配置生成和发送API请求，并处理响应
 */
export class RequestBuilder {
  private authManager: AuthManager;

  constructor() {
    this.authManager = new AuthManager();
  }

  /**
   * 发送API请求
   * @param request API请求配置
   * @returns Promise<ApiResponse> API响应
   */
  async sendRequest(request: ApiRequest): Promise<ApiResponse> {
    const requestId = request.id || uuidv4();
    const startTime = Date.now();

    try {
      // 构建Axios请求配置
      const axiosConfig = await this.buildAxiosConfig(request);

      // 发送请求
      const response = await this.executeWithRetry(request, axiosConfig);

      // 构建响应对象
      const apiResponse: ApiResponse = {
        id: uuidv4(),
        requestId: requestId,
        statusCode: response.status,
        headers: this.normalizeHeaders(response.headers),
        body: response.data,
        duration: Date.now() - startTime,
        timestamp: Date.now()
      };

      return apiResponse;
    } catch (error) {
      // 处理错误
      const apiError = this.handleError(error, requestId, startTime);
      throw apiError;
    }
  }

  /**
   * 构建Axios请求配置
   * @param request API请求配置
   * @returns Promise<AxiosRequestConfig> Axios请求配置
   */
  private async buildAxiosConfig(request: ApiRequest): Promise<AxiosRequestConfig> {
    const axiosConfig: AxiosRequestConfig = {
      url: request.url,
      method: request.method,
      headers: {
        'Content-Type': 'application/json',
        ...request.headers
      },
      params: request.queryParams,
      data: request.body,
      timeout: request.timeout || 30000
    };

    // 应用认证配置
    if (request.auth) {
      await this.authManager.applyAuth(axiosConfig, request.auth);
    }

    return axiosConfig;
  }

  /**
   * 执行带有重试机制的请求
   * @param request API请求配置
   * @param axiosConfig Axios请求配置
   * @returns Promise<AxiosResponse> Axios响应
   */
  private async executeWithRetry(request: ApiRequest, axiosConfig: AxiosRequestConfig): Promise<AxiosResponse> {
    const maxRetries = request.retryConfig?.maxRetries || 0;
    const initialDelay = request.retryConfig?.delay || 1000;
    const exponentialBackoff = request.retryConfig?.exponentialBackoff || false;

    let lastError: AxiosError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await axios(axiosConfig);
      } catch (error) {
        lastError = error as AxiosError;

        if (attempt < maxRetries) {
          // 计算重试延迟
          const delay = exponentialBackoff 
            ? initialDelay * Math.pow(2, attempt)
            : initialDelay;

          // 等待指定时间后重试
          await this.sleep(delay);
        }
      }
    }

    throw lastError;
  }

  /**
   * 处理请求错误
   * @param error 错误对象
   * @param requestId 请求ID
   * @param startTime 请求开始时间
   * @returns ApiError API错误对象
   */
  private handleError(error: any, requestId: string, startTime: number): ApiError {
    const apiError: ApiError = {
      id: uuidv4(),
      requestId: requestId,
      message: '请求失败',
      timestamp: Date.now()
    };

    if (axios.isAxiosError(error)) {
      // 处理Axios错误
      if (error.response) {
        // 请求已发出，服务器返回错误状态码
        apiError.statusCode = error.response.status;
        apiError.message = error.response.data?.message || `请求失败，状态码：${error.response.status}`;
      } else if (error.request) {
        // 请求已发出，但没有收到响应
        apiError.message = '请求超时，没有收到响应';
      } else {
        // 请求配置错误
        apiError.message = `请求配置错误：${error.message}`;
      }
    } else {
      // 处理其他错误
      apiError.message = `请求失败：${error.message}`;
      apiError.stack = error.stack;
    }

    return apiError;
  }

  /**
   * 归一化响应头
   * @param headers Axios响应头
   * @returns Record<string, string> 归一化后的响应头
   */
  private normalizeHeaders(headers: any): Record<string, string> {
    const normalizedHeaders: Record<string, string> = {};
    
    for (const [key, value] of Object.entries(headers)) {
      if (typeof value === 'string') {
        normalizedHeaders[key] = value;
      } else if (Array.isArray(value)) {
        normalizedHeaders[key] = value.join(', ');
      } else {
        normalizedHeaders[key] = String(value);
      }
    }
    
    return normalizedHeaders;
  }

  /**
   * 睡眠指定时间
   * @param ms 毫秒数
   * @returns Promise<void>
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 创建请求构建器实例
   */
  static create(): RequestBuilder {
    return new RequestBuilder();
  }
}
