import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { HttpRequestConfig, HttpResponse, AuthenticationConfig } from '../../types';
import { AuthManager } from '../auth/AuthManager';

/**
 * API请求构建器，负责构建和发送API请求
 */
export class RequestBuilder {
  private axiosInstance: AxiosInstance;
  private authManager: AuthManager;

  constructor(authManager: AuthManager, baseConfig?: Partial<AxiosRequestConfig>) {
    this.authManager = authManager;
    this.axiosInstance = axios.create({
      timeout: 30000,
      ...baseConfig,
    });

    // 添加请求拦截器
    this.axiosInstance.interceptors.request.use(
      (config) => {
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // 添加响应拦截器
    this.axiosInstance.interceptors.response.use(
      (response) => {
        return response;
      },
      (error) => {
        return Promise.reject(error);
      }
    );
  }

  /**
   * 构建HTTP请求
   */
  buildRequest(requestConfig: HttpRequestConfig, authConfig: AuthenticationConfig): AxiosRequestConfig {
    // 应用认证
    const authenticatedRequest = this.authManager.applyAuthToRequest(requestConfig, authConfig);

    return {
      method: authenticatedRequest.method,
      url: authenticatedRequest.url,
      headers: authenticatedRequest.headers,
      params: authenticatedRequest.params,
      data: authenticatedRequest.body,
      timeout: authenticatedRequest.timeout,
    };
  }

  /**
   * 发送单个API请求
   */
  async sendRequest(requestConfig: HttpRequestConfig, authConfig: AuthenticationConfig): Promise<HttpResponse> {
    const startTime = Date.now();

    try {
      // 检查并刷新过期的令牌
      const updatedAuthConfig = await this.refreshExpiredTokens(authConfig);
      
      // 构建请求
      const axiosConfig = this.buildRequest(requestConfig, updatedAuthConfig);
      
      // 发送请求
      const response = await this.axiosInstance.request(axiosConfig);
      
      const duration = Date.now() - startTime;
      
      return {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers as Record<string, string>,
        data: response.data,
        duration,
      };
    } catch (error: any) {
      const duration = Date.now() - startTime;
      
      if (error.response) {
        // 服务器返回错误响应
        return {
          status: error.response.status,
          statusText: error.response.statusText,
          headers: error.response.headers as Record<string, string>,
          data: error.response.data,
          duration,
        };
      } else if (error.request) {
        // 请求已发送但没有收到响应
        throw new Error(`No response received: ${error.message}`);
      } else {
        // 请求配置错误
        throw new Error(`Request error: ${error.message}`);
      }
    }
  }

  /**
   * 带重试机制的请求发送
   */
  async sendRequestWithRetry(requestConfig: HttpRequestConfig, authConfig: AuthenticationConfig): Promise<HttpResponse> {
    const { retry } = requestConfig;
    const maxAttempts = retry?.attempts || 1;
    const initialDelay = retry?.delay || 1000;
    const maxDelay = retry?.maxDelay || 30000;

    let lastError: Error | undefined;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return await this.sendRequest(requestConfig, authConfig);
      } catch (error: any) {
        lastError = error;
        
        if (attempt < maxAttempts - 1) {
          // 计算重试延迟（指数退避）
          const delay = Math.min(initialDelay * Math.pow(2, attempt), maxDelay);
          await this.delay(delay);
        }
      }
    }

    if (!lastError) {
      throw new Error('Request failed with no error information');
    }

    throw lastError;
  }

  /**
   * 刷新过期的令牌
   */
  private async refreshExpiredTokens(authConfig: AuthenticationConfig): Promise<AuthenticationConfig> {
    if (authConfig.type === 'jwt' && this.authManager.isJwtExpired(authConfig)) {
      // JWT令牌过期，需要刷新
      throw new Error('JWT token expired. Please provide a new token.');
    }

    if (authConfig.type === 'oauth2' && this.authManager.isOAuth2Expired(authConfig)) {
      // OAuth2令牌过期，尝试刷新
      if (!authConfig.refreshToken) {
        throw new Error('OAuth2 token expired and no refresh token available.');
      }

      // 确保refreshToken存在
      const refreshToken = authConfig.refreshToken;
      const refreshedTokens = await this.authManager.refreshOAuth2Token({
        type: 'oauth2',
        clientId: authConfig.clientId,
        clientSecret: authConfig.clientSecret,
        refreshToken,
        tokenUrl: authConfig.tokenUrl,
      });
      
      return {
        ...authConfig,
        accessToken: refreshedTokens.accessToken,
        refreshToken: refreshedTokens.refreshToken,
        expiresAt: refreshedTokens.expiresAt,
      };
    }

    return authConfig;
  }

  /**
   * 延迟函数
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 批量发送请求
   */
  async sendBatchRequests(requests: Array<{ config: HttpRequestConfig; auth: AuthenticationConfig }>, concurrency: number = 5): Promise<HttpResponse[]> {
    const results: HttpResponse[] = [];
    const queue = [...requests];
    const activeRequests: Promise<void>[] = [];

    const processQueue = async () => {
      while (queue.length > 0) {
        const request = queue.shift();
        if (!request) break;

        try {
          const result = await this.sendRequest(request.config, request.auth);
          results.push(result);
        } catch (error) {
          // 批量请求中的错误不中断整个批量处理
          console.error(`Batch request error: ${error}`);
          // 可以选择将错误结果也添加到结果中
          results.push({
            status: 500,
            statusText: 'Internal Server Error',
            headers: {},
            data: { error: (error as Error).message },
            duration: 0,
          });
        }
      }
    };

    // 启动并发请求
    for (let i = 0; i < Math.min(concurrency, requests.length); i++) {
      activeRequests.push(processQueue());
    }

    await Promise.all(activeRequests);
    return results;
  }

  /**
   * 设置Axios实例的配置
   */
  setAxiosConfig(config: Partial<AxiosRequestConfig>): void {
    // 只合并配置，不直接赋值
    if (this.axiosInstance.defaults) {
      Object.assign(this.axiosInstance.defaults, config);
    }
  }

  /**
   * 获取Axios实例
   */
  getAxiosInstance(): AxiosInstance {
    return this.axiosInstance;
  }
}
