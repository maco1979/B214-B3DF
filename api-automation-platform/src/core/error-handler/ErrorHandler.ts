import { HttpRequestConfig, HttpResponse } from '../../types';

/**
 * 错误类型定义
 */
export enum ErrorType {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  AUTH_ERROR = 'AUTH_ERROR',
  SERVER_ERROR = 'SERVER_ERROR',
  CLIENT_ERROR = 'CLIENT_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

/**
 * 错误信息接口
 */
export interface ApiError {
  type: ErrorType;
  message: string;
  code?: string;
  originalError?: any;
  request?: HttpRequestConfig;
  response?: HttpResponse;
  retryable?: boolean;
}

/**
 * 重试策略类型
 */
export type RetryStrategy = 'exponential' | 'fixed' | 'random';

/**
 * 错误处理配置
 */
export interface ErrorHandlerConfig {
  retryStrategy?: RetryStrategy;
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
 熔断?: {
    enabled: boolean;
    threshold: number;
    resetTime: number;
  };
  notification?: {
    enabled: boolean;
    channels: Array<'email' | 'slack' | 'webhook'>;
    webhookUrls?: string[];
    emailRecipients?: string[];
    slackWebhookUrl?: string;
  };
}

/**
 * 错误处理机制，负责处理API调用过程中的各种错误
 */
export class ErrorHandler {
  private config: ErrorHandlerConfig;
  private errorCounts: Map<string, { count: number; lastErrorTime: number }> = new Map();
  private circuitStates: Map<string, { state: 'closed' | 'open' | 'half-open'; lastStateChange: number }> = new Map();

  constructor(config: ErrorHandlerConfig = {}) {
    this.config = {
      retryStrategy: 'exponential',
      maxRetries: 3,
      initialDelay: 1000,
      maxDelay: 30000,
      ...config,
    };
  }

  /**
   * 分类错误类型
   */
  classifyError(error: any, request?: HttpRequestConfig, response?: HttpResponse): ApiError {
    let type: ErrorType;
    let message: string;
    let retryable = true;

    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      type = ErrorType.NETWORK_ERROR;
      message = `Network error: ${error.message}`;
    } else if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      type = ErrorType.TIMEOUT_ERROR;
      message = `Request timeout: ${error.message}`;
    } else if (response) {
      if (response.status >= 400 && response.status < 500) {
        type = ErrorType.CLIENT_ERROR;
        message = `Client error: ${response.status} ${response.statusText}`;
        // 客户端错误通常不需要重试
        retryable = [408, 429].includes(response.status);
      } else if (response.status >= 500) {
        type = ErrorType.SERVER_ERROR;
        message = `Server error: ${response.status} ${response.statusText}`;
      } else {
        type = ErrorType.UNKNOWN_ERROR;
        message = `Unknown error: ${error.message}`;
      }
    } else if (error.message?.includes('authentication') || error.message?.includes('auth')) {
      type = ErrorType.AUTH_ERROR;
      message = `Authentication error: ${error.message}`;
      retryable = false;
    } else {
      type = ErrorType.UNKNOWN_ERROR;
      message = `Unknown error: ${error.message || JSON.stringify(error)}`;
    }

    return {
      type,
      message,
      originalError: error,
      request,
      response,
      retryable,
    };
  }

  /**
   * 计算重试延迟
   */
  calculateRetryDelay(attempt: number): number {
    const { retryStrategy, initialDelay, maxDelay } = this.config;
    
    switch (retryStrategy) {
      case 'exponential':
        return Math.min(initialDelay! * Math.pow(2, attempt), maxDelay!);
      case 'fixed':
        return initialDelay!;
      case 'random':
        return Math.random() * (maxDelay! - initialDelay!) + initialDelay!;
      default:
        return initialDelay!;
    }
  }

  /**
   * 检查是否应该重试
   */
  shouldRetry(error: ApiError, attempt: number): boolean {
    if (!error.retryable) {
      return false;
    }

    if (attempt >= this.config.maxRetries!) {
      return false;
    }

    // 检查熔断状态
    const circuitKey = this.getCircuitKey(error.request!);
    const circuitState = this.circuitStates.get(circuitKey);
    
    if (circuitState) {
      if (circuitState.state === 'open') {
        // 检查是否可以尝试恢复
        if (Date.now() - circuitState.lastStateChange > this.config.熔断?.resetTime!) {
          // 进入半开状态
          this.circuitStates.set(circuitKey, {
            state: 'half-open',
            lastStateChange: Date.now(),
          });
          return true;
        }
        return false;
      } else if (circuitState.state === 'half-open') {
        // 半开状态下只允许一个请求尝试
        return true;
      }
    }

    return true;
  }

  /**
   * 记录错误并检查熔断状态
   */
  recordError(error: ApiError): void {
    const circuitKey = this.getCircuitKey(error.request!);
    const now = Date.now();
    
    // 更新错误计数
    const errorCount = this.errorCounts.get(circuitKey) || { count: 0, lastErrorTime: now };
    
    if (now - errorCount.lastErrorTime > this.config.熔断?.resetTime!) {
      // 重置错误计数
      this.errorCounts.set(circuitKey, { count: 1, lastErrorTime: now });
    } else {
      // 增加错误计数
      this.errorCounts.set(circuitKey, {
        count: errorCount.count + 1,
        lastErrorTime: now,
      });
    }

    // 检查是否需要熔断
    if (this.config.熔断?.enabled) {
      const updatedErrorCount = this.errorCounts.get(circuitKey)!;
      if (updatedErrorCount.count >= this.config.熔断.threshold) {
        // 打开熔断
        this.circuitStates.set(circuitKey, {
          state: 'open',
          lastStateChange: now,
        });
        
        // 发送熔断通知
        this.sendNotification(`Circuit breaker opened for ${circuitKey} due to ${updatedErrorCount.count} errors`, error);
      }
    }
  }

  /**
   * 记录成功请求并关闭熔断
   */
  recordSuccess(request: HttpRequestConfig): void {
    const circuitKey = this.getCircuitKey(request);
    
    // 重置错误计数
    this.errorCounts.set(circuitKey, { count: 0, lastErrorTime: Date.now() });
    
    // 关闭熔断
    const circuitState = this.circuitStates.get(circuitKey);
    if (circuitState && circuitState.state !== 'closed') {
      this.circuitStates.set(circuitKey, {
        state: 'closed',
        lastStateChange: Date.now(),
      });
      
      // 发送熔断恢复通知
      this.sendNotification(`Circuit breaker closed for ${circuitKey}`, { request });
    }
  }

  /**
   * 发送错误通知
   */
  async sendNotification(message: string, error: Partial<ApiError>): Promise<void> {
    if (!this.config.notification?.enabled) {
      return;
    }

    const channels = this.config.notification.channels;
    
    for (const channel of channels) {
      try {
        switch (channel) {
          case 'webhook':
            await this.sendWebhookNotification(message, error);
            break;
          case 'email':
            await this.sendEmailNotification(message, error);
            break;
          case 'slack':
            await this.sendSlackNotification(message, error);
            break;
        }
      } catch (notificationError) {
        console.error(`Failed to send notification via ${channel}:`, notificationError);
      }
    }
  }

  /**
   * 发送Webhook通知
   */
  private async sendWebhookNotification(message: string, error: Partial<ApiError>): Promise<void> {
    if (!this.config.notification?.webhookUrls?.length) {
      return;
    }

    for (const url of this.config.notification.webhookUrls) {
      try {
        await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message,
            error,
            timestamp: new Date().toISOString(),
          }),
        });
      } catch (error) {
        console.error(`Failed to send webhook notification to ${url}:`, error);
      }
    }
  }

  /**
   * 发送邮件通知
   */
  private async sendEmailNotification(message: string, error: Partial<ApiError>): Promise<void> {
    // 邮件发送实现，这里仅作示例
    console.log(`Sending email notification to ${this.config.notification?.emailRecipients?.join(', ')}:`);
    console.log(`Subject: API Automation Error: ${message}`);
    console.log(`Body: ${JSON.stringify(error, null, 2)}`);
  }

  /**
   * 发送Slack通知
   */
  private async sendSlackNotification(message: string, error: Partial<ApiError>): Promise<void> {
    if (!this.config.notification?.slackWebhookUrl) {
      return;
    }

    try {
      await fetch(this.config.notification.slackWebhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: `*API Automation Error*\n\n${message}\n\n${JSON.stringify(error, null, 2)}`,
        }),
      });
    } catch (error) {
      console.error('Failed to send Slack notification:', error);
    }
  }

  /**
   * 获取熔断键（基于请求的URL和方法）
   */
  private getCircuitKey(request: HttpRequestConfig): string {
    return `${request.method}:${request.url}`;
  }

  /**
   * 格式化错误信息
   */
  formatError(error: ApiError): string {
    return `${error.type}: ${error.message}${error.code ? ` (Code: ${error.code})` : ''}`;
  }

  /**
   * 创建标准化错误对象
   */
  createError(error: any, request?: HttpRequestConfig, response?: HttpResponse): ApiError {
    return this.classifyError(error, request, response);
  }
}
