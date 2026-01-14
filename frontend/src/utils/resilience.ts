/**
 * 请求韧性工具
 * 包含指数退避重试和熔断机制，用于提高请求的可靠性和保护后端
 */

/**
 * 重试配置
 */
export interface RetryConfig {

  /** 最大重试次数 */
  maxRetries?: number;

  /** 初始延迟时间（毫秒） */
  initialDelay?: number;

  /** 最大延迟时间（毫秒） */
  maxDelay?: number;

  /** 退避乘数 */
  backoffFactor?: number;

  /** 哪些状态码需要重试 */
  retryableStatusCodes?: number[];

  /** 哪些错误需要重试 */
  retryableErrors?: Array<(error: any) => boolean>;

  /** 是否启用随机抖动 */
  enableJitter?: boolean;
}

/**
 * 熔断状态
 */
export enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open'
}

/**
 * 熔断配置
 */
export interface CircuitBreakerConfig {

  /** 失败阈值，超过该值则熔断 */
  failureThreshold?: number;

  /** 熔断时间窗口（毫秒） */
  timeout?: number;

  /** 半开状态下的测试请求数量 */
  halfOpenMaxRequests?: number;

  /** 成功阈值，半开状态下达到该值则恢复 */
  successThreshold?: number;
}

/**
 * 指数退避重试实现
 */
export class RetryHandler {
  private readonly defaultConfig: RetryConfig = {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 30000,
    backoffFactor: 2,
    retryableStatusCodes: [500, 502, 503, 504, 429],
    retryableErrors: [
      error => error.code === 'ECONNRESET',
      error => error.code === 'ETIMEDOUT',
      error => error.message.includes('timeout'),
      error => error.message.includes('network'),
    ],
    enableJitter: true,
  };

  /**
   * 执行带重试的异步操作
   * @param operation 要执行的异步操作
   * @param config 重试配置
   * @returns 操作结果
   */
  async executeWithRetry<T>(
    operation: () => Promise<T>,
    config?: RetryConfig,
  ): Promise<T> {
    const mergedConfig = { ...this.defaultConfig, ...config };
    const { maxRetries, initialDelay, maxDelay, backoffFactor, retryableStatusCodes, retryableErrors, enableJitter } = mergedConfig;

    let retries = 0;
    let delay = initialDelay;

    while (true) {
      try {
        const result = await operation();
        return result;
      } catch (error: any) {
        retries++;

        // 检查是否需要重试
        const shouldRetry = this.shouldRetry(error, retries, maxRetries!, retryableStatusCodes!, retryableErrors!);

        if (!shouldRetry) {
          throw error;
        }

        // 计算延迟时间
        const waitTime = this.calculateDelay(delay, backoffFactor!, maxDelay!, enableJitter);

        console.log(`Retrying in ${waitTime}ms (attempt ${retries}/${maxRetries!})...`);
        await this.delay(waitTime);

        // 更新延迟时间
        delay = Math.min(delay * backoffFactor!, maxDelay!);
      }
    }
  }

  /**
   * 检查是否需要重试
   */
  private shouldRetry(
    error: any,
    retries: number,
    maxRetries: number,
    retryableStatusCodes: number[],
    retryableErrors: Array<(error: any) => boolean>,
  ): boolean {
    // 检查是否超过最大重试次数
    if (retries > maxRetries) {
      return false;
    }

    // 检查是否是可重试的状态码
    if (error.response?.status) {
      return retryableStatusCodes.includes(error.response.status);
    }

    // 检查是否是可重试的错误
    if (retryableErrors.some(condition => condition(error))) {
      return true;
    }

    return false;
  }

  /**
   * 计算延迟时间
   */
  private calculateDelay(
    delay: number,
    backoffFactor: number,
    maxDelay: number,
    enableJitter: boolean,
  ): number {
    let waitTime = delay;

    // 添加随机抖动，避免请求风暴
    if (enableJitter) {
      waitTime = Math.random() * waitTime;
    }

    return Math.min(waitTime, maxDelay);
  }

  /**
   * 延迟执行
   */
  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * 熔断机制实现
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenRequests = 0;
  private readonly defaultConfig: CircuitBreakerConfig = {
    failureThreshold: 5,
    timeout: 30000,
    halfOpenMaxRequests: 3,
    successThreshold: 2,
  };

  constructor(private readonly config?: CircuitBreakerConfig) {
    this.config = { ...this.defaultConfig, ...config };
  }

  /**
   * 执行带熔断保护的异步操作
   * @param operation 要执行的异步操作
   * @returns 操作结果
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    const { timeout, halfOpenMaxRequests, successThreshold } = this.config!;

    // 检查熔断状态
    this.checkCircuitState();

    // 如果熔断已打开，直接抛出错误
    if (this.state === CircuitState.OPEN) {
      throw new Error('Circuit breaker is open');
    }

    // 半开状态下限制请求数量
    if (this.state === CircuitState.HALF_OPEN) {
      if (this.halfOpenRequests >= halfOpenMaxRequests!) {
        throw new Error('Circuit breaker is half-open, too many requests');
      }
      this.halfOpenRequests++;
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    } finally {
      if (this.state === CircuitState.HALF_OPEN) {
        this.halfOpenRequests--;
      }
    }
  }

  /**
   * 检查并更新熔断状态
   */
  private checkCircuitState(): void {
    const { timeout } = this.config!;

    if (this.state === CircuitState.OPEN) {
      const now = Date.now();
      if (now - this.lastFailureTime > timeout!) {
        // 熔断时间已过，进入半开状态
        this.state = CircuitState.HALF_OPEN;
        this.successCount = 0;
        this.halfOpenRequests = 0;
      }
    }
  }

  /**
   * 处理成功请求
   */
  private onSuccess(): void {
    const { successThreshold } = this.config!;

    if (this.state === CircuitState.CLOSED) {
      // 关闭状态下，成功则重置失败计数
      this.failureCount = 0;
    } else if (this.state === CircuitState.HALF_OPEN) {
      // 半开状态下，成功计数增加
      this.successCount++;

      if (this.successCount >= successThreshold!) {
        // 达到成功阈值，关闭熔断
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
      }
    }
  }

  /**
   * 处理失败请求
   */
  private onFailure(): void {
    const { failureThreshold } = this.config!;

    if (this.state === CircuitState.CLOSED) {
      // 关闭状态下，失败计数增加
      this.failureCount++;
      this.lastFailureTime = Date.now();

      if (this.failureCount >= failureThreshold!) {
        // 达到失败阈值，打开熔断
        this.state = CircuitState.OPEN;
      }
    } else if (this.state === CircuitState.HALF_OPEN) {
      // 半开状态下，任何失败都会重新打开熔断
      this.state = CircuitState.OPEN;
      this.lastFailureTime = Date.now();
      this.successCount = 0;
    }
  }

  /**
   * 获取当前熔断状态
   */
  getState(): CircuitState {
    this.checkCircuitState();
    return this.state;
  }

  /**
   * 手动重置熔断状态
   */
  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = 0;
    this.halfOpenRequests = 0;
  }
}

/**
 * 请求韧性管理器
 * 结合重试和熔断机制
 */
export class ResilienceManager {
  private readonly retryHandler: RetryHandler;
  private readonly circuitBreaker: CircuitBreaker;

  constructor(
    retryConfig?: RetryConfig,
    circuitBreakerConfig?: CircuitBreakerConfig,
  ) {
    this.retryHandler = new RetryHandler();
    this.circuitBreaker = new CircuitBreaker(circuitBreakerConfig);
  }

  /**
   * 执行带韧性保护的异步操作
   * @param operation 要执行的异步操作
   * @param retryConfig 重试配置（可选，用于覆盖默认配置）
   * @returns 操作结果
   */
  async execute<T>(
    operation: () => Promise<T>,
    retryConfig?: RetryConfig,
  ): Promise<T> {
    // 首先通过熔断检查
    return this.circuitBreaker.execute(async () =>
      // 然后执行带重试的操作
       this.retryHandler.executeWithRetry(operation, retryConfig),
    );
  }

  /**
   * 获取当前熔断状态
   */
  getCircuitState(): CircuitState {
    return this.circuitBreaker.getState();
  }

  /**
   * 手动重置熔断状态
   */
  resetCircuit(): void {
    this.circuitBreaker.reset();
  }
}

// 创建默认的韧性管理器实例
export const resilienceManager = new ResilienceManager();
