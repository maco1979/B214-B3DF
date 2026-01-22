import winston from 'winston';
import { MonitoringMetrics, ApiCallLog, WorkflowExecution } from '../../types';

/**
 * 监控指标存储接口
 */
export interface MetricsStorage {
  recordApiCallLog(log: ApiCallLog): Promise<void>;
  recordWorkflowExecution(execution: WorkflowExecution): Promise<void>;
  getMetrics(startTime: Date, endTime: Date): Promise<MonitoringMetrics>;
  getApiCallLogs(filter: {
    startTime?: Date;
    endTime?: Date;
    status?: 'success' | 'failed';
    endpointId?: string;
    limit?: number;
    offset?: number;
  }): Promise<ApiCallLog[]>;
  getWorkflowExecutions(filter: {
    startTime?: Date;
    endTime?: Date;
    status?: string;
    workflowId?: string;
    limit?: number;
    offset?: number;
  }): Promise<WorkflowExecution[]>;
}

/**
 * 内存存储实现（用于演示）
 */
export class MemoryMetricsStorage implements MetricsStorage {
  private apiCallLogs: ApiCallLog[] = [];
  private workflowExecutions: WorkflowExecution[] = [];

  async recordApiCallLog(log: ApiCallLog): Promise<void> {
    this.apiCallLogs.push(log);
    // 限制内存中日志数量
    if (this.apiCallLogs.length > 10000) {
      this.apiCallLogs.shift();
    }
  }

  async recordWorkflowExecution(execution: WorkflowExecution): Promise<void> {
    this.workflowExecutions.push(execution);
    // 限制内存中执行记录数量
    if (this.workflowExecutions.length > 1000) {
      this.workflowExecutions.shift();
    }
  }

  async getMetrics(startTime: Date, endTime: Date): Promise<MonitoringMetrics> {
    const filteredLogs = this.apiCallLogs.filter(log => {
      return log.timestamp >= startTime && log.timestamp <= endTime;
    });

    const totalCalls = filteredLogs.length;
    const successCalls = filteredLogs.filter(log => log.status === 'success').length;
    const failedCalls = totalCalls - successCalls;
    const averageResponseTime = totalCalls > 0 ? 
      filteredLogs.reduce((sum, log) => sum + log.duration, 0) / totalCalls : 0;

    // 计算错误率
    const errorRates: Record<string, number> = {};
    const statusCodeDistribution: Record<number, number> = {};

    for (const log of filteredLogs) {
      if (log.status === 'failed') {
        const errorType = log.error?.type || 'UNKNOWN_ERROR';
        errorRates[errorType] = (errorRates[errorType] || 0) + 1;
      }
      
      if (log.response) {
        statusCodeDistribution[log.response.status] = 
          (statusCodeDistribution[log.response.status] || 0) + 1;
      }
    }

    // 计算错误率百分比
    for (const [errorType, count] of Object.entries(errorRates)) {
      errorRates[errorType] = (count / failedCalls) * 100;
    }

    return {
      totalCalls,
      successCalls,
      failedCalls,
      averageResponseTime,
      errorRates,
      statusCodeDistribution,
      timestamp: new Date(),
    };
  }

  async getApiCallLogs(filter: {
    startTime?: Date;
    endTime?: Date;
    status?: 'success' | 'failed';
    endpointId?: string;
    limit?: number;
    offset?: number;
  }): Promise<ApiCallLog[]> {
    let logs = [...this.apiCallLogs];

    // 应用过滤条件
    if (filter.startTime) {
      logs = logs.filter(log => log.timestamp >= filter.startTime!);
    }
    if (filter.endTime) {
      logs = logs.filter(log => log.timestamp <= filter.endTime!);
    }
    if (filter.status) {
      logs = logs.filter(log => log.status === filter.status);
    }
    if (filter.endpointId) {
      logs = logs.filter(log => log.endpointId === filter.endpointId);
    }

    // 排序（按时间倒序）
    logs.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    // 应用分页
    const offset = filter.offset || 0;
    const limit = filter.limit || logs.length;
    return logs.slice(offset, offset + limit);
  }

  async getWorkflowExecutions(filter: {
    startTime?: Date;
    endTime?: Date;
    status?: string;
    workflowId?: string;
    limit?: number;
    offset?: number;
  }): Promise<WorkflowExecution[]> {
    let executions = [...this.workflowExecutions];

    // 应用过滤条件
    if (filter.startTime) {
      executions = executions.filter(exec => exec.startTime >= filter.startTime!);
    }
    if (filter.endTime) {
      executions = executions.filter(exec => exec.endTime && exec.endTime <= filter.endTime!);
    }
    if (filter.status) {
      executions = executions.filter(exec => exec.status === filter.status);
    }
    if (filter.workflowId) {
      executions = executions.filter(exec => exec.workflowId === filter.workflowId);
    }

    // 排序（按时间倒序）
    executions.sort((a, b) => b.startTime.getTime() - a.startTime.getTime());

    // 应用分页
    const offset = filter.offset || 0;
    const limit = filter.limit || executions.length;
    return executions.slice(offset, offset + limit);
  }
}

/**
 * 监控与日志系统
 */
export class MonitoringSystem {
  private logger: winston.Logger;
  private metricsStorage: MetricsStorage;
  private metricsCache: Map<string, { metrics: MonitoringMetrics; timestamp: Date }> = new Map();

  constructor(metricsStorage: MetricsStorage = new MemoryMetricsStorage()) {
    this.metricsStorage = metricsStorage;
    
    // 初始化Winston日志记录器
    this.logger = winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          )
        }),
        new winston.transports.File({
          filename: 'logs/api-automation.log',
          maxsize: 5242880, // 5MB
          maxFiles: 5,
          tailable: true
        }),
        new winston.transports.File({
          filename: 'logs/api-automation-errors.log',
          level: 'error',
          maxsize: 5242880, // 5MB
          maxFiles: 5,
          tailable: true
        })
      ]
    });
  }

  /**
   * 记录API调用日志
   */
  async logApiCall(log: ApiCallLog): Promise<void> {
    // 记录到存储
    await this.metricsStorage.recordApiCallLog(log);
    
    // 记录到Winston日志
    const logLevel = log.status === 'success' ? 'info' : 'error';
    this.logger.log(logLevel, {
      type: 'api_call',
      ...log,
      timestamp: log.timestamp.toISOString(),
    });
  }

  /**
   * 记录工作流执行日志
   */
  async logWorkflowExecution(execution: WorkflowExecution): Promise<void> {
    // 记录到存储
    await this.metricsStorage.recordWorkflowExecution(execution);
    
    // 记录到Winston日志
    const logLevel = execution.status === 'success' ? 'info' : 'error';
    this.logger.log(logLevel, {
      type: 'workflow_execution',
      ...execution,
      startTime: execution.startTime.toISOString(),
      endTime: execution.endTime?.toISOString(),
    });
  }

  /**
   * 记录调试日志
   */
  debug(message: string, metadata?: any): void {
    this.logger.debug(message, metadata);
  }

  /**
   * 记录信息日志
   */
  info(message: string, metadata?: any): void {
    this.logger.info(message, metadata);
  }

  /**
   * 记录警告日志
   */
  warn(message: string, metadata?: any): void {
    this.logger.warn(message, metadata);
  }

  /**
   * 记录错误日志
   */
  error(message: string, metadata?: any): void {
    this.logger.error(message, metadata);
  }

  /**
   * 获取监控指标
   */
  async getMetrics(startTime: Date, endTime: Date): Promise<MonitoringMetrics> {
    const cacheKey = `${startTime.getTime()}-${endTime.getTime()}`;
    
    // 检查缓存
    const cached = this.metricsCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp.getTime() < 60000) { // 缓存1分钟
      return cached.metrics;
    }
    
    // 从存储获取指标
    const metrics = await this.metricsStorage.getMetrics(startTime, endTime);
    
    // 更新缓存
    this.metricsCache.set(cacheKey, { metrics, timestamp: new Date() });
    
    return metrics;
  }

  /**
   * 获取API调用日志
   */
  async getApiCallLogs(filter: {
    startTime?: Date;
    endTime?: Date;
    status?: 'success' | 'failed';
    endpointId?: string;
    limit?: number;
    offset?: number;
  }): Promise<ApiCallLog[]> {
    return this.metricsStorage.getApiCallLogs(filter);
  }

  /**
   * 获取工作流执行记录
   */
  async getWorkflowExecutions(filter: {
    startTime?: Date;
    endTime?: Date;
    status?: string;
    workflowId?: string;
    limit?: number;
    offset?: number;
  }): Promise<WorkflowExecution[]> {
    return this.metricsStorage.getWorkflowExecutions(filter);
  }

  /**
   * 获取实时监控数据（最近5分钟）
   */
  async getRealtimeMetrics(): Promise<MonitoringMetrics> {
    const endTime = new Date();
    const startTime = new Date(endTime.getTime() - 5 * 60 * 1000); // 5分钟前
    return this.getMetrics(startTime, endTime);
  }

  /**
   * 获取今日监控数据
   */
  async getTodayMetrics(): Promise<MonitoringMetrics> {
    const endTime = new Date();
    const startTime = new Date();
    startTime.setHours(0, 0, 0, 0);
    return this.getMetrics(startTime, endTime);
  }

  /**
   * 清理旧日志
   */
  async cleanOldLogs(daysToKeep: number = 7): Promise<void> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
    
    // 这里可以实现清理旧日志的逻辑
    this.logger.info(`Cleaning old logs before ${cutoffDate.toISOString()}`);
  }
}
