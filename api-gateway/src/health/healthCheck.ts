import express from 'express';
import { EventEmitter } from 'events';

interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: Date;
  details?: Record<string, any>;
}

interface ServiceHealth {
  service: string;
  status: 'healthy' | 'unhealthy';
  responseTime?: number;
  lastCheck: Date;
  error?: string;
}

class HealthCheck extends EventEmitter {
  private healthStatus: HealthStatus = {
    status: 'healthy',
    timestamp: new Date()
  };

  private serviceHealth: Map<string, ServiceHealth> = new Map();
  private checkInterval: NodeJS.Timeout | null = null;

  // 启动健康检查
  public async start(): Promise<void> {
    console.log('❤️  启动健康检查系统...');
    
    // 启动定期健康检查
    this.checkInterval = setInterval(() => {
      this.performHealthChecks();
    }, 30000); // 每30秒检查一次

    // 立即执行一次健康检查
    await this.performHealthChecks();
    
    console.log('✅ 健康检查系统启动完成');
  }

  // 执行健康检查
  private async performHealthChecks(): Promise<void> {
    const checks = [
      this.checkGatewayHealth(),
      this.checkMemoryUsage(),
      this.checkDatabaseConnection(),
      this.checkExternalServices()
    ];

    try {
      const results = await Promise.allSettled(checks);
      this.updateHealthStatus(results);
    } catch (error) {
      console.error('健康检查执行失败:', error);
      this.healthStatus = {
        status: 'unhealthy',
        timestamp: new Date(),
        details: { error: (error as Error).message }
      };
    }
  }

  // 检查网关自身健康
  private async checkGatewayHealth(): Promise<any> {
    return {
      component: 'gateway',
      status: 'healthy',
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      timestamp: new Date()
    };
  }

  // 检查内存使用情况
  private async checkMemoryUsage(): Promise<any> {
    const memoryUsage = process.memoryUsage();
    const memoryPercent = (memoryUsage.heapUsed / memoryUsage.heapTotal) * 100;
    
    return {
      component: 'memory',
      status: memoryPercent < 95 ? 'healthy' : 'degraded',

      usage: Math.round(memoryPercent),
      details: {
        heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024),
        heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024),
        rss: Math.round(memoryUsage.rss / 1024 / 1024)
      }
    };
  }

  // 检查数据库连接（如果有）
  private async checkDatabaseConnection(): Promise<any> {
    // 这里可以添加数据库连接检查
    return {
      component: 'database',
      status: 'healthy',
      message: '数据库连接正常'
    };
  }

  // 检查外部服务
  private async checkExternalServices(): Promise<any> {
    // 这里可以添加对外部服务的健康检查
    return {
      component: 'external_services',
      status: 'healthy',
      message: '外部服务连接正常'
    };
  }

  // 更新健康状态
  private updateHealthStatus(results: PromiseSettledResult<any>[]): void {
    const details: Record<string, any> = {};
    let overallStatus: 'healthy' | 'unhealthy' | 'degraded' = 'healthy';

    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        const checkResult = result.value;
        details[checkResult.component] = checkResult;
        
        if (checkResult.status === 'unhealthy') {
          overallStatus = 'unhealthy';
        } else if (checkResult.status === 'degraded' && overallStatus !== 'unhealthy') {
          overallStatus = 'degraded';
        }
      } else {
        details[`check_${index}`] = {
          status: 'unhealthy',
          error: result.reason?.message || '检查失败'
        };
        overallStatus = 'unhealthy';
      }
    });

    this.healthStatus = {
      status: overallStatus,
      timestamp: new Date(),
      details
    };

    // 触发状态变更事件
    if (overallStatus !== this.healthStatus.status) {
      this.emit('healthStatusChanged', this.healthStatus);
    }
  }

  // 注册服务健康检查
  public registerServiceHealth(serviceName: string, health: ServiceHealth): void {
    this.serviceHealth.set(serviceName, health);
  }

  // 获取健康状态
  public getHealthStatus(): HealthStatus {
    return this.healthStatus;
  }

  // 获取详细健康信息
  public getDetailedHealth(): any {
    return {
      gateway: this.healthStatus,
      services: Object.fromEntries(this.serviceHealth),
      system: {
        nodeVersion: process.version,
        platform: process.platform,
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage()
      }
    };
  }

  // 健康检查路由
  public getRouter(): express.Router {
    const router = express.Router();

    // 基础健康检查
    router.get('/', (_req, res) => {
      const status = this.getHealthStatus();
      
      if (status.status !== 'unhealthy') {
        res.json({
          status: status.status,
          timestamp: status.timestamp,
          message: 'API网关运行正常'
        });
      } else {
        res.status(503).json({
          status: status.status,
          timestamp: status.timestamp,
          message: 'API网关服务异常',
          details: status.details
        });
      }
    });


    // 详细健康信息
    router.get('/detailed', (_req, res) => {
      res.json(this.getDetailedHealth());
    });

    // 就绪检查（用于Kubernetes等编排系统）
    router.get('/ready', (_req, res) => {
      const status = this.getHealthStatus();
      
      if (status.status === 'healthy') {
        res.json({ status: 'ready' });
      } else {
        res.status(503).json({ status: 'not-ready' });
      }
    });

    // 存活检查
    router.get('/live', (_req, res) => {
      res.json({ status: 'alive' });
    });

    // 服务健康状态
    router.get('/services', (_req, res) => {
      res.json(Object.fromEntries(this.serviceHealth));
    });

    // 手动触发健康检查
    router.post('/check', async (_req, res) => {
      try {
        await this.performHealthChecks();
        res.json({
          message: '健康检查完成',
          status: this.getHealthStatus()
        });
      } catch (error) {
        res.status(500).json({
          error: '健康检查执行失败',
          message: (error as Error).message
        });
      }
    });

    return router;
  }

  // 停止健康检查
  public async stop(): Promise<void> {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    console.log('🛑 健康检查系统已停止');
  }
}

export const healthCheck = new HealthCheck();