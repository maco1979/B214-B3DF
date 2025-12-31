import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import compression from 'compression';
import { routerManager } from './routing/router.js';
import { authMiddleware } from './middleware/auth.js';
import { loggingMiddleware } from './middleware/logging.js';
import { errorHandler } from './middleware/error.js';
import { metricsMiddleware } from './middleware/metrics.js';
import { loadBalancer } from './load-balancing/loadBalancer.js';
import { serviceDiscovery } from './discovery/serviceDiscovery.js';
import { healthCheck } from './health/healthCheck.js';

class ApiGatewayServer {
  private app: express.Application;
  private port: number;

  constructor(port: number = 8080) {
    this.app = express();
    this.port = port;
    this.setupMiddlewares();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  private setupMiddlewares(): void {
    // 安全中间件
    this.app.use(helmet({
      crossOriginResourcePolicy: false,
    }));
    this.app.use(cors({
      origin: '*', 
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'X-API-KEY', 'Authorization']
    }));


    // 请求限制
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15分钟
      max: 1000, // 限制每个IP最多1000个请求
      message: '请求过于频繁，请稍后再试'
    });
    this.app.use(limiter);

    // 压缩和解析
    this.app.use(compression());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // 自定义中间件
    this.app.use(loggingMiddleware);
    this.app.use(metricsMiddleware);
    // 注意：authMiddleware不应用到所有路由，只应用到API路由
  }

  private setupRoutes(): void {
    // 健康检查路由
    this.app.use('/health', healthCheck.getRouter());

    // API路由 - 明确区分决策服务和后端服务
    this.app.use('/api/decision', authMiddleware, routerManager.decisionProxy);
    this.app.use('/api', authMiddleware, routerManager.backendProxy);


    // 服务发现路由
    this.app.use('/discovery', serviceDiscovery.getRouter());

    // 负载均衡器状态路由
    this.app.use('/load-balancer', loadBalancer.getStatusRouter());
  }

  private setupErrorHandling(): void {
    this.app.use(errorHandler);
  }

  public async start(): Promise<void> {
    try {
      // 初始化服务发现
      await serviceDiscovery.initialize();
      
      // 启动健康检查
      await healthCheck.start();

      // 启动服务器
      this.app.listen(this.port, () => {
        console.log(`🚀 API Gateway 服务启动成功，端口: ${this.port}`);
        console.log(`📊 监控面板: http://localhost:${this.port}/metrics`);
        console.log(`❤️  健康检查: http://localhost:${this.port}/health`);
      });
    } catch (error) {
      console.error('❌ API Gateway 启动失败:', error);
      process.exit(1);
    }
  }

  public async stop(): Promise<void> {
    await serviceDiscovery.shutdown();
    await healthCheck.stop();
    console.log('🛑 API Gateway 服务已停止');
  }
}

export { ApiGatewayServer };