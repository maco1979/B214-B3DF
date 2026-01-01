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
    app;
    port;
    constructor(port = 8080) {
        this.app = express();
        this.port = port;
        this.setupMiddlewares();
        this.setupRoutes();
        this.setupErrorHandling();
    }
    setupMiddlewares() {
        this.app.use(helmet({
            crossOriginResourcePolicy: false,
        }));
        this.app.use(cors({
            origin: '*',
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'X-API-KEY', 'Authorization']
        }));
        const limiter = rateLimit({
            windowMs: 1 * 60 * 1000,
            max: 500,
            message: '请求过于频繁，请稍后重试',
            standardHeaders: true,
            legacyHeaders: false,
            skipSuccessfulRequests: false,
            skipFailedRequests: false,
        });
        this.app.use(limiter);
        this.app.use(compression());
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
        this.app.use(loggingMiddleware);
        this.app.use(metricsMiddleware);
    }
    setupRoutes() {
        this.app.use('/health', healthCheck.getRouter());
        this.app.use('/api/decision', authMiddleware, routerManager.decisionProxy);
        this.app.use('/api', authMiddleware, routerManager.backendProxy);
        this.app.use('/discovery', serviceDiscovery.getRouter());
        this.app.use('/load-balancer', loadBalancer.getStatusRouter());
    }
    setupErrorHandling() {
        this.app.use(errorHandler);
    }
    async start() {
        try {
            await serviceDiscovery.initialize();
            await healthCheck.start();
            this.app.listen(this.port, () => {
                console.log(`🚀 API Gateway 服务启动成功，端口: ${this.port}`);
                console.log(`📊 监控面板: http://localhost:${this.port}/metrics`);
                console.log(`❤️  健康检查: http://localhost:${this.port}/health`);
            });
        }
        catch (error) {
            console.error('❌ API Gateway 启动失败:', error);
            process.exit(1);
        }
    }
    async stop() {
        await serviceDiscovery.shutdown();
        await healthCheck.stop();
        console.log('🛑 API Gateway 服务已停止');
    }
}
export { ApiGatewayServer };
//# sourceMappingURL=server.js.map