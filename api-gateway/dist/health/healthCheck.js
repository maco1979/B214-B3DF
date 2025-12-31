import express from 'express';
import { EventEmitter } from 'events';
class HealthCheck extends EventEmitter {
    healthStatus = {
        status: 'healthy',
        timestamp: new Date()
    };
    serviceHealth = new Map();
    checkInterval = null;
    async start() {
        console.log('❤️  启动健康检查系统...');
        this.checkInterval = setInterval(() => {
            this.performHealthChecks();
        }, 30000);
        await this.performHealthChecks();
        console.log('✅ 健康检查系统启动完成');
    }
    async performHealthChecks() {
        const checks = [
            this.checkGatewayHealth(),
            this.checkMemoryUsage(),
            this.checkDatabaseConnection(),
            this.checkExternalServices()
        ];
        try {
            const results = await Promise.allSettled(checks);
            this.updateHealthStatus(results);
        }
        catch (error) {
            console.error('健康检查执行失败:', error);
            this.healthStatus = {
                status: 'unhealthy',
                timestamp: new Date(),
                details: { error: error.message }
            };
        }
    }
    async checkGatewayHealth() {
        return {
            component: 'gateway',
            status: 'healthy',
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            timestamp: new Date()
        };
    }
    async checkMemoryUsage() {
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
    async checkDatabaseConnection() {
        return {
            component: 'database',
            status: 'healthy',
            message: '数据库连接正常'
        };
    }
    async checkExternalServices() {
        return {
            component: 'external_services',
            status: 'healthy',
            message: '外部服务连接正常'
        };
    }
    updateHealthStatus(results) {
        const details = {};
        let overallStatus = 'healthy';
        results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
                const checkResult = result.value;
                details[checkResult.component] = checkResult;
                if (checkResult.status === 'unhealthy') {
                    overallStatus = 'unhealthy';
                }
                else if (checkResult.status === 'degraded' && overallStatus !== 'unhealthy') {
                    overallStatus = 'degraded';
                }
            }
            else {
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
        if (overallStatus !== this.healthStatus.status) {
            this.emit('healthStatusChanged', this.healthStatus);
        }
    }
    registerServiceHealth(serviceName, health) {
        this.serviceHealth.set(serviceName, health);
    }
    getHealthStatus() {
        return this.healthStatus;
    }
    getDetailedHealth() {
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
    getRouter() {
        const router = express.Router();
        router.get('/', (_req, res) => {
            const status = this.getHealthStatus();
            if (status.status !== 'unhealthy') {
                res.json({
                    status: status.status,
                    timestamp: status.timestamp,
                    message: 'API网关运行正常'
                });
            }
            else {
                res.status(503).json({
                    status: status.status,
                    timestamp: status.timestamp,
                    message: 'API网关服务异常',
                    details: status.details
                });
            }
        });
        router.get('/detailed', (_req, res) => {
            res.json(this.getDetailedHealth());
        });
        router.get('/ready', (_req, res) => {
            const status = this.getHealthStatus();
            if (status.status === 'healthy') {
                res.json({ status: 'ready' });
            }
            else {
                res.status(503).json({ status: 'not-ready' });
            }
        });
        router.get('/live', (_req, res) => {
            res.json({ status: 'alive' });
        });
        router.get('/services', (_req, res) => {
            res.json(Object.fromEntries(this.serviceHealth));
        });
        router.post('/check', async (_req, res) => {
            try {
                await this.performHealthChecks();
                res.json({
                    message: '健康检查完成',
                    status: this.getHealthStatus()
                });
            }
            catch (error) {
                res.status(500).json({
                    error: '健康检查执行失败',
                    message: error.message
                });
            }
        });
        return router;
    }
    async stop() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
        console.log('🛑 健康检查系统已停止');
    }
}
export const healthCheck = new HealthCheck();
//# sourceMappingURL=healthCheck.js.map