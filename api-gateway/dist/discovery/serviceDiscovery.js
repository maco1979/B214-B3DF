import express from 'express';
import { EventEmitter } from 'events';
import { loadBalancer } from '../load-balancing/loadBalancer.js';
class ServiceDiscovery extends EventEmitter {
    services = new Map();
    heartbeatInterval = null;
    heartbeatTimeout = 30000;
    async initialize() {
        console.log('🔍 初始化服务发现系统...');
        this.startHeartbeatCheck();
        await this.preRegisterServices();
        console.log('✅ 服务发现系统初始化完成');
    }
    registerService(serviceInfo) {
        const instanceId = this.generateInstanceId(serviceInfo);
        const instance = {
            ...serviceInfo,
            id: instanceId,
            lastHeartbeat: new Date(),
            health: true,
            weight: 1
        };
        if (!this.services.has(serviceInfo.name)) {
            this.services.set(serviceInfo.name, []);
        }
        const instances = this.services.get(serviceInfo.name);
        const existingIndex = instances.findIndex(i => i.id === instanceId);
        if (existingIndex >= 0) {
            instances[existingIndex] = instance;
        }
        else {
            instances.push(instance);
        }
        loadBalancer.registerInstance(serviceInfo.name, {
            id: instanceId,
            host: serviceInfo.host,
            port: serviceInfo.port,
            weight: instance.weight,
            health: instance.health,
            lastHealthCheck: instance.lastHeartbeat,
            responseTime: 0,
            activeConnections: 0
        });
        this.emit('serviceRegistered', instance);
        console.log(`✅ 服务注册成功: ${serviceInfo.name} (${instanceId})`);
        return instanceId;
    }
    unregisterService(serviceName, instanceId) {
        const instances = this.services.get(serviceName);
        if (instances) {
            const index = instances.findIndex(i => i.id === instanceId);
            if (index >= 0) {
                instances.splice(index, 1);
                loadBalancer.unregisterInstance(serviceName, instanceId);
                this.emit('serviceUnregistered', { serviceName, instanceId });
                console.log(`❌ 服务注销: ${serviceName} (${instanceId})`);
                return true;
            }
        }
        return false;
    }
    async getServiceInstances(serviceName) {
        const instances = this.services.get(serviceName) || [];
        return instances.filter(instance => instance.health);
    }
    updateHeartbeat(serviceName, instanceId) {
        const instances = this.services.get(serviceName);
        if (instances) {
            const instance = instances.find(i => i.id === instanceId);
            if (instance) {
                instance.lastHeartbeat = new Date();
                instance.health = true;
                loadBalancer.updateInstanceHealth(serviceName, instanceId, true);
                return true;
            }
        }
        return false;
    }
    startHeartbeatCheck() {
        this.heartbeatInterval = setInterval(() => {
            this.checkHeartbeats();
        }, 10000);
    }
    checkHeartbeats() {
        const now = new Date();
        for (const [serviceName, instances] of this.services) {
            for (const instance of instances) {
                const timeSinceLastHeartbeat = now.getTime() - instance.lastHeartbeat.getTime();
                if (timeSinceLastHeartbeat > this.heartbeatTimeout) {
                    if (instance.health) {
                        instance.health = false;
                        loadBalancer.updateInstanceHealth(serviceName, instance.id, false);
                        console.log(`⚠️  服务心跳超时: ${serviceName} (${instance.id})`);
                        this.emit('serviceUnhealthy', instance);
                    }
                }
            }
        }
    }
    async preRegisterServices() {
        const predefinedServices = [
            { name: 'decision', port: 8009 },
            { name: 'blockchain', port: 8002 },
            { name: 'federated', port: 8003 },
            { name: 'edge', port: 8004 },
            { name: 'performance', port: 8005 },
            { name: 'model', port: 8006 },
            { name: 'data', port: 8007 }
        ];
        for (const service of predefinedServices) {
            this.registerService({
                name: service.name,
                version: '1.0.0',
                host: 'localhost',
                port: service.port,
                endpoints: [`/api/${service.name}`]
            });
        }
    }
    generateInstanceId(serviceInfo) {
        return `${serviceInfo.name}-${serviceInfo.host}-${serviceInfo.port}-${Date.now()}`;
    }
    getStatus() {
        const status = {
            totalServices: this.services.size,
            services: {}
        };
        for (const [serviceName, instances] of this.services) {
            status.services[serviceName] = {
                totalInstances: instances.length,
                healthyInstances: instances.filter(i => i.health).length,
                instances: instances.map(i => ({
                    id: i.id,
                    host: i.host,
                    port: i.port,
                    health: i.health,
                    lastHeartbeat: i.lastHeartbeat
                }))
            };
        }
        return status;
    }
    getRouter() {
        const router = express.Router();
        router.post('/register', (req, res) => {
            try {
                const serviceInfo = req.body;
                const instanceId = this.registerService(serviceInfo);
                res.json({ instanceId, message: '服务注册成功' });
            }
            catch (error) {
                res.status(400).json({ error: '服务注册失败', message: error.message });
            }
        });
        router.post('/unregister', (req, res) => {
            const { serviceName, instanceId } = req.body;
            const success = this.unregisterService(serviceName, instanceId);
            if (success) {
                res.json({ message: '服务注销成功' });
            }
            else {
                res.status(404).json({ error: '服务实例未找到' });
            }
        });
        router.post('/heartbeat', (req, res) => {
            const { serviceName, instanceId } = req.body;
            const success = this.updateHeartbeat(serviceName, instanceId);
            if (success) {
                res.json({ message: '心跳更新成功' });
            }
            else {
                res.status(404).json({ error: '服务实例未找到' });
            }
        });
        router.get('/status', (_req, res) => {
            res.json(this.getStatus());
        });
        router.get('/services/:serviceName', async (req, res) => {
            const { serviceName } = req.params;
            const instances = await this.getServiceInstances(serviceName);
            res.json({ serviceName, instances });
        });
        return router;
    }
    async shutdown() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        console.log('🛑 服务发现系统已关闭');
    }
}
export const serviceDiscovery = new ServiceDiscovery();
//# sourceMappingURL=serviceDiscovery.js.map