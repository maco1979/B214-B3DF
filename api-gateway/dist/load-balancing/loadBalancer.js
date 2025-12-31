import express from 'express';
class LoadBalancer {
    serviceInstances = new Map();
    currentIndices = new Map();
    strategy = 'round-robin';
    registerInstance(serviceName, instance) {
        if (!this.serviceInstances.has(serviceName)) {
            this.serviceInstances.set(serviceName, []);
            this.currentIndices.set(serviceName, 0);
        }
        const instances = this.serviceInstances.get(serviceName);
        const existingIndex = instances.findIndex(i => i.id === instance.id);
        if (existingIndex >= 0) {
            instances[existingIndex] = instance;
        }
        else {
            instances.push(instance);
        }
        console.log(`✅ 注册服务实例: ${serviceName} -> ${instance.host}:${instance.port}`);
    }
    unregisterInstance(serviceName, instanceId) {
        const instances = this.serviceInstances.get(serviceName);
        if (instances) {
            const index = instances.findIndex(i => i.id === instanceId);
            if (index >= 0) {
                instances.splice(index, 1);
                console.log(`❌ 注销服务实例: ${serviceName} -> ${instanceId}`);
            }
        }
    }
    selectInstance(serviceName, instances) {
        const healthyInstances = instances.filter(instance => instance.health);
        if (healthyInstances.length === 0) {
            throw new Error(`服务 ${serviceName} 没有可用的健康实例`);
        }
        switch (this.strategy) {
            case 'round-robin':
                return this.roundRobinSelect(serviceName, healthyInstances);
            case 'least-connections':
                return this.leastConnectionsSelect(healthyInstances);
            case 'weighted-round-robin':
                return this.weightedRoundRobinSelect(serviceName, healthyInstances);
            case 'response-time':
                return this.responseTimeSelect(healthyInstances);
            default:
                return this.roundRobinSelect(serviceName, healthyInstances);
        }
    }
    roundRobinSelect(serviceName, instances) {
        if (instances.length === 0) {
            throw new Error('No instances available for service');
        }
        let currentIndex = this.currentIndices.get(serviceName) || 0;
        const selectedInstance = instances[currentIndex % instances.length];
        this.currentIndices.set(serviceName, (currentIndex + 1) % instances.length);
        return selectedInstance;
    }
    leastConnectionsSelect(instances) {
        return instances.reduce((prev, current) => prev.activeConnections < current.activeConnections ? prev : current);
    }
    weightedRoundRobinSelect(serviceName, instances) {
        if (instances.length === 0) {
            throw new Error('No instances available for service');
        }
        const totalWeight = instances.reduce((sum, instance) => sum + instance.weight, 0);
        let currentIndex = this.currentIndices.get(serviceName) || 0;
        let currentWeight = 0;
        let selectedInstance = null;
        for (let i = 0; i < instances.length; i++) {
            const instance = instances[(currentIndex + i) % instances.length];
            currentWeight += instance.weight;
            if (currentWeight >= totalWeight) {
                selectedInstance = instance;
                break;
            }
        }
        this.currentIndices.set(serviceName, (currentIndex + 1) % instances.length);
        return selectedInstance || instances[0];
    }
    responseTimeSelect(instances) {
        return instances.reduce((prev, current) => prev.responseTime < current.responseTime ? prev : current);
    }
    updateInstanceHealth(serviceName, instanceId, health) {
        const instances = this.serviceInstances.get(serviceName);
        if (instances) {
            const instance = instances.find(i => i.id === instanceId);
            if (instance) {
                instance.health = health;
                instance.lastHealthCheck = new Date();
            }
        }
    }
    updateResponseTime(serviceName, instanceId, responseTime) {
        const instances = this.serviceInstances.get(serviceName);
        if (instances) {
            const instance = instances.find(i => i.id === instanceId);
            if (instance) {
                instance.responseTime = responseTime;
            }
        }
    }
    incrementConnections(serviceName, instanceId) {
        const instances = this.serviceInstances.get(serviceName);
        if (instances) {
            const instance = instances.find(i => i.id === instanceId);
            if (instance) {
                instance.activeConnections++;
            }
        }
    }
    decrementConnections(serviceName, instanceId) {
        const instances = this.serviceInstances.get(serviceName);
        if (instances) {
            const instance = instances.find(i => i.id === instanceId);
            if (instance && instance.activeConnections > 0) {
                instance.activeConnections--;
            }
        }
    }
    setStrategy(strategy) {
        this.strategy = strategy;
        console.log(`🔧 负载均衡策略已更新: ${strategy}`);
    }
    getStatus() {
        const status = {
            strategy: this.strategy,
            services: {}
        };
        for (const [serviceName, instances] of this.serviceInstances) {
            status.services[serviceName] = {
                totalInstances: instances.length,
                healthyInstances: instances.filter(i => i.health).length,
                instances: instances.map(i => ({
                    id: i.id,
                    host: i.host,
                    port: i.port,
                    health: i.health,
                    activeConnections: i.activeConnections,
                    responseTime: i.responseTime
                }))
            };
        }
        return status;
    }
    getStatusRouter() {
        const router = express.Router();
        router.get('/status', (_req, res) => {
            res.json(this.getStatus());
        });
        router.get('/strategy', (_req, res) => {
            res.json({ strategy: this.strategy });
        });
        router.post('/strategy', (req, res) => {
            const { strategy } = req.body;
            if (['round-robin', 'least-connections', 'weighted-round-robin', 'response-time'].includes(strategy)) {
                this.setStrategy(strategy);
                res.json({ message: '策略更新成功', strategy });
            }
            else {
                res.status(400).json({ error: '无效的负载均衡策略' });
            }
        });
        return router;
    }
}
export const loadBalancer = new LoadBalancer();
//# sourceMappingURL=loadBalancer.js.map