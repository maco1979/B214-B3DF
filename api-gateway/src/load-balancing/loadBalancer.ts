import express from 'express';

interface ServiceInstance {
  id: string;
  host: string;
  port: number;
  weight: number;
  health: boolean;
  lastHealthCheck: Date;
  responseTime: number;
  activeConnections: number;
}

type LoadBalancingStrategy = 'round-robin' | 'least-connections' | 'weighted-round-robin' | 'response-time';

class LoadBalancer {
  private serviceInstances: Map<string, ServiceInstance[]> = new Map();
  private currentIndices: Map<string, number> = new Map();
  private strategy: LoadBalancingStrategy = 'round-robin';

  // 注册服务实例
  public registerInstance(serviceName: string, instance: ServiceInstance): void {
    if (!this.serviceInstances.has(serviceName)) {
      this.serviceInstances.set(serviceName, []);
      this.currentIndices.set(serviceName, 0);
    }

    const instances = this.serviceInstances.get(serviceName)!;
    const existingIndex = instances.findIndex(i => i.id === instance.id);
    
    if (existingIndex >= 0) {
      instances[existingIndex] = instance;
    } else {
      instances.push(instance);
    }

    console.log(`✅ 注册服务实例: ${serviceName} -> ${instance.host}:${instance.port}`);
  }

  // 注销服务实例
  public unregisterInstance(serviceName: string, instanceId: string): void {
    const instances = this.serviceInstances.get(serviceName);
    if (instances) {
      const index = instances.findIndex(i => i.id === instanceId);
      if (index >= 0) {
        instances.splice(index, 1);
        console.log(`❌ 注销服务实例: ${serviceName} -> ${instanceId}`);
      }
    }
  }

  // 选择实例
  public selectInstance(serviceName: string, instances: ServiceInstance[]): ServiceInstance {
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

  // 轮询选择
  private roundRobinSelect(serviceName: string, instances: ServiceInstance[]): ServiceInstance {
    if (instances.length === 0) {
      throw new Error('No instances available for service');
    }
    let currentIndex = this.currentIndices.get(serviceName) || 0;
    const selectedInstance = instances[currentIndex % instances.length]!;
    
    this.currentIndices.set(serviceName, (currentIndex + 1) % instances.length);
    return selectedInstance;
  }

  // 最少连接选择
  private leastConnectionsSelect(instances: ServiceInstance[]): ServiceInstance {
    return instances.reduce((prev, current) => 
      prev.activeConnections < current.activeConnections ? prev : current
    );
  }

  // 加权轮询选择
  private weightedRoundRobinSelect(serviceName: string, instances: ServiceInstance[]): ServiceInstance {
    if (instances.length === 0) {
      throw new Error('No instances available for service');
    }
    const totalWeight = instances.reduce((sum, instance) => sum + instance.weight, 0);
    let currentIndex = this.currentIndices.get(serviceName) || 0;
    
    // 加权轮询算法
    let currentWeight = 0;
    let selectedInstance: ServiceInstance | null = null;
    
    for (let i = 0; i < instances.length; i++) {
      const instance = instances[(currentIndex + i) % instances.length]!;
      currentWeight += instance.weight;
      
      if (currentWeight >= totalWeight) {
        selectedInstance = instance;
        break;
      }
    }
    
    this.currentIndices.set(serviceName, (currentIndex + 1) % instances.length);
    // 确保selectedInstance不为null
    return selectedInstance || instances[0]!;
  }

  // 响应时间选择
  private responseTimeSelect(instances: ServiceInstance[]): ServiceInstance {
    return instances.reduce((prev, current) => 
      prev.responseTime < current.responseTime ? prev : current
    );
  }

  // 更新实例健康状态
  public updateInstanceHealth(serviceName: string, instanceId: string, health: boolean): void {
    const instances = this.serviceInstances.get(serviceName);
    if (instances) {
      const instance = instances.find(i => i.id === instanceId);
      if (instance) {
        instance.health = health;
        instance.lastHealthCheck = new Date();
      }
    }
  }

  // 更新实例响应时间
  public updateResponseTime(serviceName: string, instanceId: string, responseTime: number): void {
    const instances = this.serviceInstances.get(serviceName);
    if (instances) {
      const instance = instances.find(i => i.id === instanceId);
      if (instance) {
        instance.responseTime = responseTime;
      }
    }
  }

  // 增加连接计数
  public incrementConnections(serviceName: string, instanceId: string): void {
    const instances = this.serviceInstances.get(serviceName);
    if (instances) {
      const instance = instances.find(i => i.id === instanceId);
      if (instance) {
        instance.activeConnections++;
      }
    }
  }

  // 减少连接计数
  public decrementConnections(serviceName: string, instanceId: string): void {
    const instances = this.serviceInstances.get(serviceName);
    if (instances) {
      const instance = instances.find(i => i.id === instanceId);
      if (instance && instance.activeConnections > 0) {
        instance.activeConnections--;
      }
    }
  }

  // 设置负载均衡策略
  public setStrategy(strategy: LoadBalancingStrategy): void {
    this.strategy = strategy;
    console.log(`🔧 负载均衡策略已更新: ${strategy}`);
  }

  // 获取负载均衡器状态
  public getStatus(): any {
    const status: any = {
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

  // 获取状态路由
  public getStatusRouter(): express.Router {
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
        this.setStrategy(strategy as LoadBalancingStrategy);
        res.json({ message: '策略更新成功', strategy });
      } else {
        res.status(400).json({ error: '无效的负载均衡策略' });
      }
    });

    return router;
  }
}

export const loadBalancer = new LoadBalancer();