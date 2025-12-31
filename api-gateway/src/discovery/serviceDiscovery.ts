import express from 'express';
import { EventEmitter } from 'events';
import { loadBalancer } from '../load-balancing/loadBalancer.js';

interface ServiceInfo {
  name: string;
  version: string;
  host: string;
  port: number;
  endpoints: string[];
  metadata?: Record<string, any>;
}

interface ServiceInstance extends ServiceInfo {
  id: string;
  lastHeartbeat: Date;
  health: boolean;
  weight: number;
}

class ServiceDiscovery extends EventEmitter {
  private services: Map<string, ServiceInstance[]> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private heartbeatTimeout = 30000; // 30秒心跳超时

  // 初始化服务发现
  public async initialize(): Promise<void> {
    console.log('🔍 初始化服务发现系统...');
    
    // 启动心跳检测
    this.startHeartbeatCheck();
    
    // 预注册已知服务（从配置或环境变量）
    await this.preRegisterServices();
    
    console.log('✅ 服务发现系统初始化完成');
  }

  // 注册服务
  public registerService(serviceInfo: ServiceInfo): string {
    const instanceId = this.generateInstanceId(serviceInfo);
    
    const instance: ServiceInstance = {
      ...serviceInfo,
      id: instanceId,
      lastHeartbeat: new Date(),
      health: true,
      weight: 1
    };

    if (!this.services.has(serviceInfo.name)) {
      this.services.set(serviceInfo.name, []);
    }

    const instances = this.services.get(serviceInfo.name)!;
    const existingIndex = instances.findIndex(i => i.id === instanceId);
    
    if (existingIndex >= 0) {
      instances[existingIndex] = instance;
    } else {
      instances.push(instance);
    }

    // 注册到负载均衡器
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

  // 注销服务
  public unregisterService(serviceName: string, instanceId: string): boolean {
    const instances = this.services.get(serviceName);
    if (instances) {
      const index = instances.findIndex(i => i.id === instanceId);
      if (index >= 0) {
        instances.splice(index, 1);
        
        // 从负载均衡器注销
        loadBalancer.unregisterInstance(serviceName, instanceId);
        
        this.emit('serviceUnregistered', { serviceName, instanceId });
        console.log(`❌ 服务注销: ${serviceName} (${instanceId})`);
        return true;
      }
    }
    return false;
  }

  // 获取服务实例
  public async getServiceInstances(serviceName: string): Promise<ServiceInstance[]> {
    const instances = this.services.get(serviceName) || [];
    return instances.filter(instance => instance.health);
  }

  // 心跳更新
  public updateHeartbeat(serviceName: string, instanceId: string): boolean {
    const instances = this.services.get(serviceName);
    if (instances) {
      const instance = instances.find(i => i.id === instanceId);
      if (instance) {
        instance.lastHeartbeat = new Date();
        instance.health = true;
        
        // 更新负载均衡器健康状态
        loadBalancer.updateInstanceHealth(serviceName, instanceId, true);
        
        return true;
      }
    }
    return false;
  }

  // 启动心跳检测
  private startHeartbeatCheck(): void {
    this.heartbeatInterval = setInterval(() => {
      this.checkHeartbeats();
    }, 10000); // 每10秒检查一次
  }

  // 检查心跳
  private checkHeartbeats(): void {
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

  // 预注册服务
  private async preRegisterServices(): Promise<void> {
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

  // 生成实例ID
  private generateInstanceId(serviceInfo: ServiceInfo): string {
    return `${serviceInfo.name}-${serviceInfo.host}-${serviceInfo.port}-${Date.now()}`;
  }

  // 获取服务发现状态
  public getStatus(): any {
    const status: any = {
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

  // 获取路由
  public getRouter(): express.Router {
    const router = express.Router();

    // 服务注册接口
    router.post('/register', (req, res) => {
      try {
        const serviceInfo: ServiceInfo = req.body;
        const instanceId = this.registerService(serviceInfo);
        res.json({ instanceId, message: '服务注册成功' });
      } catch (error) {
        res.status(400).json({ error: '服务注册失败', message: (error as Error).message });
      }
    });

    // 服务注销接口
    router.post('/unregister', (req, res) => {
      const { serviceName, instanceId } = req.body;
      const success = this.unregisterService(serviceName, instanceId);
      
      if (success) {
        res.json({ message: '服务注销成功' });
      } else {
        res.status(404).json({ error: '服务实例未找到' });
      }
    });

    // 心跳接口
    router.post('/heartbeat', (req, res) => {
      const { serviceName, instanceId } = req.body;
      const success = this.updateHeartbeat(serviceName, instanceId);
      
      if (success) {
        res.json({ message: '心跳更新成功' });
      } else {
        res.status(404).json({ error: '服务实例未找到' });
      }
    });

    // 服务发现状态
    router.get('/status', (_req, res) => {
      res.json(this.getStatus());
    });

    // 获取服务实例
    router.get('/services/:serviceName', async (req, res) => {
      const { serviceName } = req.params;
      const instances = await this.getServiceInstances(serviceName);
      res.json({ serviceName, instances });
    });

    return router;
  }

  // 关闭服务发现
  public async shutdown(): Promise<void> {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    console.log('🛑 服务发现系统已关闭');
  }
}

export const serviceDiscovery = new ServiceDiscovery();