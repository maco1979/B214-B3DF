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
declare class LoadBalancer {
    private serviceInstances;
    private currentIndices;
    private strategy;
    registerInstance(serviceName: string, instance: ServiceInstance): void;
    unregisterInstance(serviceName: string, instanceId: string): void;
    selectInstance(serviceName: string, instances: ServiceInstance[]): ServiceInstance;
    private roundRobinSelect;
    private leastConnectionsSelect;
    private weightedRoundRobinSelect;
    private responseTimeSelect;
    updateInstanceHealth(serviceName: string, instanceId: string, health: boolean): void;
    updateResponseTime(serviceName: string, instanceId: string, responseTime: number): void;
    incrementConnections(serviceName: string, instanceId: string): void;
    decrementConnections(serviceName: string, instanceId: string): void;
    setStrategy(strategy: LoadBalancingStrategy): void;
    getStatus(): any;
    getStatusRouter(): express.Router;
}
export declare const loadBalancer: LoadBalancer;
export {};
//# sourceMappingURL=loadBalancer.d.ts.map