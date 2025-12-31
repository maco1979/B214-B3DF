import express from 'express';
import { EventEmitter } from 'events';
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
declare class ServiceDiscovery extends EventEmitter {
    private services;
    private heartbeatInterval;
    private heartbeatTimeout;
    initialize(): Promise<void>;
    registerService(serviceInfo: ServiceInfo): string;
    unregisterService(serviceName: string, instanceId: string): boolean;
    getServiceInstances(serviceName: string): Promise<ServiceInstance[]>;
    updateHeartbeat(serviceName: string, instanceId: string): boolean;
    private startHeartbeatCheck;
    private checkHeartbeats;
    private preRegisterServices;
    private generateInstanceId;
    getStatus(): any;
    getRouter(): express.Router;
    shutdown(): Promise<void>;
}
export declare const serviceDiscovery: ServiceDiscovery;
export {};
//# sourceMappingURL=serviceDiscovery.d.ts.map