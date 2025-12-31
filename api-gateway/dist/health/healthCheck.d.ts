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
declare class HealthCheck extends EventEmitter {
    private healthStatus;
    private serviceHealth;
    private checkInterval;
    start(): Promise<void>;
    private performHealthChecks;
    private checkGatewayHealth;
    private checkMemoryUsage;
    private checkDatabaseConnection;
    private checkExternalServices;
    private updateHealthStatus;
    registerServiceHealth(serviceName: string, health: ServiceHealth): void;
    getHealthStatus(): HealthStatus;
    getDetailedHealth(): any;
    getRouter(): express.Router;
    stop(): Promise<void>;
}
export declare const healthCheck: HealthCheck;
export {};
//# sourceMappingURL=healthCheck.d.ts.map