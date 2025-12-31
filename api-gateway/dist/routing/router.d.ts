import { Request, Response, NextFunction } from 'express';
export declare const routeRules: {
    '/api/decision': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/models': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/inference': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/training': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/system': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/edge': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/federated': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/agriculture': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/performance': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/blockchain': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    '/api/health': {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
};
export declare const decisionProxy: import("http-proxy-middleware").RequestHandler;
export declare const backendProxy: import("http-proxy-middleware").RequestHandler;
export declare function getTargetProxy(req: Request): import("http-proxy-middleware").RequestHandler;
export declare const routerManager: {
    decisionProxy: import("http-proxy-middleware").RequestHandler;
    backendProxy: import("http-proxy-middleware").RequestHandler;
    getTargetService(req: Request): {
        baseUrl: string;
        prefix: string;
        healthCheck: string;
    };
    buildTargetUrl(req: Request, serviceConfig: any): string;
    routeMiddleware(req: Request, res: Response, next: NextFunction): void;
    healthCheck(_req: Request, res: Response): void;
};
//# sourceMappingURL=router.d.ts.map