import { createProxyMiddleware } from 'http-proxy-middleware';
const serviceRoutes = {
    backend: {
        baseUrl: 'http://127.0.0.1:8002',
        prefix: '/api',
        healthCheck: '/api/health'
    },
    decisionService: {
        baseUrl: 'http://127.0.0.1:8009',
        prefix: '/api/decision',
        healthCheck: '/api/decision/health'
    }
};
export const routeRules = {
    '/api/decision': serviceRoutes.decisionService,
    '/api/models': serviceRoutes.backend,
    '/api/inference': serviceRoutes.backend,
    '/api/training': serviceRoutes.backend,
    '/api/system': serviceRoutes.backend,
    '/api/edge': serviceRoutes.backend,
    '/api/federated': serviceRoutes.backend,
    '/api/agriculture': serviceRoutes.backend,
    '/api/performance': serviceRoutes.backend,
    '/api/blockchain': serviceRoutes.backend,
    '/api/health': serviceRoutes.backend
};
export const decisionProxy = createProxyMiddleware({
    target: serviceRoutes.decisionService.baseUrl + '/api/decision',
    changeOrigin: true,
    pathRewrite: {
        '^/api/decision': '',
    },
    logLevel: 'debug'
});
export const backendProxy = createProxyMiddleware({
    target: serviceRoutes.backend.baseUrl + '/api',
    changeOrigin: true,
    pathRewrite: {
        '^/api': '',
    },
    logLevel: 'debug'
});
export function getTargetProxy(req) {
    const path = req.path;
    if (path.startsWith('/api/decision')) {
        return decisionProxy;
    }
    return backendProxy;
}
export const routerManager = {
    decisionProxy,
    backendProxy,
    getTargetService(req) {
        const path = req.path;
        for (const [routePrefix, serviceConfig] of Object.entries(routeRules)) {
            if (path.startsWith(routePrefix)) {
                return serviceConfig;
            }
        }
        return serviceRoutes.backend;
    },
    buildTargetUrl(req, serviceConfig) {
        return `${serviceConfig.baseUrl}${req.path}`;
    },
    routeMiddleware(req, res, next) {
        try {
            console.log(`路由中间件处理请求: ${req.path}`);
            if (req.path.startsWith('/decision')) {
                console.log(`转发到决策服务: /api${req.path}`);
                return decisionProxy(req, res, next);
            }
            console.log(`转发到后端服务: /api${req.path}`);
            return backendProxy(req, res, next);
        }
        catch (error) {
            console.error('路由转发错误:', error);
            next(error);
        }
    },
    healthCheck(_req, res) {
        res.json({
            status: 'healthy',
            service: 'api-gateway',
            timestamp: new Date().toISOString(),
            version: '1.0.0'
        });
    }
};
//# sourceMappingURL=router.js.map