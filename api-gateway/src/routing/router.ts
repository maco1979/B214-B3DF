/**
 * 路由管理器
 */

import { Request, Response, NextFunction } from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';

// 服务路由配置
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


// 路由规则
export const routeRules = {
  // 决策服务路由 - 更具体的规则放在前面
  '/api/decision': serviceRoutes.decisionService,
  
  // 后端服务路由
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

// 创建代理中间件
export const decisionProxy = createProxyMiddleware({
  target: serviceRoutes.decisionService.baseUrl + '/api/decision',
  changeOrigin: true,
  pathRewrite: {
    '^/api/decision': '', // 如果是直接使用的话
  },
  logLevel: 'debug'
});

// 后端服务代理
export const backendProxy = createProxyMiddleware({
  target: serviceRoutes.backend.baseUrl + '/api',
  changeOrigin: true,
  pathRewrite: {
    '^/api': '', // 如果是直接使用的话
  },
  logLevel: 'debug'
});


/**
 * 获取目标服务代理
 */
export function getTargetProxy(req: Request) {
  const path = req.path;
  
  // 决策服务路由
  if (path.startsWith('/api/decision')) {
    return decisionProxy;
  }
  
  // 默认使用后端服务代理
  return backendProxy;
}

export const routerManager = {
  decisionProxy,
  backendProxy,
  
  /**
   * 获取目标服务配置

   */
  getTargetService(req: Request) {
    const path = req.path;
    
    // 查找匹配的路由规则
    for (const [routePrefix, serviceConfig] of Object.entries(routeRules)) {
      if (path.startsWith(routePrefix)) {
        return serviceConfig;
      }
    }
    
    // 默认路由到后端服务
    return serviceRoutes.backend;
  },
  
  /**
   * 构建目标URL
   */
  buildTargetUrl(req: Request, serviceConfig: any): string {
    // 保留完整路径，因为后端服务已经配置了完整的路由前缀
    return `${serviceConfig.baseUrl}${req.path}`;
  },
  
  /**
   * 路由中间件
   */
  routeMiddleware(req: Request, res: Response, next: NextFunction) {
    try {
      console.log(`路由中间件处理请求: ${req.path}`);
      
      // 决策服务路由 - 注意：req.path不包含/api前缀，因为中间件挂载在/api路径下
      if (req.path.startsWith('/decision')) {
        console.log(`转发到决策服务: /api${req.path}`);
        return decisionProxy(req, res, next);
      }
      
      // 默认使用后端服务代理
      console.log(`转发到后端服务: /api${req.path}`);
      return backendProxy(req, res, next);
    } catch (error) {
      console.error('路由转发错误:', error);
      next(error);
    }
  },
  
  /**
   * 健康检查路由
   */
  healthCheck(_req: Request, res: Response) {
    res.json({
      status: 'healthy',
      service: 'api-gateway',
      timestamp: new Date().toISOString(),
      version: '1.0.0'
    });
  }
};