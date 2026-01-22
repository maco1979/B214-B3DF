/**
 * 指标中间件
 */

import { Request, Response, NextFunction } from 'express';

export const metricsMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  
  // 记录请求指标
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const statusCode = res.statusCode;
    
    // 在实际应用中，这里应该将指标发送到Prometheus或其他监控系统
    console.log(`METRICS: ${req.method} ${req.path} - ${statusCode} - ${duration}ms`);
  });
  
  next();
};