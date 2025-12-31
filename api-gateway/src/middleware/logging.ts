/**
 * 日志中间件
 */

import { Request, Response, NextFunction } from 'express';

export const loggingMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  
  // 记录请求开始
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} - 开始处理`);
  
  // 监听响应完成
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} - ${res.statusCode} - ${duration}ms`);
  });
  
  next();
};