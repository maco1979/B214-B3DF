/**
 * 错误处理中间件
 */

import { Request, Response, NextFunction } from 'express';

export const errorHandler = (err: any, _req: Request, res: Response, _next: NextFunction): void => {
  console.error('API网关错误:', err);
  
  // 处理不同类型的错误
  if (err.code === 'ECONNREFUSED') {
    res.status(503).json({
      error: '服务不可用',
      message: '后端服务暂时不可用，请稍后重试'
    });
    return;
  }
  
  if (err.response && err.response.status) {
    // 代理错误
    res.status(err.response.status).json({
      error: '后端服务错误',
      message: err.response.data?.message || '后端服务处理错误'
    });
    return;
  }
  
  // 默认错误处理
  res.status(500).json({
    error: '内部服务器错误',
    message: '服务器遇到意外错误'
  });
};