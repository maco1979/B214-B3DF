/**
 * 认证中间件
 */

import { Request, Response, NextFunction } from 'express';

export const authMiddleware = (req: Request, res: Response, next: NextFunction): void => {
  // 放行预检请求
  if (req.method === 'OPTIONS') {
    return next();
  }

  // 简单的API密钥认证
  const apiKey = req.headers['x-api-key'];

  
  if (!apiKey) {
    res.status(401).json({
      error: '未提供API密钥',
      message: '请在请求头中提供有效的X-API-KEY'
    });
    return;
  }
  
  // 在实际应用中，这里应该验证API密钥的有效性
  const validApiKeys = ['your-api-key-here', 'default-api-key'];

  
  if (!validApiKeys.includes(apiKey as string)) {
    res.status(403).json({
      error: '无效的API密钥',
      message: '提供的API密钥无效'
    });
    return;
  }
  
  next();
};