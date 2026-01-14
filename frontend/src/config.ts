// API配置（环境变量优先，保留默认回退）
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL) || '/api';
export const API_KEY = (import.meta.env.VITE_API_KEY as string) || 'default-api-key';

// 其他配置
export const APP_VERSION = '1.0.0';
export const APP_NAME = 'AI农业系统';
