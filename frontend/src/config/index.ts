// 应用配置

// API基础URL
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';

// API密钥
export const API_KEY = import.meta.env.VITE_API_KEY || 'your-api-key-here'

// 应用名称
export const APP_NAME = 'AI项目管理系统'

// 版本信息
export const APP_VERSION = '1.0.0'

// 主题配置
export const THEME_CONFIG = {
  primaryColor: '#3b82f6',
  secondaryColor: '#6366f1',
  accentColor: '#8b5cf6',
  successColor: '#10b981',
  warningColor: '#f59e0b',
  errorColor: '#ef4444',
  infoColor: '#06b6d4',
}

// 功能开关
export const FEATURE_FLAGS = {
  ENABLE_BLOCKCHAIN: true,
  ENABLE_FEDERATED_LEARNING: true,
  ENABLE_EDGE_COMPUTING: true,
  ENABLE_MONITORING: true,
}

// API端点配置
export const API_ENDPOINTS = {
  MODELS: '/models',
  INFERENCE: '/inference',
  TRAINING: '/training',
  BLOCKCHAIN: '/blockchain',
  SYSTEM: '/system',
  EDGE: '/edge',
}

// 本地存储键名
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  USER_PREFERENCES: 'user_preferences',
  RECENT_MODELS: 'recent_models',
  THEME_MODE: 'theme_mode',
}

// 默认配置
export const DEFAULT_CONFIG = {
  INFERENCE_TIMEOUT: 30000, // 30秒
  TRAINING_TIMEOUT: 600000, // 10分钟
  POLLING_INTERVAL: 5000, // 5秒
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
}

// 导出所有配置
export default {
  API_BASE_URL,
  API_KEY,
  APP_NAME,
  APP_VERSION,
  THEME_CONFIG,
  FEATURE_FLAGS,
  API_ENDPOINTS,
  STORAGE_KEYS,
  DEFAULT_CONFIG,
}