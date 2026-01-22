// API请求相关类型
export interface ApiRequest {
  id: string;
  name: string;
  url: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  queryParams?: Record<string, string | number | boolean>;
  body?: any;
  auth?: AuthConfig;
  timeout?: number;
  retryConfig?: RetryConfig;
}

// 认证配置类型
export type AuthType = 'none' | 'apiKey' | 'basic' | 'oauth2' | 'jwt';

export interface AuthConfig {
  type: AuthType;
  config: {
    [key: string]: any;
  };
}

// API Key认证配置
export interface ApiKeyAuthConfig {
  type: 'apiKey';
  config: {
    key: string;
    value: string;
    in: 'header' | 'query';
  };
}

// Basic认证配置
export interface BasicAuthConfig {
  type: 'basic';
  config: {
    username: string;
    password: string;
  };
}

// OAuth2认证配置
export interface OAuth2AuthConfig {
  type: 'oauth2';
  config: {
    clientId: string;
    clientSecret: string;
    tokenUrl: string;
    scope?: string;
    grantType?: 'client_credentials' | 'password' | 'authorization_code' | 'refresh_token';
    username?: string;
    password?: string;
    refreshToken?: string;
  };
}

// JWT认证配置
export interface JwtAuthConfig {
  type: 'jwt';
  config: {
    token: string;
    in?: 'header' | 'query';
    prefix?: string;
  };
}

// 重试配置类型
export interface RetryConfig {
  maxRetries: number;
  delay: number;
  exponentialBackoff?: boolean;
}

// API响应类型
export interface ApiResponse {
  id: string;
  requestId: string;
  statusCode: number;
  headers: Record<string, string>;
  body: any;
  duration: number;
  timestamp: number;
}

// 错误处理类型
export interface ApiError {
  id: string;
  requestId: string;
  message: string;
  statusCode?: number;
  stack?: string;
  timestamp: number;
}

// 工作流相关类型
export interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps: WorkflowStep[];
  triggers?: WorkflowTrigger[];
  createdAt: number;
  updatedAt: number;
}

export type WorkflowStepType = 'apiCall' | 'condition' | 'loop' | 'parallel';

export interface WorkflowStep {
  id: string;
  type: WorkflowStepType;
  name?: string;
  config: {
    [key: string]: any;
  };
  nextStepId?: string;
  branchSteps?: Record<string, string>;
}

export type TriggerType = 'schedule' | 'webhook' | 'event';

export interface WorkflowTrigger {
  id: string;
  type: TriggerType;
  config: {
    [key: string]: any;
  };
}

// 调度配置类型
export interface ScheduleTriggerConfig {
  type: 'schedule';
  config: {
    cron: string;
    timezone?: string;
  };
}

// Webhook配置类型
export interface WebhookTriggerConfig {
  type: 'webhook';
  config: {
    path: string;
    method: string;
    secret?: string;
  };
}

// 事件配置类型
export interface EventTriggerConfig {
  type: 'event';
  config: {
    eventName: string;
  };
}

// 监控指标类型
export interface MonitoringMetrics {
  totalRequests: number;
  successRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  errorRate: number;
  requestDistribution: {
    [method: string]: number;
  };
  statusCodeDistribution: {
    [code: string]: number;
  };
}

// 环境配置类型
export interface EnvironmentConfig {
  id: string;
  name: string;
  variables: Record<string, string>;
  isDefault?: boolean;
}
