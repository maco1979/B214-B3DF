// 核心类型定义

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD' | 'OPTIONS';

export interface HttpRequestConfig {
  method: HttpMethod;
  url: string;
  headers?: Record<string, string>;
  params?: Record<string, any>;
  body?: any;
  timeout?: number;
  retry?: {
    attempts: number;
    delay: number;
    maxDelay?: number;
  };
}

export interface HttpResponse {
  status: number;
  statusText: string;
  headers: Record<string, string>;
  data: any;
  duration: number;
}

export type AuthenticationType = 'apiKey' | 'oauth2' | 'jwt' | 'basic' | 'none';

export interface ApiKeyAuth {
  type: 'apiKey';
  key: string;
  value: string;
  location: 'header' | 'query' | 'cookie';
}

export interface OAuth2Auth {
  type: 'oauth2';
  clientId: string;
  clientSecret: string;
  accessToken: string;
  refreshToken?: string;
  tokenUrl: string;
  scope?: string[];
  expiresAt?: number;
}

export interface JwtAuth {
  type: 'jwt';
  token: string;
  algorithm?: string;
  expiresAt?: number;
}

export interface BasicAuth {
  type: 'basic';
  username: string;
  password: string;
}

export interface NoneAuth {
  type: 'none';
}

export type AuthenticationConfig = ApiKeyAuth | OAuth2Auth | JwtAuth | BasicAuth | NoneAuth;

export interface ApiEndpoint {
  id: string;
  name: string;
  description?: string;
  config: HttpRequestConfig;
  auth: AuthenticationConfig;
  tags?: string[];
}

export type WorkflowStepType = 'apiCall' | 'condition' | 'loop' | 'parallel' | 'delay' | 'custom';

export interface WorkflowStep {
  id: string;
  type: WorkflowStepType;
  name: string;
  description?: string;
  dependsOn?: string[];
  retry?: {
    attempts: number;
    delay: number;
  };
}

export interface ApiCallStep extends WorkflowStep {
  type: 'apiCall';
  endpointId: string;
  requestMapping?: Record<string, string>;
  responseMapping?: Record<string, string>;
}

export interface ConditionStep extends WorkflowStep {
  type: 'condition';
  expression: string;
  trueBranch: string;
  falseBranch: string;
}

export interface LoopStep extends WorkflowStep {
  type: 'loop';
  loopType: 'for' | 'while' | 'forEach';
  expression: string;
  steps: string[];
  maxIterations?: number;
}

export interface ParallelStep extends WorkflowStep {
  type: 'parallel';
  steps: string[];
  waitForAll?: boolean;
}

export interface DelayStep extends WorkflowStep {
  type: 'delay';
  duration: number;
  unit: 'ms' | 's' | 'm' | 'h';
}

export interface CustomStep extends WorkflowStep {
  type: 'custom';
  script: string;
  language: 'javascript' | 'typescript';
}

export type WorkflowStepConfig = ApiCallStep | ConditionStep | LoopStep | ParallelStep | DelayStep | CustomStep;

export type WorkflowTriggerType = 'manual' | 'cron' | 'webhook' | 'event';

export interface WorkflowTrigger {
  type: WorkflowTriggerType;
  config: {
    cronExpression?: string;
    webhookPath?: string;
    eventName?: string;
  };
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps: Record<string, WorkflowStepConfig>;
  trigger: WorkflowTrigger;
  variables?: Record<string, any>;
  tags?: string[];
  enabled: boolean;
}

export type WorkflowStatus = 'idle' | 'running' | 'success' | 'failed' | 'paused';

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: WorkflowStatus;
  startTime: Date;
  endTime?: Date;
  duration?: number;
  steps: Record<string, {
    status: WorkflowStatus;
    startTime: Date;
    endTime?: Date;
    duration?: number;
    result?: any;
    error?: any;
  }>;
  variables: Record<string, any>;
  error?: any;
}

export interface ApiCallLog {
  id: string;
  endpointId: string;
  request: HttpRequestConfig;
  response?: HttpResponse;
  error?: any;
  timestamp: Date;
  duration: number;
  status: 'success' | 'failed';
}

export interface ErrorConfig {
  retryable?: boolean;
  errorCode?: string;
  message?: string;
}

export interface MonitoringMetrics {
  totalCalls: number;
  successCalls: number;
  failedCalls: number;
  averageResponseTime: number;
  errorRates: Record<string, number>;
  statusCodeDistribution: Record<number, number>;
  timestamp: Date;
}

export interface ApiAutomationConfig {
  baseUrl?: string;
  timeout?: number;
  defaultAuth?: AuthenticationConfig;
  logging?: {
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'text';
  };
  monitoring?: {
    enabled: boolean;
    interval: number;
  };
}
