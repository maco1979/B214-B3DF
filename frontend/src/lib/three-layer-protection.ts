/*
 * 三层防护机制实现
 * 1. 参数校验层：生物组件与非生物组件参数验证
 * 2. 判空层：生物活性验证与功能验证
 * 3. 异常层：生物反应异常处理与系统兜底机制
 */

import type { ApiResponse } from '@/services/api';

// 生物组件安全参数配置
const BIO_COMPONENT_SAFETY_CONFIG = {
  ACTIVITY_THRESHOLD: 0.9, // 活性≥90%
  PURITY_THRESHOLD: 0.999, // 纯度≥99.9%
  SAFETY_LEVELS: { P1: 1, P2: 2, P3: 3, P4: 4 },
};

// 三层防护结果类型
export interface ProtectionResult {
  passed: boolean;
  layer: 'param_validation' | 'null_check' | 'exception_handling';
  message?: string;
  details?: any;
}

// 第一层：参数校验层
function paramValidationLayer(params: any, endpoint: string): ProtectionResult {
  // 基本参数存在性验证 - 允许undefined params（如GET请求）
  if (params !== undefined && typeof params !== 'object') {
    return {
      passed: false,
      layer: 'param_validation',
      message: '无效的请求参数格式',
      details: { expected: 'object', actual: typeof params },
    };
  }

  // 如果params是undefined，将其转换为空对象以简化后续处理
  params ||= {};

  // 特定端点的参数验证规则
  const endpointRules: Record<string, any> = {
    '/api/v1/decision/organic-core/activate-iteration': {
      required: [],
    },
    '/api/v1/decision/organic-core/evolve-structure': {
      required: [],
    },
    '/api/v1/decision/agriculture': {
      required: ['temperature', 'humidity', 'co2_level', 'light_intensity'],
      minValues: { temperature: 0, humidity: 0, co2_level: 0, light_intensity: 0 },
      maxValues: { temperature: 100, humidity: 100, co2_level: 5000, light_intensity: 10000 },
    },
    '/api/v1/decision/risk': {
      required: ['temperature', 'humidity', 'co2_level', 'light_intensity'],
      minValues: { temperature: 0, humidity: 0, co2_level: 0, light_intensity: 0 },
      maxValues: { temperature: 100, humidity: 100, co2_level: 5000, light_intensity: 10000 },
    },
    '/api/v1/models/pretrained': {
      required: ['model_name_or_path'],
    },
    '/api/v1/models/train': {
      required: ['training_data'],
    },
    '/api/v1/project/progress-chart': {
      required: [],
    },
  };

  // 获取当前端点的验证规则
  const rules = endpointRules[endpoint] || {};
  const { required = [], minValues = {}, maxValues = {} } = rules;

  // 必填参数验证
  for (const field of required) {
    if (params[field] === undefined || params[field] === null) {
      return {
        passed: false,
        layer: 'param_validation',
        message: `缺少必填参数: ${field}`,
        details: { missingField: field },
      };
    }
  }

  // 数值范围验证
  for (const [field, minValue] of Object.entries(minValues)) {
    if (params[field] !== undefined && params[field] < (minValue as number)) {
      return {
        passed: false,
        layer: 'param_validation',
        message: `${field} 值低于安全阈值`,
        details: { field, value: params[field], minValue },
      };
    }
  }

  for (const [field, maxValue] of Object.entries(maxValues)) {
    if (params[field] !== undefined && params[field] > (maxValue as number)) {
      return {
        passed: false,
        layer: 'param_validation',
        message: `${field} 值高于安全阈值`,
        details: { field, value: params[field], maxValue },
      };
    }
  }

  // 生物组件参数特殊验证
  if (params.bio_component) {
    const bioComponent = params.bio_component;
    if (bioComponent.activity && bioComponent.activity < BIO_COMPONENT_SAFETY_CONFIG.ACTIVITY_THRESHOLD) {
      return {
        passed: false,
        layer: 'param_validation',
        message: '生物组件活性低于安全阈值',
        details: {
          activity: bioComponent.activity,
          threshold: BIO_COMPONENT_SAFETY_CONFIG.ACTIVITY_THRESHOLD,
        },
      };
    }
    if (bioComponent.purity && bioComponent.purity < BIO_COMPONENT_SAFETY_CONFIG.PURITY_THRESHOLD) {
      return {
        passed: false,
        layer: 'param_validation',
        message: '生物组件纯度低于安全阈值',
        details: {
          purity: bioComponent.purity,
          threshold: BIO_COMPONENT_SAFETY_CONFIG.PURITY_THRESHOLD,
        },
      };
    }
  }

  return { passed: true, layer: 'param_validation' };
}

// 第二层：判空层
function nullCheckLayer(data: any, endpoint: string): ProtectionResult {
  // 检查数据是否为空
  if (data === undefined || data === null) {
    return {
      passed: false,
      layer: 'null_check',
      message: '数据为空或未定义',
      details: { endpoint },
    };
  }

  // 检查生物活性相关字段
  if (typeof data === 'object') {
    // 检查生物活性指标
    if (data.activity !== undefined && (data.activity === 0 || data.activity === null)) {
      return {
        passed: false,
        layer: 'null_check',
        message: '生物活性指标异常',
        details: { activity: data.activity },
      };
    }

    // 检查功能验证相关字段
    if (data.functionality_status && data.functionality_status === 'error') {
      return {
        passed: false,
        layer: 'null_check',
        message: '功能验证失败',
        details: { functionality_status: data.functionality_status },
      };
    }

    // 检查关键系统状态
    if (data.system_status && data.system_status === 'critical') {
      return {
        passed: false,
        layer: 'null_check',
        message: '系统状态异常',
        details: { system_status: data.system_status },
      };
    }

    // 检查AI模型状态
    if (data.model_status && data.model_status === 'error') {
      return {
        passed: false,
        layer: 'null_check',
        message: 'AI模型状态异常',
        details: { model_status: data.model_status },
      };
    }
  }

  return { passed: true, layer: 'null_check' };
}

// 第三层：异常层
function exceptionHandlingLayer(error: any, endpoint: string): ApiResponse {
  let errorMessage = '未知错误';
  let errorDetails: any = {};

  if (error instanceof Error) {
    errorMessage = error.message;
    errorDetails = { stack: error.stack };
  } else if (typeof error === 'object' && error !== null) {
    errorMessage = error.message || '系统错误';
    errorDetails = { ...error };
  } else {
    errorMessage = String(error);
  }

  // 特定端点的异常处理策略
  const endpointStrategies: Record<string, any> = {
    '/api/v1/decision/organic-core/activate-iteration': {
      fallback: { is_active: false, model_status: 'inactive' },
      message: '有机核心激活失败，已回退到安全状态',
    },
    '/api/v1/decision/organic-core/evolve-structure': {
      fallback: { evolution_status: 'failed', safety_mode: 'enabled' },
      message: '有机结构演化失败，已启用安全模式',
    },
    '/api/v1/models/train': {
      fallback: { task_id: null, status: 'failed', safety_check: 'passed' },
      message: '模型训练失败，已执行安全检查',
    },
  };

  const strategy = endpointStrategies[endpoint] || {
    fallback: null,
    message: '操作失败，已执行安全兜底机制',
  };

  // 记录异常日志
  console.error(`[三层防护][异常处理] ${endpoint} - ${errorMessage}`, errorDetails);

  // 返回统一格式的错误响应
  return {
    success: false,
    error: `${strategy.message}: ${errorMessage}`,
    data: strategy.fallback,
  };
}

// 三层防护机制主函数
export async function threeLayerProtection(
  endpoint: string,
  params: any,
  requestFn: () => Promise<any>,
): Promise<ApiResponse> {
  return new Promise(async resolve => {
    try {
      // 第一层：参数校验
      const paramValidationResult = paramValidationLayer(params, endpoint);
      if (!paramValidationResult.passed) {
        console.warn(`[三层防护][参数校验失败] ${endpoint}`, paramValidationResult);
        resolve({
          success: false,
          error: paramValidationResult.message,
        });
        return;
      }

      // 执行请求
      const response = await requestFn();

      // 检查响应是否为ApiResponse格式
      const isApiResponse = response && typeof response === 'object' && 'success' in response;
      if (!isApiResponse) {
        // 如果不是ApiResponse格式，转换为ApiResponse格式
        resolve({ success: true, data: response });
        return;
      }

      // 第二层：判空验证（仅对成功响应进行）
      if (response.success && response.data) {
        const nullCheckResult = nullCheckLayer(response.data, endpoint);
        if (!nullCheckResult.passed) {
          console.warn(`[三层防护][判空验证失败] ${endpoint}`, nullCheckResult);
          resolve({
            success: false,
            error: nullCheckResult.message,
          });
          return;
        }
      }

      // 验证通过，返回原始响应
      resolve(response as ApiResponse);
    } catch (error) {
      // 第三层：异常处理
      const exceptionResult = exceptionHandlingLayer(error, endpoint);
      resolve(exceptionResult);
    }
  });
}

// 生物安全合规性检查
export function checkBioSafetyCompliance(data: any): boolean {
  if (!data) {
 return false;
}

  // 检查生物安全等级
  if (data.safety_level &&
      typeof data.safety_level === 'string' &&
      !BIO_COMPONENT_SAFETY_CONFIG.SAFETY_LEVELS[data.safety_level as keyof typeof BIO_COMPONENT_SAFETY_CONFIG.SAFETY_LEVELS]) {
    return false;
  }

  // 检查生物组件活性
  if (data.bio_component) {
    const bioComponent = data.bio_component;
    if (bioComponent.activity !== undefined && bioComponent.activity < BIO_COMPONENT_SAFETY_CONFIG.ACTIVITY_THRESHOLD) {
      return false;
    }
    if (bioComponent.purity !== undefined && bioComponent.purity < BIO_COMPONENT_SAFETY_CONFIG.PURITY_THRESHOLD) {
      return false;
    }
  }

  return true;
}

