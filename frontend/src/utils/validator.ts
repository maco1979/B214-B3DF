/**
 * 请求参数验证工具
 * 用于在发送请求前验证参数合法性，拦截无效请求
 */

/**
 * 验证错误类型
 */
export enum ValidationErrorType {
  REQUIRED = 'required',
  INVALID_TYPE = 'invalid_type',
  OUT_OF_RANGE = 'out_of_range',
  INVALID_ENUM = 'invalid_enum',
  INVALID_FORMAT = 'invalid_format'
}

/**
 * 验证错误信息
 */
export interface ValidationError {
  field: string;
  type: ValidationErrorType;
  message: string;
  actualValue?: any;
}

/**
 * 验证规则
 */
export interface ValidationRule {

  /** 是否必填 */
  required?: boolean;

  /** 数据类型 */
  type?: 'string' | 'number' | 'boolean' | 'array' | 'object';

  /** 最小值（适用于数字） */
  min?: number;

  /** 最大值（适用于数字） */
  max?: number;

  /** 最小值长度（适用于字符串和数组） */
  minLength?: number;

  /** 最大值长度（适用于字符串和数组） */
  maxLength?: number;

  /** 枚举值（适用于所有类型） */
  enum?: any[];

  /** 正则表达式（适用于字符串） */
  pattern?: RegExp;

  /** 自定义验证函数 */
  customValidator?: (value: any) => boolean;

  /** 自定义错误信息 */
  message?: string;
}

/**
 * 验证规则映射
 */
export interface ValidationRules {
  [field: string]: ValidationRule;
}

/**
 * 参数验证器类
 */
export class Validator {
  /**
   * 验证单个值
   * @param value 要验证的值
   * @param rule 验证规则
   * @param field 字段名
   * @returns 验证错误，null表示验证通过
   */
  static validateValue(
    value: any,
    rule: ValidationRule,
    field: string,
  ): ValidationError | null {
    // 必填验证
    if (rule.required && (value === undefined || value === null || value === '')) {
      return {
        field,
        type: ValidationErrorType.REQUIRED,
        message: rule.message || `${field} is required`,
      };
    }

    // 如果值为undefined或null且不是必填项，则跳过其他验证
    if (value === undefined || value === null) {
      return null;
    }

    // 类型验证
    if (rule.type) {
      let isValidType = true;

      switch (rule.type) {
        case 'string':
          isValidType = typeof value === 'string';
          break;
        case 'number':
          isValidType = typeof value === 'number' && !isNaN(value);
          break;
        case 'boolean':
          isValidType = typeof value === 'boolean';
          break;
        case 'array':
          isValidType = Array.isArray(value);
          break;
        case 'object':
          isValidType = typeof value === 'object' && value !== null && !Array.isArray(value);
          break;
      }

      if (!isValidType) {
        return {
          field,
          type: ValidationErrorType.INVALID_TYPE,
          message: rule.message || `${field} must be a ${rule.type}`,
          actualValue: value,
        };
      }
    }

    // 数字范围验证
    if (typeof value === 'number') {
      if (rule.min !== undefined && value < rule.min) {
        return {
          field,
          type: ValidationErrorType.OUT_OF_RANGE,
          message: rule.message || `${field} must be greater than or equal to ${rule.min}`,
          actualValue: value,
        };
      }

      if (rule.max !== undefined && value > rule.max) {
        return {
          field,
          type: ValidationErrorType.OUT_OF_RANGE,
          message: rule.message || `${field} must be less than or equal to ${rule.max}`,
          actualValue: value,
        };
      }
    }

    // 字符串/数组长度验证
    if (typeof value === 'string' || Array.isArray(value)) {
      const { length } = value;

      if (rule.minLength !== undefined && length < rule.minLength) {
        return {
          field,
          type: ValidationErrorType.OUT_OF_RANGE,
          message: rule.message || `${field} must be at least ${rule.minLength} characters/items long`,
          actualValue: value,
        };
      }

      if (rule.maxLength !== undefined && length > rule.maxLength) {
        return {
          field,
          type: ValidationErrorType.OUT_OF_RANGE,
          message: rule.message || `${field} must be at most ${rule.maxLength} characters/items long`,
          actualValue: value,
        };
      }
    }

    // 枚举值验证
    if (rule.enum && rule.enum.length > 0 && !rule.enum.includes(value)) {
      return {
        field,
        type: ValidationErrorType.INVALID_ENUM,
        message: rule.message || `${field} must be one of ${rule.enum.join(', ')}`,
        actualValue: value,
      };
    }

    // 正则表达式验证
    if (typeof value === 'string' && rule.pattern) {
      if (!rule.pattern.test(value)) {
        return {
          field,
          type: ValidationErrorType.INVALID_FORMAT,
          message: rule.message || `${field} has an invalid format`,
          actualValue: value,
        };
      }
    }

    // 自定义验证函数
    if (rule.customValidator && typeof rule.customValidator === 'function') {
      if (!rule.customValidator(value)) {
        return {
          field,
          type: ValidationErrorType.INVALID_FORMAT,
          message: rule.message || `${field} is invalid`,
          actualValue: value,
        };
      }
    }

    return null;
  }

  /**
   * 验证对象
   * @param data 要验证的数据对象
   * @param rules 验证规则
   * @returns 验证错误列表，空数组表示验证通过
   */
  static validate(
    data: any,
    rules: ValidationRules,
  ): ValidationError[] {
    const errors: ValidationError[] = [];

    // 遍历所有规则
    for (const [field, rule] of Object.entries(rules)) {
      const value = data[field];
      const error = this.validateValue(value, rule, field);

      if (error) {
        errors.push(error);
      }
    }

    return errors;
  }

  /**
   * 注意力相关请求的验证规则
   */
  static getAttentionValidationRules() {
    return {
      sampleId: {
        required: true,
        type: 'string',
        minLength: 1,
        maxLength: 100,
      },
      layer: {
        required: true,
        type: 'number',
        min: 0,
        max: 100,
      },
      head: {
        required: true,
        type: 'number',
        min: 0,
        max: 100,
      },
      domainType: {
        type: 'string',
        enum: ['intra', 'inter'],
      },
      seqLen: {
        type: 'number',
        min: 1,
        max: 2048,
      },
      precision: {
        type: 'number',
        min: 1,
        max: 10,
      },
      visualization_type: {
        type: 'string',
        enum: ['heatmap', 'distribution', 'graph'],
      },
    };
  }

  /**
   * 迁移矩阵请求的验证规则
   */
  static getTransferMatrixValidationRules() {
    return {
      source_domain_id: {
        required: true,
        type: 'number',
        min: 0,
        max: 100,
      },
      target_domain_id: {
        required: true,
        type: 'number',
        min: 0,
        max: 100,
      },
      precision: {
        type: 'number',
        min: 1,
        max: 10,
      },
    };
  }

  /**
   * 域特征请求的验证规则
   */
  static getDomainFeaturesValidationRules() {
    return {
      sample_id: {
        required: true,
        type: 'string',
        minLength: 1,
        maxLength: 100,
      },
      domain_id: {
        required: true,
        type: 'number',
        min: 0,
        max: 100,
      },
      seq_len: {
        type: 'number',
        min: 1,
        max: 2048,
      },
      precision: {
        type: 'number',
        min: 1,
        max: 10,
      },
    };
  }
}

// 导出默认验证器实例
export default new Validator();
