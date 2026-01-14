import { Job, Queue } from 'bull';
import { CronJob } from 'cron';
import { Workflow, WorkflowExecution, WorkflowStatus, WorkflowStepConfig, ApiEndpoint, AuthenticationConfig, HttpRequestConfig } from '../../types';
import { RequestBuilder } from '../request-builder/RequestBuilder';
import { AuthManager } from '../auth/AuthManager';
import { ResponseProcessor } from '../response-processor/ResponseProcessor';
import { ErrorHandler } from '../error-handler/ErrorHandler';

/**
 * 工作流引擎，负责编排和执行API工作流
 */
export class WorkflowEngine {
  private workflows: Map<string, Workflow> = new Map();
  private workflowExecutions: Map<string, WorkflowExecution> = new Map();
  private requestBuilder: RequestBuilder;
  private authManager: AuthManager;
  private responseProcessor: ResponseProcessor;
  private errorHandler: ErrorHandler;
  private jobQueue: Queue;
  private cronJobs: Map<string, CronJob> = new Map();

  constructor(
    authManager: AuthManager,
    requestBuilder: RequestBuilder,
    responseProcessor: ResponseProcessor,
    errorHandler: ErrorHandler
  ) {
    this.authManager = authManager;
    this.requestBuilder = requestBuilder;
    this.responseProcessor = responseProcessor;
    this.errorHandler = errorHandler;
    
    // 初始化任务队列
    this.jobQueue = new Queue('workflow-jobs', {
      redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
      },
    });

    // 设置队列处理器
    this.jobQueue.process(async (job: Job) => {
      const { workflowId, executionId, stepId } = job.data;
      await this.executeWorkflowStep(workflowId, executionId, stepId);
    });
  }

  /**
   * 注册工作流
   */
  registerWorkflow(workflow: Workflow): void {
    this.workflows.set(workflow.id, workflow);
    
    // 如果工作流启用且有定时触发器，创建Cron作业
    if (workflow.enabled && workflow.trigger.type === 'cron' && workflow.trigger.config.cronExpression) {
      this.createCronJob(workflow);
    }
  }

  /**
   * 创建Cron作业
   */
  private createCronJob(workflow: Workflow): void {
    if (!workflow.trigger.config.cronExpression) {
      return;
    }

    const cronJob = new CronJob(workflow.trigger.config.cronExpression, async () => {
      await this.startWorkflowExecution(workflow.id);
    });

    cronJob.start();
    this.cronJobs.set(workflow.id, cronJob);
  }

  /**
   * 启动工作流执行
   */
  async startWorkflowExecution(workflowId: string, variables?: Record<string, any>): Promise<string> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow with id ${workflowId} not found`);
    }

    if (!workflow.enabled) {
      throw new Error(`Workflow with id ${workflowId} is disabled`);
    }

    // 创建工作流执行实例
    const executionId = this.generateId();
    const execution: WorkflowExecution = {
      id: executionId,
      workflowId,
      status: 'running',
      startTime: new Date(),
      steps: {},
      variables: variables || { ...workflow.variables },
    };

    this.workflowExecutions.set(executionId, execution);

    // 查找工作流的起始步骤（没有依赖的步骤）
    const startSteps = Object.values(workflow.steps).filter(step => !step.dependsOn || step.dependsOn.length === 0);
    
    // 执行起始步骤
    for (const step of startSteps) {
      await this.queueWorkflowStep(workflowId, executionId, step.id);
    }

    return executionId;
  }

  /**
   * 排队执行工作流步骤
   */
  private async queueWorkflowStep(workflowId: string, executionId: string, stepId: string): Promise<void> {
    await this.jobQueue.add({
      workflowId,
      executionId,
      stepId,
    });
  }

  /**
   * 执行工作流步骤
   */
  private async executeWorkflowStep(workflowId: string, executionId: string, stepId: string): Promise<void> {
    const workflow = this.workflows.get(workflowId);
    const execution = this.workflowExecutions.get(executionId);
    
    if (!workflow || !execution) {
      return;
    }

    const step = workflow.steps[stepId];
    if (!step) {
      return;
    }

    // 更新步骤状态
    execution.steps[stepId] = {
      status: 'running',
      startTime: new Date(),
    };

    try {
      let result: any;
      
      // 根据步骤类型执行不同的逻辑
      switch (step.type) {
        case 'apiCall':
          result = await this.executeApiCallStep(step, execution);
          break;
        case 'condition':
          result = await this.executeConditionStep(step, execution);
          break;
        case 'loop':
          result = await this.executeLoopStep(step, workflow, execution);
          break;
        case 'parallel':
          result = await this.executeParallelStep(step, workflow, execution);
          break;
        case 'delay':
          result = await this.executeDelayStep(step);
          break;
        case 'custom':
          result = await this.executeCustomStep(step, execution);
          break;
        default:
          throw new Error(`Unsupported step type: ${step.type}`);
      }

      // 更新步骤状态为成功
      execution.steps[stepId] = {
        ...execution.steps[stepId],
        status: 'success',
        endTime: new Date(),
        duration: Date.now() - execution.steps[stepId].startTime.getTime(),
        result,
      };

      // 执行后续步骤
      await this.executeNextSteps(workflow, execution, stepId);
    } catch (error: any) {
      // 更新步骤状态为失败
      execution.steps[stepId] = {
        ...execution.steps[stepId],
        status: 'failed',
        endTime: new Date(),
        duration: Date.now() - execution.steps[stepId].startTime.getTime(),
        error: error.message,
      };

      // 更新工作流状态为失败
      execution.status = 'failed';
      execution.endTime = new Date();
      execution.duration = Date.now() - execution.startTime.getTime();
      execution.error = error.message;

      // 处理错误
      this.errorHandler.recordError(this.errorHandler.createError(error));
    }

    // 更新工作流执行记录
    this.workflowExecutions.set(executionId, execution);

    // 检查工作流是否已完成
    if (this.isWorkflowCompleted(execution)) {
      execution.status = 'success';
      execution.endTime = new Date();
      execution.duration = Date.now() - execution.startTime.getTime();
    }
  }

  /**
   * 执行API调用步骤
   */
  private async executeApiCallStep(step: WorkflowStepConfig & { type: 'apiCall' }, execution: WorkflowExecution): Promise<any> {
    // 这里假设我们有一个API端点管理器，用于获取API端点配置
    // 为了简化，这里我们直接使用步骤中的配置
    const endpoint: ApiEndpoint = {
      id: step.endpointId,
      name: `Endpoint ${step.endpointId}`,
      config: {
        method: 'GET',
        url: 'https://api.example.com',
        // 这里应该从实际的API端点配置中获取
      },
      auth: { type: 'none' },
    };

    // 应用请求映射（替换变量）
    const requestConfig = this.applyRequestMapping(endpoint.config, step.requestMapping || {}, execution.variables);
    
    // 发送API请求
    const response = await this.requestBuilder.sendRequestWithRetry(requestConfig, endpoint.auth);
    
    // 处理响应
    if (!this.responseProcessor.isSuccessResponse(response)) {
      throw new Error(`API call failed: ${this.responseProcessor.getErrorMessage(response)}`);
    }
    
    // 应用响应映射
    const mappedResult = this.applyResponseMapping(response, step.responseMapping || {}, execution.variables);
    
    // 更新执行变量
    Object.assign(execution.variables, mappedResult);
    
    return mappedResult;
  }

  /**
   * 执行条件步骤
   */
  private async executeConditionStep(step: WorkflowStepConfig & { type: 'condition' }, execution: WorkflowExecution): Promise<boolean> {
    // 评估条件表达式
    const result = this.evaluateCondition(step.expression, execution.variables);
    
    // 根据条件结果执行不同的分支
    const nextStepId = result ? step.trueBranch : step.falseBranch;
    
    // 这里不需要立即执行下一步，因为executeNextSteps会处理
    
    return result;
  }

  /**
   * 执行循环步骤
   */
  private async executeLoopStep(step: WorkflowStepConfig & { type: 'loop' }, workflow: Workflow, execution: WorkflowExecution): Promise<any[]> {
    const results: any[] = [];
    let iteration = 0;
    let shouldContinue = true;
    
    while (shouldContinue) {
      // 检查最大迭代次数
      if (step.maxIterations && iteration >= step.maxIterations) {
        break;
      }
      
      // 评估循环条件
      if (step.loopType === 'while') {
        shouldContinue = this.evaluateCondition(step.expression, execution.variables);
        if (!shouldContinue) {
          break;
        }
      }
      
      // 执行循环内的步骤
      for (const loopStepId of step.steps) {
        const loopStep = workflow.steps[loopStepId];
        if (loopStep) {
          // 这里简化处理，实际应该递归执行步骤
          await this.queueWorkflowStep(workflow.id, execution.id, loopStepId);
        }
      }
      
      iteration++;
      
      // 对于for循环，检查迭代条件
      if (step.loopType === 'for') {
        shouldContinue = iteration < (step.expression as any) || 0;
      }
    }
    
    return results;
  }

  /**
   * 执行并行步骤
   */
  private async executeParallelStep(step: WorkflowStepConfig & { type: 'parallel' }, workflow: Workflow, execution: WorkflowExecution): Promise<any[]> {
    const promises = step.steps.map(stepId => {
      return this.queueWorkflowStep(workflow.id, execution.id, stepId);
    });
    
    // 等待所有并行步骤完成
    await Promise.all(promises);
    
    return [];
  }

  /**
   * 执行延迟步骤
   */
  private async executeDelayStep(step: WorkflowStepConfig & { type: 'delay' }): Promise<void> {
    const durationMs = this.convertToMilliseconds(step.duration, step.unit);
    await this.delay(durationMs);
  }

  /**
   * 执行自定义脚本步骤
   */
  private async executeCustomStep(step: WorkflowStepConfig & { type: 'custom' }, execution: WorkflowExecution): Promise<any> {
    // 安全地执行自定义脚本
    // 这里使用Function构造函数作为简单实现，实际应该使用更安全的沙箱环境
    const scriptFunction = new Function('variables', 'context', step.script);
    
    const context = {
      requestBuilder: this.requestBuilder,
      responseProcessor: this.responseProcessor,
      authManager: this.authManager,
      errorHandler: this.errorHandler,
    };
    
    return scriptFunction(execution.variables, context);
  }

  /**
   * 执行后续步骤
   */
  private async executeNextSteps(workflow: Workflow, execution: WorkflowExecution, completedStepId: string): Promise<void> {
    // 查找所有依赖于当前完成步骤的后续步骤
    const nextSteps = Object.values(workflow.steps).filter(step => 
      step.dependsOn && step.dependsOn.includes(completedStepId)
    );
    
    for (const nextStep of nextSteps) {
      // 检查所有依赖步骤是否都已完成
      const allDependenciesCompleted = nextStep.dependsOn?.every(depStepId => {
        const depStep = execution.steps[depStepId];
        return depStep && depStep.status === 'success';
      }) || true;
      
      if (allDependenciesCompleted) {
        // 执行后续步骤
        await this.queueWorkflowStep(workflow.id, execution.id, nextStep.id);
      }
    }
  }

  /**
   * 应用请求映射
   */
  private applyRequestMapping(requestConfig: HttpRequestConfig, mappings: Record<string, string>, variables: Record<string, any>): HttpRequestConfig {
    const result = { ...requestConfig };
    
    // 应用URL映射
    result.url = this.replaceVariables(result.url, variables);
    
    // 应用headers映射
    if (result.headers) {
      result.headers = Object.fromEntries(
        Object.entries(result.headers).map(([key, value]) => [key, this.replaceVariables(value, variables)])
      );
    }
    
    // 应用params映射
    if (result.params) {
      result.params = this.applyObjectMapping(result.params, mappings, variables);
    }
    
    // 应用body映射
    if (result.body) {
      result.body = this.applyObjectMapping(result.body, mappings, variables);
    }
    
    return result;
  }

  /**
   * 应用响应映射
   */
  private applyResponseMapping(response: any, mappings: Record<string, string>, variables: Record<string, any>): Record<string, any> {
    const result: Record<string, any> = {};
    
    for (const [key, path] of Object.entries(mappings)) {
      result[key] = this.responseProcessor.extractDataWithJsonPath(response.data, path);
    }
    
    return result;
  }

  /**
   * 应用对象映射
   */
  private applyObjectMapping(obj: any, mappings: Record<string, string>, variables: Record<string, any>): any {
    if (typeof obj === 'string') {
      return this.replaceVariables(obj, variables);
    } else if (Array.isArray(obj)) {
      return obj.map(item => this.applyObjectMapping(item, mappings, variables));
    } else if (typeof obj === 'object' && obj !== null) {
      const result: Record<string, any> = {};
      for (const [key, value] of Object.entries(obj)) {
        result[key] = this.applyObjectMapping(value, mappings, variables);
      }
      return result;
    }
    return obj;
  }

  /**
   * 替换变量
   */
  private replaceVariables(str: string, variables: Record<string, any>): string {
    return str.replace(/\{\{([^}]+)\}\}/g, (match, variableName) => {
      return variables[variableName.trim()] || match;
    });
  }

  /**
   * 评估条件表达式
   */
  private evaluateCondition(expression: string, variables: Record<string, any>): boolean {
    // 安全地评估条件表达式
    // 这里使用Function构造函数作为简单实现，实际应该使用更安全的表达式解析器
    const conditionFunction = new Function('variables', `return ${expression};`);
    return conditionFunction(variables);
  }

  /**
   * 检查工作流是否已完成
   */
  private isWorkflowCompleted(execution: WorkflowExecution): boolean {
    const steps = execution.steps;
    const allStepsCompleted = Object.values(steps).every(step => 
      step.status === 'success' || step.status === 'failed'
    );
    
    const anyStepFailed = Object.values(steps).some(step => step.status === 'failed');
    
    return allStepsCompleted && !anyStepFailed;
  }

  /**
   * 转换时间单位为毫秒
   */
  private convertToMilliseconds(duration: number, unit: 'ms' | 's' | 'm' | 'h'): number {
    switch (unit) {
      case 'ms':
        return duration;
      case 's':
        return duration * 1000;
      case 'm':
        return duration * 60 * 1000;
      case 'h':
        return duration * 60 * 60 * 1000;
      default:
        return duration;
    }
  }

  /**
   * 延迟函数
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 生成唯一ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 获取工作流
   */
  getWorkflow(workflowId: string): Workflow | undefined {
    return this.workflows.get(workflowId);
  }

  /**
   * 获取工作流执行
   */
  getWorkflowExecution(executionId: string): WorkflowExecution | undefined {
    return this.workflowExecutions.get(executionId);
  }

  /**
   * 暂停工作流
   */
  pauseWorkflow(workflowId: string): void {
    const workflow = this.workflows.get(workflowId);
    if (workflow) {
      workflow.enabled = false;
      this.workflows.set(workflowId, workflow);
      
      // 停止Cron作业
      const cronJob = this.cronJobs.get(workflowId);
      if (cronJob) {
        cronJob.stop();
        this.cronJobs.delete(workflowId);
      }
    }
  }

  /**
   * 恢复工作流
   */
  resumeWorkflow(workflowId: string): void {
    const workflow = this.workflows.get(workflowId);
    if (workflow) {
      workflow.enabled = true;
      this.workflows.set(workflowId, workflow);
      
      // 重新创建Cron作业
      if (workflow.trigger.type === 'cron' && workflow.trigger.config.cronExpression) {
        this.createCronJob(workflow);
      }
    }
  }

  /**
   * 终止工作流执行
   */
  terminateWorkflowExecution(executionId: string): void {
    const execution = this.workflowExecutions.get(executionId);
    if (execution) {
      execution.status = 'failed';
      execution.endTime = new Date();
      execution.duration = Date.now() - execution.startTime.getTime();
      execution.error = 'Workflow execution terminated by user';
      this.workflowExecutions.set(executionId, execution);
    }
  }

  /**
   * 关闭工作流引擎
   */
  async shutdown(): Promise<void> {
    // 停止所有Cron作业
    for (const cronJob of this.cronJobs.values()) {
      cronJob.stop();
    }
    
    // 关闭任务队列
    await this.jobQueue.close();
  }
}
