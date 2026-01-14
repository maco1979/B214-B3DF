/**
 * 语音云服务类
 * 实现边缘与云端协同功能，本地优先，复杂任务云端处理
 */

import { apiClient } from './api';
import type { IntentResult } from './nlpService';

export interface CloudProcessRequest {
  text: string;
  context: Record<string, any>;
  entities?: any[];
  intent?: string;
}

export interface CloudProcessResponse {
  success: boolean;
  result?: {
    intent: string;
    entities: any[];
    response: string;
    action: string;
  };
  error?: string;
  message?: string;
}

export class VoiceCloudService {
  private isOnline: boolean;
  private readonly cloudEndpoint: string;
  private retryCount: number;
  private readonly maxRetries: number;

  constructor() {
    this.isOnline = navigator.onLine;
    this.cloudEndpoint = '/api/v1/voice/process';
    this.retryCount = 0;
    this.maxRetries = 3;

    // 监听网络状态变化
    this.setupNetworkListeners();
  }

  /**
   * 设置网络状态监听器
   */
  private setupNetworkListeners(): void {
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.retryCount = 0;
      console.log('网络连接已恢复，切换到在线模式');
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      console.log('网络连接已断开，切换到离线模式');
    });
  }

  /**
   * 检查网络状态
   */
  private checkNetworkStatus(): boolean {
    return this.isOnline;
  }

  /**
   * 处理复杂语音请求，发送到云端
   */
  async processComplexRequest(request: CloudProcessRequest): Promise<CloudProcessResponse> {
    if (!this.checkNetworkStatus()) {
      return {
        success: false,
        message: '网络连接不可用，无法处理复杂请求',
        error: 'network_error',
      };
    }

    try {
      // 调用云端API
      const response = await apiClient.post<CloudProcessResponse>(
        this.cloudEndpoint,
        request,
      );

      this.retryCount = 0;
      return response.data;
    } catch (error) {
      this.retryCount++;

      // 如果重试次数未达到最大值，进行重试
      if (this.retryCount < this.maxRetries) {
        // 指数退避策略
        const delay = 2 ** this.retryCount * 1000;
        console.log(`请求失败，${delay}ms后重试 (${this.retryCount}/${this.maxRetries})`);

        await new Promise(resolve => setTimeout(resolve, delay));
        return this.processComplexRequest(request);
      }

      // 重试次数已达最大值
      this.retryCount = 0;
      console.error('云端请求失败，已达到最大重试次数');

      return {
        success: false,
        message: '云端请求失败',
        error: 'cloud_request_failed',
      };
    }
  }

  /**
   * 本地优先处理策略
   * 简单请求本地处理，复杂请求云端处理
   */
  async processWithLocalFirst(
    text: string,
    localResult: IntentResult,
  ): Promise<CloudProcessResponse> {
    // 如果本地识别到明确意图，本地处理
    if (localResult.intent !== 'UNKNOWN' && localResult.confidence > 0.7) {
      return {
        success: true,
        result: {
          intent: localResult.intent,
          entities: localResult.entities,
          response: '本地处理成功',
          action: this.mapIntentToAction(localResult.intent),
        },
      };
    }

    // 本地无法处理，发送到云端
    return this.processComplexRequest({
      text,
      context: localResult.context,
      entities: localResult.entities,
      intent: localResult.intent,
    });
  }

  /**
   * 将意图映射到具体动作
   */
  private mapIntentToAction(intent: string): string {
    const actionMap: Record<string, string> = {
      TOGGLE_MASTER_CONTROL: 'toggle_master_control',
      OPEN_CAMERA: 'open_camera',
      CLOSE_CAMERA: 'close_camera',
      PTZ_CONTROL: 'ptz_control',
      START_AI: 'start_ai',
      STOP_AI: 'stop_ai',
      UNKNOWN: 'unknown_action',
    };

    return actionMap[intent] || 'unknown_action';
  }

  /**
   * 获取当前网络状态
   */
  getNetworkStatus(): boolean {
    return this.isOnline;
  }

  /**
   * 刷新云端配置
   */
  async refreshCloudConfig(): Promise<boolean> {
    if (!this.checkNetworkStatus()) {
      return false;
    }

    try {
      // 调用云端配置刷新API
      const response = await apiClient.get('/api/v1/voice/config');
      // 处理配置刷新结果
      console.log('云端配置刷新成功:', response.data);
      return true;
    } catch (error) {
      console.error('云端配置刷新失败:', error);
      return false;
    }
  }

  /**
   * 清除云端会话
   */
  async clearCloudSession(): Promise<boolean> {
    if (!this.checkNetworkStatus()) {
      return false;
    }

    try {
      const response = await apiClient.post('/api/v1/voice/session/clear', {});
      return response.data.success;
    } catch (error) {
      console.error('清除云端会话失败:', error);
      return false;
    }
  }
}

