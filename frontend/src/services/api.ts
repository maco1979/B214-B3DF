import { http } from '@/lib/api-client'


// API响应类型定义
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}


// 认证相关类型
export interface User {
  id: string
  username: string
  email: string
  avatar?: string
  source: string
  role: string
  created_at: string
}

export interface LoginResponse {
  message: string
  user_info: User
  access_token: string
  token_type: string
}

// 设备相关类型
export interface Device {
  id: number
  name: string
  type: string
  status: 'online' | 'offline'
  connected: boolean
  connection_type: string
  signal: number
  battery: number
  location: string
  lastSeen: string
  permissions: string[]
  isCompliant: boolean
  connection_details: {
    [key: string]: any
  }
}

// 模型相关类型
export interface Model {
  id: string
  name: string
  version: string
  description: string
  status: 'training' | 'ready' | 'deployed' | 'error'
  accuracy?: number
  size?: number
  created_at: string
  updated_at: string
}

// 推理请求类型
export interface InferenceRequest {
  model_id: string
  input_data: any
  parameters?: {
    temperature?: number
    max_tokens?: number
    top_p?: number
  }
}

// 推理响应类型
export interface InferenceResponse {
  result: any
  inference_time: number
  model_version: string
}

// 训练任务类型
export interface TrainingTask {
  id: string
  model_name: string
  dataset: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  start_time: string
  end_time?: string
}

// 训练状态响应类型
export interface TrainingStatusResponse {
  task_id: string
  model_id: string
  status: string
  progress: number
  stage: string
  current_step: number
  total_steps: number
  started_at: string
  completed_at?: string
  metrics?: { [key: string]: number }
}

// JEPA数据响应类型
export interface JEPAData {
  cv_prediction: number[];
  jepa_prediction: number[];
  fused_prediction: number[];
  energy: number;
  weight: number;
  embedding_dynamics?: number[][];
  timestamp: string;
}

// 启动训练请求类型
export interface StartTrainingRequest {
  training_data: { [key: string]: any }
}

// 系统指标类型
export interface SystemMetrics {
  inference_requests: number
  active_models: number
  edge_nodes: number
  neural_latency: number
  memory_saturation: number
  active_connections: number
  ai_service_status: 'healthy' | 'degraded' | 'critical'
  database_status: 'healthy' | 'degraded' | 'critical'
}

// 区块链状态类型
export interface BlockchainStatus {
  status: string
  initialized: boolean
  latest_block?: {
    block_number: number
    transaction_count: number
  }
  timestamp: string
}

// API客户端类
class ApiClient {

  async get<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'GET' })
  }

  private async request<T>(
    endpoint: string,
    options: { method?: string; body?: any; headers?: any } = {}
  ): Promise<ApiResponse<T>> {
    try {
      const { method = 'GET', body } = options
      // 直接使用body作为axios的data参数，让axios自动处理序列化
      const requestData = body
      // 处理headers类型问题
      const headers = options.headers as any
      let axiosHeaders: any = {
        'Content-Type': 'application/json'
      }
      
      // 转换headers为axios接受的格式
      if (headers) {
        if (Array.isArray(headers)) {
          headers.forEach(([key, value]) => {
            axiosHeaders[key] = value
          })
        } else {
          axiosHeaders = { ...axiosHeaders, ...headers }
        }
      }
      
      const response = await http.request({
        url: endpoint,
        method: method as any,
        data: requestData,
        headers: axiosHeaders,
      })

      return response as unknown as ApiResponse<T>
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }


  // 模型管理API
  async getModels(): Promise<ApiResponse<Model[]>> {
    return this.request<Model[]>('/api/models', {
      method: 'GET'
    })
  }

  async getModel(id: string): Promise<ApiResponse<Model>> {
    return this.request<Model>(`/api/models/${id}`, {
      method: 'GET'
    })
  }

  async createModel(modelData: Partial<Model>): Promise<ApiResponse<Model>> {
    return this.request<Model>('/api/models/', {
      method: 'POST',
      body: JSON.stringify(modelData),
    })
  }

  // 模型版本API
  async getModelVersions(modelId: string): Promise<ApiResponse<Model[]>> {
    return this.request<Model[]>(`/api/models/${modelId}/versions`, {
      method: 'GET'
    })
  }

  async createModelVersion(modelId: string, versionData: { [key: string]: any }): Promise<ApiResponse<Model>> {
    return this.request<Model>(`/api/models/${modelId}/versions`, {
      method: 'POST',
      body: JSON.stringify(versionData),
    })
  }

  async importModel(modelData: { name: string }, file: File): Promise<ApiResponse<Model>> {
    const formData = new FormData()
    formData.append('name', modelData.name)
    formData.append('file', file)

    try {
      const response = await http.request({
        url: '/api/models/import',
        method: 'POST',
        data: formData,
        // 让浏览器自动设置 multipart 边界
        headers: {},
      })

      return response as unknown as ApiResponse<Model>
    } catch (error) {
      console.error('API request failed for /api/models/import/', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    }
  }


  async loadPretrainedModel(modelData: {
    model_name_or_path: string;
    model_format?: string;
    model_type?: string;
    name?: string;
    metadata?: Record<string, any>;
  }): Promise<ApiResponse<Model>> {
    return this.request<Model>('/api/models/pretrained', {
      method: 'POST',
      body: JSON.stringify(modelData),
    })
  }

  async updateModel(id: string, modelData: Partial<Model>): Promise<ApiResponse<Model>> {
    return this.request<Model>(`/api/models/${id}`, {
      method: 'PUT',
      body: JSON.stringify(modelData),
    })
  }

  async deleteModel(id: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/api/models/${id}`, {
      method: 'DELETE',
    })
  }

  async startModel(modelId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/models/${modelId}/start`, {
      method: 'POST',
    })
  }

  async pauseModel(modelId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/models/${modelId}/pause`, {
      method: 'POST',
    })
  }

  // 推理服务API
  async runInference(request: InferenceRequest): Promise<ApiResponse<InferenceResponse>> {
    return this.request<InferenceResponse>('/api/inference', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async getInferenceHistory(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/api/inference/history', {
      method: 'GET'
    })
  }

  // 训练任务API
  async getTrainingTasks(): Promise<ApiResponse<TrainingTask[]>> {
    return this.request<TrainingTask[]>('/api/training/tasks', {
      method: 'GET'
    })
  }

  async createTrainingTask(taskData: Partial<TrainingTask>): Promise<ApiResponse<TrainingTask>> {
    return this.request<TrainingTask>('/api/training/tasks', {
      method: 'POST',
      body: JSON.stringify(taskData),
    })
  }

  async getTrainingTask(id: string): Promise<ApiResponse<TrainingTask>> {
    return this.request<TrainingTask>(`/api/training/tasks/${id}`, {
      method: 'GET'
    })
  }

  // 模型训练API
  async startModelTraining(modelId: string, trainingData: { [key: string]: any }): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/models/${modelId}/train`, {
      method: 'POST',
      body: JSON.stringify({ training_data: trainingData }),
    })
  }

  async getModelTrainingStatus(taskId: string): Promise<ApiResponse<TrainingStatusResponse>> {
    return this.request<TrainingStatusResponse>(`/api/models/training/${taskId}`, {
      method: 'GET',
    })
  }

  // 区块链API
  async getBlockchainStatus(): Promise<ApiResponse<BlockchainStatus>> {
    return this.request<BlockchainStatus>('/api/blockchain/status', {
      method: 'GET'
    })
  }

  async verifyModelIntegrity(modelId: string, modelHash: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/blockchain/models/${modelId}/verify`, {
      method: 'POST',
      body: JSON.stringify({ model_hash: modelHash }),
    })
  }

  async getModelHistory(modelId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/blockchain/models/${modelId}/history`, {
      method: 'GET'
    })
  }

  // 权限管理API
  async grantPermission(permissionData: {
    resource_id: string,
    permission: string,
    granted_to: string,
    granted_by: string,
    expires_at?: string
  }): Promise<ApiResponse<any>> {
    return this.request<any>('/api/blockchain/access/grant', {
      method: 'POST',
      body: permissionData
    });
  }
  
  async revokePermission(permissionData: {
    resource_id: string,
    permission: string,
    granted_to: string
  }): Promise<ApiResponse<any>> {
    return this.request<any>('/api/blockchain/access/revoke', {
      method: 'POST',
      body: permissionData
    });
  }
  
  async checkPermission(permissionData: {
    resource_id: string,
    permission: string,
    user_id: string
  }): Promise<ApiResponse<boolean>> {
    return this.request<boolean>('/api/blockchain/access/check', {
      method: 'POST',
      body: permissionData
    });
  }
  
  async createRole(roleData: {
    role_id: string,
    role_name: string,
    permissions: string[],
    description: string
  }): Promise<ApiResponse<any>> {
    return this.request<any>('/api/blockchain/roles/create', {
      method: 'POST',
      body: roleData
    });
  }
  
  async assignRole(roleData: {
    user_id: string,
    role_id: string,
    assigned_by?: string
  }): Promise<ApiResponse<any>> {
    return this.request<any>('/api/blockchain/roles/assign', {
      method: 'POST',
      body: roleData
    });
  }
  
  async getContractStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/blockchain/contracts/status', {
      method: 'GET',
    });
  }

  // 联邦学习API
  async getFederatedClients(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/api/federated/clients', {
      method: 'GET'
    })
  }

  async registerFederatedClient(clientInfo: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/federated/clients', {
      method: 'POST',
      body: JSON.stringify(clientInfo),
    })
  }

  async getFederatedRounds(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/api/federated/rounds', {
      method: 'GET'
    })
  }

  async startFederatedRound(config: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/federated/rounds', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  }

  async aggregateFederatedRound(roundId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/federated/rounds/${roundId}/aggregate`, {
      method: 'POST'
    })
  }

  async getFederatedStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/federated/status', {
      method: 'GET'
    })
  }

  async getFederatedPrivacyStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/federated/privacy/status', {
      method: 'GET'
    })
  }

  async updateFederatedPrivacyConfig(config: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/federated/privacy/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    })
  }

  // 系统监控API
  async getSystemMetrics(): Promise<ApiResponse<SystemMetrics>> {
    return this.request<SystemMetrics>('/api/system/metrics', {
      method: 'GET'
    })
  }

  async getHealthStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/system/health', {
      method: 'GET'
    })
  }

  // 通用API方法

  async post<T>(endpoint: string, data: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  // 边缘计算API
  async getEdgeDevices(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/api/edge/devices', {
      method: 'GET'
    })
  }

  async syncEdgeDevice(deviceId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/edge/devices/${deviceId}/sync`, {
      method: 'POST'
    })
  }

  // 农业相关API
  async getCropConfigs(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/agriculture/crop-configs', {
      method: 'GET'
    })
  }

  async generateLightRecipe(data: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/agriculture/light-recipe', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async predictGrowth(data: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/agriculture/growth-prediction', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async createPlantingPlan(data: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/agriculture/planting-plans', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async contributeAgricultureData(data: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/agriculture/contribute-data', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  // 性能监控API
  async getPerformanceMetrics(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/summary')
  }

  // JEPA-DT-MPC集成API
  async getJepaDtmpcStatus(): Promise<ApiResponse<{ is_active: boolean; model_status: string }>> {
    return this.request<{ is_active: boolean; model_status: string }>('/api/jepa-dtmpc/status', {
      method: 'GET'
    })
  }

  async getJepaPrediction(): Promise<ApiResponse<JEPAData>> {
    return this.request<JEPAData>('/api/jepa-dtmpc/prediction', {
      method: 'GET'
    })
  }

  async activateJepaDtmpc(params: {
    controller_params?: { [key: string]: any };
    mv_params?: { [key: string]: any };
    cv_params?: { [key: string]: any };
    model_params?: { [key: string]: any };
    jepa_params?: { [key: string]: any };
  }): Promise<ApiResponse<void>> {
    return this.request<void>('/api/jepa-dtmpc/activate', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  }

  async trainJepaModel(): Promise<ApiResponse<{ task_id: string }>> {
    return this.request<{ task_id: string }>('/api/jepa-dtmpc/train', {
      method: 'POST'
    })
  }

  async getJepaTrainingStatus(taskId: string): Promise<ApiResponse<TrainingStatusResponse>> {
    return this.request<TrainingStatusResponse>(`/api/jepa-dtmpc/train/${taskId}`, {
      method: 'GET'
    })
  }

  async getPerformanceSummary(timeRange: string = "1h"): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/performance/summary?time_range=${timeRange}`, {
      method: 'GET'
    })
  }

  // 设备控制API
  async getDevices(): Promise<ApiResponse<Device[]>> {
    return this.request<Device[]>('/api/ai-control/devices', {
      method: 'GET'
    })
  }

  async scanDevices(): Promise<ApiResponse<Device[]>> {
    return this.request<Device[]>('/api/ai-control/scan-devices', {
      method: 'GET'
    })
  }

  async controlDevice(deviceId: number, controlParams: any): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/ai-control/device/${deviceId}`, {
      method: 'POST',
      body: JSON.stringify(controlParams),
    })
  }

  async getDeviceStatus(deviceId: number): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/ai-control/device/${deviceId}/status`, {
      method: 'GET'
    })
  }

  async activateMasterControl(activate: boolean): Promise<ApiResponse<any>> {
    return this.request<any>('/api/ai-control/master-control', {
      method: 'POST',
      body: JSON.stringify({ activate }),
    })
  }

  async toggleDeviceConnection(deviceId: number, connect: boolean): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/ai-control/device/${deviceId}/connection`, {
      method: 'POST',
      body: JSON.stringify({ connect }),
    })
  }

  async getIntegrationPerformanceSummary(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/integration-summary', {
      method: 'GET'
    })
  }

  async getOptimizationStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/optimization-status', {
      method: 'GET'
    })
  }

  async getOptimizationRecommendations(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/api/performance/optimization-recommendations', {
      method: 'GET'
    })
  }

  async applyOptimization(component: string, recommendationType: string): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/optimization/apply', {
      method: 'POST',
      body: JSON.stringify({ component, recommendation_type: recommendationType }),
    })
  }

  async getPerformanceAlerts(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/alerts', {
      method: 'GET'
    })
  }

  async acknowledgeAlert(alertId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/performance/alerts/${alertId}/acknowledge`, {
      method: 'POST'
    })
  }

  async getBenchmarkReport(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/benchmark/report', {
      method: 'GET'
    })
  }

  async runBenchmarkTest(testType: string, parameters: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/benchmark/run', {
      method: 'POST',
      body: JSON.stringify({ test_type: testType, parameters }),
    })
  }

  async enableAutoOptimization(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/auto-optimization/enable', {
      method: 'POST'
    })
  }

  async disableAutoOptimization(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/auto-optimization/disable', {
      method: 'POST'
    })
  }

  async runAutoOptimization(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/optimization/auto/run', {
      method: 'POST'
    })
  }

  async getMetricDetails(metricType: string, timeRange: string = "1h"): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/performance/metrics/${metricType}?time_range=${timeRange}`, {
      method: 'GET'
    })
  }

  async recordPerformanceMetric(metricData: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/metrics', {
      method: 'POST',
      body: JSON.stringify(metricData),
    })
  }

  async recordIntegrationPerformance(performanceData: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/integration-metrics', {
      method: 'POST',
      body: JSON.stringify(performanceData),
    })
  }

  async recordMigrationLearningPerformance(performanceData: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/migration-learning-metrics', {
      method: 'POST',
      body: JSON.stringify(performanceData),
    })
  }

  async recordEdgeComputingPerformance(performanceData: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/performance/edge-computing-metrics', {
      method: 'POST',
      body: JSON.stringify(performanceData),
    })
  }

  // 社区API
  async getCommunityLiveStreams(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/api/community/live-streams')
  }

  async likePost(postId: number): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/community/posts/${postId}/like`, {
      method: 'POST',
    })
  }

  async submitComment(streamId: number, content: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/community/live-streams/${streamId}/comments`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    })
  }

  // 设置API
  async saveSettings(settings: any): Promise<ApiResponse<any>> {
    return this.request<any>('/api/system/settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    })
  }

  // 推理服务控制API
  async toggleInferenceService(active: boolean): Promise<ApiResponse<any>> {
    return this.request<any>('/api/ai-control/device/5', {
      method: 'POST',
      body: JSON.stringify({ action: active ? 'start_inference' : 'stop_inference' }),
    })
  }

  async testInferenceService(serviceId: number): Promise<ApiResponse<any>> {
    return this.request<any>('/api/ai-control/device/5', {
      method: 'POST',
      body: JSON.stringify({ action: 'test_inference', service_id: serviceId }),
    })
  }

  // 决策API
  async makeDecision(type: string, inputData: any): Promise<ApiResponse<any>> {
    // 根据类型选择不同的端点
    const endpoint = type === 'agriculture' ? '/api/decision/agriculture' : '/api/decision/risk';
    
    // 转换请求格式以匹配后端DecisionRequest模型
    // 根据type设置不同的任务类型和目标
    const isRiskAnalysis = type === 'risk';
    
    // 从inputData中提取需要的字段，过滤掉不应该传递给后端的字段
    const filteredInputData = inputData?.data || inputData || {};
    
    // 确保requestBody只包含后端DecisionRequest模型定义的字段
    // 注意：当task_type为'routine_monitoring'时，risk_level必须为'low'
    const requestBody = {
      // 默认值作为回退
      temperature: filteredInputData.temperature || 25.5,
      humidity: filteredInputData.humidity || 65.0,
      co2_level: filteredInputData.co2_level || 400.0,
      light_intensity: filteredInputData.light_intensity || 1500.0,
      spectrum_config: filteredInputData.spectrum_config || {
        uv_380nm: 0.05,
        far_red_720nm: 0.1,
        white_light: 0.7,
        red_660nm: 0.15
      },
      crop_type: filteredInputData.crop_type || 'tomato',
      growth_day: filteredInputData.growth_day || 30,
      growth_rate: filteredInputData.growth_rate || 0.8,
      health_score: filteredInputData.health_score || 0.9,
      yield_potential: filteredInputData.yield_potential || 0.85,
      energy_consumption: filteredInputData.energy_consumption || 120.0,
      resource_utilization: filteredInputData.resource_utilization || 0.75,
      objective: filteredInputData.objective || (isRiskAnalysis ? 'enhance_resistance' : 'maximize_yield'),
      task_type: filteredInputData.task_type || (isRiskAnalysis ? 'high_priority' : 'routine_monitoring'),
      risk_level: filteredInputData.risk_level || (isRiskAnalysis ? 'high' : 'low')
    };
    
    return this.request<any>(endpoint, {
      method: 'POST',
      body: requestBody
    });
  }
  
  // 有机体AI核心API
  async activateOrganicAIIteration(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/decision/organic-core/activate-iteration', {
      method: 'POST',
    });
  }
  
  async deactivateOrganicAIIteration(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/decision/organic-core/deactivate-iteration', {
      method: 'POST',
    });
  }
  
  async getOrganicAIStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/decision/organic-core/status', {
      method: 'GET',
    });
  }
  
  async evolveOrganicAIStructure(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/decision/organic-core/evolve-structure', {
      method: 'POST',
    });
  }

  // 认证API
  async registerWithCode(code: string, email: string, password: string): Promise<ApiResponse<User>> {
    return this.request<User>('/api/auth/register/code', {
      method: 'POST',
      body: JSON.stringify({ code, email, password }),
    })
  }

  async login(email: string, password: string): Promise<ApiResponse<LoginResponse>> {
    return this.request<LoginResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
  }

  // 摄像头控制API
  async openCamera(cameraIndex: number = 0): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/open', {
      method: 'POST',
      body: JSON.stringify({ camera_index: cameraIndex }),
    })
  }

  async closeCamera(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/close', {
      method: 'POST'
    })
  }

  async getCameraStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/status', {
      method: 'GET'
    })
  }

  // 系统日志API
  async getSystemLogs(limit: number = 100): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/system/logs?limit=${limit}`, {
      method: 'GET'
    })
  }

  async getCameraFrame(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/frame', {
      method: 'GET'
    })
  }

  async startTracking(trackerType: string = 'CSRT'): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/tracking/start', {
      method: 'POST',
      body: JSON.stringify({ tracker_type: trackerType }),
    })
  }

  async stopTracking(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/tracking/stop', {
      method: 'POST'
    })
  }

  async startRecognition(modelType: string = 'haar'): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/recognition/start', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType }),
    })
  }

  async stopRecognition(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/recognition/stop', {
      method: 'POST'
    })
  }

  async listCameras(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/camera/list', {
      method: 'GET'
    })
  }
}

// 创建全局API客户端实例
export const apiClient = new ApiClient()

