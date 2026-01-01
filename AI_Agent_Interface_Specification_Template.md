# 🛠️ 智能体接口规范模板（国家标准版）

> **版本**：v1.0.0  
> **制定依据**：《国家人工智能产业综合标准化体系建设指南(2024版)》+ ITU-T F.748.46 + 智能体互联协议标准  
> **适用范围**：企业级AI智能体API设计与实现  
> **制定日期**：2025-01-01  
> **目标**：确保智能体接口符合国家标准和国际标准要求

---

## 📖 目录

1. [接口设计原则](#接口设计原则)
2. [通用接口规范](#通用接口规范)
3. [智能体核心接口](#智能体核心接口)
4. [互联协作接口](#互联协作接口)
5. [安全认证接口](#安全认证接口)
6. [数据格式规范](#数据格式规范)
7. [错误处理规范](#错误处理规范)
8. [性能指标要求](#性能指标要求)
9. [测试验证规范](#测试验证规范)
10. [文档生成标准](#文档生成标准)

---

## 🎯 1. 接口设计原则

### 1.1 标准化原则
- **遵循RESTful设计**：采用RESTful API设计原则
- **统一数据格式**：使用JSON作为主要数据交换格式
- **版本管理**：采用URL路径版本管理（如/v1/、/v2/）
- **命名规范**：使用小写字母和连字符，见名知意

### 1.2 安全性原则
- **认证授权**：所有接口必须实现认证授权机制
- **数据加密**：敏感数据必须加密传输
- **输入验证**：所有输入参数必须验证
- **权限控制**：实现细粒度权限控制

### 1.3 可靠性原则
- **错误处理**：统一错误处理机制
- **重试机制**：支持客户端重试
- **限流控制**：实现API限流机制
- **监控告警**：提供接口监控能力

---

## 🌐 2. 通用接口规范

### 2.1 基础路径规范
```
API基础路径：https://api.yourdomain.com/ai-agent/v1/
```

### 2.2 HTTP方法规范
| 方法 | 用途 | 幂等性 | 安全性 |
|------|------|--------|--------|
| GET | 获取资源 | 是 | 是 |
| POST | 创建资源 | 否 | 否 |
| PUT | 更新资源 | 是 | 否 |
| PATCH | 部分更新 | 否 | 否 |
| DELETE | 删除资源 | 是 | 否 |

### 2.3 请求头规范
```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer {token}
User-Agent: YourAgent/1.0
X-Request-ID: {unique-request-id}
X-Client-Version: 1.0.0
```

### 2.4 响应格式规范
```json
{
  "success": true,
  "code": 200,
  "message": "操作成功",
  "data": {},
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "unique-request-id"
}
```

---

## 🤖 3. 智能体核心接口

### 3.1 智能体管理接口

#### 3.1.1 创建智能体
```
POST /agents
```

**请求参数**：
```json
{
  "name": "客服智能体",
  "description": "智能客服助手",
  "type": "customer_service",
  "model_config": {
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "capabilities": [
    "text_generation",
    "dialogue_management",
    "knowledge_base_query"
  ],
  "permissions": [
    "read_user_profile",
    "access_knowledge_base"
  ]
}
```

**响应示例**：
```json
{
  "success": true,
  "code": 201,
  "message": "智能体创建成功",
  "data": {
    "agent_id": "agt_1234567890",
    "name": "客服智能体",
    "status": "active",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 3.1.2 获取智能体列表
```
GET /agents?page=1&size=10&status=active
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "获取成功",
  "data": {
    "items": [
      {
        "agent_id": "agt_1234567890",
        "name": "客服智能体",
        "description": "智能客服助手",
        "type": "customer_service",
        "status": "active",
        "created_at": "2025-01-01T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 3.1.3 获取智能体详情
```
GET /agents/{agent_id}
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "获取成功",
  "data": {
    "agent_id": "agt_1234567890",
    "name": "客服智能体",
    "description": "智能客服助手",
    "type": "customer_service",
    "status": "active",
    "model_config": {
      "model_name": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 2048
    },
    "capabilities": [
      "text_generation",
      "dialogue_management",
      "knowledge_base_query"
    ],
    "permissions": [
      "read_user_profile",
      "access_knowledge_base"
    ],
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 3.1.4 更新智能体
```
PUT /agents/{agent_id}
```

**请求参数**：
```json
{
  "name": "更新后的客服智能体",
  "description": "更新后的智能客服助手",
  "model_config": {
    "model_name": "gpt-4-turbo",
    "temperature": 0.5
  }
}
```

#### 3.1.5 删除智能体
```
DELETE /agents/{agent_id}
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "智能体删除成功",
  "data": {
    "agent_id": "agt_1234567890"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

### 3.2 对话交互接口

#### 3.2.1 发送消息
```
POST /agents/{agent_id}/chat
```

**请求参数**：
```json
{
  "message": "你好，我想咨询一下产品信息",
  "user_id": "user_123456",
  "session_id": "sess_789012",
  "context": {
    "user_profile": {
      "name": "张三",
      "age": 30,
      "preferences": ["科技", "运动"]
    },
    "conversation_history": [
      {
        "role": "user",
        "content": "你好",
        "timestamp": "2025-01-01T00:00:00Z"
      }
    ]
  },
  "options": {
    "stream": false,
    "temperature": 0.7
  }
}
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "消息处理成功",
  "data": {
    "response": "您好！很高兴为您服务。请问您想了解哪方面的产品信息呢？",
    "session_id": "sess_789012",
    "conversation_id": "conv_345678",
    "tokens_used": {
      "input": 15,
      "output": 25,
      "total": 40
    },
    "execution_time": 1.2,
    "confidence": 0.95
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 3.2.2 流式对话（SSE）
```
GET /agents/{agent_id}/chat/stream?message=你好&user_id=user_123456
```

**SSE响应格式**：
```
data: {"type": "start", "conversation_id": "conv_345678"}

data: {"type": "chunk", "content": "您好"}

data: {"type": "chunk", "content": "！很高兴为您服务"}

data: {"type": "end", "tokens_used": {"input": 15, "output": 25}}
```

#### 3.2.3 获取对话历史
```
GET /agents/{agent_id}/conversations/{conversation_id}?user_id=user_123456
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "获取成功",
  "data": {
    "conversation_id": "conv_345678",
    "messages": [
      {
        "role": "user",
        "content": "你好，我想咨询一下产品信息",
        "timestamp": "2025-01-01T00:00:00Z",
        "user_id": "user_123456"
      },
      {
        "role": "assistant",
        "content": "您好！很高兴为您服务。请问您想了解哪方面的产品信息呢？",
        "timestamp": "2025-01-01T00:00:01Z",
        "agent_id": "agt_1234567890"
      }
    ],
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:01Z"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

### 3.3 任务执行接口

#### 3.3.1 提交任务
```
POST /agents/{agent_id}/tasks
```

**请求参数**：
```json
{
  "task_type": "data_analysis",
  "description": "分析用户行为数据",
  "input_data": {
    "dataset_id": "ds_123456",
    "analysis_type": "trend_analysis",
    "time_range": {
      "start": "2024-12-01T00:00:00Z",
      "end": "2024-12-31T23:59:59Z"
    }
  },
  "priority": "high",
  "callback_url": "https://your-callback.com/task-result"
}
```

**响应示例**：
```json
{
  "success": true,
  "code": 202,
  "message": "任务已提交",
  "data": {
    "task_id": "task_789012",
    "status": "pending",
    "estimated_completion": "2025-01-01T00:05:00Z",
    "created_at": "2025-01-01T00:00:00Z"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 3.3.2 获取任务状态
```
GET /agents/{agent_id}/tasks/{task_id}
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "获取成功",
  "data": {
    "task_id": "task_789012",
    "status": "completed",
    "progress": 100,
    "result": {
      "summary": "用户行为分析完成",
      "key_findings": ["用户活跃度上升20%", "转化率提升15%"],
      "recommendations": ["增加营销投入", "优化用户体验"]
    },
    "started_at": "2025-01-01T00:00:00Z",
    "completed_at": "2025-01-01T00:02:30Z",
    "execution_time": 150
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

---

## 🔗 4. 互联协作接口

### 4.1 智能体发现接口

#### 4.1.1 搜索可用智能体
```
GET /discovery/agents?capabilities=text_generation&status=active&region=cn-east
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "搜索成功",
  "data": {
    "agents": [
      {
        "agent_id": "agt_1234567890",
        "name": "文本生成智能体",
        "capabilities": ["text_generation", "content_moderation"],
        "status": "active",
        "region": "cn-east",
        "latency": 50,
        "availability": 0.999,
        "supported_languages": ["zh", "en"]
      }
    ]
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 4.1.2 注册智能体
```
POST /discovery/register
```

**请求参数**：
```json
{
  "agent_id": "agt_new_agent_123",
  "name": "新智能体",
  "capabilities": ["image_generation", "text_to_speech"],
  "endpoint": "https://new-agent.yourdomain.com",
  "health_check_url": "https://new-agent.yourdomain.com/health",
  "region": "cn-east",
  "supported_languages": ["zh", "en", "ja"],
  "tags": ["multimodal", "high_performance"]
}
```

### 4.2 协作接口

#### 4.2.1 分配子任务
```
POST /collaboration/{main_agent_id}/subtasks
```

**请求参数**：
```json
{
  "subtask_id": "sub_123456",
  "target_agent_id": "agt_specialized_789",
  "task_type": "knowledge_query",
  "input_data": {
    "query": "查询产品技术规格",
    "product_id": "prod_12345"
  },
  "timeout": 30,
  "callback_url": "https://main-agent.com/subtask-result"
}
```

#### 4.2.2 获取协作状态
```
GET /collaboration/{main_agent_id}/status
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "获取成功",
  "data": {
    "main_agent_id": "agt_main_123",
    "collaboration_id": "collab_456789",
    "status": "in_progress",
    "subtasks": [
      {
        "subtask_id": "sub_123456",
        "target_agent_id": "agt_specialized_789",
        "status": "completed",
        "result": "查询结果数据",
        "completed_at": "2025-01-01T00:00:15Z"
      },
      {
        "subtask_id": "sub_234567",
        "target_agent_id": "agt_another_890",
        "status": "pending",
        "assigned_at": "2025-01-01T00:00:00Z"
      }
    ],
    "overall_progress": 50,
    "created_at": "2025-01-01T00:00:00Z"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

---

## 🔐 5. 安全认证接口

### 5.1 认证接口

#### 5.1.1 获取访问令牌
```
POST /auth/token
```

**请求参数**：
```json
{
  "grant_type": "client_credentials",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "scope": "agent:read agent:write"
}
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "认证成功",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "scope": "agent:read agent:write"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

#### 5.1.2 刷新令牌
```
POST /auth/refresh
```

**请求参数**：
```json
{
  "refresh_token": "refresh_token_here",
  "client_id": "your_client_id"
}
```

### 5.2 权限管理接口

#### 5.2.1 获取用户权限
```
GET /auth/permissions?user_id=user_123456
```

**响应示例**：
```json
{
  "success": true,
  "code": 200,
  "message": "获取成功",
  "data": {
    "user_id": "user_123456",
    "permissions": [
      "agent:create",
      "agent:read",
      "chat:send",
      "task:submit"
    ],
    "roles": ["user", "developer"],
    "scopes": ["public", "internal"]
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

---

## 📊 6. 数据格式规范

### 6.1 通用数据类型

#### 6.1.1 时间格式
- **格式**：ISO 8601标准格式
- **示例**：`2025-01-01T00:00:00Z`

#### 6.1.2 ID格式
- **智能体ID**：`agt_[a-zA-Z0-9]{10}`
- **用户ID**：`user_[a-zA-Z0-9]{6-12}`
- **会话ID**：`sess_[a-zA-Z0-9]{6-12}`
- **任务ID**：`task_[a-zA-Z0-9]{6-12}`

#### 6.1.3 枚举值定义
```json
{
  "agent_status": ["active", "inactive", "maintenance", "suspended"],
  "task_status": ["pending", "in_progress", "completed", "failed", "cancelled"],
  "priority": ["low", "normal", "high", "urgent"],
  "role": ["user", "assistant", "system"],
  "capability": [
    "text_generation", "image_generation", "audio_generation",
    "dialogue_management", "knowledge_base_query", "task_execution"
  ]
}
```

### 6.2 复杂数据结构

#### 6.2.1 模型配置
```json
{
  "model_name": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop_sequences": ["\n\n", "###"],
  "response_format": {
    "type": "json_object",
    "schema": {}
  }
}
```

#### 6.2.2 上下文数据
```json
{
  "user_profile": {
    "user_id": "user_123456",
    "name": "张三",
    "age": 30,
    "preferences": ["科技", "运动"],
    "history": {
      "last_login": "2025-01-01T00:00:00Z",
      "total_interactions": 150
    }
  },
  "conversation_context": {
    "topic": "产品咨询",
    "current_intent": "information_request",
    "entities": {
      "product": "智能手表",
      "brand": "Apple"
    },
    "sentiment": "neutral"
  },
  "external_data": {
    "knowledge_base": ["kb_123", "kb_456"],
    "user_data": {
      "purchases": ["prod_789", "prod_012"],
      "support_tickets": ["ticket_345"]
    }
  }
}
```

---

## ⚠️ 7. 错误处理规范

### 7.1 错误响应格式
```json
{
  "success": false,
  "code": 400,
  "message": "请求参数错误",
  "error": {
    "type": "VALIDATION_ERROR",
    "details": [
      {
        "field": "name",
        "code": "MISSING_FIELD",
        "message": "名称字段不能为空"
      }
    ],
    "trace_id": "err_1234567890"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

### 7.2 错误码定义

#### 7.2.1 通用错误码
| 错误码 | HTTP状态码 | 错误类型 | 说明 |
|--------|------------|----------|------|
| 10000 | 400 | VALIDATION_ERROR | 参数验证错误 |
| 10001 | 401 | AUTHENTICATION_ERROR | 认证失败 |
| 10002 | 403 | AUTHORIZATION_ERROR | 权限不足 |
| 10003 | 404 | NOT_FOUND | 资源不存在 |
| 10004 | 429 | RATE_LIMIT_ERROR | 请求频率超限 |
| 10005 | 500 | INTERNAL_ERROR | 内部服务器错误 |
| 10006 | 503 | SERVICE_UNAVAILABLE | 服务不可用 |

#### 7.2.2 业务错误码
| 错误码 | HTTP状态码 | 错误类型 | 说明 |
|--------|------------|----------|------|
| 20001 | 400 | AGENT_CREATION_FAILED | 智能体创建失败 |
| 20002 | 400 | TASK_EXECUTION_FAILED | 任务执行失败 |
| 20003 | 400 | CONVERSATION_EXPIRED | 对话已过期 |
| 20004 | 400 | INSUFFICIENT_CREDITS | 信用额度不足 |
| 20005 | 400 | UNSUPPORTED_CAPABILITY | 不支持的功能 |

### 7.3 重试机制
- **客户端重试**：对于5xx错误和网络错误实现指数退避重试
- **重试次数**：默认3次重试
- **重试间隔**：1s, 2s, 4s
- **幂等性**：确保重试操作的幂等性

---

## ⚡ 8. 性能指标要求

### 8.1 响应时间要求
| 接口类型 | P95响应时间 | P99响应时间 | 说明 |
|----------|-------------|-------------|------|
| 智能体创建/更新 | ≤2s | ≤5s | 配置复杂度影响 |
| 消息发送/接收 | ≤500ms | ≤1s | 实时对话要求 |
| 任务提交/查询 | ≤1s | ≤3s | 异步处理特性 |
| 智能体发现 | ≤200ms | ≤500ms | 发现服务要求 |
| 认证授权 | ≤100ms | ≤200ms | 安全性要求 |

### 8.2 吞吐量要求
- **并发用户数**：≥10,000
- **每秒请求数**：≥10,000 RPS
- **每秒任务数**：≥1,000 任务/s

### 8.3 可用性要求
- **系统可用性**：≥99.9%
- **数据一致性**：强一致性
- **故障恢复时间**：≤5分钟

### 8.4 资源使用要求
- **CPU使用率**：≤70%
- **内存使用率**：≤80%
- **磁盘使用率**：≤85%

---

## 🧪 9. 测试验证规范

### 9.1 功能测试

#### 9.1.1 接口功能测试
```json
{
  "test_case": "智能体创建接口测试",
  "api": "POST /agents",
  "input": {
    "name": "测试智能体",
    "type": "test_agent"
  },
  "expected_response": {
    "success": true,
    "code": 201
  },
  "assertions": [
    "响应状态码为201",
    "返回的agent_id格式正确",
    "智能体状态为active"
  ]
}
```

#### 9.1.2 业务逻辑测试
- **权限验证测试**：验证不同角色的权限控制
- **数据完整性测试**：验证数据的完整性和一致性
- **边界条件测试**：测试各种边界条件和异常情况

### 9.2 性能测试

#### 9.2.1 负载测试
- **并发测试**：模拟多用户并发访问
- **压力测试**：逐步增加负载直到系统极限
- **稳定性测试**：长时间运行系统稳定性

#### 9.2.2 性能监控指标
```json
{
  "response_time": {
    "p50": 100,
    "p95": 500,
    "p99": 1000
  },
  "throughput": {
    "requests_per_second": 1000
  },
  "error_rate": {
    "percentage": 0.1
  },
  "resource_usage": {
    "cpu": 65,
    "memory": 75,
    "disk_io": 80
  }
}
```

### 9.3 安全测试

#### 9.3.1 认证授权测试
- **无效令牌测试**：使用无效令牌访问
- **权限提升测试**：尝试访问未授权资源
- **令牌泄露测试**：验证令牌安全性

#### 9.3.2 数据安全测试
- **敏感数据泄露**：检查是否泄露敏感数据
- **数据加密测试**：验证数据传输和存储加密
- **输入验证测试**：验证输入参数验证机制

---

## 📚 10. 文档生成标准

### 10.1 API文档规范
- **OpenAPI 3.0**：使用OpenAPI 3.0标准
- **自动生成**：支持代码注释自动生成文档
- **版本管理**：文档版本与API版本同步
- **示例代码**：提供多种语言的示例代码

### 10.2 文档内容要求
- **接口描述**：详细的功能描述
- **参数说明**：完整的参数类型和验证规则
- **响应示例**：标准和错误响应示例
- **使用场景**：典型使用场景说明
- **性能指标**：性能要求和限制说明

### 10.3 SDK生成规范
- **多语言支持**：支持主流编程语言
- **异步支持**：提供异步调用支持
- **错误处理**：内置错误处理机制
- **重试机制**：内置重试逻辑
- **日志记录**：提供详细的日志记录

---

## 🛠️ 附录A：代码示例

### A.1 Python SDK示例
```python
from ai_agent_client import AIAgentClient

# 初始化客户端
client = AIAgentClient(
    api_key="your_api_key",
    base_url="https://api.yourdomain.com/ai-agent/v1/"
)

# 创建智能体
agent_config = {
    "name": "客服智能体",
    "type": "customer_service",
    "model_config": {
        "model_name": "gpt-4",
        "temperature": 0.7
    }
}

try:
    response = client.agents.create(agent_config)
    agent_id = response.data["agent_id"]
    print(f"智能体创建成功: {agent_id}")
except AIAgentError as e:
    print(f"创建失败: {e.message}")
```

### A.2 JavaScript SDK示例
```javascript
import { AIAgentClient } from '@your-company/ai-agent-sdk';

const client = new AIAgentClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.yourdomain.com/ai-agent/v1/'
});

// 发送消息
try {
  const response = await client.chat.sendMessage('agt_123456', {
    message: '你好',
    userId: 'user_123456'
  });
  
  console.log('回复:', response.data.response);
} catch (error) {
  console.error('发送失败:', error.message);
}
```

---

## 📋 附录B：合规检查清单

### B.1 接口合规检查
- [ ] 遵循RESTful设计原则
- [ ] 使用标准HTTP状态码
- [ ] 实现统一的响应格式
- [ ] 包含必要的请求头验证
- [ ] 实现输入参数验证
- [ ] 提供详细的错误信息
- [ ] 实现认证授权机制
- [ ] 遵循数据格式规范
- [ ] 满足性能指标要求
- [ ] 通过安全测试验证

### B.2 标准符合性检查
- [ ] 符合《国家人工智能产业综合标准化体系建设指南》要求
- [ ] 符合ITU-T F.748.46国际标准
- [ ] 符合智能体互联协议标准
- [ ] 符合行业特定标准（如适用）
- [ ] 通过第三方合规验证

---

**文档版本**：v1.0.0  
**制定日期**：2025-01-01  
**标准依据**：国家及国际智能体相关标准  
**文档状态**：正式实施

**备注**：
1. 本规范基于2025年最新智能体标准制定
2. 企业可根据具体需求调整接口设计
3. 建议定期更新规范以适应标准变化
4. 重要：所有接口实现必须遵循本规范
