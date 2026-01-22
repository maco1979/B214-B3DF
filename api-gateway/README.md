# API网关服务 (API Gateway)

## 概述

API网关服务是AI农业微服务系统的统一入口，负责请求路由、负载均衡、认证授权、限流熔断等核心功能。

## 功能特性

### 核心功能
- **统一入口**：所有外部请求统一通过网关访问
- **动态路由**：基于服务发现的路由配置
- **负载均衡**：多种负载均衡策略
- **认证授权**：JWT认证和权限控制
- **限流熔断**：请求限流和服务熔断
- **监控日志**：请求追踪和性能监控

### 高级功能
- **API聚合**：多个后端API聚合为单个接口
- **请求转换**：请求/响应格式转换
- **缓存策略**：API响应缓存
- **安全防护**：API安全防护机制

## 技术栈

- **框架**：Node.js + Express.js + TypeScript
- **服务发现**：Consul客户端
- **认证**：JWT + Redis
- **限流**：express-rate-limit
- **监控**：Prometheus + Grafana
- **容器化**：Docker + Kubernetes

## 快速开始

### 安装依赖
```bash
cd api-gateway
npm install
```

### 开发环境
```bash
npm run dev
```

### 生产环境
```bash
npm run build
npm start
```

## 配置说明

### 环境变量
```bash
# 服务配置
PORT=8080
NODE_ENV=development

# 服务发现
CONSUL_HOST=localhost
CONSUL_PORT=8500

# 认证配置
JWT_SECRET=your-secret-key
REDIS_URL=redis://localhost:6379

# 限流配置
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
```

### 路由配置
```yaml
# routes.yaml
routes:
  - path: /api/decision/**
    service: decision-engine
    methods: [GET, POST]
    auth: true
    rate_limit: 100
    
  - path: /api/models/**
    service: model-training
    methods: [GET, POST, PUT, DELETE]
    auth: true
    
  - path: /api/blockchain/**
    service: blockchain-service
    methods: [GET, POST]
    auth: true
```

## 架构设计

### 请求处理流程
```
客户端请求 → API网关 → 认证中间件 → 限流中间件 → 路由匹配 → 服务发现 → 负载均衡 → 后端服务 → 响应处理 → 客户端
```

### 核心组件
- **GatewayServer**：网关服务器
- **RouteManager**：路由管理器
- **AuthMiddleware**：认证中间件
- **RateLimitMiddleware**：限流中间件
- **ServiceDiscovery**：服务发现
- **LoadBalancer**：负载均衡器

## 监控指标

### 基础指标
- 请求总数
- 请求成功率
- 平均响应时间
- 错误率

### 业务指标
- API调用次数
- 用户活跃度
- 服务健康状态

## 部署说明

### Docker部署
```bash
docker build -t api-gateway .
docker run -p 8080:8080 api-gateway
```

### Kubernetes部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: api-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: CONSUL_HOST
          value: "consul-service"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## 开发指南

### 添加新路由
1. 在 `src/routes` 目录创建路由文件
2. 配置路由规则
3. 注册路由到网关

### 自定义中间件
1. 在 `src/middleware` 目录创建中间件
2. 实现中间件逻辑
3. 注册中间件到网关

### 测试
```bash
npm test
npm run test:coverage
```

## 故障排除

### 常见问题
- **服务不可达**：检查服务发现配置
- **认证失败**：检查JWT配置和Redis连接
- **限流生效**：调整限流参数或检查业务逻辑

### 日志分析
- 查看网关日志定位问题
- 分析监控指标识别瓶颈
- 检查服务健康状态

## 性能优化

### 网关优化
- 启用响应压缩
- 配置连接池
- 优化中间件顺序

### 后端优化
- 服务实例扩容
- 数据库优化
- 缓存策略优化