# Kubernetes 部署指南

## 项目架构

本项目采用微服务架构，支持百万级并发用户。主要组件包括：

- API网关 (Kong)
- 多个微服务 (认证、模型管理、推理、决策等)
- 数据库集群 (PostgreSQL + PgBouncer)
- 缓存集群 (Redis)
- 消息队列 (Kafka)
- 监控和日志系统

## 部署步骤

### 1. 准备工作

```bash
# 创建命名空间
kubectl create namespace ai-platform

# 创建密钥 (需要根据实际情况修改)
kubectl create secret generic ai-platform-secrets \
  --namespace=ai-platform \
  --from-literal=database-password=your_password \
  --from-literal=database-url=postgresql://ai_user:your_password@ai-platform-postgres:5432/ai_platform \
  --from-literal=registry-credentials=your_registry_credentials
```

### 2. 部署基础服务

```bash
# 部署命名空间和资源配置
kubectl apply -f namespace.yaml

# 部署数据库和连接池
kubectl apply -f database-hpa.yaml

# 部署Redis集群
kubectl apply -f redis-hpa.yaml

# 部署消息队列 (Kafka和RabbitMQ)
kubectl apply -f service-mesh.yaml
kubectl apply -f task-queue.yaml

# 部署微服务
kubectl apply -f microservices.yaml

# 部署服务网格 (Istio配置)
kubectl apply -f istio-config.yaml

# 部署API网关
kubectl apply -f service-mesh.yaml

# 部署CDN和边缘计算配置
kubectl apply -f cdn-edge-config.yaml

# 部署监控系统
kubectl apply -f monitoring.yaml

# 部署后端和前端服务的HPA
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml
```

### 3. 验证部署

```bash
# 检查所有Pod状态
kubectl get pods -n ai-platform

# 检查服务状态
kubectl get services -n ai-platform

# 检查HPA状态
kubectl get hpa -n ai-platform

# 查看日志
kubectl logs -n ai-platform deployment/ai-platform-backend

# 检查Istio服务
kubectl get virtualservices,destinationrules,gateways -n ai-platform
```

## 高并发配置说明

### 自动扩缩容配置

1. **后端服务 (ai-platform-backend)**:
   - 最小副本数: 50
   - 最大副本数: 500
   - CPU使用率阈值: 70%
   - 内存使用率阈值: 80%

2. **前端服务 (ai-platform-frontend)**:
   - 最小副本数: 10
   - 最大副本数: 100
   - CPU使用率阈值: 70%
   - 内存使用率阈值: 80%

3. **微服务 (认证、模型、推理、决策)**:
   - 根据服务类型配置不同的副本数和资源限制
   - 支持基于CPU和内存的自动扩缩容

4. **任务队列服务 (Celery Worker)**:
   - 最小副本数: 5
   - 最大副本数: 50
   - 支持基于队列长度的自定义指标扩缩容

5. **边缘计算节点**:
   - 最小副本数: 10
   - 最大副本数: 100
   - 支持基于请求率的扩缩容

### 数据库优化

- PostgreSQL配置优化，支持2000个并发连接
- 使用PgBouncer连接池，减少数据库连接开销
- 主从复制配置，提高读取性能和可用性
- 支持读写分离，分散数据库负载

### 缓存策略

- Redis集群，支持6个节点(3主3从)
- 配置最大内存2GB，使用LRU策略
- 支持故障转移和自动恢复
- 实现多级缓存策略

### 消息队列

- Kafka集群，支持高吞吐量消息处理
- RabbitMQ作为备选方案，支持复杂路由
- Zookeeper协调服务，确保集群稳定性
- 用于异步任务处理和事件驱动架构

### 服务网格

- Istio服务网格，提供流量管理
- 智能路由和负载均衡
- 服务间安全通信
- 流量控制和熔断机制

### CDN和边缘计算

- 全球CDN分发，减少延迟
- 边缘计算节点，就近处理请求
- 静态资源缓存优化
- 降低中心服务器负载

### 性能监控

系统集成了Prometheus和Grafana监控：

- 应用性能指标
- 资源使用情况
- 服务健康状态
- 自动告警机制
- 分布式追踪

## 故障恢复

- Pod自动重启策略
- 服务健康检查
- 自动故障转移
- 数据备份和恢复机制
- 服务熔断和降级

## 安全配置

- 网络策略限制
- RBAC权限控制
- TLS加密通信
- 安全密钥管理
- 服务间认证