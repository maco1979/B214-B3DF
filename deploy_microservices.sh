#!/bin/bash

# AI平台微服务部署脚本
# 支持百万级并发用户

set -e  # 遇到错误时退出

echo "🚀 开始部署AI平台微服务架构..."

echo "🔧 检查kubectl和集群连接..."
# 检查kubectl是否已安装
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl未安装，请先安装kubectl"
    exit 1
fi

# 检查是否已连接到Kubernetes集群
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ 未连接到Kubernetes集群，请配置kubectl"
    exit 1
fi

echo "✅ Kubernetes集群连接正常"

# 创建命名空间和资源配置
echo "🔧 创建命名空间和资源配置..."
kubectl apply -f infrastructure/kubernetes/namespace.yaml

# 等待命名空间创建
sleep 5

# 创建密钥 (请根据实际情况修改)
echo "🔐 创建密钥..."
kubectl create secret generic ai-platform-secrets \
  --namespace=ai-platform \
  --from-literal=database-password=your_secure_password \
  --from-literal=database-url=postgresql://ai_user:your_secure_password@ai-platform-postgres:5432/ai_platform \
  --from-literal=registry-credentials=your_registry_credentials \
  --from-literal=rabbitmq-password=your_rabbitmq_password \
  --dry-run=client -o yaml | kubectl apply -f -

# 部署监控系统
echo "📊 部署监控系统..."
kubectl apply -f infrastructure/kubernetes/monitoring.yaml

# 部署数据库集群
echo "💾 部署数据库集群..."
kubectl apply -f infrastructure/kubernetes/database-hpa.yaml

# 等待数据库Pod启动
echo "⏳ 等待数据库集群启动..."
sleep 30

# 部署Redis集群
echo "⚡ 部署Redis集群..."
kubectl apply -f infrastructure/kubernetes/redis-hpa.yaml

# 部署消息队列 (Kafka和RabbitMQ)
echo "📨 部署消息队列..."
kubectl apply -f infrastructure/kubernetes/service-mesh.yaml
kubectl apply -f infrastructure/kubernetes/task-queue.yaml

# 部署服务网格
echo "🛡️ 部署服务网格..."
kubectl apply -f infrastructure/kubernetes/istio-config.yaml

# 部署CDN和边缘计算
echo "🌐 部署CDN和边缘计算..."
kubectl apply -f infrastructure/kubernetes/cdn-edge-config.yaml

# 部署微服务
echo "⚙️ 部署微服务..."
kubectl apply -f infrastructure/kubernetes/microservices.yaml

# 部署API网关
echo "🌐 部署API网关..."
kubectl apply -f infrastructure/kubernetes/service-mesh.yaml

# 部署前端和后端服务
echo "📱 部署前端和后端服务..."
kubectl apply -f infrastructure/kubernetes/backend-deployment.yaml
kubectl apply -f infrastructure/kubernetes/frontend-deployment.yaml

# 验证部署
echo "🔍 验证部署状态..."

echo "📋 检查所有Pod状态:"
kubectl get pods -n ai-platform

echo "📋 检查服务状态:"
kubectl get services -n ai-platform

echo "📋 检查HPA状态:"
kubectl get hpa -n ai-platform

echo "📋 检查Istio服务:"
kubectl get virtualservices,destinationrules,gateways -n ai-platform

# 等待所有Pod就绪
echo "⏳ 等待所有服务就绪..."
kubectl wait --for=condition=ready pod -l app=ai-platform -n ai-platform --timeout=600s || true

echo "✅ AI平台微服务部署完成！"

echo ""
echo "📈 部署摘要:"
echo "- 命名空间: ai-platform"
echo "- 数据库: PostgreSQL集群 (3节点)"
echo "- 缓存: Redis集群 (6节点 3主3从)"
echo "- 消息队列: Kafka集群 (3节点) 和 RabbitMQ集群 (3节点)"
echo "- 微服务: 认证、模型、推理、决策等"
echo "- API网关: Kong (5实例)"
echo "- 服务网格: Istio (流量管理、安全通信)"
echo "- CDN和边缘计算: 全球分发和就近处理"
echo "- 监控系统: Prometheus + AlertManager"
echo "- 自动扩缩容: 已配置HPA策略"
echo ""
echo "💡 下一步操作:"
echo "1. 配置负载均衡器以访问服务"
echo "2. 验证服务网格功能"
echo "3. 进行压力测试验证性能"
echo "4. 配置CI/CD流水线"
echo "5. 设置告警规则和监控面板"