# 容器化环境搭建方案

## 一、环境概述

### 1.1 环境层次

| 环境类型 | 用途 | 特点 |
|----------|------|------|
| 开发环境 | 开发人员日常开发和调试 | 配置灵活，便于调试，数据可重置 |
| 测试环境 | 功能测试、集成测试、性能测试 | 配置接近生产，数据相对稳定 |
| 预生产环境 | 上线前验证、回归测试 | 配置与生产一致，使用真实数据镜像 |

### 1.2 环境目标

1. **一致性**：确保各环境配置一致，减少环境差异导致的问题
2. **隔离性**：各环境相互隔离，避免相互影响
3. **可复制性**：能够快速复制和重建环境
4. **可扩展性**：支持根据需求扩展环境资源
5. **自动化**：环境搭建和管理自动化，减少人工操作
6. **安全性**：各环境具备适当的安全措施
7. **监控性**：具备完善的监控和日志系统

## 二、基础设施要求

### 2.1 硬件要求

| 环境类型 | 服务器数量 | CPU | 内存 | 存储 | 网络 |
|----------|------------|-----|------|------|------|
| 开发环境 | 1-3台 | 8核+ | 16GB+ | 500GB+ SSD | 千兆网卡 |
| 测试环境 | 3-5台 | 16核+ | 32GB+ | 1TB+ SSD | 千兆网卡/万兆网卡 |
| 预生产环境 | 5-8台 | 16核+ | 64GB+ | 2TB+ SSD | 万兆网卡 |

### 2.2 网络要求

1. **网络隔离**：各环境之间网络隔离，通过VPN或专线访问
2. **域名管理**：为各环境配置独立域名
3. **SSL证书**：为各环境配置SSL证书，确保通信安全
4. **负载均衡**：测试和预生产环境配置负载均衡
5. **防火墙**：配置防火墙规则，限制访问

## 三、技术栈

| 类别 | 技术选型 | 版本 | 用途 |
|------|----------|------|------|
| 容器引擎 | Docker | 24.0+ | 容器构建和运行 |
| 容器编排 | Kubernetes | 1.27+ | 容器编排和管理 |
| 容器运行时 | containerd | 1.7+ | 容器运行时 |
| 镜像仓库 | Harbor | 2.8+ | 私有容器镜像管理 |
| 配置管理 | Argo CD | 2.8+ | GitOps配置管理 |
| 网络插件 | Calico | 3.26+ | Kubernetes网络插件 |
| 存储插件 | Longhorn | 1.5+ | 持久化存储 |
| 监控系统 | Prometheus + Grafana | 2.47+/10.1+ | 系统监控和可视化 |
| 日志系统 | ELK Stack | 8.8+ | 日志收集、存储和分析 |
| CI/CD | Jenkins/GitLab CI | 2.414+/16.2+ | 持续集成和部署 |

## 四、环境搭建步骤

### 4.1 开发环境搭建

#### 4.1.1 单节点Kubernetes集群搭建（使用Kind或Minikube）

**步骤1：安装Docker**

```bash
# 更新系统
apt-get update && apt-get upgrade -y

# 安装Docker
apt-get install -y docker.io

# 启动Docker服务
systemctl start docker
systemctl enable docker

# 验证Docker安装
docker --version
```

**步骤2：安装Kind**

```bash
# 下载Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
mv ./kind /usr/local/bin/kind

# 验证Kind安装
kind --version
```

**步骤3：创建Kind集群**

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30000
    hostPort: 30000
    protocol: TCP
  - containerPort: 30001
    hostPort: 30001
    protocol: TCP
```

```bash
# 创建集群
kind create cluster --name dev-cluster --config kind-config.yaml

# 验证集群创建
kubectl cluster-info
```

**步骤4：安装Longhorn存储**

```bash
# 安装Longhorn
kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.5.1/deploy/longhorn.yaml

# 验证Longhorn安装
kubectl get pods -n longhorn-system
```

**步骤5：配置Harbor镜像仓库**

```bash
# 安装Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 下载Harbor离线安装包
wget https://github.com/goharbor/harbor/releases/download/v2.8.0/harbor-offline-installer-v2.8.0.tgz
tar xvf harbor-offline-installer-v2.8.0.tgz
cd harbor

# 配置Harbor
cp harbor.yml.tmpl harbor.yml
# 编辑harbor.yml，修改hostname、port、admin密码等

# 安装Harbor
./install.sh

# 验证Harbor安装
# 访问http://<harbor-host>:80
```

**步骤6：配置开发工具**

- 安装kubectl：用于管理Kubernetes集群
- 安装helm：用于部署应用
- 配置IDE插件：如VS Code的Kubernetes插件

### 4.2 测试环境搭建

#### 4.2.1 多节点Kubernetes集群搭建

**步骤1：准备服务器**

- 准备3-5台服务器
- 安装Ubuntu 22.04 LTS
- 配置静态IP
- 关闭防火墙和SELinux（或配置允许相关端口）
- 配置SSH免密码登录

**步骤2：安装Docker和containerd**

```bash
# 更新系统
apt-get update && apt-get upgrade -y

# 安装依赖包
apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# 添加Docker GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# 添加Docker源
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 安装Docker和containerd
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

# 配置containerd
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
# 编辑/etc/containerd/config.toml，配置sandbox_image为registry.k8s.io/pause:3.9

# 重启containerd
systemctl restart containerd
```

**步骤3：安装Kubernetes组件**

```bash
# 添加Kubernetes GPG密钥
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# 添加Kubernetes源
cat <<EOF > /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF

# 安装Kubernetes组件
apt-get update && apt-get install -y kubelet=1.27.0-00 kubeadm=1.27.0-00 kubectl=1.27.0-00

# 锁定版本，防止自动更新
apt-mark hold kubelet kubeadm kubectl
```

**步骤4：初始化Kubernetes集群**

```bash
# 在主节点初始化集群
kubeadm init --control-plane-endpoint="<master-host>:6443" --pod-network-cidr=192.168.0.0/16 --kubernetes-version=v1.27.0

# 配置kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# 安装Calico网络插件
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.0/manifests/calico.yaml

# 添加工作节点
# 在工作节点执行kubeadm join命令
```

**步骤5：安装Longhorn存储**

```bash
# 安装Longhorn
kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.5.1/deploy/longhorn.yaml

# 验证Longhorn安装
kubectl get pods -n longhorn-system
```

**步骤6：安装Harbor镜像仓库**

- 参考开发环境Harbor安装步骤
- 配置HTTPS，使用有效的SSL证书

**步骤7：安装监控和日志系统**

```bash
# 安装Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace

# 安装ELK Stack
# 可以使用Helm部署，或使用ECK（Elastic Cloud on Kubernetes）
```

### 4.3 预生产环境搭建

**预生产环境搭建步骤与测试环境类似，但需要注意以下几点：**

1. **配置与生产一致**：确保预生产环境的配置与生产环境完全一致
2. **使用真实数据镜像**：使用生产数据的镜像，确保测试的真实性
3. **更严格的安全措施**：配置与生产环境相同的安全措施
4. **完善的监控系统**：部署与生产环境相同的监控和日志系统
5. **容量规划**：根据生产环境的负载情况，合理规划预生产环境的资源

## 五、配置管理

### 5.1 使用Argo CD进行GitOps管理

**步骤1：安装Argo CD**

```bash
# 创建Argo CD命名空间
kubectl create namespace argocd

# 安装Argo CD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 暴露Argo CD服务
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "NodePort", "ports": [{"name": "http", "port": 80, "targetPort": 8080, "nodePort": 30080}, {"name": "https", "port": 443, "targetPort": 8443, "nodePort": 30443}]}}'

# 获取初始密码
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

**步骤2：配置Argo CD**

- 访问Argo CD UI：http://<argocd-host>:30080
- 创建应用：连接Git仓库，配置应用部署
- 配置自动同步：设置应用自动同步，实现GitOps

### 5.2 环境配置管理

| 配置类型 | 管理方式 | 工具 |
|----------|----------|------|
| 应用配置 | GitOps | Argo CD + Helm |
| 密钥管理 | 加密存储 | Vault/Sealed Secrets |
| 环境变量 | 配置文件 | ConfigMap |
| 集群配置 | 基础设施即代码 | Terraform/Ansible |

## 六、安全配置

### 6.1 容器安全

1. **镜像安全**：
   - 使用Harbor的镜像扫描功能，扫描镜像漏洞
   - 只使用可信镜像源
   - 定期更新镜像

2. **运行时安全**：
   - 部署Falco监控容器运行时
   - 限制容器权限，使用非root用户运行容器
   - 配置容器资源限制

### 6.2 集群安全

1. **访问控制**：
   - 配置RBAC，限制用户和服务账户的权限
   - 使用kubeconfig文件管理访问凭证
   - 定期轮换证书和密钥

2. **网络安全**：
   - 配置Network Policies，限制Pod间通信
   - 启用Pod Security Standards
   - 配置Ingress和Egress规则

3. **审计日志**：
   - 启用Kubernetes审计日志
   - 配置审计日志存储和分析

### 6.3 环境隔离

| 环境类型 | 隔离方式 |
|----------|----------|
| 开发环境 | 网络隔离，独立Kubernetes集群 |
| 测试环境 | 网络隔离，独立Kubernetes集群 |
| 预生产环境 | 网络隔离，独立Kubernetes集群，与生产环境物理隔离 |

## 七、环境管理

### 7.1 环境生命周期管理

| 操作 | 频率 | 工具 |
|------|------|------|
| 环境备份 | 每日 | Velero |
| 环境更新 | 每周/按需 | Argo CD |
| 环境清理 | 每周 | 自定义脚本 |
| 资源优化 | 每月 | Kubernetes HPA |

### 7.2 环境监控

1. **监控指标**：
   - 集群指标：CPU、内存、磁盘、网络
   - Pod指标：CPU、内存、重启次数
   - 应用指标：请求数、响应时间、错误率

2. **告警规则**：
   - CPU使用率 > 80% 持续5分钟
   - 内存使用率 > 85% 持续5分钟
   - Pod重启次数 > 3次 持续1分钟
   - 应用错误率 > 1% 持续5分钟

3. **日志管理**：
   - 集中收集所有环境的日志
   - 配置日志保留策略
   - 实现日志检索和分析

### 7.3 环境自动化

1. **自动化脚本**：
   - 环境搭建脚本
   - 环境备份和恢复脚本
   - 环境清理脚本

2. **CI/CD集成**：
   - 开发环境：代码提交自动部署
   - 测试环境：合并请求自动部署和测试
   - 预生产环境：手动触发部署，自动测试

## 八、开发工作流

### 8.1 本地开发流程

1. 开发人员在本地开发代码
2. 使用Docker Compose运行本地服务
3. 测试通过后提交代码到Git仓库
4. CI/CD流水线自动构建镜像，推送到Harbor
5. Argo CD自动同步到开发环境
6. 开发人员在开发环境验证

### 8.2 测试流程

1. 代码合并到测试分支
2. CI/CD流水线构建镜像，推送到Harbor
3. Argo CD自动同步到测试环境
4. 测试人员进行功能测试和集成测试
5. 性能测试团队进行性能测试
6. 发现问题后，开发人员修复并重新提交

### 8.3 预生产验证流程

1. 代码合并到预生产分支
2. CI/CD流水线构建镜像，推送到Harbor
3. 手动触发Argo CD同步到预生产环境
4. 进行回归测试和上线前验证
5. 验证通过后，准备上线生产

## 九、最佳实践

### 9.1 环境一致性

- 使用相同版本的Kubernetes和容器运行时
- 使用相同的Helm charts部署应用
- 使用相同的配置模板，只修改环境特定的变量
- 定期同步环境配置

### 9.2 资源管理

- 为每个环境配置资源配额
- 使用水平Pod自动伸缩（HPA）
- 定期清理无用资源
- 监控资源使用情况，优化资源配置

### 9.3 安全最佳实践

- 遵循最小权限原则
- 定期扫描镜像漏洞
- 启用审计日志
- 定期进行安全审计

### 9.4 自动化最佳实践

- 自动化环境搭建和配置
- 自动化测试和部署
- 自动化监控和告警
- 自动化备份和恢复

### 9.5 文档最佳实践

- 维护环境搭建文档
- 记录环境配置和变更
- 编写环境使用指南
- 维护故障处理手册

## 十、故障处理

### 10.1 常见故障

| 故障类型 | 可能原因 | 解决方法 |
|----------|----------|----------|
| Pod调度失败 | 资源不足 | 增加节点资源或调整Pod资源请求 |
| 服务不可访问 | 网络配置问题 | 检查Ingress、Service配置，检查Network Policies |
| 镜像拉取失败 | 镜像仓库不可访问或权限问题 | 检查Harbor配置，检查Pod的ImagePullSecrets |
| 数据丢失 | 存储配置问题 | 检查Longhorn配置，恢复备份 |

### 10.2 故障恢复流程

1. 定位故障：使用kubectl、Prometheus、Kibana等工具定位故障
2. 分析原因：分析日志和监控数据，确定故障原因
3. 实施修复：根据故障原因实施修复措施
4. 验证恢复：验证服务是否恢复正常
5. 记录故障：记录故障原因、修复过程和预防措施

## 十一、预期效果

1. **开发效率提升**：开发人员可以快速获取和使用开发环境
2. **测试质量提高**：测试环境配置接近生产，减少环境差异导致的问题
3. **上线风险降低**：预生产环境验证，确保上线前问题得到解决
4. **环境管理自动化**：减少人工操作，提高环境管理效率
5. **资源利用率优化**：根据需求弹性伸缩资源，提高资源利用率
6. **安全性增强**：各环境具备适当的安全措施，提高系统安全性

## 十二、后续维护

1. **定期更新**：定期更新Kubernetes版本和组件
2. **性能优化**：根据监控数据优化环境配置
3. **安全加固**：定期进行安全审计和加固
4. **文档更新**：及时更新环境文档，记录环境变更
5. **培训**：对开发和运维团队进行环境使用培训

## 十三、结论

本容器化环境搭建方案提供了从开发环境到预生产环境的完整搭建指南，采用了Kubernetes、Docker、Harbor、Argo CD等云原生技术，实现了环境的一致性、隔离性、可复制性和自动化管理。

通过本方案的实施，可以提高开发效率，降低测试和上线风险，优化资源利用率，增强系统安全性。同时，本方案遵循了云原生最佳实践，为后续系统的持续演进和扩展奠定了基础。