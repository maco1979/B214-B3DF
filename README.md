# AI农业决策系统

AI农业决策系统是一个融合强化学习、区块链与边缘计算的前沿项目，已在全国20多个省市落地应用。系统通过"空天地一体化感知+云端智能决策+终端自主执行"的技术闭环，实现从种植管理到市场预判的全流程数字化。

## 软件架构说明

### 整体架构
```mermaid
graph TB
    subgraph "客户端层"
        A1[Web前端 - React/Vite]
        A2[移动端APP - React Native]
        A3[智能设备 - IoT]
    end
    
    subgraph "API网关层"
        B1[API Gateway - Node.js]
        B2[服务发现 - Consul]
        B3[负载均衡 - Nginx]
    end
    
    subgraph "核心服务层"
        C1[决策引擎服务 - FastAPI]
        C2[模型训练服务 - FastAPI]
        C3[区块链服务 - Hyperledger Fabric]
        C4[边缘计算服务 - FastAPI]
        C5[数据服务 - FastAPI]
    end
    
    subgraph "数据存储层"
        D1[PostgreSQL - 主数据]
        D2[Redis - 缓存]
        D3[TimescaleDB - 时序数据]
        D4[MinIO - 对象存储]
        D5[区块链网络 - Hyperledger]
    end
    
    subgraph "基础设施层"
        E1[Kubernetes集群]
        E2[Docker容器]
        E3[监控系统 - Prometheus/Grafana]
        E4[日志系统 - ELK Stack]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> C1
    B1 --> C2
    B1 --> C3
    B1 --> C4
    B1 --> C5
    C1 --> D1
    C1 --> D2
    C2 --> D1
    C2 --> D2
    C2 --> D4
    C3 --> D5
    C4 --> D2
    C4 --> D3
    C5 --> D1
    C5 --> D3
    E1 --> C1
    E1 --> C2
    E1 --> C3
    E1 --> C4
    E1 --> C5
    E3 --> C1
    E3 --> C2
    E3 --> C3
    E3 --> C4
    E3 --> C5
```

### 核心技术栈
- **后端**: FastAPI (Python 3.10+) + JAX/Flax (AI/ML)
- **前端**: React 18 + TypeScript + Vite 5 + Tailwind CSS
- **数据库**: PostgreSQL 15 + Redis 7 + TimescaleDB
- **区块链**: Hyperledger Fabric 2.5
- **容器化**: Docker + Kubernetes
- **监控**: Prometheus + Grafana

## 安装教程

### 环境要求
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL (可选，用于开发)
- Redis (可选，用于开发)

### 本地开发环境安装

1. **克隆项目**
```bash
git clone https://gitee.com/Organism_AI/AI-Agricultural-Decision-System.git
cd AI-Agricultural-Decision-System
```

2. **安装后端依赖**
```bash
cd backend
pip install -r requirements.txt
```

3. **安装前端依赖**
```bash
cd ../frontend
npm install
```

4. **配置环境变量**
```bash
# backend/.env
ENVIRONMENT=development
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:password@localhost:5432/ai_agriculture
REDIS_URL=redis://localhost:6379
BLOCKCHAIN_ENDPOINT=http://localhost:7050
```

5. **启动服务**
```bash
# 启动后端服务
cd backend
python src/main.py

# 启动前端服务
cd frontend
npm run dev
```

### Docker部署
```bash
# 启动完整环境
docker-compose up -d

# 访问服务
- 前端: http://localhost:3000
- 后端API: http://localhost:8000
- 文档: http://localhost:8000/docs
```

## 使用说明

### 系统功能模块

1. **智能决策引擎**
   - 访问: `/api/decision`
   - 功能: 基于强化学习的农业参数优化决策

2. **区块链溯源系统**
   - 访问: `/api/blockchain`
   - 功能: 农业数据上链与溯源查询

3. **边缘计算模块**
   - 访问: `/api/edge`
   - 功能: 边缘节点管理与模型部署

4. **设备控制模块**
   - 访问: `/api/camera` 和 `/api/ai_control`
   - 功能: PTZ云台控制与智能设备联动

5. **数据服务**
   - 访问: `/api/models`, `/api/inference`, `/api/training`
   - 功能: 模型管理、推理服务、训练管理

### API使用示例

```javascript
// 获取决策建议
fetch('/api/decision/agriculture', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    crop_type: 'wheat',
    weather_data: {...},
    soil_data: {...}
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### 系统管理

1. **监控系统**
   - 访问: http://localhost:3000 (Grafana)
   - 监控指标: CPU、内存、请求量、错误率等

2. **日志查询**
   - 通过Grafana或ELK Stack进行日志查询

3. **性能调优**
   - 根据监控数据调整资源配置
   - 优化数据库查询和缓存策略

### 常见问题

1. **端口冲突**
   - 检查8000、3000、5432、6379端口是否被占用
   - 修改配置文件中的端口设置

2. **依赖安装失败**
   - 确保网络连接正常
   - 检查Python和Node.js版本

3. **数据库连接失败**
   - 检查数据库服务是否启动
   - 验证数据库连接配置

## 项目结构

```
AI-Agricultural-Decision-System/
├── backend/                 # 后端服务
│   ├── src/
│   │   ├── api/           # API路由
│   │   ├── core/          # 核心业务逻辑
│   │   ├── blockchain/    # 区块链集成
│   │   └── services/      # 服务层
│   ├── requirements.txt
│   └── main.py
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/    # 组件
│   │   ├── pages/         # 页面
│   │   └── services/      # API服务
│   ├── package.json
│   └── vite.config.ts
├── api-gateway/            # API网关
├── docker-compose.yml      # 容器编排
└── README.md
```

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- **项目主页**: https://gitee.com/Organism_AI/AI-Agricultural-Decision-System
- **技术文档**: https://docs.ai-agri.com
- **商务合作**: contact@ai-agri.com