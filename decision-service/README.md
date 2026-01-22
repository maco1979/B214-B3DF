# 决策服务微服务 (Decision Service)

AI农业平台的决策服务微服务，提供基于强化学习的农业参数优化决策功能。

## 功能特性

- ✅ 农业参数优化决策
- ✅ 批量决策处理
- ✅ 决策反馈收集
- ✅ 决策性能监控
- ✅ 决策历史查询
- ✅ 策略参数优化

## 技术栈

- **框架**: FastAPI + Uvicorn
- **机器学习**: JAX + Flax
- **数据库**: PostgreSQL (决策历史存储)
- **消息队列**: Redis (异步任务处理)
- **监控**: Prometheus + Grafana

## 服务架构

```
API网关 → 决策服务 → [决策引擎] → [农业AI模型] → [数据库/Redis]
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn src.main:app --host 0.0.0.0 --port 8001
```

## API文档

启动服务后访问: http://localhost:8001/docs

## 环境配置

复制 `.env.example` 为 `.env` 并配置相应环境变量。