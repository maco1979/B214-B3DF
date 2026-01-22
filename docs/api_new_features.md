# API文档 - 新增功能

本文档描述了近期新增的性能优化、智能体权限管理和监控集成功能的API端点。

## 目录

1. [监控API](#监控api)
2. [异步缓存服务](#异步缓存服务)
3. [智能体权限管理](#智能体权限管理)
4. [性能指标](#性能指标)

---

## 监控API

监控API提供了与Prometheus/Grafana集成的能力，用于收集和导出系统指标。

### 基础信息

- **前缀**: `/api/monitoring`
- **标签**: 监控

### 端点列表

#### GET /api/monitoring/metrics

获取Prometheus格式的指标数据。

**响应格式**: `text/plain`

**响应示例**:
```
# HELP agent_actions_total Counter metric
# TYPE agent_actions_total counter
agent_actions_total{agent_id="agent_1",agent_type="decision_agent",action="process",status="success"} 150

# HELP cache_hits_total Counter metric
# TYPE cache_hits_total counter
cache_hits_total{cache_type="memory"} 1250
```

---

#### GET /api/monitoring/dashboard

获取监控仪表板数据。

**响应示例**:
```json
{
  "metrics": {
    "counters": {
      "agent_actions_total{...}": 150
    },
    "gauges": {
      "active_agents_count{...}": 5
    },
    "histograms": {}
  },
  "active_alerts": [],
  "alert_rules": [...],
  "timestamp": "2025-12-31T23:00:00.000000"
}
```

---

#### GET /api/monitoring/alerts

获取告警历史。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| hours | int | 否 | 查询最近多少小时的告警，默认24 |

**响应示例**:
```json
{
  "alerts": [
    {
      "name": "HighPermissionDenialRate",
      "severity": "warning",
      "value": 0.35,
      "timestamp": "2025-12-31T22:30:00.000000"
    }
  ],
  "count": 1
}
```

---

#### GET /api/monitoring/health

监控服务健康检查。

**响应示例**:
```json
{
  "status": "healthy",
  "service": "audit_monitoring",
  "timestamp": "2025-12-31T23:00:00.000000"
}
```

---

#### POST /api/monitoring/agent/action

记录智能体动作。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| agent_id | string | 是 | 智能体ID |
| agent_type | string | 是 | 智能体类型 |
| action | string | 是 | 动作名称 |
| status | string | 否 | 状态，默认"success" |
| duration_ms | float | 否 | 持续时间(毫秒) |

**响应示例**:
```json
{
  "success": true,
  "message": "Action recorded"
}
```

---

#### POST /api/monitoring/permission/check

记录权限检查。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| agent_id | string | 是 | 智能体ID |
| permission | string | 是 | 权限名称 |
| granted | bool | 是 | 是否授予 |

---

#### POST /api/monitoring/cache/access

记录缓存访问。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| cache_type | string | 是 | 缓存类型 |
| hit | bool | 是 | 是否命中 |

---

#### POST /api/monitoring/decision

记录决策。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| module | string | 是 | 模块名称 |
| objective | string | 是 | 目标 |
| status | string | 是 | 状态 |
| latency_ms | float | 否 | 延迟(毫秒) |

---

## 异步缓存服务

提供高性能的异步缓存功能，支持内存缓存和Redis缓存。

### 类: AsyncMemoryCache

LRU内存缓存，支持TTL过期。

**初始化参数**:
```python
AsyncMemoryCache(max_size=1000, default_ttl=300)
```

**方法**:

| 方法 | 说明 |
|------|------|
| `async get(key: str)` | 获取缓存值 |
| `async set(key: str, value: Any, ttl: int = None)` | 设置缓存值 |
| `async delete(key: str)` | 删除缓存 |
| `async clear()` | 清空缓存 |
| `async get_stats()` | 获取统计信息 |

**性能指标**:
- 平均读写时间: < 0.01ms
- 并发吞吐: > 28万 ops/秒

---

### 类: AsyncRedisCache

异步Redis缓存，自动降级到内存缓存。

**初始化参数**:
```python
AsyncRedisCache(redis_url="redis://localhost:6379", default_ttl=300)
```

**方法**:

| 方法 | 说明 |
|------|------|
| `async get(key: str)` | 获取缓存值 |
| `async set(key: str, value: Any, ttl: int = None)` | 设置缓存值 |
| `async get_or_set(key: str, factory: Callable, ttl: int = None)` | 获取或创建缓存 |
| `async is_connected()` | 检查Redis连接 |

---

### 类: DecisionResultCache

决策结果专用缓存。

**方法**:

| 方法 | 说明 |
|------|------|
| `async get_cached_decision(module, state, objective)` | 获取缓存的决策 |
| `async cache_decision(module, state, objective, decision, ttl)` | 缓存决策结果 |
| `async invalidate_module(module)` | 失效某模块的所有缓存 |

---

## 智能体权限管理

提供细粒度的智能体权限控制和行为审计。

### 智能体类型

| 类型 | 说明 | 默认权限 |
|------|------|---------|
| DECISION_AGENT | 决策智能体 | decision.read, decision.create, model.read, metrics.read |
| CONTROL_AGENT | 控制智能体 | device.read, device.control, decision.read, metrics.read |
| MONITOR_AGENT | 监控智能体 | metrics.read, logs.read, system.read, device.read |
| LEARNING_AGENT | 学习智能体 | model.read, model.train, data.read, metrics.read |
| DATA_AGENT | 数据智能体 | data.read, data.write, metrics.read |
| SYSTEM_AGENT | 系统智能体 | system.read, system.configure, system.admin, metrics.read, logs.read |

### 权限列表

| 权限 | 说明 |
|------|------|
| decision.read | 读取决策 |
| decision.create | 创建决策 |
| decision.execute | 执行决策 |
| device.read | 读取设备状态 |
| device.control | 控制设备 |
| device.configure | 配置设备 |
| model.read | 读取模型 |
| model.train | 训练模型 |
| model.deploy | 部署模型 |
| model.delete | 删除模型 |
| data.read | 读取数据 |
| data.write | 写入数据 |
| data.delete | 删除数据 |
| data.export | 导出数据 |
| system.read | 读取系统信息 |
| system.configure | 配置系统 |
| system.admin | 系统管理员 |
| metrics.read | 读取指标 |
| logs.read | 读取日志 |
| alerts.manage | 管理告警 |

### 类: AgentPermissionManager

**使用示例**:

```python
from backend.src.core.services.agent_permission_manager import (
    AgentPermissionManager, AgentType
)

# 创建管理器
manager = AgentPermissionManager()

# 注册智能体
agent = await manager.register_agent(
    agent_id="agent_001",
    agent_type=AgentType.DECISION_AGENT,
    name="决策智能体1",
    rate_limit=100  # 每分钟最大请求数
)

# 检查权限
has_permission = await manager.check_permission(
    agent_id="agent_001",
    permission="decision.read"
)

# 授予额外权限
await manager.grant_permission(
    agent_id="agent_001",
    permission="model.train",
    granted_by="admin"
)

# 撤销权限
await manager.revoke_permission(
    agent_id="agent_001",
    permission="model.train",
    revoked_by="admin"
)

# 获取审计摘要
summary = await manager.get_audit_summary(hours=24)
```

---

## 性能指标

### 基准测试结果

| 测试项 | 性能指标 |
|--------|---------|
| 异步内存缓存读写 | 0.003ms 平均延迟 |
| 决策结果缓存 | 100% 命中率 |
| 权限检查 | 0.012ms 平均延迟 |
| 审计日志记录 | 0.01ms 平均延迟 |
| 并发缓存访问 | 280,000+ ops/秒 |
| Prometheus导出 | < 0.1ms |

### Prometheus指标

系统导出以下Prometheus指标：

| 指标名称 | 类型 | 说明 |
|----------|------|------|
| agent_actions_total | counter | 智能体动作总数 |
| agent_permission_checks_total | counter | 权限检查总数 |
| active_agents_count | gauge | 活跃智能体数量 |
| agent_action_duration_seconds | histogram | 智能体动作耗时 |
| cache_hits_total | counter | 缓存命中数 |
| cache_misses_total | counter | 缓存未命中数 |
| cache_size_bytes | gauge | 缓存大小 |
| decisions_total | counter | 决策总数 |
| decision_latency_seconds | histogram | 决策延迟 |
| system_health_status | gauge | 系统健康状态 |

---

## Prometheus配置

将以下配置添加到`prometheus.yml`以抓取监控指标：

```yaml
scrape_configs:
  - job_name: 'ai-platform-monitoring'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/api/monitoring/metrics'
    scrape_interval: 10s
```

---

## Grafana仪表板

建议创建以下面板：

1. **智能体活动面板**
   - 智能体动作趋势图
   - 权限检查成功/失败比例
   - 活跃智能体数量

2. **缓存性能面板**
   - 缓存命中率
   - 缓存大小趋势
   - 缓存操作延迟

3. **决策性能面板**
   - 决策数量趋势
   - 决策延迟分布
   - 各模块决策统计

4. **告警面板**
   - 活跃告警列表
   - 告警历史趋势
   - 按严重程度分类

---

## 更新日志

### v1.1.0 (2025-12-31)

**新功能**:
- 添加异步缓存服务（AsyncMemoryCache, AsyncRedisCache, DecisionResultCache）
- 添加智能体权限管理器（细粒度权限控制+速率限制）
- 添加审计日志监控服务（Prometheus集成）
- 添加监控API端点（/api/monitoring/*）

**性能优化**:
- 将阻塞式psutil调用改为异步
- 添加决策结果缓存减少重复计算
- 并发缓存访问优化

**文档**:
- 添加新功能API文档
