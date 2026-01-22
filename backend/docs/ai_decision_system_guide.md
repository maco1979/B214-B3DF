# AI自主决策系统使用指南

## 系统概述

AI自主决策系统是一个基于强化学习的完全自主决策平台，为现有AI农业平台提供全方位的智能决策能力。系统支持秒级实时响应，覆盖农业参数优化、区块链积分分配、AI模型自动训练和系统资源动态分配等核心功能。

## 核心功能模块

### 1. 农业参数优化决策模块
- **功能**: 基于强化学习的农业生长参数自动优化
- **应用场景**: 作物生长环境参数调整（温度、湿度、光照等）
- **API端点**: `POST /api/v1/decision/agriculture/optimize`

### 2. 区块链积分分配决策模块
- **功能**: 基于风险评估的区块链积分智能分配
- **应用场景**: 用户贡献度评估和积分分配
- **API端点**: `POST /api/v1/decision/blockchain/allocate`

### 3. AI模型自动训练决策模块
- **功能**: 智能模型训练参数和策略优化
- **应用场景**: 自动选择训练参数、优化训练策略
- **API端点**: `POST /api/v1/decision/model_training/optimize`

### 4. 系统资源动态分配模块
- **功能**: 实时系统资源分配和负载均衡
- **应用场景**: 计算资源动态调度和优化
- **API端点**: `POST /api/v1/decision/resource/allocate`

## API使用说明

### 基础配置

```python
import requests
import json

# API基础URL
BASE_URL = "http://localhost:8000"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your_token_here"
}
```

### 农业参数优化决策

```python
# 请求示例
payload = {
    "crop_type": "tomato",
    "current_params": {
        "temperature": 25.0,
        "humidity": 0.6,
        "light_intensity": 800,
        "soil_ph": 6.5
    },
    "historical_data": {
        "yield_history": [0.8, 0.9, 0.85, 0.92],
        "quality_scores": [85, 88, 82, 90]
    }
}

response = requests.post(
    f"{BASE_URL}/api/v1/decision/agriculture/optimize",
    headers=headers,
    json=payload
)

# 响应示例
{
    "optimized_params": {
        "temperature": 26.5,
        "humidity": 0.65,
        "light_intensity": 850,
        "soil_ph": 6.6
    },
    "confidence": 0.87,
    "expected_improvement": 0.12,
    "execution_time": 0.045
}
```

### 区块链积分分配决策

```python
# 请求示例
payload = {
    "user_id": "user_123",
    "contribution_score": 85,
    "risk_level": "medium",
    "available_points": 1000,
    "historical_allocations": [
        {"date": "2024-01-01", "points": 100},
        {"date": "2024-01-02", "points": 150}
    ]
}

response = requests.post(
    f"{BASE_URL}/api/v1/decision/blockchain/allocate",
    headers=headers,
    json=payload
)

# 响应示例
{
    "allocated_points": 120,
    "allocation_reason": "balanced_contribution",
    "risk_adjustment": 0.15,
    "confidence": 0.92,
    "next_review_period": 7
}
```

### 性能监控API

系统提供实时性能监控接口：

```python
# 获取性能指标
response = requests.get(f"{BASE_URL}/api/v1/decision/monitoring/performance")

# 响应示例
{
    "total_decisions": 1245,
    "avg_response_time": 0.045,
    "max_response_time": 0.123,
    "avg_throughput": 22.2,
    "error_rate": 0.003,
    "performance_targets": {
        "response_time": true,
        "throughput": true,
        "error_rate": true
    }
}
```

## 系统配置

### 性能优化配置

系统支持多种环境配置，可通过环境变量调整：

```bash
# 环境变量配置
export DECISION_ENVIRONMENT="production"  # production/development/testing
export MAX_CONCURRENT_DECISIONS=20
export CACHE_TTL=300
export RESPONSE_TIMEOUT=3.0
```

### 强化学习配置

```python
from backend.config.performance_config import PerformanceConfig

# 获取生产环境配置
config = PerformanceConfig.get_optimized_config("production")

# 自定义配置
custom_config = {
    "decision_engine": {
        "max_concurrent_decisions": 15,
        "decision_timeout": 2.0
    },
    "reinforcement_learning": {
        "learning_rate": 0.001,
        "exploration_rate": 0.1
    }
}
```

## 性能测试

系统提供完整的性能测试工具：

```bash
# 运行性能测试
cd backend/tests
python decision_performance_test.py

# 测试输出示例
============================================================
AI自主决策系统性能测试结果
============================================================
测试时间: 2024-12-20 14:30:25
总迭代次数: 80
总错误数: 0
错误率: 0.00%
总体平均响应时间: 45.23ms
总体最小响应时间: 23.15ms
总体最大响应时间: 89.67ms
✅ 系统达到秒级实时响应要求!
```

## 故障排除

### 常见问题

1. **响应时间过长**
   - 检查系统负载
   - 调整并发决策数量
   - 优化缓存配置

2. **决策准确性下降**
   - 检查模型训练数据
   - 调整强化学习参数
   - 验证特征工程

3. **系统资源不足**
   - 监控CPU和内存使用率
   - 调整资源分配策略
   - 启用异步处理

### 日志监控

系统提供详细的日志记录：

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("ai_decision_system")
```

## 最佳实践

1. **批量处理**: 对于大量决策请求，使用批量处理接口
2. **缓存策略**: 合理配置缓存生存时间和大小
3. **监控告警**: 设置性能阈值告警机制
4. **定期优化**: 定期重新训练强化学习模型
5. **容错处理**: 实现降级决策机制

## 技术支持

如有技术问题，请联系：
- 系统架构师: tech-support@example.com
- 文档维护: docs@example.com
- 紧急支持: emergency@example.com

---

*最后更新: 2024年12月20日*
*版本: v1.0.0*