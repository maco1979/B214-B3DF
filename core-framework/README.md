# 核心框架层 (Core Framework)

## 概述

核心框架层为AI农业微服务系统提供通用的基础功能和工具库，包括配置管理、日志记录、错误处理、数据验证等核心功能。

## 模块结构

```
core-framework/
├── src/
│   ├── config/          # 配置管理
│   ├── logging/         # 日志系统
│   ├── errors/          # 错误处理
│   ├── validation/      # 数据验证
│   ├── utils/           # 工具函数
│   ├── middleware/      # 中间件
│   └── types/           # 类型定义
├── tests/               # 测试代码
├── pyproject.toml       # 项目配置
├── requirements.txt     # 依赖管理
└── README.md           # 项目说明
```

## 功能特性

### 1. 配置管理
- 环境变量支持
- 配置文件管理
- 热重载配置
- 类型安全的配置访问

### 2. 日志系统
- 结构化日志
- 多级别日志支持
- 日志轮转和归档
- 分布式追踪支持

### 3. 错误处理
- 统一的错误类型
- 错误码管理
- 异常捕获和转换
- 错误报告和监控

### 4. 数据验证
- 请求数据验证
- 响应数据格式化
- 数据转换和序列化
- 类型安全的数据访问

### 5. 工具函数
- 日期时间处理
- 字符串操作
- 加密解密
- 文件操作

## 安装使用

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用
```python
from core_framework.config import Config
from core_framework.logging import setup_logging

# 初始化配置
config = Config()

# 设置日志
setup_logging(config)

# 使用配置
database_url = config.get('database.url')
```

## 依赖管理

核心框架设计为轻量级，尽量减少外部依赖，主要依赖：
- Python 3.9+
- Pydantic (数据验证)
- Loguru (日志)
- PyYAML (配置文件)

## 开发指南

### 代码规范
- 使用类型注解
- 遵循PEP 8规范
- 编写单元测试
- 提供详细的文档

### 测试
```bash
pytest tests/
```

### 构建
```bash
python -m build
```

## 贡献指南

欢迎提交Issue和Pull Request来改进核心框架。