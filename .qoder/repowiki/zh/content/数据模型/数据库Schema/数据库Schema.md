# 数据库Schema

<cite>
**本文档引用的文件**
- [init.sql](file://backend/init.sql)
- [decision_models.py](file://decision-service/src/models/decision_models.py)
- [performance_config.py](file://backend/config/performance_config.py)
- [database-deployment.yaml](file://infrastructure/kubernetes/database-deployment.yaml)
</cite>

## 目录
1. [数据库表结构设计](#数据库表结构设计)
2. [业务含义与使用场景](#业务含义与使用场景)
3. [表间关联关系与扩展建议](#表间关联关系与扩展建议)
4. [ORM映射实现](#orm映射实现)
5. [数据库管理指南](#数据库管理指南)
6. [结论](#结论)

## 数据库表结构设计

根据`init.sql`脚本，系统包含三张核心表：`users`、`ai_models`和`performance_metrics`。以下是各表的详细字段定义：

### users表
该表用于用户认证与权限管理，其结构设计如下：

- **id**: `SERIAL`类型，主键，自增标识符
- **username**: `VARCHAR(50)`类型，唯一且非空，存储用户名
- **email**: `VARCHAR(100)`类型，唯一且非空，存储用户邮箱
- **password_hash**: `VARCHAR(255)`类型，非空，存储密码哈希值
- **created_at**: `TIMESTAMP`类型，默认值为`CURRENT_TIMESTAMP`，记录创建时间
- **updated_at**: `TIMESTAMP`类型，默认值为`CURRENT_TIMESTAMP`，记录更新时间

该表通过`username`和`email`字段的唯一约束确保用户信息的唯一性，同时使用`password_hash`字段安全存储密码哈希而非明文密码。

### ai_models表
该表用于存储AI模型的元信息，其结构设计如下：

- **id**: `SERIAL`类型，主键，自增标识符
- **model_name**: `VARCHAR(100)`类型，非空，存储模型名称
- **model_type**: `VARCHAR(50)`类型，非空，存储模型类型（如分类、检测等）
- **version**: `VARCHAR(20)`类型，非空，存储模型版本号
- **accuracy**: `DECIMAL(5,4)`类型，可为空，存储模型准确率（范围0-1）
- **created_at**: `TIMESTAMP`类型，默认值为`CURRENT_TIMESTAMP`，记录创建时间

该表通过`model_name`、`model_type`和`version`字段组合描述一个具体的AI模型实例，`accuracy`字段用于记录模型性能指标。

### performance_metrics表
该表用于记录系统性能指标，其结构设计如下：

- **id**: `SERIAL`类型，主键，自增标识符
- **metric_name**: `VARCHAR(50)`类型，非空，存储指标名称（如响应时间、吞吐量等）
- **metric_value**: `DECIMAL(10,4)`类型，非空，存储指标数值
- **timestamp**: `TIMESTAMP`类型，默认值为`CURRENT_TIMESTAMP`，记录指标采集时间

该表设计为通用性能指标存储结构，可通过`metric_name`区分不同类型的性能数据，支持灵活的监控需求。

**Section sources**
- [init.sql](file://backend/init.sql#L10-L35)

## 业务含义与使用场景

### users表：用户认证与权限管理
`users`表是系统安全体系的核心，承担用户身份认证和权限管理功能。在实际使用中，该表支持以下场景：
- 用户注册时，系统将用户名、邮箱和密码哈希存入此表
- 用户登录时，系统通过比对邮箱和密码哈希完成身份验证
- 系统根据用户信息实现访问控制和权限管理
- `created_at`和`updated_at`字段用于审计用户账户的生命周期

该表的设计遵循安全最佳实践，使用`VARCHAR(255)`存储密码哈希以兼容各种哈希算法（如bcrypt、scrypt），同时通过唯一约束防止用户名和邮箱重复。

### ai_models表：模型元信息存储
`ai_models`表是AI模型管理的核心，用于存储和管理平台上的所有AI模型元数据。其主要使用场景包括：
- 模型注册：当新模型部署到平台时，将其基本信息存入此表
- 模型版本管理：通过`version`字段支持模型的版本控制和迭代
- 模型选择：系统根据`model_type`字段选择适合特定任务的模型
- 性能跟踪：通过`accuracy`字段记录和比较不同模型的性能表现

该表的设计支持平台的多模型管理需求，为模型的全生命周期管理提供数据支持。

### performance_metrics表：系统性能监控
`performance_metrics`表是系统监控体系的基础，用于收集和存储各种性能指标。其主要使用场景包括：
- 实时监控：系统定期采集关键性能指标并存入此表
- 性能分析：通过历史数据趋势分析系统性能变化
- 告警触发：当某些指标超出阈值时触发告警
- 容量规划：基于历史性能数据进行系统扩容决策

该表的通用设计使其能够灵活适应不同类型的性能数据，支持系统的可扩展性需求。

**Section sources**
- [init.sql](file://backend/init.sql#L9-L35)

## 表间关联关系与扩展建议

### 潜在关联关系分析
虽然当前表结构中未显式定义外键约束，但存在以下潜在的业务关联：

1. **用户与模型归属关系**：虽然`ai_models`表目前没有用户关联字段，但在业务逻辑上，某些模型可能是由特定用户创建或拥有的。这种隐式的归属关系可以通过应用层逻辑维护。

2. **性能指标与模型关联**：`performance_metrics`表记录的性能数据可能与特定AI模型相关，例如某个模型的推理延迟或准确率变化。目前这种关联需要通过`metric_name`的命名约定来隐式表达。

3. **操作审计关联**：用户的操作可能会触发模型的使用或产生性能数据，形成用户-模型-性能的间接关联链。

### 外键约束扩展建议
为了增强数据完整性和查询效率，建议未来考虑以下外键约束扩展：

1. **添加模型归属关系**：在`ai_models`表中添加`created_by`字段，作为外键关联到`users`表的`id`字段，明确模型的创建者。

```sql
ALTER TABLE ai_models ADD COLUMN created_by INTEGER REFERENCES users(id);
```

2. **添加模型性能关联**：在`performance_metrics`表中添加`model_id`字段，作为外键关联到`ai_models`表的`id`字段，明确性能指标对应的模型。

```sql
ALTER TABLE performance_metrics ADD COLUMN model_id INTEGER REFERENCES ai_models(id);
```

3. **创建复合索引**：在扩展外键后，为常用查询模式创建复合索引，如`(model_id, timestamp)`索引以优化按模型和时间范围查询性能。

这些扩展将使表间关系更加明确，支持更复杂的查询需求，同时通过外键约束保证数据一致性。

**Section sources**
- [init.sql](file://backend/init.sql#L9-L35)

## ORM映射实现

### ORM框架与配置
系统使用SQLAlchemy作为ORM框架，通过`decision_models.py`文件中的类定义实现数据库表的Python对象映射。ORM配置遵循最佳实践，包括：
- 使用`declarative_base()`创建基类，支持声明式模型定义
- 通过`__tablename__`属性明确指定数据库表名
- 使用`Column`类定义字段，明确指定数据类型和约束
- 利用`server_default`和`onupdate`参数处理数据库级默认值和更新逻辑

### 核心模型类映射
虽然`users`和`ai_models`表的ORM类未在当前代码中直接找到，但`performance_metrics`表的ORM映射在`decision_models.py`中有明确实现：

```python
class PerformanceMetrics(Base):
    """性能指标表（ORM模型）"""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
```

该映射类准确反映了数据库表结构，使用`Float`类型映射`DECIMAL`字段，`DateTime`类型映射`TIMESTAMP`字段，并通过`server_default=func.now()`确保时间戳由数据库生成。

### 数据库连接配置
系统通过`performance_config.py`中的`DATABASE_CONFIG`配置数据库连接池，关键参数包括：
- `pool_size`: 连接池大小，生产环境设置为50
- `max_overflow`: 最大溢出连接数，允许在高峰期创建额外连接
- `pool_recycle`: 连接回收时间，防止长时间空闲连接失效
- `pool_pre_ping`: 预检查连接，确保从池中获取的连接有效

这些配置优化了数据库连接管理，支持高并发场景下的性能需求。

**Section sources**
- [decision_models.py](file://decision-service/src/models/decision_models.py#L74-L78)
- [performance_config.py](file://backend/config/performance_config.py#L43-L51)

## 数据库管理指南

### 索引优化策略
为了确保查询性能，建议实施以下索引策略：

1. **主键索引**：所有表的`id`字段已自动创建主键索引，无需额外操作。

2. **唯一索引**：`users`表的`username`和`email`字段已通过`UNIQUE`约束创建唯一索引，确保数据完整性。

3. **复合索引建议**：
   - 在`ai_models`表上创建`(model_type, version)`复合索引，优化按类型和版本查询的性能
   - 在`performance_metrics`表上创建`(metric_name, timestamp)`复合索引，优化按指标名称和时间范围查询的性能

4. **部分索引**：对于`ai_models`表中`accuracy`字段，可考虑创建部分索引，仅索引非空值，提高特定查询效率。

### 查询性能调优
实施以下查询优化措施以提升系统性能：

1. **查询缓存**：利用Redis缓存频繁查询的结果，特别是`ai_models`表的模型列表查询。

2. **分页查询**：对于`performance_metrics`表的大数据量查询，实施分页机制，避免一次性加载过多数据。

3. **只读副本**：对于分析型查询，考虑使用数据库只读副本，减轻主库负载。

4. **查询计划分析**：定期使用`EXPLAIN`分析慢查询，识别性能瓶颈。

### 数据备份与恢复
实施全面的数据保护策略：

1. **定期备份**：配置每日全量备份和每小时增量备份，确保数据可恢复性。

2. **备份验证**：定期测试备份文件的恢复流程，确保备份有效性。

3. **灾难恢复**：在异地数据中心维护数据库副本，支持灾难恢复。

4. **备份加密**：对备份文件进行加密存储，保护数据安全。

5. **保留策略**：实施备份保留策略，平衡存储成本和恢复需求。

**Section sources**
- [performance_config.py](file://backend/config/performance_config.py#L43-L51)
- [database-deployment.yaml](file://infrastructure/kubernetes/database-deployment.yaml#L1-L56)

## 结论

本文档详细分析了AI决策系统的数据库Schema设计，涵盖了`users`、`ai_models`和`performance_metrics`三张核心表的结构、业务含义和使用场景。系统采用PostgreSQL作为数据库，通过合理的表结构设计支持用户管理、模型管理和性能监控三大核心功能。

ORM映射实现基于SQLAlchemy框架，通过清晰的Python类定义将数据库表映射为对象模型。数据库配置经过优化，支持高并发场景下的稳定运行。针对未来扩展，建议添加外键约束以明确表间关系，并实施索引优化、查询调优和完善的备份恢复策略，确保系统的可维护性和可靠性。