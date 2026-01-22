# 农业文本学习模块

## 概述

农业文本学习模块是一个智能化的农业知识管理系统，能够自动从农业相关文本中提取、学习和构建知识图谱。该模块支持多种文本来源，提供强大的知识搜索和推荐功能，帮助农业决策系统持续学习和优化。

## 核心功能

### 1. 文本学习
- **单个文本学习**：从单段农业文本中提取知识
- **批量文本学习**：高效处理大量农业文本
- **智能分类**：自动将知识分类到合适的类别（病虫害防治、施肥管理、灌溉技术等）
- **实体识别**：识别作物、病虫害、肥料、农药、技术等农业实体
- **关键词提取**：自动提取文本中的关键信息

### 2. 知识管理
- **知识存储**：结构化存储农业知识
- **知识搜索**：基于关键词和实体的智能搜索
- **知识推荐**：基于知识图谱的相关知识推荐
- **知识图谱**：构建农业实体之间的关联关系
- **类别管理**：按类别组织和管理知识

### 3. 文本收集
- **多来源收集**：支持文件、目录、URL、列表等多种来源
- **文本过滤**：自动过滤低质量和无关文本
- **文本去重**：智能去重，避免重复知识
- **文本预处理**：清理和标准化文本内容

### 4. 数据持久化
- **知识库导出**：将知识库导出为 JSON 格式
- **知识库导入**：从 JSON 文件导入知识库
- **统计信息**：提供详细的学习和使用统计

## 技术架构

### 核心组件

1. **AgricultureTextLearner** ([agriculture_text_learner.py](file:///d:/1.5/backend/src/core/services/agriculture_text_learner.py))
   - 文本学习器核心类
   - 负责知识提取、存储和搜索

2. **AgricultureTextCollector** ([agriculture_text_collector.py](file:///d:/1.5/backend/src/core/services/agriculture_text_collector.py))
   - 文本收集器
   - 支持从多种来源收集文本

3. **AgricultureTextPreprocessor** ([agriculture_text_collector.py](file:///d:/1.5/backend/src/core/services/agriculture_text_collector.py))
   - 文本预处理器
   - 清理和标准化文本

4. **API 路由** ([agriculture_text_learning.py](file:///d:/1.5/backend/src/api/routes/agriculture_text_learning.py))
   - RESTful API 接口
   - 提供完整的 API 服务

### 农业实体词典

系统内置了丰富的农业实体词典：

- **作物实体**：番茄、黄瓜、辣椒、茄子、西瓜、水稻、小麦等
- **害虫实体**：蚜虫、白粉虱、红蜘蛛、蓟马、斜纹夜蛾等
- **病害实体**：白粉病、霜霉病、灰霉病、疫病、病毒病等
- **肥料实体**：氮肥、磷肥、钾肥、复合肥、有机肥等
- **农药实体**：吡虫啉、噻虫嗪、阿维菌素、多菌灵等
- **技术实体**：滴灌、喷灌、水肥一体化、覆膜栽培等

## API 接口

### 1. 学习文本

**POST** `/api/agriculture-text-learning/learn`

从单个文本学习知识

**请求体：**
```json
{
  "text": "番茄种植过程中，要注意防治蚜虫和白粉虱。可以使用吡虫啉进行防治。",
  "source": "manual"
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "knowledge_id": "agri_know_abc123",
    "category": "病虫害防治",
    "title": "番茄种植过程中，要注意防治蚜虫和白粉虱...",
    "keywords": ["番茄种植过程中", "要注意防治蚜虫"],
    "entities": {
      "crops": ["番茄"],
      "pests": ["蚜虫", "白粉虱"],
      "pesticides": ["吡虫啉"]
    },
    "confidence": 0.9,
    "created_at": "2026-01-03T10:00:00"
  },
  "message": "文本学习成功"
}
```

### 2. 批量学习

**POST** `/api/agriculture-text-learning/learn/batch`

批量学习文本

**请求体：**
```json
{
  "texts": [
    "番茄种植技术要点：选择优质种子，适时播种。",
    "黄瓜施肥管理：基肥为主，追肥为辅。"
  ],
  "source": "batch"
}
```

### 3. 搜索知识

**POST** `/api/agriculture-text-learning/search`

搜索农业知识

**请求体：**
```json
{
  "query": "番茄 病虫害防治",
  "category": null,
  "limit": 10
}
```

### 4. 获取知识详情

**GET** `/api/agriculture-text-learning/knowledge/{knowledge_id}`

获取指定知识的详细信息

### 5. 按类别获取知识

**GET** `/api/agriculture-text-learning/knowledge/category/{category}?limit=20`

按类别获取知识列表

### 6. 获取统计信息

**GET** `/api/agriculture-text-learning/statistics`

获取学习统计信息

### 7. 导出知识库

**POST** `/api/agriculture-text-learning/export`

导出知识库为 JSON 文件

### 8. 导入知识库

**POST** `/api/agriculture-text-learning/import`

从 JSON 文件导入知识库

### 9. 获取知识图谱

**GET** `/api/agriculture-text-learning/knowledge-graph?entity=白粉病`

获取知识图谱或指定实体的相关实体

## 使用示例

### Python 代码示例

```python
from src.core.services.agriculture_text_learner import AgricultureTextLearner

# 创建学习器
learner = AgricultureTextLearner()

# 学习文本
text = "番茄种植过程中，要注意防治蚜虫和白粉虱。可以使用吡虫啉进行防治。"
knowledge = learner.learn_from_text(text, "示例")

# 添加到知识库
learner.add_knowledge(knowledge)

# 搜索知识
results = learner.search_knowledge("番茄 病虫害", limit=5)
for knowledge in results:
    print(f"{knowledge.title} (置信度: {knowledge.confidence})")

# 获取统计信息
stats = learner.get_statistics()
print(f"总知识数: {stats['total_knowledge']}")
```

### API 调用示例

```bash
# 学习文本
curl -X POST http://localhost:8003/api/agriculture-text-learning/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "番茄种植技术要点：选择优质种子，适时播种。", "source": "api"}'

# 搜索知识
curl -X POST http://localhost:8003/api/agriculture-text-learning/search \
  -H "Content-Type: application/json" \
  -d '{"query": "番茄 病虫害", "limit": 10}'

# 获取统计信息
curl http://localhost:8003/api/agriculture-text-learning/statistics
```

## 测试

运行测试脚本：

```bash
cd d:/1.5/backend
python tests/test_agriculture_text_learning.py
```

运行使用示例：

```bash
cd d:/1.5/backend
python examples/agriculture_text_learning_examples.py
```

## 文件结构

```
backend/
├── src/
│   ├── core/
│   │   └── services/
│   │       ├── agriculture_text_learner.py      # 文本学习器
│   │       └── agriculture_text_collector.py    # 文本收集器
│   └── api/
│       └── routes/
│           └── agriculture_text_learning.py      # API 路由
├── tests/
│   └── test_agriculture_text_learning.py        # 测试脚本
├── examples/
│   └── agriculture_text_learning_examples.py     # 使用示例
└── data/
    └── agriculture_knowledge_*.json             # 知识库导出文件
```

## 知识类别

系统支持以下知识类别：

- **病虫害防治**：病虫害的防治方法和药物使用
- **病虫害识别**：病虫害的识别和诊断
- **施肥管理**：肥料的选择、配比和使用方法
- **灌溉技术**：灌溉系统和技术
- **种植技术**：作物种植和管理技术
- **作物管理**：作物的日常管理
- **综合管理**：综合性的农业管理知识

## 知识图谱

知识图谱构建了农业实体之间的关联关系，例如：

- **白粉病** → 多菌灵、代森锰锌、番茄、黄瓜
- **番茄** → 白粉病、蚜虫、氮肥
- **滴灌** → 水肥一体化、草莓、黄瓜

这些关联关系可以帮助系统推荐相关知识，提供更全面的农业决策支持。

## 性能特点

- **高效学习**：支持批量处理大量文本
- **智能分类**：自动识别知识类别
- **精准搜索**：基于关键词和实体的多维度搜索
- **知识关联**：通过知识图谱实现相关知识推荐
- **可扩展性**：支持自定义实体词典和分类规则

## 未来改进

- [ ] 集成更先进的 NLP 模型（如 BERT、RoBERTa）
- [ ] 支持多语言文本学习
- [ ] 增加知识验证和更新机制
- [ ] 实现知识图谱可视化
- [ ] 支持知识版本管理
- [ ] 集成到 AI 核心决策系统

## 许可证

本模块是 AI 农业决策系统的一部分，遵循项目的许可证协议。
