# 蔬菜瓜果病虫害防御权威标准系统

## 概述

蔬菜瓜果病虫害防御权威标准系统是一个基于国标/行标的智能化标准管理系统，提供权威的病虫害防御标准数据、速查表生成、知识库管理和验证更新机制。系统确保所有数据都基于国家现行有效标准，支持农业决策的准确性和权威性。

## 核心功能

### 1. 权威标准数据管理
- **11项权威标准**：包括 GB/T 23416 系列、NY/T 4023 系列、GB 2763 等
- **10项病虫害预警阈值**：蚜虫、白粉虱、霜霉病、晚疫病等
- **3类预防措施**：农业防治、物理防治、种子处理
- **2类生物防治**：天敌释放、生物药剂
- **9项化学防治标准**：按叶菜类、茄果类、瓜类分类
- **31种禁用农药**：国家明令禁止的高毒农药清单

### 2. 速查表生成
- **通用速查表**：包含所有标准数据
- **分类速查表**：按作物类别（叶菜类、茄果类、瓜类等）生成
- **区域适配速查表**：按地区（华南、东北、华北、西南）生成适配标准

### 3. 知识库集成
- **自动导入**：将标准数据导入到农业文本学习知识库
- **智能搜索**：基于关键词和实体的多维度搜索
- **知识图谱**：构建标准之间的关联关系
- **数据导出**：支持 JSON 格式导出

### 4. 验证和更新机制
- **数据验证**：验证标准结构、阈值格式、化学防治数据等
- **更新检查**：定期检查标准更新（每90天）
- **版本管理**：建立版本号制度（V1.0、V1.1）
- **更新记录**：记录所有标准更新历史

## 技术架构

### 核心组件

1. **pest_disease_control_standards.py** ([pest_disease_control_standards.py](file:///d:/1.5/backend/src/data/pest_disease_control_standards.py))
   - 权威标准数据结构
   - 查询函数
   - 速查表格式化

2. **standards_data_importer.py** ([standards_data_importer.py](file:///d:/1.5/backend/src/data/standards_data_importer.py))
   - 标准数据导入器
   - 自动生成知识文本
   - 批量导入到知识库

3. **standards_validator.py** ([standards_validator.py](file:///d:/1.5/backend/src/data/standards_validator.py))
   - 标准验证器
   - 标准更新器
   - 数据管理器

4. **pest_disease_quick_reference.py** ([pest_disease_quick_reference.py](file:///d:/1.5/backend/src/api/routes/pest_disease_quick_reference.py))
   - RESTful API 接口
   - 速查表查询
   - 标准数据查询

## API 接口

### 1. 获取元数据

**GET** `/api/pest-disease-quick-reference/metadata`

获取速查表元数据

**响应示例：**
```json
{
  "success": true,
  "data": {
    "table_name": "蔬菜瓜果病虫害防御速查表（权威版）",
    "data_sources": ["GB/T 23416.1-2009", "GB 2763-2021"],
    "reviewed_by": "农业技术推广中心/农产品质量安全检测中心",
    "version": "V1.0",
    "update_date": "2026-01-03",
    "next_review_date": "2026-04-03",
    "warnings": [
      "⚠️ 本标准数据依据国家现行有效标准制定",
      "❌ 严禁使用国家明令禁止的33种高毒农药",
      "✅ 优先推荐生物防治、绿色防控技术"
    ]
  }
}
```

### 2. 生成速查表

**POST** `/api/pest-disease-quick-reference/quick-reference`

生成定制化速查表

**请求体：**
```json
{
  "crop_category": "茄果类",
  "region": "华南地区",
  "pest_disease": "晚疫病"
}
```

### 3. 查询预警阈值

**POST** `/api/pest-disease-quick-reference/threshold`

查询病虫害预警阈值

**请求体：**
```json
{
  "pest_disease_name": "蚜虫"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "pest_disease": "蚜虫",
    "threshold": "有翅蚜迁飞期，黄板日诱虫量≥20头/板",
    "level": "一般",
    "standard": "GB/T 23416.1-2009"
  }
}
```

### 4. 查询化学防治标准

**POST** `/api/pest-disease-quick-reference/chemical-control`

查询化学防治标准

**请求体：**
```json
{
  "crop_category": "茄果类",
  "pest_disease": "晚疫病"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "crop_category": "茄果类",
    "pest_disease": "晚疫病",
    "recommended_pesticides": ["氰霜唑", "烯酰吗啉"],
    "safety_interval": "5天",
    "max_dosage": "50g/亩",
    "standard": "GB 2763-2021",
    "application_method": "预防性施药，发病初期重点防治"
  }
}
```

### 5. 检查农药是否禁用

**POST** `/api/pest-disease-quick-reference/pesticide-check`

检查农药是否禁用

**请求体：**
```json
{
  "pesticide_name": "甲胺磷"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "pesticide_name": "甲胺磷",
    "is_banned": true,
    "warning": "⚠️ 该农药为国家明令禁止的高毒农药，严禁使用！",
    "total_banned_count": 31
  }
}
```

### 6. 获取禁用农药清单

**GET** `/api/pest-disease-quick-reference/banned-pesticides`

获取所有禁用农药清单

### 7. 获取区域适配标准

**GET** `/api/pest-disease-quick-reference/regional/{region}`

获取区域适配标准

**响应示例：**
```json
{
  "success": true,
  "data": {
    "region": "华南地区",
    "focus": "高温高湿条件下的病害防控",
    "additional_standards": {
      "霜霉病": "增加遮阳降温、排水防涝标准",
      "细菌性角斑病": "加强通风降湿，避免叶面长时间湿润"
    }
  }
}
```

### 8. 获取作物类别

**GET** `/api/pest-disease-quick-reference/crop-categories`

获取所有作物类别

### 9. 根据作物名称获取类别

**GET** `/api/pest-disease-quick-reference/crop/{crop_name}/category`

根据作物名称获取所属类别

### 10. 获取所有阈值

**GET** `/api/pest-disease-quick-reference/thresholds`

获取所有病虫害预警阈值

### 11. 获取施药技术规范

**GET** `/api/pest-disease-quick-reference/application-standards`

获取施药时间、方法和管理的标准规范

### 12. 获取效果评估标准

**GET** `/api/pest-disease-quick-reference/evaluation-standards`

获取病害防效、虫害防效和农药残留的评估标准

## 数据导入

### 导入所有标准数据

```bash
cd d:/1.5/backend
python src/data/standards_data_importer.py
```

**导入结果：**
- 权威标准：11 条
- 预警阈值：10 条
- 预防措施：3 条
- 化学防治：9 条
- 生物防治：2 条
- 禁用农药：1 条
- 施药规范：3 条
- 效果评估：3 条
- **总计：42 条**

### 导入到知识库

标准数据会自动导入到农业文本学习知识库，支持：
- 智能搜索
- 知识推荐
- 知识图谱查询

## 测试

运行测试脚本：

```bash
cd d:/1.5/backend
python tests/test_pest_disease_standards.py
```

**测试覆盖：**
1. 标准数据结构测试
2. 病虫害阈值查询测试
3. 化学防治标准查询测试
4. 农药禁用检查测试
5. 作物分类查询测试
6. 速查表生成测试
7. 标准数据验证测试
8. 标准数据导入测试
9. 知识搜索功能测试
10. 知识库导出功能测试

**测试结果：**
```
✓ 所有测试完成
总计导入: 42 条
处理时间: 0.01 秒
知识库统计:
  总知识数: 42
  类别分布: 8 个类别
  知识图谱大小: 23
```

## 权威标准来源

所有标准数据均基于以下权威来源：

| 标准编号 | 标准名称 | 适用范围 |
|---------|---------|---------|
| GB/T 23416.1-2009 | 蔬菜病虫害安全防治技术规范 第1部分：总则 | 全品类病虫害监测、预警、防治的通用原则 |
| GB/T 23416.2-2009 | 蔬菜病虫害安全防治技术规范 第2部分：茄果类 | 茄果类病虫害清单、防治阈值、技术规范 |
| GB/T 23416.3-2009 | 蔬菜病虫害安全防治技术规范 第3部分：瓜类 | 瓜类病虫害清单、防治阈值、技术规范 |
| GB/T 23416.4-2009 | 蔬菜病虫害安全防治技术规范 第4部分：叶菜类 | 叶菜类病虫害清单、防治阈值、技术规范 |
| NY/T 4023-2021 | 豇豆主要病虫害绿色防控技术规程 | 生物防治、理化诱控等绿色技术的操作规范 |
| NY/T 4024-2021 | 韭菜主要病虫害绿色防控技术规程 | 生物防治、理化诱控等绿色技术的操作规范 |
| NY/T 4025-2021 | 芹菜主要病虫害绿色防控技术规程 | 生物防治、理化诱控等绿色技术的操作规范 |
| GB 2763-2021 | 食品中农药最大残留限量 | 农药种类、用量、安全间隔期的强制要求 |
| GB 2763.1-2022 | 食品中农药最大残留限量 第1部分：植物源性食品 | 农药种类、用量、安全间隔期的强制要求 |
| GB/T 8321 系列 | 农药合理使用准则（共9部分） | 农药轮换、施用方法、抗性管理的技术规范 |

## 作物分类

系统支持以下作物类别：

- **叶菜类**：白菜、菠菜、生菜、油菜、芹菜、韭菜、青菜、甘蓝
- **茄果类**：番茄、辣椒、茄子、甜椒
- **瓜类**：黄瓜、西瓜、甜瓜、南瓜、冬瓜、丝瓜、苦瓜
- **豆类**：豇豆、四季豆、豌豆、蚕豆
- **根茎类**：萝卜、胡萝卜、土豆、红薯、山药、芋头

## 区域适配

系统支持以下区域的适配标准：

### 华南地区
- **重点**：高温高湿条件下的病害防控
- **附加标准**：
  - 霜霉病：增加遮阳降温、排水防涝标准
  - 细菌性角斑病：加强通风降湿，避免叶面长时间湿润
  - 疫病：雨季前预防性施药，雨后及时补喷

### 东北地区
- **重点**：大棚育苗期病害防控
- **附加标准**：
  - 低温寡照：补光、增温标准
  - 苗期病害：强化猝倒病、立枯病防控
  - 灰霉病：控制棚内湿度，避免结露

### 华北地区
- **重点**：露地蔬菜病虫害防控
- **附加标准**：
  - 蚜虫：春季重点防治，使用防虫网
  - 小菜蛾：成虫期使用性诱剂
  - 霜霉病：雨季前预防性施药

### 西南地区
- **重点**：山地蔬菜病虫害防控
- **附加标准**：
  - 土传病害：加强轮作，土壤消毒
  - 病毒病：防治传毒媒介（蚜虫、粉虱）
  - 根结线虫：使用抗病品种，土壤处理

## 验证和更新机制

### 数据验证

系统会对所有标准数据进行验证：

1. **标准结构验证**：
   - 检查必需字段（name、category、scope）
   - 检查标准代码格式
   - 检查类别有效性
   - 检查量化指标

2. **阈值数据验证**：
   - 检查必需字段（threshold、level、standard）
   - 检查阈值格式
   - 检查预警级别
   - 检查标准引用

3. **化学防治数据验证**：
   - 检查必需字段（recommended_pesticides、safety_interval、max_dosage、standard）
   - 检查安全间隔期格式
   - 检查最大用量格式
   - 检查推荐药剂列表

### 更新机制

- **更新检查频率**：每90天检查一次标准更新
- **版本管理**：建立版本号制度（V1.0、V1.1）
- **更新记录**：记录所有标准更新历史
- **动态更新**：跟进新品种、新药剂、新防控技术

## 使用示例

### Python 代码示例

```python
from src.data.pest_disease_control_standards import (
    get_all_standards,
    get_pest_disease_threshold,
    get_chemical_control_standards,
    is_pesticide_banned
)

# 获取所有标准
standards = get_all_standards()

# 查询病虫害阈值
threshold_info = get_pest_disease_threshold("蚜虫")
print(f"阈值: {threshold_info['threshold']}")
print(f"级别: {threshold_info['level']}")

# 查询化学防治标准
control_info = get_chemical_control_standards("茄果类", "晚疫病")
print(f"推荐药剂: {control_info['recommended_pesticides']}")
print(f"安全间隔期: {control_info['safety_interval']}")

# 检查农药是否禁用
is_banned = is_pesticide_banned("甲胺磷")
if is_banned:
    print("⚠️ 该农药为国家明令禁止的高毒农药，严禁使用！")
```

### API 调用示例

```bash
# 查询病虫害阈值
curl -X POST http://localhost:8003/api/pest-disease-quick-reference/threshold \
  -H "Content-Type: application/json" \
  -d '{"pest_disease_name": "蚜虫"}'

# 查询化学防治标准
curl -X POST http://localhost:8003/api/pest-disease-quick-reference/chemical-control \
  -H "Content-Type: application/json" \
  -d '{"crop_category": "茄果类", "pest_disease": "晚疫病"}'

# 检查农药是否禁用
curl -X POST http://localhost:8003/api/pest-disease-quick-reference/pesticide-check \
  -H "Content-Type: application/json" \
  -d '{"pesticide_name": "甲胺磷"}'

# 生成速查表
curl -X POST http://localhost:8003/api/pest-disease-quick-reference/quick-reference \
  -H "Content-Type: application/json" \
  -d '{"crop_category": "茄果类", "region": "华南地区"}'
```

## 文件结构

```
backend/
├── src/
│   ├── data/
│   │   ├── pest_disease_control_standards.py      # 权威标准数据
│   │   ├── standards_data_importer.py          # 标准数据导入器
│   │   └── standards_validator.py              # 标准验证和更新机制
│   └── api/
│       └── routes/
│           └── pest_disease_quick_reference.py    # API 路由
├── tests/
│   └── test_pest_disease_standards.py          # 测试脚本
└── data/
    └── standards_update_records.json             # 更新记录
```

## 性能特点

- **权威性**：所有数据基于国标/行标
- **准确性**：量化指标明确，可操作性强
- **时效性**：定期检查标准更新，确保数据最新
- **可追溯性**：每条数据都标注标准来源
- **可验证性**：提供验证机制，确保数据质量
- **可扩展性**：支持自定义标准和区域适配
- **智能化**：集成知识库，支持智能搜索和推荐

## 未来改进

- [ ] 集成更多权威标准（如地方标准、行业标准）
- [ ] 实现标准更新自动通知
- [ ] 添加标准版本对比功能
- [ ] 实现标准数据可视化
- [ ] 支持多语言标准
- [ ] 集成到农业决策引擎
- [ ] 实现标准推荐算法

## 许可证

本模块是 AI 农业决策系统的一部分，遵循项目的许可证协议。

## 联系方式

- **项目代码**: [GitHub](https://github.com/ai-agri-system)
- **技术文档**: [Docs](https://docs.ai-agri.com)
- **商务合作**: contact@ai-agri.com
