# DT-MPC控制组件使用文档

## 1. 概述

DT-MPC（Dynamic Technology Model Predictive Control）是一个用于过程工业的多变量预测控制组件，具有自抗扰、鲁棒控制和自动测试功能。

### 1.1 核心功能

- **多变量预测控制**：协调控制MV（操纵变量）、CV（被控变量）和DV（扰动变量）
- **自抗扰功能**：实时估计和补偿外部扰动
- **鲁棒控制**：处理模型不确定性和系统变化
- **PRBS自动测试**：自动进行系统辨识和模型验证
- **数据预处理**：实时数据采集、滤波和异常检测

## 2. 安装

### 2.1 依赖要求

- Python 3.7+
- NumPy

### 2.2 安装步骤

```bash
# 克隆或下载代码
cd dt_mpc
# 安装依赖
pip install numpy
```

## 3. 快速开始

以下是一个简单的DT-MPC控制器使用示例：

```python
import numpy as np
from dt_mpc_core import DTMpcController

# 配置参数
controller_params = {
    'control_switch': True,
    'robust_control_switch': True,
    'prbs_amplitude': 15.0,
    'prbs_period': 3,
    'prbs_steps': 50
}

mv_params = {
    'operation_range': [-50, 50],
    'rate_limits': [-8, 8],
    'action_cycle': 1.0
}

cv_params = {
    'setpoint': 20.0,
    'safety_range': [-50, 50],
    'weights': 1.0
}

model_params = {
    'prediction_horizon': 20,
    'control_horizon': 10,
    'system_gain': 1.2,
    'time_delay': 2,
    'time_constant': 8
}

robust_params = {
    'robust_control_switch': True,
    'uncertainty_range': 0.2,
    'disturbance_gain': 0.1,
    'error_window': 15
}

data_config = {
    'sampling_rate': 1.0,
    'filter_enabled': True,
    'filter_window': 5,
    'anomaly_threshold': 3.0
}

# 初始化控制器
controller = DTMpcController(
    controller_params=controller_params,
    mv_params=mv_params,
    cv_params=cv_params,
    model_params=model_params,
    robust_params=robust_params,
    data_config=data_config
)

# 运行控制循环
for step in range(10):
    result = controller.step()
    print(f"步 {step+1} - 控制输出: {result['control_output']:.2f}")

# 启动PRBS自动测试
controller.start_auto_test(steps=20)
for step in range(20):
    result = controller.step()
    print(f"测试步 {step+1} - 控制输出: {result['control_output']:.2f}")

# 获取测试结果
test_results = controller.get_auto_test_results()
print(f"测试总步数: {test_results['total_steps']}")
```

## 4. 核心概念

### 4.1 变量定义

- **MV（Manipulated Variables）**：可操作变量，控制器直接调整的变量
- **CV（Controlled Variables）**：被控变量，需要保持在设定值附近的变量
- **DV（Disturbance Variables）**：扰动变量，影响系统但控制器无法直接调整的变量

### 4.2 FOPDT模型

系统使用一阶加纯滞后（FOPDT）模型进行预测：

```
G(s) = Kp * e^(-Td*s) / (1 + Tc*s)
```

其中：
- `Kp`：系统增益
- `Td`：纯滞后时间
- `Tc`：时间常数

### 4.3 MPC预测算法

1. **预测**：基于FOPDT模型预测未来输出
2. **优化**：通过网格搜索寻找最优控制序列
3. **执行**：只应用第一个控制量，在下一个采样周期重复

## 5. 配置参数

### 5.1 控制器参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| control_switch | bool | True | 控制开关 |
| startup_mode | str | 'cold' | 启动模式（'cold'或'warm'） |
| auto_test_switch | bool | False | 自动测试开关 |
| robust_control_switch | bool | True | 鲁棒控制开关 |
| prbs_amplitude | float | 20.0 | PRBS信号幅度 |
| prbs_period | int | 5 | PRBS信号周期 |
| prbs_steps | int | 100 | PRBS测试总步数 |
| prbs_seed | int | 42 | PRBS随机种子 |

### 5.2 MV参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| operation_range | list | [-50, 50] | MV操作范围 |
| rate_limits | list | [-10, 10] | MV变化率限制 |
| action_cycle | float | 1.0 | 控制周期（秒） |

### 5.3 CV参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| setpoint | float | 0.0 | CV设定值 |
| safety_range | list | [-50, 50] | CV安全范围 |
| weights | float | 1.0 | CV权重 |

### 5.4 模型参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| prediction_horizon | int | 20 | 预测时域 |
| control_horizon | int | 10 | 控制时域 |
| system_gain | float | 1.0 | 系统增益 |
| time_delay | int | 1 | 纯滞后时间 |
| time_constant | int | 5 | 时间常数 |

### 5.5 鲁棒参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| robust_control_switch | bool | True | 鲁棒控制开关 |
| uncertainty_range | float | 0.2 | 模型不确定性范围 |
| disturbance_gain | float | 0.1 | 扰动估计增益 |
| error_window | int | 10 | 误差窗口大小 |
| safety_margin | float | 0.1 | 安全余量 |

### 5.6 数据配置

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| sampling_rate | float | 1.0 | 采样频率（秒） |
| filter_enabled | bool | True | 滤波开关 |
| filter_window | int | 5 | 滤波窗口大小 |
| anomaly_threshold | float | 3.0 | 异常检测阈值（σ） |

## 6. API参考

### 6.1 DTMpcController类

#### 初始化

```python
def __init__(self, controller_params=None, mv_params=None, cv_params=None, model_params=None, robust_params=None, data_config=None)
```

#### 主要方法

- **step()**：执行单步控制计算
  - 返回：控制结果字典

- **start_auto_test(steps=None)**：启动PRBS自动测试
  - 参数：steps - 测试步数（可选）

- **stop_auto_test()**：停止自动测试

- **get_auto_test_results()**：获取自动测试结果
  - 返回：测试结果字典

- **set_setpoint(setpoint)**：设置CV设定值
  - 参数：setpoint - 新的设定值
  - 返回：是否成功

- **data_acquisition(data_source=None)**：数据采集
  - 参数：data_source - 数据源（可选）
  - 返回：处理后的数据字典

### 6.2 MPCore类

#### 主要方法

- **predict(current_state, mv_sequence)**：基于FOPDT模型预测未来输出
  - 参数：
    - current_state：当前状态
    - mv_sequence：控制序列
  - 返回：预测的输出序列

- **optimize(current_state, setpoint, mv_current, mv_limits, mv_rate_limits, cv_weights)**：优化控制序列
  - 参数：
    - current_state：当前状态
    - setpoint：设定值
    - mv_current：当前控制量
    - mv_limits：控制量范围
    - mv_rate_limits：控制量变化率限制
    - cv_weights：被控变量权重
  - 返回：最优控制增量

### 6.3 DTMpcRobustController类

#### 主要方法

- **estimate_disturbance(current_output, predicted_output)**：估计外部扰动
  - 参数：
    - current_output：当前输出
    - predicted_output：预测输出
  - 返回：估计的扰动值

- **robust_adjustment(predicted_output, setpoint, safety_range)**：鲁棒调整
  - 参数：
    - predicted_output：预测输出
    - setpoint：设定值
    - safety_range：安全范围
  - 返回：调整后的预测输出

- **model_correction(model_params)**：模型修正
  - 参数：model_params - 当前模型参数
  - 返回：修正后的模型参数

### 6.4 DTMpcDataProcessor类

#### 主要方法

- **acquire_data(data_source=None)**：采集原始数据
  - 参数：data_source - 数据源（可选）
  - 返回：原始数据字典

- **preprocess_data(raw_data)**：数据预处理
  - 参数：raw_data - 原始数据
  - 返回：处理后的数据

- **detect_anomaly(data, threshold=3.0)**：异常检测
  - 参数：
    - data - 数据数组
    - threshold - 检测阈值
  - 返回：是否异常

## 7. 高级功能

### 7.1 PRBS自动测试

PRBS（伪随机二进制序列）测试用于系统辨识：

```python
# 启动测试
controller.start_auto_test(steps=50)

# 运行测试循环
for step in range(50):
    result = controller.step()

# 获取测试结果
test_results = controller.get_auto_test_results()

# 分析结果
print(f"MV范围: [{test_results['mv_min']:.2f}, {test_results['mv_max']:.2f}]")
print(f"CV标准差: {test_results['cv_std']:.2f}")
```

### 7.2 自抗扰功能

控制器实时估计和补偿外部扰动：

```python
# 在step()方法中自动执行
result = controller.step()
disturbance = result['disturbance_estimate']
print(f"估计的扰动值: {disturbance:.2f}")
```

### 7.3 鲁棒控制

处理模型不确定性：

```python
# 调整鲁棒控制参数
robust_params = {
    'robust_control_switch': True,
    'uncertainty_range': 0.3,  # 增加不确定性范围
    'disturbance_gain': 0.15,   # 增加扰动估计增益
    'safety_margin': 0.2        # 增加安全余量
}

controller = DTMpcController(robust_params=robust_params, ...)
```

## 8. 典型应用场景

### 8.1 工业过程控制

- 化工反应器温度和压力控制
- 精馏塔多变量协调控制
- 锅炉燃烧优化

### 8.2 能源系统控制

- 风力发电场功率控制
- 太阳能电池板跟踪控制
- 电网频率稳定控制

### 8.3 机器人控制

- 工业机器人轨迹跟踪
- 移动机器人路径规划
- 机器人抓取系统

## 9. 故障排除

### 9.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 控制输出振荡 | 模型参数不准确 | 重新辨识模型或调整预测/控制时域 |
| 响应缓慢 | 时间常数设置过大 | 减小时间常数或增加控制时域 |
| PRBS信号不变 | PRBS周期设置过大 | 减小prbs_period参数 |
| 异常检测误报 | 阈值设置过小 | 增加anomaly_threshold参数 |

### 9.2 调试技巧

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.INFO)

# 检查内部状态
print(f"当前控制量: {controller.current_mv}")
print(f"当前状态: {controller.current_state}")
print(f"PRBS状态: {controller.prbs_lfsr}")
```

## 10. 性能优化

### 10.1 计算效率

- 减小预测时域和控制时域
- 使用更高效的优化算法
- 减少数据采样频率

### 10.2 稳定性提升

- 增加鲁棒控制的安全余量
- 减小控制量变化率限制
- 增加模型修正频率

## 11. 版本历史

### v1.0.0 (2025-12-25)

- 初始版本
- 实现基本MPC功能
- 添加自抗扰和鲁棒控制
- 实现PRBS自动测试
- 创建示例应用和文档

## 12. 联系方式

如有问题或建议，请联系开发团队。
