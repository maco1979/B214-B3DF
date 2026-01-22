import time
import numpy as np
from dt_mpc_core import DTMpcController

# 创建示例应用
def dt_mpc_example():
    """
    DT-MPC控制器示例应用
    """
    print("=== DT-MPC控制组件示例应用 ===")
    
    # 配置参数
    controller_params = {
        'control_switch': True,
        'startup_mode': 'cold',
        'auto_test_switch': False,
        'robust_control_switch': True,
        'prbs_amplitude': 15.0,  # PRBS信号幅度
        'prbs_period': 3,         # PRBS信号周期
        'prbs_steps': 50          # 自动测试总步数
    }
    
    mv_params = {
        'operation_range': [-50, 50],  # MV操作范围
        'rate_limits': [-8, 8],        # MV变化率限制
        'action_cycle': 1.0            # 控制周期
    }
    
    cv_params = {
        'setpoint': 20.0,              # CV设定值
        'safety_range': [-50, 50],     # CV安全范围
        'weights': 1.0                 # CV权重
    }
    
    model_params = {
        'prediction_horizon': 20,      # 预测时域
        'control_horizon': 10,         # 控制时域
        'system_gain': 1.2,            # 系统增益
        'time_delay': 2,               # 时滞
        'time_constant': 8             # 时间常数
    }
    
    robust_params = {
        'robust_control_switch': True,
        'uncertainty_range': 0.2,      # 模型不确定性范围
        'disturbance_gain': 0.1,       # 扰动估计增益
        'error_window': 15             # 误差窗口大小
    }
    
    data_config = {
        'sampling_rate': 1.0,          # 采样频率
        'filter_enabled': True,        # 启用滤波
        'filter_window': 5,            # 滤波窗口大小
        'anomaly_threshold': 3.0       # 异常检测阈值
    }
    
    # 初始化DT-MPC控制器
    controller = DTMpcController(
        controller_params=controller_params,
        mv_params=mv_params,
        cv_params=cv_params,
        model_params=model_params,
        robust_params=robust_params,
        data_config=data_config
    )
    
    print("DT-MPC控制器初始化完成")
    
    # 运行控制循环
    print("\n=== 1. 正常控制模式测试 ===")
    print("运行10步控制循环...")
    
    for step in range(10):
        result = controller.step()
        print(f"步 {step+1}:")
        print(f"  控制输出: {result['control_output']:.2f}")
        print(f"  控制状态: {result['control_status']}")
        print(f"  当前CV: {result['cv_measured']:.2f}")
        print(f"  扰动估计: {result['disturbance_estimate']:.2f}")
        print(f"  预测CV第一个点: {result['cv_prediction'][0]:.2f}")
        time.sleep(0.5)  # 模拟控制周期
    
    # 启动自动测试
    print("\n=== 2. 自动测试模式测试 ===")
    print("启动PRBS自动测试...")
    controller.start_auto_test(steps=20)
    
    print(f"自动测试状态: {controller.control_status}")
    print(f"开始PRBS测试，共 {controller.prbs_steps} 步...")
    
    # 运行自动测试
    test_results = []
    for step in range(20):
        result = controller.step()
        test_results.append(result)
        print(f"测试步 {step+1}:")
        print(f"  控制输出: {result['control_output']:.2f}")
        print(f"  控制状态: {result['control_status']}")
        print(f"  当前CV: {result['cv_measured']:.2f}")
        time.sleep(0.3)  # 模拟控制周期
    
    # 获取自动测试结果
    print("\n=== 3. 自动测试结果分析 ===")
    auto_test_results = controller.get_auto_test_results()
    
    if auto_test_results:
        print(f"测试总步数: {auto_test_results['total_steps']}")
        print(f"MV统计:")
        print(f"  最小值: {auto_test_results['mv_min']:.2f}")
        print(f"  最大值: {auto_test_results['mv_max']:.2f}")
        print(f"  平均值: {auto_test_results['mv_mean']:.2f}")
        
        print(f"CV统计:")
        print(f"  最小值: {auto_test_results['cv_min']:.2f}")
        print(f"  最大值: {auto_test_results['cv_max']:.2f}")
        print(f"  平均值: {auto_test_results['cv_mean']:.2f}")
        print(f"  标准差: {auto_test_results['cv_std']:.2f}")
    
    # 测试设定值变化
    print("\n=== 4. 设定值变化测试 ===")
    print(f"当前设定值: {controller.cv_setpoint}")
    
    if controller.set_setpoint(30.0):
        print(f"成功将设定值更改为: {controller.cv_setpoint}")
    else:
        print("设定值更改失败")
    
    # 运行5步控制循环，观察设定值变化响应
    print("\n运行5步控制循环，观察设定值变化响应...")
    for step in range(5):
        result = controller.step()
        print(f"步 {step+1}:")
        print(f"  控制输出: {result['control_output']:.2f}")
        print(f"  控制状态: {result['control_status']}")
        print(f"  当前CV: {result['cv_measured']:.2f}")
        print(f"  预测CV第一个点: {result['cv_prediction'][0]:.2f}")
        time.sleep(0.5)  # 模拟控制周期
    
    print("\n=== 示例应用结束 ===")

# 主函数
if __name__ == "__main__":
    dt_mpc_example()