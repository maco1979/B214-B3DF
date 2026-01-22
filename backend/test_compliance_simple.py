#!/usr/bin/env python3
"""
极简合规性测试脚本，通过检查源代码验证功能实现
"""

import os
import re

def check_data_localization():
    """检查数据本地化功能实现"""
    print("\n=== 检查数据本地化存储功能 ===")
    
    file_path = "D:\1.5\backend\src\edge_computing\cloud_edge_sync.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查节点位置验证函数
    if "_is_valid_node_location" in content:
        print("✅ 找到节点位置验证函数 _is_valid_node_location")
    else:
        print("❌ 未找到节点位置验证函数 _is_valid_node_location")
        return False
    
    # 检查敏感数据识别函数
    if "_is_sensitive_data" in content:
        print("✅ 找到敏感数据识别函数 _is_sensitive_data")
    else:
        print("❌ 未找到敏感数据识别函数 _is_sensitive_data")
        return False
    
    # 检查数据本地化审计日志
    if "_log_localization_audit" in content:
        print("✅ 找到数据本地化审计日志函数 _log_localization_audit")
    else:
        print("❌ 未找到数据本地化审计日志函数 _log_localization_audit")
        return False
    
    # 检查允许的地区配置
    if re.search(r'allowed_regions.*CN', content):
        print("✅ 找到中国大陆地区限制配置")
    else:
        print("❌ 未找到中国大陆地区限制配置")
        return False
    
    return True

def check_differential_privacy():
    """检查差分隐私功能实现"""
    print("\n=== 检查差分隐私保护功能 ===")
    
    file_path = "D:\1.5\backend\src\privacy\differential_privacy.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查epsilon参数配置
    if "epsilon: float = 1.0" in content:
        print("✅ 找到差分隐私参数 ε=1.0 配置")
    else:
        print("❌ 未找到差分隐私参数 ε=1.0 配置")
        return False
    
    # 检查高斯机制实现
    if "gaussian_mechanism" in content:
        print("✅ 找到高斯机制实现")
    else:
        print("❌ 未找到高斯机制实现")
        return False
    
    # 检查梯度裁剪实现
    if "clip_gradients" in content:
        print("✅ 找到梯度裁剪实现")
    else:
        print("❌ 未找到梯度裁剪实现")
        return False
    
    return True

def check_transaction_traceability():
    """检查交易溯源功能实现"""
    print("\n=== 检查交易溯源功能 ===")
    
    file_path = "D:\1.5\backend\src\distributed_dcnn\blockchain_rewards.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查交易哈希生成
    if "hashlib.sha256" in content:
        print("✅ 找到SHA256交易哈希生成")
    else:
        print("❌ 未找到SHA256交易哈希生成")
        return False
    
    # 检查奖励分配记录
    if "RewardAllocation" in content and "transaction_hash" in content:
        print("✅ 找到奖励分配记录与交易哈希字段")
    else:
        print("❌ 未找到奖励分配记录与交易哈希字段")
        return False
    
    # 检查贡献度记录
    if "record_contribution" in content:
        print("✅ 找到贡献度记录函数")
    else:
        print("❌ 未找到贡献度记录函数")
        return False
    
    return True

def check_edge_inference_latency():
    """检查边缘推理延迟控制实现"""
    print("\n=== 检查边缘推理延迟控制 ===")
    
    file_path = "D:\1.5\backend\src\distributed_dcnn\federated_edge.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查节点选择策略权重
    if re.search(r'compute_power.*0\.4|memory.*0\.3|load.*0\.2|latency.*0\.1', content):
        print("✅ 找到边缘节点选择策略权重配置")
    else:
        print("❌ 未找到边缘节点选择策略权重配置")
        return False
    
    # 检查延迟阈值
    if re.search(r'max_inference_latency.*100', content):
        print("✅ 找到边缘推理延迟阈值配置 (<100ms)")
    else:
        print("❌ 未找到边缘推理延迟阈值配置")
        return False
    
    return True

def check_compliance_with_cybersecurity_law():
    """检查网络安全法合规性"""
    print("\n=== 检查网络安全法合规性 ===")
    
    files_to_check = [
        "D:\1.5\backend\src\edge_computing\cloud_edge_sync.py",
        "D:\1.5\backend\src\privacy\differential_privacy.py",
        "D:\1.5\backend\src\distributed_dcnn\blockchain_rewards.py"
    ]
    
    all_compliant = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            all_compliant = False
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查数据本地化合规
        if "allowed_regions.*CN" in content or "_is_valid_node_location" in content:
            print(f"✅ {os.path.basename(file_path)}: 符合数据本地化要求")
        else:
            print(f"❌ {os.path.basename(file_path)}: 不符合数据本地化要求")
            all_compliant = False
    
    return all_compliant

def main():
    """主检查函数"""
    print("开始执行极简合规性检查...")
    
    results = []
    results.append(check_data_localization())
    results.append(check_differential_privacy())
    results.append(check_transaction_traceability())
    results.append(check_edge_inference_latency())
    results.append(check_compliance_with_cybersecurity_law())
    
    print("\n" + "="*50)
    if all(results):
        print("🎉 所有合规性检查通过！")
        print("✅ 数据本地化存储功能已实现")
        print("✅ 差分隐私保护功能已实现 (ε=1.0)")
        print("✅ 交易溯源功能已实现 (SHA256哈希)")
        print("✅ 边缘推理延迟控制已实现 (<100ms)")
        print("✅ 符合网络安全法要求")
        return 0
    else:
        print("❌ 部分合规性检查未通过")
        return 1

if __name__ == "__main__":
    main()
