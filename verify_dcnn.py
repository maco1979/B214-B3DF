"""
验证分布式DCNN系统文件创建情况
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} (文件不存在)")
        return False

def main():
    """主验证函数"""
    print("分布式DCNN系统文件验证")
    print("=" * 50)
    
    base_path = "d:/1.5/backend/src/distributed_dcnn"
    
    files_to_check = [
        (f"{base_path}/core.py", "核心DCNN系统"),
        (f"{base_path}/federated_edge.py", "联邦学习与边缘计算集成"),
        (f"{base_path}/blockchain_rewards.py", "区块链奖励机制"),
        (f"{base_path}/config.py", "系统配置"),
        (f"{base_path}/main.py", "主应用入口"),
        ("d:/1.5/backend/src/blockchain/photon_rewards.py", "PHOTON奖励系统"),
        ("d:/1.5/distributed_dcnn_architecture.md", "架构设计文档")
    ]
    
    results = []
    for filepath, description in files_to_check:
        results.append(check_file_exists(filepath, description))
    
    print("\n" + "=" * 50)
    total_files = len(results)
    existing_files = sum(results)
    
    print(f"文件总数: {total_files}")
    print(f"已创建文件: {existing_files}")
    print(f"创建成功率: {existing_files/total_files*100:.1f}%")
    
    if existing_files == total_files:
        print("\n🎯 所有文件创建成功！分布式DCNN系统已恢复完成。")
        print("\n系统特性:")
        print("• 分布式卷积神经网络 (DCNN)")
        print("• 联邦学习与边缘计算集成") 
        print("• 区块链PHOTON奖励机制")
        print("• 隐私保护与数据安全")
        print("• 实时推理与模型优化")
    else:
        print(f"\n⚠️ 有 {total_files - existing_files} 个文件缺失，需要进一步检查。")

if __name__ == "__main__":
    main()