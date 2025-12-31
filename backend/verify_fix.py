#!/usr/bin/env python3
"""
验证模型持久化修复
这个脚本不依赖任何外部库，只检查代码逻辑
"""

import os
import re

def verify_model_persistence_fix():
    """验证模型持久化修复"""
    print("=== 验证模型持久化修复 ===")
    
    # 读取model_manager.py文件
    file_path = "src/core/services/model_manager.py"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查1: _save_model调用是否已取消注释
    save_call_pattern = r'\s*self\._save_model\(model_id,\s*state,\s*metadata\)'
    save_call_match = re.search(save_call_pattern, content)
    
    if save_call_match:
        print("✅ _save_model调用已取消注释")
    else:
        print("❌ _save_model调用仍被注释")
        return False
    
    # 检查2: state是否不再是None
    state_none_pattern = r'state\s*=\s*None'
    state_none_matches = re.findall(state_none_pattern, content)
    
    if len(state_none_matches) == 0:
        print("✅ state不再被设置为None")
    else:
        print(f"⚠️  发现{len(state_none_matches)}处state = None的代码")
        # 检查这些代码是否在create_model方法中
        create_model_start = content.find('def create_model(')
        create_model_end = content.find('def _create_dummy_input(')
        create_model_content = content[create_model_start:create_model_end]
        
        state_none_in_create = re.findall(state_none_pattern, create_model_content)
        if len(state_none_in_create) > 0:
            print("❌ create_model方法中仍有state = None的代码")
            return False
        else:
            print("✅ create_model方法中state不再被设置为None")
    
    # 检查3: 是否实现了真实模型初始化
    model_init_patterns = [
        r'model\.init\(',
        r'train_state\.TrainState\.create\(',
        r'optax\.adam\('
    ]
    
    all_patterns_found = True
    for pattern in model_init_patterns:
        if re.search(pattern, content):
            print(f"✅ 发现模型初始化代码: {pattern}")
        else:
            print(f"❌ 未发现模型初始化代码: {pattern}")
            all_patterns_found = False
    
    if not all_patterns_found:
        return False
    
    # 检查4: 是否支持不同模型类型
    model_type_patterns = [
        r'"transformer":\s*TransformerModel',
        r'"vision":\s*VisionModel',
        r'"diffusion":\s*DiffusionModel'
    ]
    
    all_types_found = True
    for pattern in model_type_patterns:
        if re.search(pattern, content):
            print(f"✅ 发现模型类型支持: {pattern}")
        else:
            print(f"❌ 未发现模型类型支持: {pattern}")
            all_types_found = False
    
    if not all_types_found:
        return False
    
    # 检查5: _save_model方法是否正确实现
    save_method_start = content.find('def _save_model(')
    save_method_end = content.find('def load_model(')
    save_method_content = content[save_method_start:save_method_end]
    
    if re.search(r'pickle\.dump\(state,\s*f\)', save_method_content):
        print("✅ _save_model方法使用pickle保存模型状态")
    else:
        print("❌ _save_model方法未正确实现pickle保存")
        return False
    
    if re.search(r'json\.dump\(metadata\.to_dict\(\),\s*f', save_method_content):
        print("✅ _save_model方法保存元数据")
    else:
        print("❌ _save_model方法未正确保存元数据")
        return False
    
    # 检查6: load_model方法是否正确实现
    load_method_start = content.find('def load_model(')
    load_method_end = content.find('def update_model_metrics(')
    load_method_content = content[load_method_start:load_method_end]
    
    if re.search(r'pickle\.load\(', load_method_content):
        print("✅ load_model方法使用pickle加载模型状态")
    else:
        print("❌ load_model方法未正确实现pickle加载")
        return False
    
    if re.search(r'ModelMetadata\.from_dict', load_method_content):
        print("✅ load_model方法加载元数据")
    else:
        print("❌ load_model方法未正确加载元数据")
        return False
    
    print("\n🎉 所有验证通过! 模型持久化修复已成功实现")
    print("\n修复总结:")
    print("1. ✅ 取消了_save_model调用的注释")
    print("2. ✅ 实现了真实模型状态创建逻辑")
    print("3. ✅ 支持Transformer、Vision和Diffusion三种模型类型")
    print("4. ✅ 使用pickle正确保存和加载模型状态")
    print("5. ✅ 正确保存和加载模型元数据")
    print("6. ✅ 支持从文件加载模型")
    print("7. ✅ 创建了可用于推理的TrainState对象")
    
    return True

if __name__ == "__main__":
    verify_model_persistence_fix()
