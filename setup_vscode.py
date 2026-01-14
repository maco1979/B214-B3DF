#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VS Code 自动配置脚本
"""

import os
import subprocess
import json
import sys


def main():
    print("=" * 40)
    print("VS Code 自动配置脚本")
    print("=" * 40)
    
    # 检查 VS Code 是否安装
    try:
        result = subprocess.run(["code", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ VS Code 已安装: {result.stdout.split()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: 未检测到 VS Code，请先安装 VS Code")
        input("按回车键退出...")
        sys.exit(1)
    
    # 安装扩展
    print("\n[1/3] 正在安装 VS Code 扩展...")
    extensions = [
        "MS-CEINTL.vscode-language-pack-zh-hans",  # 中文语言包
        "ms-python.python"  # Python 插件
    ]
    
    for ext in extensions:
        print(f"   正在安装: {ext}")
        try:
            subprocess.run(["code", "--install-extension", ext], capture_output=True, text=True, check=True)
            print(f"   ✅ 成功: {ext}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ 失败: {ext} ({e.stderr.strip()})")
    
    # 配置 VS Code 设置
    print("\n[2/3] 正在配置 VS Code 设置...")
    
    # 获取 settings.json 路径
    appdata = os.getenv("APPDATA")
    settings_path = os.path.join(appdata, "Code", "User", "settings.json")
    
    # 读取现有设置
    settings = {}
    if os.path.exists(settings_path):
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        print(f"   ✅ 已读取现有设置: {settings_path}")
    else:
        print(f"   ⚠️ 未找到现有设置，将创建新文件: {settings_path}")
    
    # 更新设置
    new_settings = {
        "python.defaultInterpreterPath": "C:\\Python311\\python.exe",
        "workbench.displayLanguage": "zh-cn"
    }
    
    settings.update(new_settings)
    
    # 保存设置
    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print("   ✅ 成功保存设置")
    except Exception as e:
        print(f"   ❌ 保存设置失败: {e}")
    
    # 重启 VS Code
    print("\n[3/3] 正在重启 VS Code...")
    try:
        # 关闭所有 VS Code 实例
        subprocess.run(["taskkill", "/f", "/im", "Code.exe"], capture_output=True, text=True)
        # 重新启动 VS Code
        vscode_path = os.path.join(os.getenv("LOCALAPPDATA"), "Programs", "Microsoft VS Code", "Code.exe")
        if os.path.exists(vscode_path):
            subprocess.Popen([vscode_path])
            print("   ✅ VS Code 已重启")
        else:
            print("   ⚠️ 未找到 VS Code 可执行文件，需手动重启")
    except Exception as e:
        print(f"   ❌ 重启 VS Code 失败: {e}")
    
    print("\n" + "=" * 40)
    print("配置完成！")
    print("=" * 40)
    input("按回车键退出...")


if __name__ == "__main__":
    main()
