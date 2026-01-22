#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动运行桌面Python文件工具
"""

import os
import subprocess
import sys


def main():
    """主函数"""
    print("=" * 50)
    print("自动运行桌面Python文件工具")
    print("=" * 50)
    
    # 1. 检测Python环境
    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ 已检测到Python环境：")
        print(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误：未检测到Python环境，请先安装Python并配置到系统环境变量！")
        input("按回车键退出...")
        sys.exit(1)
    
    # 2. 查找桌面的Python文件
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    if not os.path.exists(desktop_path):
        print(f"❌ 错误：未找到桌面目录：{desktop_path}")
        input("按回车键退出...")
        sys.exit(1)
    
    # 获取桌面所有.py文件
    py_files = []
    for file in os.listdir(desktop_path):
        if file.endswith(".py"):
            py_files.append(os.path.join(desktop_path, file))
    
    if not py_files:
        print(f"⚠️ 警告：桌面未找到任何.py文件！")
        input("按回车键退出...")
        sys.exit(1)
    
    # 3. 列出并选择要运行的文件
    print(f"\n桌面找到的Python文件：")
    for idx, file_path in enumerate(py_files, 1):
        file_name = os.path.basename(file_path)
        print(f"{idx}. {file_name}")
    
    # 4. 选择文件并运行
    while True:
        try:
            choice = input(f"\n请输入要运行的文件序号（1-{len(py_files)}）：")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(py_files):
                selected_file = py_files[choice_idx]
                break
            else:
                print(f"❌ 错误：输入的序号无效！请输入1-{len(py_files)}之间的数字。")
        except ValueError:
            print("❌ 错误：请输入有效的数字序号！")
    
    # 5. 运行选定的文件
    print(f"\n正在运行：{selected_file}")
    print("-" * 50)
    
    try:
        # 直接运行Python文件
        subprocess.run(["python", selected_file], check=True)
        print("-" * 50)
        print("✅ 脚本运行完成！")
    except subprocess.CalledProcessError:
        print("-" * 50)
        print("❌ 脚本运行出错！")
    except KeyboardInterrupt:
        print("-" * 50)
        print("⚠️ 脚本被用户中断！")
    
    input("\n按回车键退出...")


if __name__ == "__main__":
    main()
