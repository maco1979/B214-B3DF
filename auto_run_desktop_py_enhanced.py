#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版自动运行桌面Python文件工具
功能：
1. Python环境检测
2. 桌面Python文件自动搜索
3. 文件列表展示与选择
4. 选定文件运行
5. 运行日志记录
6. 脚本运行失败自动重试
7. 常用文件收藏功能
"""

import os
import subprocess
import sys
import json
import time
from datetime import datetime

# 配置文件路径
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".py_runner")
FAVORITES_FILE = os.path.join(CONFIG_DIR, "favorites.json")
LOG_FILE = os.path.join(CONFIG_DIR, "run_logs.txt")


class PythonRunner:
    """Python文件运行器类"""
    
    def __init__(self):
        """初始化配置"""
        # 确保配置目录存在
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)
        
        # 加载收藏的文件
        self.favorites = self._load_favorites()
    
    def _load_favorites(self):
        """加载收藏的文件列表"""
        if os.path.exists(FAVORITES_FILE):
            try:
                with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_favorites(self):
        """保存收藏的文件列表"""
        try:
            with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.favorites, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️ 警告：保存收藏文件失败：{e}")
    
    def _log_run(self, file_path, exit_code, duration):
        """记录运行日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] 文件：{file_path} | 退出码：{exit_code} | 耗时：{duration:.2f}秒\n"
        
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except IOError as e:
            print(f"⚠️ 警告：写入日志文件失败：{e}")
    
    def detect_python(self):
        """检测Python环境"""
        try:
            result = subprocess.run(["python", "--version"], capture_output=True, text=True, check=True)
            print(f"✅ 已检测到Python环境：")
            print(result.stdout.strip())
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ 错误：未检测到Python环境，请先安装Python并配置到系统环境变量！")
            return False
    
    def get_desktop_py_files(self):
        """获取桌面的Python文件列表"""
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        
        if not os.path.exists(desktop_path):
            print(f"❌ 错误：未找到桌面目录：{desktop_path}")
            return []
        
        # 获取桌面所有.py文件
        py_files = []
        for file in os.listdir(desktop_path):
            if file.endswith(".py"):
                full_path = os.path.join(desktop_path, file)
                py_files.append(full_path)
        
        return py_files
    
    def show_menu(self, py_files):
        """显示主菜单"""
        print("\n" + "=" * 50)
        print("自动运行桌面Python文件工具")
        print("=" * 50)
        print("1. 运行桌面Python文件")
        
        if self.favorites:
            print("2. 运行收藏的文件")
        
        print("3. 管理收藏的文件")
        print("4. 查看运行日志")
        print("5. 退出")
        
        return self.get_menu_choice(1, 5 if self.favorites else 4)
    
    def get_menu_choice(self, min_val, max_val):
        """获取有效的菜单选择"""
        while True:
            try:
                choice = input(f"\n请输入选项（{min_val}-{max_val}）：")
                choice_val = int(choice)
                
                if min_val <= choice_val <= max_val:
                    return choice_val
                else:
                    print(f"❌ 错误：输入的选项无效！请输入{min_val}-{max_val}之间的数字。")
            except ValueError:
                print("❌ 错误：请输入有效的数字！")
    
    def run_file(self, file_path):
        """运行指定的Python文件"""
        print(f"\n正在运行：{file_path}")
        print("-" * 50)
        
        # 获取重试设置
        retry_enabled = input("是否启用自动重试？(y/n，默认n)：").lower() == 'y'
        retry_count = 0
        
        if retry_enabled:
            while True:
                try:
                    retry_input = input("请输入重试次数（1-5，默认3）：")
                    retry_count = int(retry_input) if retry_input else 3
                    
                    if 1 <= retry_count <= 5:
                        break
                    else:
                        print("❌ 错误：请输入1-5之间的数字！")
                except ValueError:
                    print("❌ 错误：请输入有效的数字！")
        
        # 运行文件
        total_attempts = 1 + retry_count
        attempt = 0
        success = False
        
        while attempt < total_attempts and not success:
            attempt += 1
            
            if attempt > 1:
                print(f"\n🔄 第 {attempt} 次重试...")
                print("-" * 50)
            
            start_time = time.time()
            
            try:
                # 直接运行Python文件
                result = subprocess.run(["python", file_path], check=True)
                exit_code = 0
                success = True
            except subprocess.CalledProcessError as e:
                exit_code = e.returncode
                success = False
            except KeyboardInterrupt:
                print("\n⚠️ 脚本被用户中断！")
                exit_code = -1
                success = False
                break
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 记录日志
            self._log_run(file_path, exit_code, duration)
            
            if success:
                break
            elif attempt < total_attempts:
                print(f"❌ 运行失败（退出码：{exit_code}），{retry_count - attempt + 1} 次重试机会剩余")
                time.sleep(1)  # 等待1秒后重试
        
        print("-" * 50)
        
        if success:
            print("✅ 脚本运行成功！")
            
            # 询问是否添加到收藏
            if file_path not in self.favorites:
                add_fav = input("\n是否将此文件添加到收藏？(y/n，默认n)：").lower() == 'y'
                if add_fav:
                    self.favorites.append(file_path)
                    self._save_favorites()
                    print("✅ 文件已添加到收藏！")
        else:
            print(f"❌ 脚本运行失败，已重试 {retry_count} 次！")
    
    def run_desktop_files(self):
        """运行桌面Python文件"""
        py_files = self.get_desktop_py_files()
        
        if not py_files:
            print(f"⚠️ 警告：桌面未找到任何.py文件！")
            return
        
        # 列出文件
        print(f"\n桌面找到的Python文件：")
        for idx, file_path in enumerate(py_files, 1):
            file_name = os.path.basename(file_path)
            is_fav = "⭐ " if file_path in self.favorites else "   "
            print(f"{idx}. {is_fav}{file_name}")
        
        # 选择文件
        choice = self.get_menu_choice(1, len(py_files))
        selected_file = py_files[choice - 1]
        
        # 运行文件
        self.run_file(selected_file)
    
    def run_favorite_files(self):
        """运行收藏的Python文件"""
        if not self.favorites:
            print(f"⚠️ 警告：未找到任何收藏的文件！")
            return
        
        # 列出收藏文件
        print(f"\n收藏的Python文件：")
        for idx, file_path in enumerate(self.favorites, 1):
            file_name = os.path.basename(file_path)
            print(f"{idx}. ⭐ {file_name}")
        
        # 选择文件
        choice = self.get_menu_choice(1, len(self.favorites))
        selected_file = self.favorites[choice - 1]
        
        # 检查文件是否存在
        if not os.path.exists(selected_file):
            print(f"❌ 错误：收藏的文件已不存在：{selected_file}")
            # 询问是否从收藏中移除
            remove = input("是否从收藏中移除该文件？(y/n，默认y)：").lower() != 'n'
            if remove:
                self.favorites.pop(choice - 1)
                self._save_favorites()
                print("✅ 文件已从收藏中移除！")
            return
        
        # 运行文件
        self.run_file(selected_file)
    
    def manage_favorites(self):
        """管理收藏的文件"""
        if not self.favorites:
            print(f"⚠️ 警告：未找到任何收藏的文件！")
            return
        
        print(f"\n收藏的文件管理：")
        for idx, file_path in enumerate(self.favorites, 1):
            file_name = os.path.basename(file_path)
            exists = "✅ " if os.path.exists(file_path) else "❌ "
            print(f"{idx}. {exists}{file_name}")
            print(f"    路径：{file_path}")
        
        # 管理选项
        print("\n1. 移除收藏的文件")
        print("2. 清空所有收藏")
        print("3. 返回主菜单")
        
        choice = self.get_menu_choice(1, 3)
        
        if choice == 1:
            # 移除单个文件
            remove_choice = self.get_menu_choice(1, len(self.favorites))
            removed_file = self.favorites.pop(remove_choice - 1)
            self._save_favorites()
            print(f"✅ 文件 {os.path.basename(removed_file)} 已从收藏中移除！")
        elif choice == 2:
            # 清空所有收藏
            confirm = input("\n⚠️ 确定要清空所有收藏的文件吗？(y/n，默认n)：").lower() == 'y'
            if confirm:
                self.favorites.clear()
                self._save_favorites()
                print("✅ 已清空所有收藏的文件！")
    
    def view_logs(self):
        """查看运行日志"""
        if not os.path.exists(LOG_FILE):
            print(f"⚠️ 警告：暂无运行日志！")
            return
        
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = f.readlines()
            
            print("\n" + "=" * 50)
            print("运行日志")
            print("=" * 50)
            
            # 显示最新的20条日志
            recent_logs = logs[-20:]
            for log in recent_logs:
                print(log.strip())
            
            if len(logs) > 20:
                print(f"\n... 显示最近20条日志，共 {len(logs)} 条记录")
                
        except IOError as e:
            print(f"❌ 错误：读取日志文件失败：{e}")
    
    def main_loop(self):
        """主循环"""
        if not self.detect_python():
            return
        
        while True:
            menu_choice = self.show_menu(self.get_desktop_py_files())
            
            if menu_choice == 1:
                self.run_desktop_files()
            elif menu_choice == 2 and self.favorites:
                self.run_favorite_files()
            elif menu_choice == 3:
                self.manage_favorites()
            elif menu_choice == 4:
                self.view_logs()
            elif menu_choice == 5:
                print("\n👋 退出程序，再见！")
                break
            
            input("\n按回车键继续...")


def main():
    """主函数"""
    runner = PythonRunner()
    runner.main_loop()


if __name__ == "__main__":
    main()
