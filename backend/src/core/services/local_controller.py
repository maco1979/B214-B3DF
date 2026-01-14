"""本地控制模块
实现电脑本地功能控制，包括文件管理、摄像头调用、系统命令执行等
"""

import os
import subprocess
import cv2
import shutil
from typing import Optional


class LocalController:
    """本地控制器类，提供本地功能控制接口"""
    
    @staticmethod
    def open_file(file_path: str) -> str:
        """打开指定文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            操作结果信息
        """
        if os.path.exists(file_path):
            try:
                # 根据操作系统选择打开方式
                if os.name == 'nt':  # Windows
                    subprocess.run(["start", file_path], shell=True, check=True)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(["open" if os.uname().sysname == 'Darwin' else "xdg-open", file_path], shell=True, check=True)
                return f"已为你打开文件: {file_path}"
            except subprocess.CalledProcessError as e:
                return f"打开文件失败: {str(e)}"
        else:
            return "文件不存在"
    
    @staticmethod
    def take_photo(save_path: str = "photo.jpg") -> str:
        """调用摄像头拍照
        
        Args:
            save_path: 照片保存路径
            
        Returns:
            操作结果信息
        """
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "摄像头调用失败，无法打开摄像头"
            
            # 拍摄照片
            ret, frame = cap.read()
            if ret:
                # 保存照片
                cv2.imwrite(save_path, frame)
                cap.release()
                return f"拍照成功，保存路径: {os.path.abspath(save_path)}"
            else:
                cap.release()
                return "摄像头调用失败，无法获取图像"
        except Exception as e:
            return f"拍照失败: {str(e)}"
    
    @staticmethod
    def run_system_cmd(cmd: str) -> str:
        """执行系统命令
        
        Args:
            cmd: 要执行的系统命令
            
        Returns:
            命令执行结果
        """
        try:
            # 限制命令范围，避免危险操作
            dangerous_commands = ["rm", "del", "format", "shutdown", "restart", "regedit", "chmod", "chown"]
            for dangerous_cmd in dangerous_commands:
                if dangerous_cmd in cmd.lower():  # 检查是否包含危险命令
                    return f"禁止执行危险命令: {dangerous_cmd}"
            
            # 执行命令
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30  # 设置超时，防止命令挂起
            )
            
            if result.returncode == 0:
                return f"命令执行成功:\n{result.stdout}"
            else:
                return f"执行失败:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "命令执行超时"
        except Exception as e:
            return f"命令错误: {str(e)}"
    
    @staticmethod
    def list_files(directory: str = ".") -> str:
        """列出目录中的文件
        
        Args:
            directory: 目录路径
            
        Returns:
            文件列表信息
        """
        if os.path.exists(directory) and os.path.isdir(directory):
            try:
                files = os.listdir(directory)
                file_list = "\n".join(f"- {f}" for f in files)
                return f"目录 {directory} 中的文件:\n{file_list}"
            except Exception as e:
                return f"列出文件失败: {str(e)}"
        else:
            return "目录不存在"
    
    @staticmethod
    def get_system_info() -> str:
        """获取系统信息
        
        Returns:
            系统信息
        """
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(["systeminfo"], shell=True, capture_output=True, text=True, check=True)
                # 提取关键信息
                info = result.stdout
                # 只返回前20行，避免信息过长
                return "系统信息:\n" + "\n".join(info.split("\n")[:20])
            else:  # macOS/Linux
                result = subprocess.run(["uname", "-a"], shell=True, capture_output=True, text=True, check=True)
                return f"系统信息:\n{result.stdout}"
        except Exception as e:
            return f"获取系统信息失败: {str(e)}"
    
    @staticmethod
    def open_url(url: str) -> str:
        """打开指定URL
        
        Args:
            url: 要打开的URL
            
        Returns:
            操作结果信息
        """
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(["start", url], shell=True, check=True)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(["open" if os.uname().sysname == 'Darwin' else "xdg-open", url], shell=True, check=True)
            return f"已为你打开URL: {url}"
        except subprocess.CalledProcessError as e:
            return f"打开URL失败: {str(e)}"
        except Exception as e:
            return f"打开URL错误: {str(e)}"
    
    @staticmethod
    def screenshot(save_path: str = "screenshot.jpg") -> str:
        """截取屏幕截图
        
        Args:
            save_path: 截图保存路径
            
        Returns:
            操作结果信息
        """
        try:
            if os.name == 'nt':  # Windows
                # 使用Windows内置命令截图
                subprocess.run(["powershell", f"Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; $bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height); $graphics = [System.Drawing.Graphics]::FromImage($bitmap); $graphics.CopyFromScreen(0, 0, 0, 0, $bitmap.Size); $bitmap.Save('{save_path}', [System.Drawing.Imaging.ImageFormat]::Jpeg); $graphics.Dispose(); $bitmap.Dispose()"], shell=True, check=True)
            elif os.name == 'posix':  # macOS/Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.run(["screencapture", save_path], shell=True, check=True)
                else:  # Linux
                    subprocess.run(["scrot", save_path], shell=True, check=True)
            return f"截图成功，保存路径: {os.path.abspath(save_path)}"
        except subprocess.CalledProcessError as e:
            return f"截图失败: {str(e)}"
        except Exception as e:
            return f"截图错误: {str(e)}"
    
    @staticmethod
    def create_file(file_path: str, content: str = "") -> str:
        """创建文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            操作结果信息
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"文件创建成功，路径: {file_path}"
        except Exception as e:
            return f"创建文件失败: {str(e)}"
    
    @staticmethod
    def copy_file(source: str, destination: str) -> str:
        """复制文件
        
        Args:
            source: 源文件路径
            destination: 目标文件路径
            
        Returns:
            操作结果信息
        """
        if os.path.exists(source):
            try:
                shutil.copy2(source, destination)
                return f"文件复制成功，从 {source} 到 {destination}"
            except Exception as e:
                return f"复制文件失败: {str(e)}"
        else:
            return "源文件不存在"
    
    @staticmethod
    def delete_file(file_path: str) -> str:
        """删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            操作结果信息
        """
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return f"文件删除成功: {file_path}"
            except Exception as e:
                return f"删除文件失败: {str(e)}"
        else:
            return "文件不存在"
    
    @staticmethod
    def start_application(app_name: str) -> str:
        """启动应用程序
        
        Args:
            app_name: 应用程序名称或路径
            
        Returns:
            操作结果信息
        """
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(["start", "", app_name], shell=True, check=True)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(["open" if os.uname().sysname == 'Darwin' else "xdg-open", app_name], shell=True, check=True)
            return f"已启动应用程序: {app_name}"
        except subprocess.CalledProcessError as e:
            return f"启动应用程序失败: {str(e)}"
        except Exception as e:
            return f"启动应用程序错误: {str(e)}"
    
    @staticmethod
    def get_process_list() -> str:
        """获取进程列表
        
        Returns:
            进程列表信息
        """
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(["tasklist"], shell=True, capture_output=True, text=True, check=True)
            else:  # macOS/Linux
                result = subprocess.run(["ps", "aux"], shell=True, capture_output=True, text=True, check=True)
            # 只返回前30行，避免信息过长
            return "进程列表:\n" + "\n".join(result.stdout.split("\n")[:30])
        except Exception as e:
            return f"获取进程列表失败: {str(e)}"
