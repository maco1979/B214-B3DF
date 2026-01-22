@echo off
setlocal enabledelayedexpansion

:: 简单的VS Code配置脚本
echo 正在安装VS Code扩展...
code --install-extension MS-CEINTL.vscode-language-pack-zh-hans
code --install-extension ms-python.python

echo 正在设置VS Code配置...
code --wait --command workbench.action.openSettings

echo 脚本执行完成！请手动在VS Code设置中配置：
echo 1. Python默认解释器路径：C:\Python311\python.exe
echo 2. 显示语言：zh-cn
echo 
echo 按任意键关闭窗口...
pause
