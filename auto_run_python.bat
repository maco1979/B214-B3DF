@echo off 
chcp 65001 > nul 2>&1  :: 设置编码为UTF-8，避免中文乱码 
echo ============== 自动运行桌面Python文件 ============== 

:: 1. 检测Python环境 
python --version > nul 2>&1 
if errorlevel 1 ( 
    echo 错误：未检测到Python环境，请先安装Python并配置到系统环境变量！ 
    pause 
    exit /b 1 
) 
echo ✅ 已检测到Python环境： 
python --version 

:: 2. 查找桌面的Python文件（.py） 
set "desktop_path=%USERPROFILE%\Desktop" 
dir /b "%desktop_path%\*.py" > nul 2>&1 
if errorlevel 1 ( 
    echo 警告：桌面未找到任何.py文件！ 
    pause 
    exit /b 1 
) 

:: 3. 列出并选择要运行的文件 
echo. 
echo 桌面找到的Python文件： 
setlocal enabledelayedexpansion 
set "file_list=" 
set "count=0" 
for %%f in ("%desktop_path%\*.py") do ( 
    set /a count+=1 
    set "file_list[!count!]=%%f" 
    echo !count!. %%~nxf 
) 

:: 4. 选择文件并运行 
set /p "choice=请输入要运行的文件序号（1-!count!）：" 
if not defined file_list[%choice%] ( 
    echo 错误：输入的序号无效！ 
    pause 
    exit /b 1 
) 
echo. 
echo 正在运行：!file_list[%choice%]! 
echo -------------------------------------------------- 
python "!file_list[%choice%]!" 
echo -------------------------------------------------- 
echo 脚本运行完成！ 
pause