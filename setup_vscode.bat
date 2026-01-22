@echo off
chcp 65001 >nul 2>&1

:: 检查VS Code是否已安装
where code >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到VS Code，请先安装VS Code
    pause
    exit /b 1
)

echo 正在安装VS Code中文语言包...
code --install-extension MS-CEINTL.vscode-language-pack-zh-hans

echo 正在安装Python插件...
code --install-extension ms-python.python

echo 正在设置中文显示...
code --wait --command workbench.action.configureDisplayLanguage "zh-cn"

echo 正在配置Python解释器...
set SETTINGS_FILE=%APPDATA%\Code\User\settings.json

:: 检查设置文件是否存在，如果存在则先备份
if exist "%SETTINGS_FILE%" (
    copy "%SETTINGS_FILE%" "%SETTINGS_FILE%.bak" >nul
    echo 已备份原有设置文件为：%SETTINGS_FILE%.bak
)

:: 使用PowerShell来安全地更新或创建settings.json
powershell -Command "
$settingsPath = '%SETTINGS_FILE%'
$settings = @{}
if (Test-Path $settingsPath) {
    $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
}
$settings | Add-Member -MemberType NoteProperty -Name 'python.defaultInterpreterPath' -Value 'C:\\Python311\\python.exe' -Force
$settings | Add-Member -MemberType NoteProperty -Name 'workbench.displayLanguage' -Value 'zh-cn' -Force
$settings | ConvertTo-Json -Depth 100 | Set-Content $settingsPath
"

echo 重启VS Code使配置生效...
taskkill /f /im Code.exe >nul 2>&1
start "" "%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"

echo 配置完成！
pause