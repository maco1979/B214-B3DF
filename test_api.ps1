# PowerShell脚本测试模型管理API
$baseUri = "http://localhost:8003/api"

Write-Host "测试模型管理API..."

# 测试1: 获取模型列表
Write-Host "`n1. 获取模型列表..."
try {
    $response = Invoke-RestMethod -Uri "$baseUri/models" -Method Get
    Write-Host "状态: 成功"
    Write-Host "模型数量: $($response.Count)"
    if ($response.Count -gt 0) {
        Write-Host "第一个模型: $($response[0].name)"
    }
} catch {
    Write-Host "错误: $($_.Exception.Message)"
}

# 测试2: 创建模型
Write-Host "`n2. 创建测试模型..."
$modelData = @{
    name = "农业分类模型"
    description = "用于识别农业图像中的作物和病虫害"
    status = "ready"
    version = "1.0.0"
    model_type = "ai"
} | ConvertTo-Json -Compress

try {
    $response = Invoke-RestMethod -Uri "$baseUri/models/" -Method Post -ContentType "application/json" -Body $modelData
    Write-Host "创建模型成功"
    Write-Host "模型ID: $($response.id)"
    
    # 测试3: 启动模型
    if ($response.id) {
        Write-Host "`n3. 启动模型 (ID: $($response.id))..."
        try {
            $startResponse = Invoke-RestMethod -Uri "$baseUri/models/$($response.id)/start" -Method Post
            Write-Host "启动模型成功"
            Write-Host "结果: $($startResponse | ConvertTo-Json -Compress)"
        } catch {
            Write-Host "启动模型失败: $($_.Exception.Message)"
        }
        
        # 测试4: 暂停模型
        Write-Host "`n4. 暂停模型 (ID: $($response.id))..."
        try {
            $pauseResponse = Invoke-RestMethod -Uri "$baseUri/models/$($response.id)/pause" -Method Post
            Write-Host "暂停模型成功"
            Write-Host "结果: $($pauseResponse | ConvertTo-Json -Compress)"
        } catch {
            Write-Host "暂停模型失败: $($_.Exception.Message)"
        }
    }
} catch {
    Write-Host "创建模型失败: $($_.Exception.Message)"
}