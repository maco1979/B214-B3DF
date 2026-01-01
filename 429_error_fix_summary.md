# 429 请求过于频繁错误修复总结

## 问题描述

前端应用在使用过程中频繁出现 **429 Too Many Requests** 错误，显示"请求过于频繁，请稍后重试"。

## 根本原因

1. **API网关速率限制过严**
   - 原配置：15分钟内每个IP最多1000个请求
   - 在开发环境中，多个页面同时轮询数据，容易触发限制

2. **前端轮询过于频繁**
   - ModelDetail页面：每2秒轮询一次训练状态
   - Dashboard页面：每1分钟更新一次图表数据
   - SystemLogs组件：每30秒刷新一次日志
   - 多个页面同时运行时，请求量迅速累积

## 解决方案

### 1. 调整API网关速率限制（开发环境优化）

**文件：** `api-gateway/src/server.ts`

**修改内容：**
```typescript
// 原配置
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15分钟
  max: 1000, // 限制每个IP最多1000个请求
  message: '请求过于频繁，请稍后再试'
});

// 新配置（开发环境）
const limiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1分钟
  max: 500, // 限制每个IP 1分钟最多500个请求
  message: '请求过于频繁，请稍后重试',
  standardHeaders: true, // 返回速率限制信息在 RateLimit-* headers
  legacyHeaders: false,
  skipSuccessfulRequests: false,
  skipFailedRequests: false,
});
```

**优势：**
- 更短的时间窗口（1分钟）让限制更灵活
- 添加标准headers，便于前端识别限流状态
- 仍保持合理的请求上限（500/分钟 = 8.3/秒）

### 2. 优化前端轮询间隔

#### ModelDetail.tsx - 训练状态轮询
**修改：** 2秒 → 5秒
```typescript
// 原代码：2秒轮询
const interval = setInterval(() => {
  checkTrainingStatus(task_id)
}, 2000)

// 优化后：5秒轮询
const interval = setInterval(() => {
  checkTrainingStatus(task_id)
}, 5000)
```
**影响：** 训练状态更新略有延迟，但不影响用户体验

#### Dashboard.tsx - 图表数据更新
**修改：** 1分钟 → 5分钟
```typescript
// 原代码：每1分钟更新
const interval = setInterval(updateChartData, 60 * 1000);

// 优化后：每5分钟更新
const interval = setInterval(updateChartData, 5 * 60 * 1000);
```
**影响：** Dashboard数据不需要实时性，5分钟更新足够

#### SystemLogs.tsx - 日志刷新
**修改：** 30秒 → 60秒
```typescript
// 原代码：每30秒刷新
const interval = setInterval(fetchLogs, 30000);

// 优化后：每60秒刷新
const interval = setInterval(fetchLogs, 60000);
```
**影响：** 日志查看延迟略微增加，但大幅减少请求量

## 效果对比

### 优化前（单用户单会话）
- ModelDetail轮询：30次/分钟
- Dashboard更新：1次/分钟
- SystemLogs刷新：2次/分钟
- **总计：约33次/分钟**

### 优化后（单用户单会话）
- ModelDetail轮询：12次/分钟
- Dashboard更新：0.2次/分钟
- SystemLogs刷新：1次/分钟
- **总计：约13次/分钟**

**请求量减少：约60%** ✅

## 部署步骤

1. **重新编译API网关：**
   ```bash
   cd api-gateway
   npm run build
   ```

2. **重新编译前端：**
   ```bash
   cd frontend
   npm run build
   ```

3. **重启服务：**
   - 后端：运行在8005端口（uvicorn --reload会自动重载）
   - 前端：刷新浏览器页面（Ctrl+F5强制刷新）
   - API网关：如果有运行，重启服务

4. **验证修复：**
   - 打开浏览器开发者工具（F12）→ Network标签
   - 查看请求频率和响应状态码
   - 应该看到：
     - 请求间隔增加
     - 无429错误
     - Response Headers包含 `RateLimit-Limit` 和 `RateLimit-Remaining`

## 生产环境建议

对于生产环境，建议：

1. **API网关速率限制：**
   ```typescript
   const limiter = rateLimit({
     windowMs: 15 * 60 * 1000, // 15分钟
     max: 1000, // 每个IP最多1000个请求
     message: '请求过于频繁，请稍后重试',
     standardHeaders: true,
     legacyHeaders: false,
     // 针对不同路径设置不同限制
     skip: (req) => {
       // 健康检查不限制
       return req.path.startsWith('/health');
     }
   });
   ```

2. **前端实现请求队列和重试机制：**
   - 检测429状态码
   - 读取 `Retry-After` header
   - 实现指数退避算法

3. **使用WebSocket替代轮询：**
   - 对于实时性要求高的数据（如训练状态）
   - 减少HTTP请求开销

4. **添加本地缓存：**
   - 使用React Query的缓存机制
   - 避免重复请求相同数据

## 监控建议

1. **API网关添加指标：**
   - 速率限制触发次数
   - 每个端点的请求频率
   - 按IP统计请求分布

2. **前端添加错误上报：**
   - 捕获429错误
   - 记录触发时间和页面
   - 分析用户行为模式

3. **设置告警：**
   - 429错误率 > 5%时告警
   - 单个IP请求量异常时告警

## 测试验证

运行以下测试确认修复：

```bash
# 1. 测试速率限制
# 使用PowerShell快速发送多个请求
for ($i=1; $i -le 10; $i++) {
  $body = '{"connect": true}'
  Invoke-RestMethod -Uri "http://127.0.0.1:8005/api/ai-control/device/1/connection" `
    -Method Post -Body $body -ContentType "application/json"
  Write-Host "Request $i completed"
}

# 2. 打开浏览器，同时访问以下页面：
# - Dashboard (/dashboard)
# - Model Management (/models)
# - AI Control (/ai-control)

# 3. 观察Network标签，确认：
# - 无429错误
# - 请求间隔符合预期
# - Response headers包含速率限制信息
```

## 参考文档

- [Express Rate Limit](https://github.com/express-rate-limit/express-rate-limit)
- [React Query - Polling](https://tanstack.com/query/latest/docs/framework/react/guides/window-focus-refetching)
- [MDN - 429 Too Many Requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429)

---

**修复时间：** 2025-12-31  
**修复状态：** ✅ 完成  
**影响范围：** API网关配置、前端轮询逻辑  
**回滚方案：** 恢复 `git diff` 中的原始代码
