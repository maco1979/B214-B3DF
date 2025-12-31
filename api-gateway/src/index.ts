import dotenv from 'dotenv';
import { ApiGatewayServer } from './server.js';

// 加载环境变量
dotenv.config();

// 获取配置
const PORT = parseInt(process.env['PORT'] || '8080');
const NODE_ENV = process.env['NODE_ENV'] || 'development';

console.log('🚀 启动API网关服务...');
console.log(`🌍 环境: ${NODE_ENV}`);
console.log(`🔌 端口: ${PORT}`);

// 创建并启动网关服务器
const gatewayServer = new ApiGatewayServer(PORT);

gatewayServer.start().catch(error => {
  console.error('❌ API网关启动失败:', error);
  process.exit(1);
});

// 优雅关闭处理
process.on('SIGINT', async () => {
  console.log('\n🛑 收到关闭信号，正在优雅关闭...');
  await gatewayServer.stop();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n🛑 收到终止信号，正在优雅关闭...');
  await gatewayServer.stop();
  process.exit(0);
});

// 未捕获异常处理
process.on('uncaughtException', (error) => {
  console.error('💥 未捕获异常:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, _promise) => {
  console.error('💥 未处理的Promise拒绝:', reason);
  process.exit(1);
});