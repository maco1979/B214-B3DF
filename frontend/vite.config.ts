import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { fileURLToPath } from 'url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
    port: 3000,
    strictPort: false,  // 允许端口被占用时自动递增
    proxy: {
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        secure: false,
        ws: true, // 兼容WebSocket，无副作用
        // 代理异常日志（报错时快速定位问题）
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.error('\x1B[31m【代理报错】\x1B[0m 连接后端失败：', err.message);
          });
        },
        // 保留 /api 前缀，适配后端路由规范 ✔️
      }
    }
  },
  optimizeDeps: {
    force: true // 强制每次启动都重新预构建依赖，彻底避免缓存问题
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2015',  // 提高浏览器兼容性
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
          query: ['@tanstack/react-query'],
        }
      }
    }
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
})