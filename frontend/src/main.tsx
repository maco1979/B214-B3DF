import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import './galaxy.css';

// 初始化全局日志过滤器，解决内存地址日志问题
import { initializeLogger } from './utils/logger';
initializeLogger();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
