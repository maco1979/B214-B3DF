import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useParams } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { QueryClientProvider } from '@tanstack/react-query';

import { MainLayout as Layout } from './components/layout/MainLayout';
import ErrorBoundary from './components/ErrorBoundary';
import { AuthProvider } from './hooks/useAuth';
import { queryClient } from './lib/query-client';
import { useOnlineStatus } from './hooks/useOnlineStatus';

import './index.css';

// 网络状态监听组件（全局生效）
function NetworkStatusMonitor() {
  // 调用hook启用网络状态监听，网络变化时自动显示toast
  useOnlineStatus();
  return null;
}

// 旧路径重定向组件
function RedirectToModelDetail() {
  const params = useParams<{ id: string }>();
  return <Navigate to={`/models/${params.id}`} replace />;
}

const Dashboard = lazy(async () => import('./pages/Dashboard').then(module => ({ default: module.Dashboard })));
const AgriculturePage = lazy(async () => import('./pages/Agriculture'));
const ModelManagement = lazy(async () => import('./pages/ModelManagement').then(module => ({ default: module.ModelManagement })));
const ModelDetail = lazy(async () => import('./pages/ModelDetail').then(module => ({ default: module.ModelDetail })));
const InferenceService = lazy(async () => import('./pages/InferenceService').then(module => ({ default: module.InferenceService })));
const Blockchain = lazy(async () => import('./pages/Blockchain').then(module => ({ default: module.Blockchain })));
const FederatedLearning = lazy(async () => import('./pages/FederatedLearning').then(module => ({ default: module.FederatedLearning })));
const MonitoringDashboard = lazy(async () => import('./pages/MonitoringDashboard').then(module => ({ default: module.MonitoringDashboard })));
const PerformanceMonitoring = lazy(async () => import('./pages/PerformanceMonitoring').then(module => ({ default: module.PerformanceMonitoring })));
const Settings = lazy(async () => import('./pages/Settings').then(module => ({ default: module.Settings })));
const Community = lazy(async () => import('./pages/Community'));
const AIControl = lazy(async () => import('./pages/AIControl').then(module => ({ default: module.AIControl })));
const LoginPage = lazy(async () => import('./pages/Login'));

function App() {
  return (
    <Router>
      <AuthProvider>
        <QueryClientProvider client={queryClient}>
          <ErrorBoundary>
            <Suspense fallback={<div className="flex h-screen items-center justify-center text-white">加载中...</div>}>
              <Routes>
                {/* 登录页面 - 不需要Layout */}
                <Route path="/login" element={<LoginPage />} />

                {/* 受保护业务路由 */}
                <Route path="/" element={
                  <Layout>
                    <Dashboard />
                  </Layout>
                } />
                <Route path="/agriculture" element={
                  <Layout>
                    <AgriculturePage />
                  </Layout>
                } />
                <Route path="/models" element={
                  <Layout>
                    <ModelManagement />
                  </Layout>
                } />
                <Route path="/models/:id" element={
                  <Layout>
                    <ModelDetail />
                  </Layout>
                } />
                <Route path="/inference" element={
                  <Layout>
                    <InferenceService />
                  </Layout>
                } />
                <Route path="/blockchain" element={
                  <Layout>
                    <Blockchain />
                  </Layout>
                } />
                <Route path="/federated" element={
                  <Layout>
                    <FederatedLearning />
                  </Layout>
                } />
                <Route path="/monitoring" element={
                  <Layout>
                    <MonitoringDashboard />
                  </Layout>
                } />
                <Route path="/performance" element={
                  <Layout>
                    <PerformanceMonitoring />
                  </Layout>
                } />
                <Route path="/settings" element={
                  <Layout>
                    <Settings />
                  </Layout>
                } />
                <Route path="/community" element={
                  <Layout>
                    <Community />
                  </Layout>
                } />
                <Route path="/ai-control" element={
                  <Layout>
                    <AIControl />
                  </Layout>
                } />

              {/* 兼容旧路径（如 /model/:id） */}
              <Route path="/model/:id" element={<RedirectToModelDetail />} />

              {/* 重定向其他路由到首页 */}
              <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1A1A1A',
                  color: '#FFFFFF',
                  border: '1px solid #2D2D2D',
                },
              }}
            />
            {/* 全局网络状态监听 */}
            <NetworkStatusMonitor />
          </ErrorBoundary>
        </QueryClientProvider>
      </AuthProvider>
    </Router>
  );
}

export default App;
