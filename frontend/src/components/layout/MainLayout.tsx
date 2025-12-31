import React, { useState, useContext } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, Navigate } from 'react-router-dom';
import { AuthContext } from '@/hooks/useAuth';

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ 
  children 
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  
  // 使用useContext直接检查AuthContext是否存在
  const authContext = useContext(AuthContext);
  
  // 如果AuthContext不可用，显示加载状态
  if (!authContext) {
    return (
      <div className="flex h-screen items-center justify-center text-white">
        <div className="text-xl">加载中...</div>
      </div>
    );
  }
  
  const { isAuthenticated } = authContext;
  
  // 认证检查
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="min-h-screen bg-cyber-black text-white flex overflow-hidden">
      {/* Background blobs */}
      <div className="organic-blob top-[-10%] left-[-10%] opacity-40" />
      <div className="organic-blob bottom-[-10%] right-[-10%] opacity-20 bg-cyber-purple" style={{ background: 'radial-gradient(circle, rgba(188, 19, 254, 0.08) 0%, transparent 70%)' }} />
      <div className="cyber-grid fixed inset-0 z-0 pointer-events-none" />

      <Sidebar 
        collapsed={collapsed} 
        setCollapsed={setCollapsed} 
      />

      <div className="flex-1 flex flex-col relative z-10 overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto custom-scrollbar p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="h-full"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
};

