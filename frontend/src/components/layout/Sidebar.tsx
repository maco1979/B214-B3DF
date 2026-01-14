import React from 'react';
import { motion } from 'framer-motion';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Cpu,
  MessageSquare,
  Settings,
  Activity,
  Database,
  Shield,
  ChevronLeft,
  ChevronRight,
  Brain,
  Sprout,
  Link2,
  BarChart3,
  Video,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { queryClient } from '@/lib/query-client';
import { http } from '@/lib/api-client';
import { fetchModels } from '@/services/modelService';


interface SidebarProps {
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

const menuItems = [
  { path: '/', icon: LayoutDashboard, label: '仪表盘' },
  { path: '/ai-control', icon: Shield, label: 'AI控制' },
  { path: '/agriculture', icon: Sprout, label: '农业AI' },
  { path: '/models', icon: Brain, label: '模型管理' },
  { path: '/inference', icon: Cpu, label: '推理服务' },
  { path: '/monitoring', icon: Activity, label: '系统监控' },
  { path: '/blockchain', icon: Link2, label: '区块链' },
  { path: '/federated', icon: Database, label: '联邦学习' },
  { path: '/community', icon: MessageSquare, label: '社区' },
  { path: '/performance', icon: BarChart3, label: '性能监控' },
  { path: '/settings', icon: Settings, label: '设置' },
];

const prefetchMap: Record<string, () => Promise<unknown>> = {
  '/': async () =>
    queryClient.prefetchQuery({
      queryKey: ['system-metrics'],
      queryFn: async () => {
        const res = await http.get('/api/system/metrics');
        if (!(res as any).success) {
 throw new Error((res as any).error || '获取系统指标失败');
}
        return (res as any).data;
      },
      staleTime: 30_000,
    }),
  '/models': async () => queryClient.prefetchQuery({ queryKey: ['models'], queryFn: fetchModels, staleTime: 60_000 }),
  '/monitoring': async () =>
    Promise.all([
      queryClient.prefetchQuery({
        queryKey: ['system-metrics'],
        queryFn: async () => {
          const res = await http.get('/api/system/metrics');
          if (!(res as any).success) {
 throw new Error((res as any).error || '获取系统指标失败');
}
          return (res as any).data;
        },
        staleTime: 30_000,
      }),
      queryClient.prefetchQuery({
        queryKey: ['edge-devices'],
        queryFn: async () => {
          const res = await http.get('/api/edge/devices');
          if (!(res as any).success) {
 throw new Error((res as any).error || '获取设备列表失败');
}
          return (res as any).data || [];
        },
        staleTime: 10_000,
      }),
    ]),
  '/performance': async () =>
    Promise.all([
      queryClient.prefetchQuery({
        queryKey: ['performance-summary'],
        queryFn: async () => {
          const res = await http.get('/api/performance/summary');
          if (!(res as any).success) {
 throw new Error((res as any).error || '获取性能概览失败');
}
          return (res as any).data;
        },
        staleTime: 30_000,
      }),
      queryClient.prefetchQuery({
        queryKey: ['performance-metrics'],
        queryFn: async () => {
          const res = await http.get('/api/performance/summary');
          if (!(res as any).success) {
 throw new Error((res as any).error || '获取性能指标失败');
}
          return (res as any).data;
        },
        staleTime: 20_000,
      }),
    ]),
  '/inference': async () =>
    queryClient.prefetchQuery({
      queryKey: ['inference-history'],
      queryFn: async () =>
        // 后端不存在该API，直接返回空数据
         [],
      staleTime: 30_000,
    }),
};


const prefetchByPath = (path: string) => {
  const task = prefetchMap[path];
  if (task) {
    task().catch(() => {});
  }
};

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  setCollapsed,
}) => (
    <motion.div

      initial={false}
      animate={{ width: collapsed ? 80 : 260 }}
      className={cn(
        'h-screen glass-morphism border-r border-white/5 flex flex-col z-50 sticky top-0 transition-all duration-300',
        collapsed ? 'items-center' : 'items-start',
      )}
    >
      {/* Logo Section */}
      <div className="h-20 flex items-center px-6 w-full mb-8 shrink-0">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyber-cyan to-cyber-purple flex items-center justify-center neon-glow-cyan shrink-0">
          <Brain className="text-white w-6 h-6" />
        </div>
        {!collapsed && (
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="ml-4 font-bold text-lg tracking-wider gradient-text uppercase whitespace-nowrap"
          >
            赛博有机体
          </motion.span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 w-full space-y-2 overflow-y-auto custom-scrollbar overflow-x-hidden">
        {menuItems.map(item => (
            <NavLink
              key={item.path}
              to={item.path}
              onMouseEnter={() => prefetchByPath(item.path)}
              className={({ isActive }) => cn(
                'w-full flex items-center rounded-xl p-3 transition-all duration-200 group relative',
                isActive ?
                  'bg-cyber-cyan/10 text-cyber-cyan' :
                  'text-gray-400 hover:bg-white/5 hover:text-white',
              )}
            >

              <item.icon className={cn('w-6 h-6 shrink-0', 'group-hover:neon-glow-cyan transition-shadow')} />
              {!collapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="ml-4 font-medium whitespace-nowrap"
                >
                  {item.label}
                </motion.span>
              )}
            </NavLink>
          ))}
      </nav>

      {/* Collapse Toggle */}
      <div className="p-4 w-full shrink-0">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full h-10 flex items-center justify-center rounded-xl bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </button>
      </div>
    </motion.div>
  );

