import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Cpu, 
  Database, 
  TrendingUp, 
  Users, 
  Clock,
  Play,
  Pause,
  RefreshCw,
  Zap,
  Activity,
  Shield,
  ArrowUpRight
} from 'lucide-react';

import { useModelsQuery } from '@/hooks/useModelsQuery'
import { 
  useSystemMetricsQuery, 
  useBlockchainStatusQuery,
  useEdgeDevicesQuery,
} from '@/hooks/useSystemQueries'

import { DecisionAgent } from '@/components/DecisionAgent';
import { BentoCard } from '@/components/ui/BentoCard';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

import { Device, apiClient } from '@/services/api';

// 生成基于当前实时时间的图表数据
const generateRealtimeChartData = () => {
  const now = new Date();
  const data = [];
  
  // 生成过去7个时间点的数据（每个间隔约1小时）
  for (let i = 6; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000); // 每小时一个点
    const hours = time.getHours().toString().padStart(2, '0');
    const minutes = time.getMinutes().toString().padStart(2, '0');
    
    // 生成模拟的性能数据（基于时间的周期性变化 + 随机波动）
    const baseValue = 500 + Math.sin(time.getHours() / 24 * Math.PI * 2) * 200;
    const randomVariation = (Math.random() - 0.5) * 100;
    const value = Math.round(Math.max(100, Math.min(1000, baseValue + randomVariation)));
    
    data.push({
      name: `${hours}:${minutes}`,
      value: value
    });
  }
  
  return data;
};

export function Dashboard() {
  const { data: models } = useModelsQuery();
  const { data: metrics, refetch: refetchMetrics, isFetching: metricsLoading, error: metricsError } = useSystemMetricsQuery();
  const { data: blockchainStatus, error: blockchainError } = useBlockchainStatusQuery();
  const { data: edgeDevices, isFetching: devicesLoading, error: devicesError } = useEdgeDevicesQuery();
  
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [chartData, setChartData] = useState(generateRealtimeChartData());

  const [masterActive, setMasterActive] = useState(false);

  // 每分钟更新图表数据以保持时间同步
  useEffect(() => {
    const updateChartData = () => {
      setChartData(generateRealtimeChartData());
    };
    
    // 计算到下一分钟的毫秒数，确保在整分钟时更新
    const now = new Date();
    const msUntilNextMinute = (60 - now.getSeconds()) * 1000 - now.getMilliseconds();
    
    // 先等到下一个整分钟，然后每分钟更新一次
    const initialTimeout = setTimeout(() => {
      updateChartData();
      const interval = setInterval(updateChartData, 60 * 1000);
      return () => clearInterval(interval);
    }, msUntilNextMinute);
    
    return () => clearTimeout(initialTimeout);
  }, []);

  useEffect(() => {
    const checkMasterStatus = async () => {
      try {
        const res = await apiClient.get<{ master_control_active: boolean }>('/api/ai-control/master-control/status');
        setMasterActive(res.data?.master_control_active || false);
      } catch (e) {
        console.error("Failed to fetch master status", e);
        // 如果获取状态失败，默认设置为关闭状态
        setMasterActive(false);
      }
    };
    checkMasterStatus();
  }, []);

  const handleMasterToggle = async () => {
    try {
      const newStatus = !masterActive;
      const res = await apiClient.activateMasterControl(newStatus);
      if (res.success) {
        setMasterActive(newStatus);
      }
    } catch (error) {
      console.error("Failed to toggle master control", error);
    }
  };


  // 确保所有渲染值都是安全的原始类型
  const safeString = (value: any): string => {
    try {
      if (value === null || value === undefined) return '—';
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
      return '—';
    } catch (e) {
      return '—';
    }
  };

  const stats = [
    { label: '活跃模型', value: safeString(Array.isArray(models) ? models.length : undefined), icon: Brain, color: 'text-cyber-cyan' },
    { label: '神经吞吐量', value: safeString(metrics?.inference_requests), icon: Zap, color: 'text-yellow-400' },
    { label: '边缘节点', value: safeString(Array.isArray(edgeDevices) ? edgeDevices.length : undefined), icon: Activity, color: 'text-cyber-emerald' },
    { label: '区块链高度', value: safeString(
        typeof blockchainStatus === 'object' && blockchainStatus !== null && 
        typeof blockchainStatus.latest_block === 'object' && blockchainStatus.latest_block !== null && 
        typeof blockchainStatus.latest_block.block_number === 'number' ? 
        blockchainStatus.latest_block.block_number : undefined
      ), icon: Shield, color: 'text-cyber-purple' },
  ];


  if (metricsError || blockchainError || devicesError) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-3">
        <div className="text-red-400 text-lg">数据加载失败</div>
        <div className="text-gray-400 text-sm">{(metricsError as Error)?.message || (blockchainError as Error)?.message || (devicesError as Error)?.message}</div>
        <button 
          onClick={() => { refetchMetrics(); }}
          className="px-4 py-2 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30"
        >
          重试
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-8">

      {/* Welcome & Global Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-4xl font-black tracking-tighter text-white mb-2 uppercase">
            系统 <span className="text-cyber-cyan">概览</span>
          </h1>
          <p className="text-gray-500 font-medium">监控有机神经网络和自主边缘节点。</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button 
            onClick={() => refetchMetrics()}
            className="p-3 rounded-xl glass-morphism border border-white/5 hover:border-cyber-cyan/30 text-gray-400 hover:text-cyber-cyan transition-all"
          >
            <RefreshCw size={20} className={(isRefreshing || metricsLoading || devicesLoading) ? "animate-spin" : ""} />
          </button>

          
          <button 
            onClick={handleMasterToggle}
            className={`px-6 py-3 rounded-xl font-bold flex items-center space-x-2 transition-all ${
              masterActive 
                ? "bg-cyber-rose/10 text-cyber-rose border border-cyber-rose/20" 
                : "bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20 neon-glow-cyan"
            }`}
          >
            {masterActive ? <Pause size={18} /> : <Play size={18} />}
            <span>{masterActive ? "关闭AI核心" : "启动AI核心"}</span>
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <BentoCard className="h-32 flex flex-col justify-center">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">{stat.label}</p>
                  <h3 className="text-3xl font-black text-white">{stat.value}</h3>
                </div>
                <div className={`p-3 rounded-xl bg-white/5 border border-white/10 ${stat.color}`}>
                  <stat.icon size={24} />
                </div>
              </div>
            </BentoCard>
          </motion.div>
        ))}
      </div>

      {/* Main Bento Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 auto-rows-[450px]">
        {/* AI Agent - Main Terminal */}
        <BentoCard 
          className="lg:col-span-8 overflow-hidden p-0 border-none" 
          title="神经智能代理"
          description="自主决策支持"
          icon={Brain}
        >
          <div className="h-full w-full bg-cyber-black/40">
             <DecisionAgent />
          </div>
        </BentoCard>

        {/* System Health / Telemetry */}
        <BentoCard 
          className="lg:col-span-4"
          title="遥测数据"
          description="实时性能"
          icon={TrendingUp}
        >
          <div className="h-64 w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f2ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00f2ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis 
                  dataKey="name" 
                  stroke="#ffffff20" 
                  fontSize={10} 
                  tickLine={false} 
                  axisLine={false} 
                />
                <YAxis hide />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#09090b', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px'
                  }}
                  itemStyle={{ color: '#00f2ff' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#00f2ff" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorValue)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-6 space-y-4">
            <div className="flex justify-between items-center p-3 rounded-xl bg-white/5 border border-white/5">
              <span className="text-sm text-gray-400 font-medium">神经延迟</span>
              <span className="text-cyber-cyan font-bold">14ms</span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-xl bg-white/5 border border-white/5">
              <span className="text-sm text-gray-400 font-medium">内存饱和度</span>
              <span className="text-cyber-purple font-bold">42.8%</span>
            </div>
          </div>
        </BentoCard>

        {/* Recent Events */}
        <BentoCard 
          className="lg:col-span-4"
          title="系统日志"
          description="事件时间线"
          icon={Clock}
        >
          <div className="mt-4 space-y-4">
            {[1, 2, 3, 4].map((_, i) => (
              <div key={i} className="flex items-start space-x-4 p-3 rounded-xl hover:bg-white/5 transition-colors border border-transparent hover:border-white/5 group">
                <div className="w-1 h-8 rounded-full bg-cyber-cyan/40 mt-1" />
                <div className="flex-1">
                  <div className="flex justify-between">
                    <p className="text-sm font-bold text-white/90">模型推理成功</p>
                    <span className="text-[10px] text-gray-500">2分钟前</span>
                  </div>
                  <p className="text-xs text-gray-500 truncate">DCNN-V3在Edge_Node_04上执行</p>
                </div>
                <ArrowUpRight size={14} className="text-gray-600 group-hover:text-cyber-cyan" />
              </div>
            ))}
          </div>
        </BentoCard>

        {/* Network Topology / Devices */}
        <BentoCard 
          className="lg:col-span-8"
          title="边缘网络"
          description="全局设备矩阵"
          icon={Users}
        >
           <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              {(Array.isArray(edgeDevices) ? edgeDevices : []).slice(0, 8).map((device: Device) => {
                try {
                  return (
                    <div key={typeof device?.id === 'number' ? device.id : Math.random().toString(36).substr(2, 9)} className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-cyber-cyan/30 transition-all text-center group">
                      <div className="w-10 h-10 rounded-full bg-cyber-cyan/10 flex items-center justify-center mx-auto mb-3 group-hover:neon-glow-cyan transition-all">
                        <Cpu size={18} className="text-cyber-cyan" />
                      </div>
                      <p className="text-xs font-bold text-white truncate">{typeof device?.name === 'string' ? device.name : 'Unknown Device'}</p>
                      <p className="text-[10px] text-cyber-emerald uppercase tracking-tighter">已连接</p>
                    </div>
                  );
                } catch (e) {
                  return (
                    <div key={`error-${Math.random().toString(36).substr(2, 9)}`} className="p-4 rounded-xl bg-white/5 border border-white/5 text-center">
                      <p className="text-xs font-bold text-gray-500">Device Error</p>
                    </div>
                  );
                }
              })}
              {(!Array.isArray(edgeDevices) || edgeDevices.length === 0) && [1,2,3,4,5,6,7,8].map(i => (
                <div key={i} className="p-4 rounded-xl bg-white/5 border border-white/5 animate-pulse">
                  <div className="w-10 h-10 rounded-full bg-white/5 mx-auto mb-3" />
                  <div className="h-2 w-12 bg-white/5 mx-auto rounded mb-2" />
                  <div className="h-1.5 w-8 bg-white/5 mx-auto rounded" />
                </div>
              ))}
           </div>
        </BentoCard>
      </div>
    </div>
  );
}
