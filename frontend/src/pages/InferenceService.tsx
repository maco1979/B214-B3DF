import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Cpu, 
  Play, 
  Pause, 
  RefreshCw,
  TrendingUp,
  Clock,
  Server,
  Activity,
  ArrowUpRight,
  Shield,
  Zap
} from 'lucide-react';
import { apiClient } from '@/services/api';
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
import { cn } from '@/lib/utils';

const chartData = [
  { time: '10:00', load: 45 },
  { time: '10:05', load: 52 },
  { time: '10:10', load: 48 },
  { time: '10:15', load: 61 },
  { time: '10:20', load: 55 },
  { time: '10:25', load: 67 },
  { time: '10:30', load: 60 },
];

export function InferenceService() {
  const [isRunning, setIsRunning] = useState(true);
  const [loading, setLoading] = useState(false);
  
  const services = [
    { id: 1, name: 'Vision_Classifier', endpoint: '/inf/v1/classify', status: '活跃', latency: '42ms', requests: '1.2K', rate: '99.8%', color: 'text-cyber-cyan' },
    { id: 2, name: 'Object_Detector_V3', endpoint: '/inf/v3/detect', status: '活跃', latency: '65ms', requests: '840', rate: '98.5%', color: 'text-cyber-purple' },
    { id: 3, name: 'LLM_Neural_Bridge', endpoint: '/inf/n1/generate', status: '维护中', latency: '120ms', requests: '560', rate: '97.2%', color: 'text-yellow-500' },
  ];

  const metrics = [
    { label: '平均延迟', value: '64ms', trend: '-12%', icon: Clock },
    { label: '错误率', value: '0.04%', trend: '-0.2%', icon: Shield },
    { label: '吞吐量', value: '2.4 GB/s', trend: '+18%', icon: Zap },
    { label: '活跃节点', value: '14/16', trend: '稳定', icon: Server },
  ];

  const toggleService = async () => {
    setLoading(true);
    const newState = !isRunning;
    const res = await apiClient.toggleInferenceService(newState);
    if (res.success) setIsRunning(newState);
    setLoading(false);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <h1 className="text-4xl font-black tracking-tighter text-white mb-2 uppercase">
            推理 <span className="text-cyber-cyan">管道</span>
          </h1>
          <p className="text-gray-500 font-medium tracking-tight uppercase text-xs">实时神经计算与节点管理</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button 
            onClick={toggleService}
            disabled={loading}
            className={cn(
              "px-6 py-3 rounded-xl font-bold flex items-center space-x-2 transition-all",
              isRunning 
                ? "bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20 neon-glow-cyan" 
                : "bg-cyber-rose/10 text-cyber-rose border border-cyber-rose/20"
            )}
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : (isRunning ? <Pause size={18} /> : <Play size={18} />)}
            <span>{isRunning ? "暂停管道" : "恢复管道"}</span>
          </button>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((m, i) => (
          <BentoCard key={i} className="h-28 flex items-center justify-between">
            <div>
              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">{m.label}</p>
              <h3 className="text-2xl font-black text-white">{m.value}</h3>
              <p className="text-[10px] text-cyber-emerald font-bold mt-1">{m.trend}</p>
            </div>
            <div className="p-3 rounded-xl bg-white/5 border border-white/10 text-cyber-cyan">
              <m.icon size={20} />
            </div>
          </BentoCard>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Service List */}
        <div className="lg:col-span-7 space-y-4">
           <h3 className="text-xs font-black text-gray-500 uppercase tracking-[0.2em] mb-4 flex items-center space-x-2">
              <Cpu size={14} className="text-cyber-cyan" />
              <span>神经微服务</span>
           </h3>
           
           {services.map((s) => (
             <motion.div 
               key={s.id}
               whileHover={{ x: 4 }}
               className="glass-card rounded-2xl p-5 border border-white/5 flex items-center justify-between group"
             >
                <div className="flex items-center space-x-4">
                   <div className={cn("p-3 rounded-xl bg-white/5 border border-white/5 group-hover:neon-glow-cyan transition-all", s.color)}>
                      <Cpu size={24} />
                   </div>
                   <div>
                      <h4 className="font-bold text-white mb-0.5">{s.name}</h4>
                      <p className="text-[10px] font-mono text-gray-600">{s.endpoint}</p>
                   </div>
                </div>

                <div className="flex items-center space-x-8">
                   <div className="hidden md:block text-right">
                      <p className="text-[10px] font-bold text-gray-600 uppercase mb-1">成功率</p>
                      <p className="text-xs font-black text-cyber-emerald">{s.rate}</p>
                   </div>
                   <div className="hidden md:block text-right">
                      <p className="text-[9px] font-bold text-gray-600 uppercase mb-1">延迟</p>
                      <p className="text-xs font-black text-white">{s.latency}</p>
                   </div>
                   <div className={cn(
                     "px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-tighter border",
                     s.status === 'Active' ? "bg-cyber-emerald/10 text-cyber-emerald border-cyber-emerald/20" : "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
                   )}>
                      {s.status}
                   </div>
                   <button className="p-2 rounded-lg bg-white/5 hover:bg-cyber-cyan hover:text-black transition-all">
                      <ArrowUpRight size={16} />
                   </button>
                </div>
             </motion.div>
           ))}
        </div>

        {/* Load Analysis Chart */}
        <BentoCard 
          className="lg:col-span-5" 
          title="计算负载" 
          description="神经集群压力"
          icon={Activity}
        >
          <div className="h-64 w-full mt-6">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#bc13fe" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#bc13fe" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="time" stroke="#ffffff20" fontSize={10} axisLine={false} tickLine={false} />
                <YAxis hide />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#09090b', border: '1px solid #ffffff10', borderRadius: '8px' }}
                  itemStyle={{ color: '#bc13fe' }}
                />
                <Area type="monotone" dataKey="load" stroke="#bc13fe" strokeWidth={3} fill="url(#colorLoad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-6 p-4 rounded-xl bg-cyber-purple/5 border border-cyber-purple/20">
             <div className="flex items-center space-x-3">
                <Shield size={20} className="text-cyber-purple" />
                <div>
                   <p className="text-[10px] font-bold text-cyber-purple uppercase">安全协议</p>
                   <p className="text-xs text-gray-400">所有推理请求均经过端到端加密，并通过去中心化共识进行验证。</p>
                </div>
             </div>
          </div>
        </BentoCard>
      </div>
    </div>
  );
}
