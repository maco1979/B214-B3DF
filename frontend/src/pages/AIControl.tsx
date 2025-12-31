import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cpu, 
  Shield, 
  Wifi, 
  Mic, 
  MicOff, 
  Power, 
  RefreshCw, 
  Video, 
  Settings,
  Activity,
  Zap,
  CheckCircle,
  Lock,
  ArrowRight,
  Maximize2
} from 'lucide-react';
import { apiClient, Device, JEPAData } from '@/services/api';
import { cn } from '@/lib/utils';
import { BentoCard } from '@/components/ui/BentoCard';
import { DeviceCard } from '@/components/ui/DeviceCard';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';

// Mock data to fallback if API fails
const mockPresets = [
  { id: 1, name: '农业智能', description: '自主灌溉和养分监测', devices: [1, 2, 3] },
  { id: 2, name: '生态节能', description: '超低功耗模式', devices: [1, 4] },
  { id: 3, name: '最大安全', description: '全方位生物识别监控', devices: [3, 4] },
];

export function AIControl() {
  const [devices, setDevices] = useState<Device[]>([]);
  const [selectedDevices, setSelectedDevices] = useState<number[]>([]);
  const [isMasterActive, setIsMasterActive] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState(mockPresets[0]);
  
  // JEPA-DT-MPC state
  const [isJepaActive, setIsJepaActive] = useState(false);
  const [jepaData, setJepaData] = useState<JEPAData | null>(null);
  const [jepaModelStatus, setJepaModelStatus] = useState('ready');

  // Voice recognition
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [lastCommand, setLastCommand] = useState('');
  const voiceRecognitionRef = useRef<any>(null);
  
  // Camera state
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraFrame, setCameraFrame] = useState('');
  const [cameraError, setCameraError] = useState('');
  
  const [isScanning, setIsScanning] = useState(false);

  // Lifecycle
  useEffect(() => {
    fetchDevices();
    fetchJepaStatus();
    checkMasterStatus();
  }, []);

  const checkMasterStatus = async () => {
     try {
       const res = await apiClient.get<{ master_control_active: boolean }>('/api/ai-control/master-control/status');
       setIsMasterActive(res.data?.master_control_active || false);
     } catch (e) {}
  };

  const fetchDevices = async () => {
    const res = await apiClient.getDevices();
    if (res.success && res.data) setDevices(res.data);
  };

  const fetchJepaStatus = async () => {
    const res = await apiClient.getJepaDtmpcStatus();
    if (res.success && res.data) {
      setIsJepaActive(res.data.is_active);
      setJepaModelStatus(res.data.model_status);
    }
  };

  const scanDevices = async () => {
    setIsScanning(true);
    const res = await apiClient.scanDevices();
    if (res.success && res.data) setDevices(res.data);
    setIsScanning(false);
  };

  // Actions
  const handleMasterToggle = async () => {
    const newStatus = !isMasterActive;
    const res = await apiClient.activateMasterControl(newStatus);
    if (res.success) setIsMasterActive(newStatus);
  };

  const toggleJepa = async () => {
    const newStatus = !isJepaActive;
    try {
      const res = await apiClient.activateJepaDtmpc({
        controller_params: { control_switch: newStatus, startup_mode: 'cold' },
        mv_params: { 
          operation_range: [-100, 100],
          rate_limits: [-10, 10],
          action_cycle: 1.0
        },
        cv_params: { 
          setpoint: 50,
          safety_range: [-200, 200],
          weights: 1.0
        },
        model_params: { 
          prediction_horizon: 10, 
          control_horizon: 5,
          system_gain: 1.0,
          time_delay: 1,
          time_constant: 5
        },
        jepa_params: { 
          enabled: newStatus, 
          embedding_dim: 10, 
          prediction_horizon: 20,
          input_dim: 3,
          output_dim: 1,
          training_steps: 100
        }
      } as any);
      if (res.success) setIsJepaActive(newStatus);
    } catch (error) {
      console.error('激活JEPA-DT-MPC失败:', error);
      // 可以添加用户友好的错误提示
    }
  };

  const toggleCamera = async () => {
    if (isCameraOpen) {
      await apiClient.closeCamera();
      setIsCameraOpen(false);
      setCameraFrame('');
    } else {
      const res = await apiClient.openCamera();
      if (res.success) setIsCameraOpen(true);
      else setCameraError("Failed to access hardware");
    }
  };

  // Camera Frame Loop
  useEffect(() => {
    let frameId: number;
    const getFrame = async () => {
      if (!isCameraOpen) return;
      try {
        const res = await apiClient.getCameraFrame();
        if (res.success && res.data?.frame_base64) {
          setCameraFrame(res.data.frame_base64);
        }
      } catch (e) {}
      frameId = requestAnimationFrame(getFrame);
    };
    if (isCameraOpen) getFrame();
    return () => cancelAnimationFrame(frameId);
  }, [isCameraOpen]);

  // Voice Control logic (simplified for brevity but functional)
  const toggleVoice = () => {
    if (isVoiceActive) {
      voiceRecognitionRef.current?.stop();
      setIsVoiceActive(false);
    } else {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.lang = 'zh-CN';
        recognition.continuous = true;
        recognition.onresult = (event: any) => {
          const cmd = event.results[event.results.length - 1][0].transcript;
          setLastCommand(cmd);
          if (cmd.includes('开启主控')) handleMasterToggle();
        };
        recognition.start();
        voiceRecognitionRef.current = recognition;
        setIsVoiceActive(true);
      }
    }
  };

  return (
    <div className="space-y-8 pb-20">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <h1 className="text-4xl font-black tracking-tighter text-white mb-2 uppercase">
            控制 <span className="text-cyber-cyan">中心</span>
          </h1>
          <p className="text-gray-500 font-medium tracking-tight uppercase text-xs">AI设备编排与优化</p>
        </div>
        
        <div className="flex items-center space-x-3 bg-white/5 p-2 rounded-2xl border border-white/5 backdrop-blur-md">
           <button 
             onClick={toggleVoice}
             className={cn(
               "p-3 rounded-xl transition-all flex items-center space-x-2",
               isVoiceActive ? "bg-cyber-rose/20 text-cyber-rose neon-glow-purple" : "text-gray-400 hover:text-white"
             )}
           >
             {isVoiceActive ? <Mic size={20} /> : <MicOff size={20} />}
             <span className="text-xs font-bold uppercase">{isVoiceActive ? "监听中" : "语音控制"}</span>
           </button>
           
           <div className="w-[1px] h-6 bg-white/10" />

           <button 
             onClick={handleMasterToggle}
             className={cn(
               "px-6 py-3 rounded-xl font-bold flex items-center space-x-2 transition-all",
               isMasterActive 
                ? "bg-cyber-rose text-white shadow-lg shadow-cyber-rose/40" 
                : "bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20 neon-glow-cyan"
             )}
           >
             <Power size={18} />
             <span>{isMasterActive ? "关闭主控" : "激活主控"}</span>
           </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Device Matrix & Presets */}
        <div className="lg:col-span-8 space-y-6">
          {/* Presets Bento */}
          <BentoCard title="AI运行模式" description="预定义神经配置文件" icon={Settings}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
              {mockPresets.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => {
                    setSelectedPreset(preset);
                    setSelectedDevices(preset.devices);
                  }}
                  className={cn(
                    "p-5 rounded-2xl border text-left transition-all relative overflow-hidden group",
                    selectedPreset.id === preset.id 
                      ? "bg-cyber-cyan/10 border-cyber-cyan/40" 
                      : "bg-white/5 border-white/5 hover:border-white/10"
                  )}
                >
                  <div className={cn(
                    "mb-4 w-10 h-10 rounded-lg flex items-center justify-center transition-all",
                    selectedPreset.id === preset.id ? "bg-cyber-cyan text-black" : "bg-white/5 text-gray-500"
                  )}>
                    <Zap size={20} />
                  </div>
                  <h4 className="font-bold text-white mb-1">{preset.name}</h4>
                  <p className="text-[10px] text-gray-500 uppercase leading-relaxed tracking-wider">{preset.description}</p>
                  
                  {selectedPreset.id === preset.id && (
                    <motion.div layoutId="active-preset" className="absolute top-2 right-2">
                      <CheckCircle size={16} className="text-cyber-cyan" />
                    </motion.div>
                  )}
                </button>
              ))}
            </div>
          </BentoCard>

          {/* Device Matrix */}
          <div className="flex items-center justify-between mb-2">
             <h3 className="text-lg font-bold text-white uppercase tracking-tighter flex items-center space-x-2">
                <Wifi size={18} className="text-cyber-cyan" />
                <span>设备矩阵</span>
             </h3>
             <button 
               onClick={scanDevices}
               disabled={isScanning}
               className="text-[10px] font-bold text-cyber-cyan uppercase tracking-widest flex items-center space-x-2 hover:opacity-80 disabled:opacity-50"
             >
                <RefreshCw size={12} className={isScanning ? "animate-spin" : ""} />
                <span>{isScanning ? "扫描中..." : "重新扫描环境"}</span>
             </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
             {devices.map((device) => (
               <DeviceCard 
                 key={device.id}
                 device={device}
                 isSelected={selectedDevices.includes(device.id)}
                 onSelect={(id) => setSelectedDevices(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])}
                 onToggleConnection={() => {}}
                 onControl={() => {}}
               />
             ))}
             {devices.length === 0 && [1,2,3,4].map(i => (
               <div key={i} className="h-48 glass-card rounded-2xl animate-pulse" />
             ))}
          </div>
        </div>

        {/* Right Column: Intelligence & Vision */}
        <div className="lg:col-span-4 space-y-6">
           {/* Visual Feed */}
           <BentoCard title="视觉智能" description="神经流馈送" icon={Video}>
              <div className="mt-4 relative rounded-xl overflow-hidden aspect-video bg-black border border-white/5 group">
                 {isCameraOpen ? (
                   cameraFrame ? (
                     <img src={`data:image/jpeg;base64,${cameraFrame}`} className="w-full h-full object-cover" alt="Feed" />
                   ) : (
                     <div className="absolute inset-0 flex items-center justify-center text-xs text-cyber-cyan animate-pulse font-mono">建立上行链路...</div>
                   )
                 ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600">
                       <Video size={40} className="mb-2 opacity-20" />
                       <span className="text-[10px] uppercase font-bold tracking-widest">馈送离线</span>
                    </div>
                 )}
                 
                 <div className="absolute bottom-4 right-4 flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={toggleCamera} className="p-2 rounded-lg bg-cyber-black/80 backdrop-blur-md border border-white/10 text-white hover:bg-cyber-cyan hover:text-black transition-all">
                       <Power size={14} />
                    </button>
                    <button className="p-2 rounded-lg bg-cyber-black/80 backdrop-blur-md border border-white/10 text-white hover:bg-cyber-cyan hover:text-black transition-all">
                       <Maximize2 size={14} />
                    </button>
                 </div>

                 {isCameraOpen && (
                   <div className="absolute top-4 left-4 flex items-center space-x-2 px-2 py-1 rounded bg-black/60 backdrop-blur-sm border border-cyber-emerald/30">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyber-emerald animate-pulse" />
                      <span className="text-[8px] font-bold text-cyber-emerald uppercase tracking-tighter">实时神经馈送</span>
                   </div>
                 )}
              </div>
              <div className="mt-4 space-y-2">
                 <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest">AI视觉状态</p>
                 <div className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5">
                    <span className="text-xs text-gray-400">对象跟踪</span>
                    <span className="text-cyber-emerald font-bold text-xs uppercase">最佳</span>
                 </div>
              </div>
           </BentoCard>

           {/* JEPA-DT-MPC Control */}
           <BentoCard title="JEPA预测" description="自主MPC集成" icon={Activity}>
              <div className="flex items-center justify-between mb-4">
                 <div className="flex items-center space-x-2">
                    <div className={cn("w-2 h-2 rounded-full", isJepaActive ? "bg-cyber-emerald animate-pulse" : "bg-gray-700")} />
                    <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">引擎 {isJepaActive ? "运行中" : "待机"}</span>
                 </div>
                 <button 
                   onClick={toggleJepa}
                   className={cn(
                     "text-[10px] font-bold px-3 py-1 rounded-full transition-all border",
                     isJepaActive 
                       ? "bg-cyber-rose/10 text-cyber-rose border-cyber-rose/20" 
                       : "bg-cyber-cyan/10 text-cyber-cyan border-cyber-cyan/20"
                   )}
                 >
                   {isJepaActive ? "断开" : "启动"}
                 </button>
              </div>

              <div className="h-40 w-full bg-white/5 rounded-xl border border-white/5 p-4 relative overflow-hidden">
                 <div className="absolute inset-0 flex items-center justify-center opacity-10">
                    <Activity size={80} className="text-cyber-cyan" />
                 </div>
                 <div className="relative z-10 flex flex-col h-full justify-between">
                    <div className="flex justify-between items-end">
                       <div>
                          <p className="text-[8px] text-gray-500 uppercase font-bold mb-1">预测权重</p>
                          <h4 className="text-2xl font-black text-white">0.842</h4>
                       </div>
                       <ArrowRight size={20} className="text-cyber-cyan mb-1" />
                    </div>
                    
                    <div className="space-y-2">
                       <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest">
                          <span className="text-gray-500">稳定性</span>
                          <span className="text-cyber-emerald">98.4%</span>
                       </div>
                       <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: "98%" }}
                            className="h-full bg-cyber-emerald"
                          />
                       </div>
                    </div>
                 </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                 <div className="p-3 rounded-xl bg-white/5 border border-white/5 text-center">
                    <p className="text-[8px] text-gray-500 uppercase font-bold mb-1">模型状态</p>
                    <p className="text-xs font-bold text-cyber-cyan uppercase">{jepaModelStatus}</p>
                 </div>
                 <div className="p-3 rounded-xl bg-white/5 border border-white/5 text-center">
                    <p className="text-[8px] text-gray-500 uppercase font-bold mb-1">能源指数</p>
                    <p className="text-xs font-bold text-cyber-purple">0.024</p>
                 </div>
              </div>
           </BentoCard>

           {/* Voice Logs */}
           <AnimatePresence>
             {lastCommand && (
               <motion.div
                 initial={{ opacity: 0, y: 10 }}
                 animate={{ opacity: 1, y: 0 }}
                 exit={{ opacity: 0, scale: 0.95 }}
                 className="p-4 rounded-xl glass-morphism border border-cyber-cyan/20 bg-cyber-cyan/5"
               >
                 <div className="flex items-center space-x-2 mb-2">
                    <Mic size={14} className="text-cyber-cyan" />
                    <span className="text-[10px] font-bold text-cyber-cyan uppercase tracking-widest">最后接收的命令</span>
                 </div>
                 <p className="text-sm font-mono text-white/90">"{lastCommand}"</p>
               </motion.div>
             )}
           </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
