import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  MessageSquare, 
  Terminal, 
  ChevronRight, 
  Loader2,
  Cpu,
  Shield,
  Zap,
  Activity,
  Maximize2,
  Trash2,
  FileText,
  Menu,
  X
} from 'lucide-react';
import { apiClient } from '@/services/api';
import { cn } from '@/lib/utils';

export function DecisionAgent() {
  const [messages, setMessages] = useState<{role: 'system' | 'ai', content: string, type?: 'reasoning' | 'decision'}[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [lastDecision, setLastDecision] = useState<any>(null);
  const [showLogs, setShowLogs] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const askAgent = async (type: string = 'agriculture') => {
    setIsThinking(true);
    setMessages(prev => [...prev, { role: 'system', content: `REQUESTING NEURAL INFERENCE [${type.toUpperCase()}]...` }]);
    
    try {
      const response = await apiClient.makeDecision(type, { 
        timestamp: new Date().toISOString(), 
        sensor_data: "optimal" 
      });
      
      if (response.success && response.data) {
        const data = response.data;
        setLastDecision(data);
        
        // Process actual backend response data
        const reasoning = `Decision ID: ${data.decision_id}\nAction: ${data.action}\nConfidence: ${(data.confidence * 100).toFixed(0)}%\nExecution Time: ${data.execution_time.toFixed(2)}ms`;
        const decision = data.recommendations ? data.recommendations.join('\n') : 'No recommendations available';
        
        // Simulated multi-step organic thought process
        setTimeout(() => {
          setMessages(prev => [...prev, { role: 'ai', content: reasoning, type: 'reasoning' }]);
          setTimeout(() => {
            setMessages(prev => [...prev, { role: 'ai', content: decision, type: 'decision' }]);
            setIsThinking(false);
          }, 800);
        }, 1200);
      } else {
        throw new Error(response.error || 'Failed to reach neural core');
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'ai', content: `CRITICAL ERROR: ${err instanceof Error ? err.message : String(err)}` }]);
      setIsThinking(false);
    }
  };

  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{ role: 'ai', content: '神经代理_V4.2 [在线] - 准备就绪' }]);
    }
  }, []);

  const clearLogs = () => {
    setMessages([{ role: 'ai', content: '终端日志已清空 - 待机' }]);
  };

  return (
    <div className="flex flex-col h-full bg-cyber-black/40 border border-white/5 rounded-2xl overflow-hidden group">
      {/* Terminal Header */}
      <div className="h-14 px-6 flex items-center justify-between border-b border-white/5 bg-cyber-black/20 backdrop-blur-md shrink-0">
        <div className="flex items-center space-x-3">
          <div className="relative">
             <div className={cn(
               "w-8 h-8 rounded-lg bg-cyber-cyan/10 flex items-center justify-center transition-all duration-500",
               isThinking && "neon-glow-cyan scale-110"
             )}>
                <Brain className={cn("w-5 h-5 text-cyber-cyan", isThinking && "animate-pulse")} />
             </div>
          </div>
          <div>
            <h3 className="text-sm font-black text-white/90 uppercase tracking-tighter">神经智能</h3>
            <p className="text-[8px] text-gray-500 uppercase tracking-widest font-bold">自主决策支持</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
           <button 
             onClick={() => askAgent('agriculture')} 
             disabled={isThinking}
             className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-[10px] font-bold text-gray-400 hover:text-cyber-cyan hover:border-cyber-cyan/30 transition-all disabled:opacity-50"
           >
             农业模块
           </button>
           <button 
             onClick={() => askAgent('risk')} 
             disabled={isThinking}
             className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-[10px] font-bold text-gray-400 hover:text-cyber-purple hover:border-cyber-purple/30 transition-all disabled:opacity-50"
           >
             风险分析
           </button>
           <div className="w-[1px] h-4 bg-white/10 mx-1" />
           <button onClick={clearLogs} className="p-1.5 rounded-lg text-gray-600 hover:text-cyber-rose hover:bg-cyber-rose/10 transition-all">
              <Trash2 size={14} />
           </button>
           <button 
             onClick={() => setShowLogs(true)} 
             className="p-1.5 rounded-lg text-gray-600 hover:text-cyber-purple hover:bg-cyber-purple/10 transition-all"
             title="查看系统日志"
           >
              <FileText size={14} />
           </button>
        </div>
      </div>

      {/* Terminal Body - 使用 flex-1 但添加最大高度限制 */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-4 font-mono text-[11px] custom-scrollbar cursor-text"
        style={{ maxHeight: 'calc(100vh - 120px)' }}
        onClick={() => {
          // 点击终端内容区域时，可以触发一个新的通用请求
          if (!isThinking) {
            askAgent('general');
          }
        }}
      >
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className={cn(
                "flex flex-col",
                msg.role === 'system' ? "items-end" : "items-start"
              )}
            >
              <div className={cn(
                "max-w-[90%] p-3 rounded-xl border transition-all duration-300",
                msg.role === 'system' 
                  ? "bg-cyber-cyan/5 text-cyber-cyan border-cyber-cyan/20 hover:border-cyber-cyan/40" 
                  : msg.type === 'decision'
                    ? "bg-cyber-emerald/10 text-cyber-emerald border-cyber-emerald/30 font-bold shadow-lg shadow-cyber-emerald/10 hover:border-cyber-emerald/50"
                    : "bg-white/5 text-gray-400 border-white/5 hover:border-white/10"
              )}>
                <div className="flex items-center space-x-2 mb-1.5 opacity-40 text-[9px] uppercase font-black">
                  {msg.role === 'ai' ? <Terminal size={10} /> : <Zap size={10} />}
                  <span>{msg.role === 'ai' ? '神经代理_01' : '上行链路系统'}</span>
                </div>
                
                <div className="leading-relaxed whitespace-pre-wrap">{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}</div>

                {msg.type === 'decision' && typeof lastDecision === 'object' && lastDecision !== null && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between text-[8px] font-bold uppercase text-gray-500"
                  >
                    <div className="flex items-center space-x-3">
                       <span className="flex items-center space-x-1">
                          <Activity size={10} className="text-cyber-cyan" />
                          <span>置信度: {String(((typeof lastDecision.confidence === 'number' ? lastDecision.confidence : 0) * 100).toFixed(1) + '%')}</span>
                       </span>
                       <span className="flex items-center space-x-1">
                          <Shield size={10} className="text-cyber-purple" />
                          <span>节点: {String(typeof lastDecision.node_name === 'string' ? lastDecision.node_name : 'DCNN_V4')}</span>
                       </span>
                    </div>
                    <span className="text-[10px] text-cyber-cyan">已验证</span>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isThinking && (
          <div className="flex items-center space-x-3 text-cyber-cyan">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-[10px] font-bold uppercase tracking-widest animate-pulse">运行神经推理中...</span>
          </div>
        )}
        
        {/* 添加提示文本，当终端为空时显示 */}
        {messages.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <Terminal size={24} className="mx-auto mb-2 opacity-50" />
            <p className="text-sm">点击任意位置或使用上方按钮开始对话</p>
          </div>
        )}
      </div>

      {/* Terminal Footer */}
      <div className="h-14 px-4 bg-cyber-black/40 border-t border-white/5 flex items-center space-x-3">
         <div className="flex-1 bg-cyber-black/60 rounded-xl px-4 py-2 border border-white/5 flex items-center space-x-2">
            <ChevronRight size={14} className="text-cyber-cyan" />
            <span className="text-[10px] text-gray-600 font-bold uppercase italic">
               {isThinking ? '神经核心繁忙' : '等待上行链路...'}
            </span>
         </div>
         <button 
           onClick={() => askAgent('general')} 
           disabled={isThinking}
           className="w-10 h-10 rounded-xl bg-cyber-cyan/10 flex items-center justify-center text-cyber-cyan border border-cyber-cyan/20 hover:bg-cyber-cyan hover:text-black transition-all disabled:opacity-50 disabled:cursor-not-allowed"
         >
            <MessageSquare size={18} />
         </button>
      </div>

      {/* 侧滑系统日志面板 */}
      <div 
        className={`fixed top-0 right-0 h-full w-full md:w-96 bg-cyber-black/90 backdrop-blur-xl border-l border-white/10 z-50 transform transition-transform duration-300 ease-in-out ${showLogs ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="h-full flex flex-col">
          <div className="h-14 px-6 flex items-center justify-between border-b border-white/5 bg-cyber-black/20 backdrop-blur-md shrink-0">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-8 h-8 rounded-lg bg-cyber-purple/10 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-cyber-purple" />
                </div>
              </div>
              <div>
                <h3 className="text-sm font-black text-white/90 uppercase tracking-tighter">系统日志</h3>
                <p className="text-[8px] text-gray-500 uppercase tracking-widest font-bold">System Logs</p>
              </div>
            </div>
            <button 
              onClick={() => setShowLogs(false)} 
              className="p-1.5 rounded-lg text-gray-600 hover:text-cyber-rose hover:bg-cyber-rose/10 transition-all"
            >
              <X size={14} />
            </button>
          </div>
          
          {/* 日志内容区域 */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-[10px] custom-scrollbar">
            <div className="text-center py-12 text-gray-500">
              <FileText className="mx-auto mb-2 opacity-50 w-6 h-6" />
              <p className="text-sm">系统日志功能</p>
              <p className="text-xs mt-1 opacity-60">点击关闭按钮返回终端</p>
            </div>
          </div>
          
          <div className="h-10 px-4 bg-cyber-black/40 border-t border-white/5 flex items-center justify-between">
            <span className="text-[9px] text-gray-500 uppercase font-bold">
              实时日志监控
            </span>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-cyber-cyan animate-pulse"></div>
              <span className="text-[9px] text-cyber-cyan font-bold uppercase">
                在线
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 遮罩层 */}
      {showLogs && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
          onClick={() => setShowLogs(false)}
        ></div>
      )}
    </div>
  );
}
