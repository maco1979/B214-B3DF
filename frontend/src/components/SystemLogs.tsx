import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { apiClient } from '@/services/api';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  component: string;
}

const SystemLogs: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchLogs = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // 从后端API获取系统日志
      const response = await apiClient.get('/api/system/logs');
      
      if (response.success && response.data) {
        const data = response.data as { logs?: LogEntry[] };
        setLogs(data.logs || []);
      } else {
        throw new Error(response.error || '获取日志失败');
      }
    } catch (err) {
      console.error('获取系统日志失败:', err);
      setError(err instanceof Error ? err.message : '获取日志时发生错误');
      // 如果API调用失败，使用模拟数据
      setLogs([
        {
          timestamp: new Date().toISOString(),
          level: 'INFO',
          message: '系统启动完成',
          component: 'system'
        },
        {
          timestamp: new Date(Date.now() - 300000).toISOString(), // 5分钟前
          level: 'INFO',
          message: 'API服务已就绪',
          component: 'api'
        },
        {
          timestamp: new Date(Date.now() - 600000).toISOString(), // 10分钟前
          level: 'WARNING',
          message: '内存使用率超过80%',
          component: 'monitoring'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    
    // 设置定时器定期刷新日志（优化：每60秒，减少请求频率）
    const interval = setInterval(fetchLogs, 60000);
    
    return () => clearInterval(interval);
  }, []);

  const getLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'text-cyber-rose bg-cyber-rose/10 border-cyber-rose/20';
      case 'WARNING':
        return 'text-cyber-yellow bg-cyber-yellow/10 border-cyber-yellow/20';
      case 'INFO':
        return 'text-cyber-cyan bg-cyber-cyan/10 border-cyber-cyan/20';
      case 'DEBUG':
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
      default:
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className="flex flex-col h-full bg-cyber-black/40 border border-white/5 rounded-2xl overflow-hidden group">
      {/* Header */}
      <div className="h-14 px-6 flex items-center justify-between border-b border-white/5 bg-cyber-black/20 backdrop-blur-md shrink-0">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="w-8 h-8 rounded-lg bg-cyber-purple/10 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-cyber-purple">
                <path d="M12 12h.01"></path>
                <path d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z"></path>
                <path d="M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4z"></path>
              </svg>
            </div>
          </div>
          <div>
            <h3 className="text-sm font-black text-white/90 uppercase tracking-tighter">系统日志</h3>
            <p className="text-[8px] text-gray-500 uppercase tracking-widest font-bold">系统日志</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button 
            onClick={fetchLogs}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-[10px] font-bold text-gray-400 hover:text-cyber-purple hover:border-cyber-purple/30 transition-all disabled:opacity-50"
          >
            {isLoading ? '刷新中...' : '刷新'}
          </button>
        </div>
      </div>

      {/* Logs Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-[10px] custom-scrollbar">
        {error && (
          <div className="p-3 rounded-xl border border-cyber-rose/30 bg-cyber-rose/10 text-cyber-rose">
            <div className="flex items-center space-x-2 mb-1.5 opacity-40 text-[9px] uppercase font-black">
              <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-terminal">
                <polyline points="4 17 10 11 4 5"></polyline>
                <line x1="12" x2="20" y1="19" y2="19"></line>
              </svg>
              <span>错误处理器</span>
            </div>
            <div className="leading-relaxed whitespace-pre-wrap">{error}</div>
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center space-x-3 text-cyber-purple/80 py-8">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span className="text-[10px] font-bold uppercase tracking-widest">加载日志中...</span>
          </div>
        ) : logs.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 opacity-50">
              <path d="M12 12h.01"></path>
              <path d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z"></path>
              <path d="M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4z"></path>
            </svg>
            <p className="text-sm">暂无系统日志</p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {logs.map((log, i) => (
              <motion.div
                key={`${log.timestamp}-${i}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="flex flex-col items-start"
              >
                <div className={cn(
                  "max-w-full p-3 rounded-xl border transition-all duration-300 w-full",
                  getLevelColor(log.level)
                )}>
                  <div className="flex justify-between items-start mb-1.5">
                    <div className="flex items-center space-x-2 opacity-40 text-[9px] uppercase font-black">
                      <span>{log.component}</span>
                      <span className="text-gray-500">•</span>
                      <span>{formatTimestamp(log.timestamp)}</span>
                    </div>
                    <span className={cn(
                      "text-[8px] font-bold uppercase px-2 py-0.5 rounded-full",
                      log.level.toUpperCase() === 'ERROR' && 'bg-cyber-rose/20 text-cyber-rose',
                      log.level.toUpperCase() === 'WARNING' && 'bg-cyber-yellow/20 text-cyber-yellow',
                      log.level.toUpperCase() === 'INFO' && 'bg-cyber-cyan/20 text-cyber-cyan',
                      log.level.toUpperCase() === 'DEBUG' && 'bg-gray-400/20 text-gray-400'
                    )}>
                      {log.level}
                    </span>
                  </div>
                  <div className="leading-relaxed whitespace-pre-wrap text-[10px]">{log.message}</div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>

      {/* Footer */}
      <div className="h-10 px-4 bg-cyber-black/40 border-t border-white/5 flex items-center justify-between">
        <span className="text-[9px] text-gray-500 uppercase font-bold">
          {logs.length} 条日志记录
        </span>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-cyber-cyan animate-pulse"></div>
          <span className="text-[9px] text-cyber-cyan font-bold uppercase">
            实时更新
          </span>
        </div>
      </div>
    </div>
  );
};

export default SystemLogs;