import React from 'react';
import { Bell, Search, User, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export const Header: React.FC = () => {
  return (
    <header className="h-20 glass-morphism border-b border-white/5 px-8 flex items-center justify-between sticky top-0 z-40">
      {/* Search Section */}
      <div className="flex items-center bg-white/5 rounded-full px-4 py-2 w-96 border border-white/5 focus-within:border-cyber-cyan/50 transition-all">
        <Search className="text-gray-400 w-5 h-5" />
        <input 
          type="text" 
          placeholder="搜索神经模式..." 
          className="bg-transparent border-none outline-none ml-3 text-sm text-white w-full placeholder:text-gray-500"
        />
      </div>

      {/* Action Section */}
      <div className="flex items-center space-x-6">
        {/* Status Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-cyber-emerald/10 border border-cyber-emerald/20 text-cyber-emerald text-xs font-medium">
          <div className="w-2 h-2 rounded-full bg-cyber-emerald animate-pulse" />
          <span>系统最佳</span>
        </div>

        <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
          <Bell size={22} />
          <span className="absolute top-2 right-2 w-2 h-2 bg-cyber-rose rounded-full" />
        </button>

        <button className="p-2 text-gray-400 hover:text-white transition-colors">
          <Zap size={22} className="text-cyber-cyan" />
        </button>

        <div className="flex items-center space-x-3 pl-4 border-l border-white/10">
          <div className="text-right">
            <p className="text-sm font-semibold text-white">Neural_Admin</p>
            <p className="text-[10px] text-gray-500 uppercase tracking-tighter">Level 4 Access</p>
          </div>
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-cyber-purple to-pink-500 p-[1px]">
            <div className="w-full h-full rounded-full bg-cyber-black flex items-center justify-center overflow-hidden">
              <User size={20} className="text-white" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
