import React from 'react';
import { motion } from 'framer-motion';
import { 
  Wifi, 
  Battery, 
  Zap, 
  Shield, 
  Settings, 
  Power,
  Cpu,
  Smartphone,
  Bluetooth,
  Radio
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Device } from '@/services/api';

interface DeviceCardProps {
  device: Device;
  onToggleConnection: (id: number) => void;
  onControl: (id: number) => void;
  isSelected: boolean;
  onSelect: (id: number) => void;
}

const getConnectionIcon = (type: string) => {
  switch (type) {
    case 'wifi': return <Wifi className="w-4 h-4" />;
    case 'bluetooth': return <Bluetooth className="w-4 h-4" />;
    case 'app': return <Smartphone className="w-4 h-4" />;
    case 'infrared': return <Radio className="w-4 h-4" />;
    default: return <Wifi className="w-4 h-4" />;
  }
};

export const DeviceCard: React.FC<DeviceCardProps> = ({ 
  device, 
  onToggleConnection, 
  onControl,
  isSelected,
  onSelect
}) => {
  const isOnline = device.status === 'online';

  return (
    <motion.div
      layout
      className={cn(
        "glass-card rounded-2xl p-5 border transition-all duration-300 group",
        isOnline ? "border-white/10" : "border-red-500/20 opacity-70",
        isSelected && "border-cyber-cyan/40 ring-1 ring-cyber-cyan/20"
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={cn(
            "p-3 rounded-xl transition-all duration-300",
            isOnline ? "bg-cyber-cyan/10 text-cyber-cyan group-hover:neon-glow-cyan" : "bg-gray-800 text-gray-500"
          )}>
            <Cpu size={20} />
          </div>
          <div>
            <h4 className="font-bold text-white/90 leading-none mb-1">{device.name}</h4>
            <div className="flex items-center space-x-2">
              <span className={cn(
                "w-1.5 h-1.5 rounded-full",
                isOnline ? "bg-cyber-emerald animate-pulse" : "bg-red-500"
              )} />
              <span className="text-[10px] uppercase tracking-tighter text-gray-500 font-bold">
                {device.type} • {device.status}
              </span>
            </div>
          </div>
        </div>
        
        <input 
          id={`device-select-${device.id}`}
          name={`device-select-${device.id}`}
          type="checkbox"
          checked={isSelected}
          onChange={() => onSelect(device.id)}
          className="w-4 h-4 rounded border-white/10 bg-white/5 checked:bg-cyber-cyan focus:ring-cyber-cyan accent-cyber-cyan"
        />
      </div>

      {/* Metrics */}
      <div className="space-y-3 mb-5">
        <div className="flex items-center justify-between text-[10px] text-gray-500 uppercase tracking-widest font-bold">
          <span>连接质量</span>
          <span className={isOnline ? "text-cyber-cyan" : ""}>{device.signal}%</span>
        </div>
        <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${device.signal}%` }}
            className={cn(
              "h-full rounded-full",
              device.signal > 70 ? "bg-cyber-cyan" : "bg-yellow-500"
            )}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-gray-400">
            {getConnectionIcon(device.connection_type || 'wifi')}
            <span className="text-xs">{device.connection_type?.toUpperCase() || 'UNKNOWN'}</span>
          </div>
          <div className="flex items-center space-x-2 text-gray-400">
            <Battery size={14} className={device.battery < 20 ? "text-red-500" : ""} />
            <span className="text-xs">{device.battery}%</span>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center space-x-2 pt-4 border-t border-white/5">
        <button
          onClick={() => onToggleConnection(device.id)}
          className={cn(
            "flex-1 py-2 rounded-lg text-xs font-bold transition-all flex items-center justify-center space-x-2",
            device.connected 
              ? "bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white" 
              : "bg-cyber-emerald/10 text-cyber-emerald border border-cyber-emerald/20 hover:bg-cyber-emerald/20"
          )}
        >
          <Power size={14} />
          <span>{device.connected ? "断开连接" : "连接"}</span>
        </button>
        
        <button
          disabled={!device.connected}
          onClick={() => onControl(device.id)}
          className={cn(
            "p-2 rounded-lg transition-all",
            device.connected 
              ? "bg-cyber-cyan/10 text-cyber-cyan hover:bg-cyber-cyan/20 border border-cyber-cyan/20" 
              : "bg-white/5 text-gray-600 cursor-not-allowed"
          )}
        >
          <Settings size={16} />
        </button>
      </div>
    </motion.div>
  );
};
