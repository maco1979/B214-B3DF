import React from 'react';

interface GalaxyCardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export const GalaxyCard: React.FC<GalaxyCardProps> = ({ children, className = '', onClick }) => {
  return (
    <div 
      className={`galaxy-card ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
};

interface GalaxyButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}

export const GalaxyButton: React.FC<GalaxyButtonProps> = ({ 
  children, 
  variant = 'primary', 
  onClick, 
  disabled = false,
  className = ''
}) => {
  return (
    <button
      className={`galaxy-button ${variant === 'secondary' ? 'galaxy-button-secondary' : ''} ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

interface GalaxyInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export const GalaxyInput: React.FC<GalaxyInputProps> = ({ label, ...props }) => {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="galaxy-font-body text-sm text-white/70">
          {label}
        </label>
      )}
      <input
        className="galaxy-input w-full"
        {...props}
      />
    </div>
  );
};

interface GalaxyBadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning';
}

export const GalaxyBadge: React.FC<GalaxyBadgeProps> = ({ 
  children, 
  variant = 'default' 
}) => {
  const variantClass = variant === 'success' 
    ? 'galaxy-badge-success' 
    : variant === 'warning' 
    ? 'galaxy-badge-warning' 
    : '';
  
  return (
    <span className={`galaxy-badge ${variantClass}`}>
      {children}
    </span>
  );
};

interface GalaxyProgressProps {
  value: number;
  max?: number;
  label?: string;
}

export const GalaxyProgress: React.FC<GalaxyProgressProps> = ({ 
  value, 
  max = 100,
  label 
}) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <span className="galaxy-font-body text-sm text-white/70">
          {label}
        </span>
      )}
      <div className="galaxy-progress-bar">
        <div 
          className="galaxy-progress-fill" 
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

interface GalaxyModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export const GalaxyModal: React.FC<GalaxyModalProps> = ({ 
  isOpen, 
  onClose, 
  title, 
  children 
}) => {
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="galaxy-modal w-full max-w-lg mx-4">
        <div className="flex items-center justify-between mb-6">
          <h2 className="galaxy-font-display text-xl font-bold galazy-text-gradient">
            {title}
          </h2>
          <button 
            onClick={onClose}
            className="text-white/50 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
};

interface GalaxyStatCardProps {
  title: string;
  value: string | number;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

export const GalaxyStatCard: React.FC<GalaxyStatCardProps> = ({ 
  title, 
  value, 
  icon,
  trend 
}) => {
  return (
    <div className="galaxy-stat-card">
      {icon && (
        <div className="absolute top-4 right-4 text-3xl galazy-icon-glow">
          {icon}
        </div>
      )}
      <div className="galaxy-metric-label mb-2">
        {title}
      </div>
      <div className="galaxy-metric-value">
        {value}
      </div>
      {trend && (
        <div className={`flex items-center gap-2 mt-2 text-sm ${
          trend.isPositive ? 'text-green-400' : 'text-red-400'
        }`}>
          <span>
            {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
          </span>
        </div>
      )}
    </div>
  );
};

interface GalaxySwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
}

export const GalaxySwitch: React.FC<GalaxySwitchProps> = ({ 
  checked, 
  onChange, 
  label 
}) => {
  return (
    <div className="flex items-center gap-3">
      {label && (
        <span className="galaxy-font-body text-sm text-white/70">
          {label}
        </span>
      )}
      <button
        className={`galaxy-switch ${checked ? 'galaxy-switch-active' : ''}`}
        onClick={() => onChange(!checked)}
      >
        <div className="galaxy-switch-thumb" />
      </button>
    </div>
  );
};

interface GalaxyAvatarProps {
  name: string;
  size?: 'sm' | 'md' | 'lg';
  src?: string;
}

export const GalaxyAvatar: React.FC<GalaxyAvatarProps> = ({ 
  name, 
  size = 'md',
  src 
}) => {
  const sizeClasses = {
    sm: 'w-8 h-8 text-sm',
    md: 'w-12 h-12 text-lg',
    lg: 'w-16 h-16 text-xl'
  };
  
  const initials = name
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
  
  return (
    <div className={`galaxy-avatar ${sizeClasses[size]}`}>
      {src ? (
        <img src={src} alt={name} className="w-full h-full rounded-full" />
      ) : (
        <span>{initials}</span>
      )}
    </div>
  );
};

interface GalaxyTagProps {
  children: React.ReactNode;
  onClick?: () => void;
}

export const GalaxyTag: React.FC<GalaxyTagProps> = ({ children, onClick }) => {
  return (
    <span 
      className="galaxy-tag cursor-pointer"
      onClick={onClick}
    >
      {children}
    </span>
  );
};

interface GalaxyTimelineItemProps {
  date: string;
  title: string;
  description?: string;
  icon?: React.ReactNode;
}

export const GalaxyTimelineItem: React.FC<GalaxyTimelineItemProps> = ({ 
  date, 
  title, 
  description,
  icon 
}) => {
  return (
    <div className="galaxy-timeline-item">
      <div className="galaxy-timeline-dot">
        {icon}
      </div>
      <div className="ml-6">
        <div className="text-sm text-white/50 mb-1 galazy-font-body">
          {date}
        </div>
        <div className="font-semibold text-white mb-1 galazy-font-display">
          {title}
        </div>
        {description && (
          <div className="text-sm text-white/70 galazy-font-body">
            {description}
          </div>
        )}
      </div>
    </div>
  );
};

interface GalaxyTimelineProps {
  children: React.ReactNode;
}

export const GalaxyTimeline: React.FC<GalaxyTimelineProps> = ({ children }) => {
  return (
    <div className="galaxy-timeline">
      {children}
    </div>
  );
};

interface GalaxyLoaderProps {
  size?: 'sm' | 'md' | 'lg';
  type?: 'spinner' | 'dots';
}

export const GalaxyLoader: React.FC<GalaxyLoaderProps> = ({ 
  size = 'md',
  type = 'spinner'
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-10 h-10',
    lg: 'w-14 h-14'
  };
  
  if (type === 'dots') {
    return (
      <div className="galaxy-dots-loader">
        <div className="galaxy-dot" />
        <div className="galaxy-dot" />
        <div className="galaxy-dot" />
      </div>
    );
  }
  
  return (
    <div className={`galaxy-loader ${sizeClasses[size]}`} />
  );
};

interface GalaxyNotificationProps {
  title: string;
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error';
  onClose?: () => void;
}

export const GalaxyNotification: React.FC<GalaxyNotificationProps> = ({ 
  title, 
  message,
  type = 'info',
  onClose 
}) => {
  const typeColors = {
    info: 'from-blue-500 to-cyan-500',
    success: 'from-green-500 to-emerald-500',
    warning: 'from-yellow-500 to-orange-500',
    error: 'from-red-500 to-pink-500'
  };
  
  return (
    <div className="galaxy-notification mb-4">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className={`font-semibold mb-1 galazy-font-display galazy-text-gradient-cyan bg-gradient-to-r ${typeColors[type]}`}>
            {title}
          </div>
          <div className="text-sm text-white/80 galazy-font-body">
            {message}
          </div>
        </div>
        {onClose && (
          <button 
            onClick={onClose}
            className="text-white/50 hover:text-white transition-colors ml-4"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

interface GalaxyContainerProps {
  children: React.ReactNode;
  className?: string;
}

export const GalaxyContainer: React.FC<GalaxyContainerProps> = ({ 
  children, 
  className = '' 
}) => {
  return (
    <div className={`galaxy-container min-h-screen ${className}`}>
      <div className="galaxy-stars" />
      <div className="galaxy-nebula galaxy-nebula-1" />
      <div className="galaxy-nebula galaxy-nebula-2" />
      <div className="galaxy-nebula galaxy-nebula-3" />
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};
