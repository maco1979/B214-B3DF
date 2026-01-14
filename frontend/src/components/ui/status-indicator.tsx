/**
 * StatusIndicator组件
 */

import React from 'react';

type Status = 'online' | 'offline' | 'idle' | 'warning' | 'error';

interface StatusIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  status: Status;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusIndicator = React.forwardRef<HTMLDivElement, StatusIndicatorProps>(
  ({ className, status, label, size = 'md', ...props }, ref) => {
    const statusConfig = {
      online: {
        color: 'bg-green-500',
        textColor: 'text-green-400',
        label: label || '在线',
      },
      offline: {
        color: 'bg-gray-600',
        textColor: 'text-gray-400',
        label: label || '离线',
      },
      idle: {
        color: 'bg-yellow-500',
        textColor: 'text-yellow-400',
        label: label || '空闲',
      },
      warning: {
        color: 'bg-orange-500',
        textColor: 'text-orange-400',
        label: label || '警告',
      },
      error: {
        color: 'bg-red-500',
        textColor: 'text-red-400',
        label: label || '错误',
      },
    }[status];

    const sizeClasses = {
      sm: 'w-2 h-2',
      md: 'w-3 h-3',
      lg: 'w-4 h-4',
    }[size];

    return (
      <div
        ref={ref}
        className={`inline-flex items-center space-x-2 status-indicator-${status} ${className}`}
        {...props}
      >
        <span className={`inline-block rounded-full ${statusConfig.color} ${sizeClasses}`}></span>
        {label && <span className={`text-xs ${statusConfig.textColor}`}>{label}</span>}
      </div>
    );
  },
);
StatusIndicator.displayName = 'StatusIndicator';
