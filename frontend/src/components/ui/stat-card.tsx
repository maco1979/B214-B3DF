import React from 'react';
import { Card, CardContent, CardHeader } from './card';
import { StatusIndicator } from './status-indicator';

interface StatCardProps {
  title: string;
  value: string | number;
  icon?: React.ReactNode;
  trend?: { value: string | number; direction: 'up' | 'down' };
  status?: 'online' | 'offline' | 'idle' | 'warning' | 'error';
  className?: string;
  children?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon,
  trend,
  status,
  className,
  children,
}) => (
    <Card className={`bg-gradient-to-br from-tech-dark via-tech-gray to-tech-light border-tech-light/30 ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-400">{title}</h3>
          <div className="flex items-center space-x-2">
            {status && <StatusIndicator status={status} size="sm" />}
            {icon && <div className="text-gray-400">{icon}</div>}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between">
          <div className="text-2xl font-bold text-white">{value}</div>
          {trend && (
            <div className={`flex items-center space-x-1 text-xs ${trend.direction === 'up' ? 'text-green-400' : 'text-red-400'}`}>
              <span>{trend.direction === 'up' ? '↑' : '↓'}</span>
              <span>{trend.value}</span>
            </div>
          )}
        </div>
        {children && <div className="mt-4">{children}</div>}
      </CardContent>
    </Card>
  );
