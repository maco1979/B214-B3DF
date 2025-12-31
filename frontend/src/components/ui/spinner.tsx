/**
 * Spinner组件 - 加载状态指示器
 */

import React from 'react';

interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'primary';
  text?: string;
  overlay?: boolean;
}

export const Spinner = React.forwardRef<HTMLDivElement, SpinnerProps>(
  ({ className, size = 'md', variant = 'default', text, overlay = false, ...props }, forwardedRef) => {
    const sizeClasses = {
      sm: 'h-4 w-4',
      md: 'h-8 w-8',
      lg: 'h-12 w-12'
    }[size];

    const variantClasses = {
      default: 'border-gray-600 border-t-gray-400',
      primary: 'border-tech-primary/50 border-t-tech-primary'
    }[variant];

    // 创建加载指示器核心
    const spinnerCore = (
      <div className="flex flex-col items-center justify-center gap-2">
        <div
          ref={forwardedRef}
          className={`animate-spin rounded-full border-4 ${variantClasses} ${sizeClasses}`}
          style={{ animationDuration: '0.8s' }} // 调整动画速度为0.8秒
          {...props}
        />
        {text && (
          <span className="text-sm text-gray-400">{text}</span>
        )}
      </div>
    );

    // 如果是overlay模式，添加全屏遮罩
    if (overlay) {
      return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          {spinnerCore}
        </div>
      );
    }

    return spinnerCore;
  }
);
Spinner.displayName = 'Spinner';
