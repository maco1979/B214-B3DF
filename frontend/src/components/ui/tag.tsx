/**
 * Tag组件
 */

import React from 'react';

interface TagProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export const Tag = React.forwardRef<HTMLDivElement, TagProps>(
  ({ className, variant = 'default', size = 'md', ...props }, ref) => {
    const variantClasses = {
      default: 'bg-gray-800 text-gray-200 border-gray-700',
      primary: 'bg-tech-primary/20 text-tech-primary border-tech-primary/40',
      success: 'bg-green-900/30 text-green-400 border-green-700/40',
      warning: 'bg-yellow-900/30 text-yellow-400 border-yellow-700/40',
      danger: 'bg-red-900/30 text-red-400 border-red-700/40'
    }[variant];

    const sizeClasses = {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-3 py-1 text-sm',
      lg: 'px-4 py-1.5 text-base'
    }[size];

    return (
      <div
        ref={ref}
        className={`inline-flex items-center rounded-full border ${variantClasses} ${sizeClasses} ${className}`}
        {...props}
      />
    );
  }
);
Tag.displayName = 'Tag';
