/**
 * Badge组件
 */

import React from 'react';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md';
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', size = 'md', ...props }, ref) => {
    const variantClasses = {
      default: 'bg-gray-800 text-gray-200',
      primary: 'bg-tech-primary text-white',
      success: 'bg-green-600 text-white',
      warning: 'bg-yellow-600 text-white',
      danger: 'bg-red-600 text-white'
    }[variant];

    const sizeClasses = {
      sm: 'h-4 min-w-[1rem] text-xs',
      md: 'h-6 min-w-[1.5rem] text-sm'
    }[size];

    return (
      <span
        ref={ref}
        className={`inline-flex items-center justify-center rounded-full font-medium ${variantClasses} ${sizeClasses} ${className}`}
        {...props}
      />
    );
  }
);
Badge.displayName = 'Badge';
