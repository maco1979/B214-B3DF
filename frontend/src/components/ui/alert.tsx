/**
 * Alert组件 - 增强的错误提示和信息通知组件
 */

import React, { useEffect } from 'react';
import {
  X,
  CheckCircle,
  AlertCircle,
  Info,
  AlertTriangle,
} from 'lucide-react';

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  title?: string;
  closable?: boolean;
  onClose?: () => void;
  autoClose?: boolean;
  autoCloseDuration?: number;
}

interface AlertTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {
  className?: string;
  children?: React.ReactNode;
}

interface AlertDescriptionProps extends React.HTMLAttributes<HTMLParagraphElement> {
  className?: string;
  children?: React.ReactNode;
}

// 获取对应的图标
const getAlertIcon = (variant: AlertProps['variant']) => {
  switch (variant) {
    case 'success':
      return <CheckCircle className="w-5 h-5" />;
    case 'warning':
      return <AlertTriangle className="w-5 h-5" />;
    case 'danger':
      return <AlertCircle className="w-5 h-5" />;
    case 'info':
      return <Info className="w-5 h-5" />;
    default:
      return <Info className="w-5 h-5" />;
  }
};

export const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({
    className,
    variant = 'default',
    title,
    children,
    closable = false,
    onClose,
    autoClose = false,
    autoCloseDuration = 5000,
    ...props
  }, ref) => {
    const variantClasses = {
      default: 'bg-gray-900/50 border-gray-700 text-gray-200',
      success: 'bg-green-900/30 border-green-700 text-green-300',
      warning: 'bg-yellow-900/30 border-yellow-700 text-yellow-300',
      danger: 'bg-red-900/30 border-red-700 text-red-300',
      info: 'bg-blue-900/30 border-blue-700 text-blue-300',
    }[variant];

    // 自动关闭功能
    useEffect(() => {
      if (autoClose && onClose) {
        const timer = setTimeout(() => {
          onClose();
        }, autoCloseDuration);

        return () => clearTimeout(timer);
      }
    }, [autoClose, autoCloseDuration, onClose]);

    return (
      <div
        ref={ref}
        className={`relative rounded-lg border p-4 ${variantClasses} ${className} transition-all duration-300 ease-in-out`}
        {...props}
      >
        <div className="flex items-start gap-3">
          {/* 图标 */}
          <div className="mt-0.5 flex-shrink-0">
            {getAlertIcon(variant)}
          </div>

          {/* 内容 */}
          <div className="flex-1">
            {title && (
              <AlertTitle>{title}</AlertTitle>
            )}
            {children && (
              <AlertDescription>{children}</AlertDescription>
            )}
          </div>

          {/* 关闭按钮 */}
          {closable && onClose && (
            <button
              type="button"
              className="ml-auto flex-shrink-0 rounded-md bg-transparent p-1 text-gray-400 hover:bg-gray-800 hover:text-white focus:outline-none"
              onClick={onClose}
              aria-label="关闭通知"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    );
  },
);
Alert.displayName = 'Alert';

export const AlertTitle = React.forwardRef<HTMLHeadingElement, AlertTitleProps>(
  ({ className, children, ...props }, ref) => (
      <h3
        ref={ref}
        className={`mb-1 font-medium text-sm ${className}`}
        {...props}
      >
        {children}
      </h3>
    ),
);
AlertTitle.displayName = 'AlertTitle';

export const AlertDescription = React.forwardRef<HTMLParagraphElement, AlertDescriptionProps>(
  ({ className, children, ...props }, ref) => (
      <p
        ref={ref}
        className={`text-sm ${className}`}
        {...props}
      >
        {children}
      </p>
    ),
);
AlertDescription.displayName = 'AlertDescription';
