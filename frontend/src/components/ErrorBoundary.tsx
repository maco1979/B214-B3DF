import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary] 捕获渲染错误', error, info);
  }

  handleReload = () => {
    this.setState({ hasError: false, error: undefined });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
 return <>{this.props.fallback}</>;
}

      return (
        <div className="flex h-screen flex-col items-center justify-center space-y-4 bg-[#0b0b0f] text-white px-6 text-center">
          <div className="flex items-center space-x-3 text-red-300">
            <AlertTriangle className="h-6 w-6" />
            <span className="text-lg font-semibold">页面出现错误</span>
          </div>
          <p className="text-gray-400 text-sm max-w-md">
            请尝试刷新页面。如果问题持续出现，请联系技术支持。
          </p>
          <Button variant="tech" onClick={this.handleReload} className="min-w-[120px]">
            刷新页面
          </Button>
          {this.state.error && (
            <p className="text-xs text-gray-500 max-w-lg break-words">
              {this.state.error.message}
            </p>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
