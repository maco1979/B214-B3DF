/**
 * 网络状态检测 Hook
 * 监听浏览器在线/离线状态变化，用于在网络断开时提供用户提示
 */

import { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';

export interface OnlineStatusOptions {

  /** 是否在状态变化时显示toast提示 */
  showToast?: boolean

  /** 离线时的提示消息 */
  offlineMessage?: string

  /** 恢复在线时的提示消息 */
  onlineMessage?: string
}

const defaultOptions: OnlineStatusOptions = {
  showToast: true,
  offlineMessage: '网络连接已断开，请检查网络设置',
  onlineMessage: '网络连接已恢复',
};

/**
 * 检测并监听网络在线状态
 * @param options 配置选项
 * @returns 当前在线状态
 */
export function useOnlineStatus(options: OnlineStatusOptions = {}): boolean {
  const { showToast, offlineMessage, onlineMessage } = { ...defaultOptions, ...options };
  const [isOnline, setIsOnline] = useState<boolean>(
    typeof navigator !== 'undefined' ? navigator.onLine : true,
  );

  const handleOnline = useCallback(() => {
    setIsOnline(true);
    if (showToast && onlineMessage) {
      toast.success(onlineMessage, {
        duration: 3000,
        icon: '🌐',
      });
    }
  }, [showToast, onlineMessage]);

  const handleOffline = useCallback(() => {
    setIsOnline(false);
    if (showToast && offlineMessage) {
      toast.error(offlineMessage, {
        duration: 5000,
        icon: '📡',
      });
    }
  }, [showToast, offlineMessage]);

  useEffect(() => {
    // 初始检查
    if (typeof navigator !== 'undefined') {
      setIsOnline(navigator.onLine);
    }

    // 添加事件监听
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // 清理事件监听
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [handleOnline, handleOffline]);

  return isOnline;
}

/**
 * 网络状态上下文值类型
 */
export interface NetworkStatusContextValue {
  isOnline: boolean

  /** 检查当前网络状态 */
  checkConnection: () => boolean
}

/**
 * 手动检查网络连接状态
 * @returns 当前是否在线
 */
export function checkNetworkStatus(): boolean {
  if (typeof navigator === 'undefined') {
 return true;
}
  return navigator.onLine;
}

export default useOnlineStatus;
