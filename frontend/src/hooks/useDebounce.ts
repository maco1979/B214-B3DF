/**
 * 防抖Hook
 * 用于限制函数在短时间内被频繁调用，只执行最后一次调用
 */

import { useState, useEffect, useCallback } from 'react';

/**
 * 防抖Hook
 * @param value 需要防抖的值
 * @param delay 延迟时间，单位毫秒
 * @returns 防抖后的值
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    // 设置定时器
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // 清理函数
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * 创建防抖函数
 * @param func 需要防抖的函数
 * @param delay 延迟时间，单位毫秒
 * @returns 防抖后的函数
 */
export function useDebounceFn<T extends(...args: any[]) => any>(
  func: T,
  delay: number,
): (...args: Parameters<T>) => void {
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  const debouncedFn = useCallback(
    (...args: Parameters<T>) => {
      // 清除之前的定时器
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      // 设置新的定时器
      const id = setTimeout(() => {
        func(...args);
      }, delay);

      setTimeoutId(id);
    },
    [func, delay, timeoutId],
  );

  // 清理函数
  useEffect(() => () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    }, [timeoutId]);

  return debouncedFn;
}

export default useDebounce;
