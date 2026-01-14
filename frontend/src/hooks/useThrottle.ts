/**
 * 节流Hook
 * 用于限制函数在一定时间内只能执行一次
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * 节流Hook
 * @param value 需要节流的值
 * @param delay 节流时间间隔，单位毫秒
 * @returns 节流后的值
 */
export function useThrottle<T>(value: T, delay: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastUpdateTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateTimeRef.current;

    if (timeSinceLastUpdate >= delay) {
      // 如果距离上次更新已经超过了delay，立即更新
      setThrottledValue(value);
      lastUpdateTimeRef.current = now;
    } else {
      // 否则设置定时器，在剩余时间后更新
      const handler = setTimeout(() => {
        setThrottledValue(value);
        lastUpdateTimeRef.current = Date.now();
      }, delay - timeSinceLastUpdate);

      // 清理函数
      return () => {
        clearTimeout(handler);
      };
    }
  }, [value, delay]);

  return throttledValue;
}

/**
 * 创建节流函数
 * @param func 需要节流的函数
 * @param delay 节流时间间隔，单位毫秒
 * @returns 节流后的函数
 */
export function useThrottleFn<T extends(...args: any[]) => any>(
  func: T,
  delay: number,
): (...args: Parameters<T>) => void {
  const lastCallTimeRef = useRef<number>(0);
  const timeoutIdRef = useRef<NodeJS.Timeout | null>(null);

  const throttledFn = useCallback(
    (...args: Parameters<T>) => {
      const now = Date.now();
      const timeSinceLastCall = now - lastCallTimeRef.current;

      // 如果距离上次调用已经超过了delay，立即执行
      if (timeSinceLastCall >= delay) {
        // 清除可能存在的定时器
        if (timeoutIdRef.current) {
          clearTimeout(timeoutIdRef.current);
          timeoutIdRef.current = null;
        }

        func(...args);
        lastCallTimeRef.current = now;
      } else if (!timeoutIdRef.current) {
        // 否则设置定时器，在剩余时间后执行
        timeoutIdRef.current = setTimeout(() => {
          func(...args);
          lastCallTimeRef.current = Date.now();
          timeoutIdRef.current = null;
        }, delay - timeSinceLastCall);
      }
    },
    [func, delay],
  );

  // 清理函数
  useEffect(() => () => {
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }
    }, []);

  return throttledFn;
}

export default useThrottle;
