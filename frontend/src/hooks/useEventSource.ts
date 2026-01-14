import { useEffect, useMemo, useRef } from 'react';
import type { QueryKey } from '@tanstack/react-query';
import { queryClient } from '@/lib/query-client';
import { API_BASE_URL, API_KEY } from '@/config';

const TOKEN_KEY = 'ai-project-token';

type UpdateStrategy<T> = 'replace' | 'append' | ((prev: any, next: T) => any)

interface UseEventSourceOptions<T = any> {
  endpoint: string

  /** 自定义 event 名称，默认 message */
  event?: string
  enabled?: boolean
  queryKey?: QueryKey

  /** 数组流推荐 append，或自定义合并方法 */
  updateStrategy?: UpdateStrategy<T>

  /** append 模式下按某字段去重 */
  dedupeBy?: string

  /** append 模式下保留最近 N 条 */
  keepLast?: number

  /** 自定义解析方法，默认 JSON.parse */
  parse?: (event: MessageEvent) => T
  onEvent?: (payload: T) => void

  /** 断线重连间隔 */
  reconnectInterval?: number
  withCredentials?: boolean
}

function buildSseUrl(endpoint: string) {
  const base = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
  const url = new URL(base);
  const token = localStorage.getItem(TOKEN_KEY);

  if (token && !url.searchParams.has('token')) {
    url.searchParams.set('token', token);
  }
  if (API_KEY && !url.searchParams.has('apiKey')) {
    url.searchParams.set('apiKey', API_KEY);
  }
  return url.toString();
}

function applyUpdateStrategy<T>(
  prev: any,
  next: T,
  strategy: UpdateStrategy<T>,
  dedupeBy?: string,
  keepLast?: number,
) {
  if (typeof strategy === 'function') {
 return strategy(prev, next);
}
  if (strategy === 'append') {
    const prevArr = Array.isArray(prev) ? prev : [];
    const incoming = Array.isArray(next) ? next : [next];
    let merged = [...prevArr, ...incoming];

    if (dedupeBy) {
      const map = new Map();
      merged.forEach(item => {
        const key = item && typeof item === 'object' ? (item)[dedupeBy] : undefined;
        if (key !== undefined) {
          map.set(key, item);
        }
      });
      merged = Array.from(map.values());
    } else {
      merged = merged.filter((item, idx, arr) => idx === arr.findIndex(other => JSON.stringify(other) === JSON.stringify(item)));
    }

    if (keepLast && keepLast > 0) {
      merged = merged.slice(-keepLast);
    }
    return merged;
  }
  return next;
}

/**
 * 通用 SSE 订阅 Hook，自动重连并可与 React Query 集成
 */
export function useEventSource<T = any>(options: UseEventSourceOptions<T>) {
  const {
    endpoint,
    event,
    enabled = true,
    queryKey,
    updateStrategy = 'replace',
    dedupeBy,
    keepLast,
    parse = (ev: MessageEvent) => JSON.parse(ev.data),
    onEvent,
    reconnectInterval = 5000,
    withCredentials = false,
  } = options;

  const sourceRef = useRef<EventSource | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const url = useMemo(() => buildSseUrl(endpoint), [endpoint]);

  useEffect(() => {
    if (!enabled) {
 return;
}

    let stopped = false;

    const connect = () => {
      sourceRef.current?.close();

      const es = new EventSource(url, { withCredentials });
      sourceRef.current = es;

      const handleMessage = (event: MessageEvent) => {
        try {
          const payload = parse(event);
          if (queryKey) {
            queryClient.setQueryData(queryKey, prev => applyUpdateStrategy(prev, payload, updateStrategy, dedupeBy, keepLast));
          }
          onEvent?.(payload);
        } catch (err) {
          console.error('[SSE] parse error', err);
        }
      };

      if (event) {
        es.addEventListener(event, handleMessage as EventListener);
      } else {
        es.onmessage = handleMessage;
      }

      es.onerror = () => {
        es.close();
        if (!stopped) {
          if (reconnectTimer.current) {
 window.clearTimeout(reconnectTimer.current);
}
          reconnectTimer.current = window.setTimeout(connect, reconnectInterval);
        }
      };
    };

    connect();

    return () => {
      stopped = true;
      sourceRef.current?.close();
      if (reconnectTimer.current) {
 window.clearTimeout(reconnectTimer.current);
}
    };
  }, [url, event, enabled, dedupeBy, keepLast, parse, queryKey, updateStrategy, reconnectInterval, withCredentials, onEvent]);
}
