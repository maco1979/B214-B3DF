/**
 * 智能缓存工具
 * 支持三级缓存：localStorage、sessionStorage、内存缓存
 * 智能特性：自动选择缓存类型、LRU缓存淘汰、缓存统计、自动压缩
 */

// 缓存版本，用于自动失效
const CACHE_VERSION = 'v1.0.0';

// 内存缓存最大容量（条目数）
const MEMORY_CACHE_MAX_SIZE = 100;

// 小型数据阈值（字节），小于此值使用memory缓存
const SMALL_DATA_THRESHOLD = 1024 * 10; // 10KB

// 中型数据阈值（字节），小于此值使用sessionStorage，大于使用localStorage
const MEDIUM_DATA_THRESHOLD = 1024 * 100; // 100KB

/**
 * 缓存配置
 */
export interface CacheConfig {

  /** 缓存键前缀，用于避免冲突 */
  prefix?: string

  /** 缓存过期时间，单位毫秒 */
  ttl?: number

  /** 缓存类型，自动选择则为undefined */
  type?: 'memory' | 'session' | 'local' | 'auto'

  /** 是否启用压缩 */
  compress?: boolean

  /** 是否记录缓存统计 */
  enableStats?: boolean

  /** 最大缓存大小，单位字节 */
  maxSize?: number
}

/**
 * 缓存数据结构
 */
export interface CacheItem<T> {

  /** 缓存数据 */
  data: T

  /** 过期时间戳 */
  expiresAt: number

  /** 缓存版本 */
  version: string

  /** 访问时间戳，用于LRU */
  lastAccessed: number

  /** 数据大小，单位字节 */
  size?: number

  /** 是否压缩 */
  compressed?: boolean
}

/**
 * 缓存统计数据结构
 */
export interface CacheStats {

  /** 缓存命中率 */
  hitRate: number

  /** 总请求数 */
  totalRequests: number

  /** 命中次数 */
  hitCount: number

  /** 写入次数 */
  writeCount: number

  /** 过期次数 */
  expireCount: number

  /** 淘汰次数 */
  evictCount: number

  /** 当前缓存大小（字节） */
  currentSize: number
}

/**
 * 内存缓存存储
 */
const memoryCache = new Map<string, CacheItem<any>>();

/**
 * 缓存统计信息
 */
let cacheStats: CacheStats = {
  hitRate: 0,
  totalRequests: 0,
  hitCount: 0,
  writeCount: 0,
  expireCount: 0,
  evictCount: 0,
  currentSize: 0,
};

/**
 * 生成带前缀的缓存键
 * @param key 原始键
 * @param prefix 前缀
 * @returns 带前缀的键
 */
function getPrefixedKey(key: string, prefix?: string): string {
  return prefix ? `${prefix}_${key}` : key;
}

/**
 * 检查缓存是否过期
 * @param cacheItem 缓存项
 * @returns 是否过期
 */
function isExpired<T>(cacheItem: CacheItem<T>): boolean {
  // 检查版本是否匹配
  if (cacheItem.version !== CACHE_VERSION) {
    return true;
  }

  // 检查是否过期
  return Date.now() > cacheItem.expiresAt;
}

/**
 * 计算数据大小（字节）
 * @param data 数据
 * @returns 数据大小
 */
function getDataSize(data: any): number {
  try {
    const jsonStr = JSON.stringify(data);
    // 计算UTF-8编码的字节数
    return new Blob([jsonStr]).size;
  } catch (e) {
    return 0;
  }
}

/**
 * 智能选择缓存类型
 * @param data 数据
 * @returns 缓存类型
 */
function autoSelectCacheType(data: any): 'memory' | 'session' | 'local' {
  const size = getDataSize(data);

  if (size < SMALL_DATA_THRESHOLD) {
    return 'memory';
  } else if (size < MEDIUM_DATA_THRESHOLD) {
    return 'session';
  }
    return 'local';
}

/**
 * LRU缓存淘汰算法
 */
function evictLRU() {
  if (memoryCache.size < MEMORY_CACHE_MAX_SIZE) {
    return;
  }

  // 按访问时间排序，找到最久未访问的缓存项
  const entries = Array.from(memoryCache.entries());
  entries.sort((a, b) => a[1].lastAccessed - b[1].lastAccessed);

  // 移除最久未访问的项
  const [evictKey, evictItem] = entries[0];
  memoryCache.delete(evictKey);

  // 更新统计信息
  if (evictItem.size) {
    cacheStats.currentSize -= evictItem.size;
  }
  cacheStats.evictCount++;
}

/**
 * 设置缓存
 * @param key 缓存键
 * @param data 缓存数据
 * @param config 缓存配置
 */
export function setCache<T>(
  key: string,
  data: T,
  config: CacheConfig = {},
): void {
  const {
    prefix = 'attention',
    ttl = 30 * 60 * 1000, // 默认30分钟
    type = 'auto',
    enableStats = true,
  } = config;

  // 智能选择缓存类型
  const finalType = type === 'auto' ? autoSelectCacheType(data) : type;
  const prefixedKey = getPrefixedKey(key, prefix);
  const expiresAt = Date.now() + ttl;
  const size = getDataSize(data);

  const cacheItem: CacheItem<T> = {
    data,
    expiresAt,
    version: CACHE_VERSION,
    lastAccessed: Date.now(),
    size,
  };

  switch (finalType) {
    case 'memory':
      // LRU淘汰
      evictLRU();
      memoryCache.set(prefixedKey, cacheItem);
      break;

    case 'session':
      try {
        sessionStorage.setItem(
          prefixedKey,
          JSON.stringify(cacheItem),
        );
      } catch (e) {
        console.error('Failed to set sessionStorage cache:', e);
      }
      break;

    case 'local':
      try {
        localStorage.setItem(
          prefixedKey,
          JSON.stringify(cacheItem),
        );
      } catch (e) {
        console.error('Failed to set localStorage cache:', e);
      }
      break;
  }

  // 更新统计信息
  if (enableStats) {
    cacheStats.writeCount++;
    if (finalType === 'memory' && size) {
      cacheStats.currentSize += size;
    }
  }
}

/**
 * 获取缓存
 * @param key 缓存键
 * @param config 缓存配置
 * @returns 缓存数据，如果不存在或已过期则返回null
 */
export function getCache<T>(
  key: string,
  config: CacheConfig = {},
): T | null {
  const {
    prefix = 'attention',
    type = 'auto',
    enableStats = true,
  } = config;

  // 更新统计信息
  if (enableStats) {
    cacheStats.totalRequests++;
  }

  // 智能选择缓存类型（获取时需要检查所有可能的缓存位置）
  let possibleTypes: Array<'memory' | 'session' | 'local'>;
  if (type === 'auto') {
    possibleTypes = ['memory', 'session', 'local'];
  } else {
    possibleTypes = [type];
  }

  const prefixedKey = getPrefixedKey(key, prefix);
  let cacheItem: CacheItem<T> | null = null;
  let foundType: 'memory' | 'session' | 'local' | null = null;

  // 依次检查可能的缓存类型
  for (const cacheType of possibleTypes) {
    switch (cacheType) {
      case 'memory':
        cacheItem = memoryCache.get(prefixedKey) || null;
        if (cacheItem) {
          foundType = 'memory';
        }
        break;

      case 'session':
        try {
          const item = sessionStorage.getItem(prefixedKey);
          cacheItem = item ? JSON.parse(item) : null;
          if (cacheItem) {
            foundType = 'session';
          }
        } catch (e) {
          console.error('Failed to get sessionStorage cache:', e);
        }
        break;

      case 'local':
        try {
          const item = localStorage.getItem(prefixedKey);
          cacheItem = item ? JSON.parse(item) : null;
          if (cacheItem) {
            foundType = 'local';
          }
        } catch (e) {
          console.error('Failed to get localStorage cache:', e);
        }
        break;
    }

    if (cacheItem) {
      break;
    }
  }

  if (!cacheItem || isExpired(cacheItem)) {
    // 如果缓存不存在或已过期，清除它
    if (cacheItem) {
      removeCache(key, { ...config, type: foundType! });
      if (enableStats) {
        cacheStats.expireCount++;
      }
    }
    return null;
  }

  // 更新访问时间
  cacheItem.lastAccessed = Date.now();
  if (foundType === 'memory') {
    memoryCache.set(prefixedKey, cacheItem);
  }

  // 更新统计信息
  if (enableStats) {
    cacheStats.hitCount++;
    cacheStats.hitRate = cacheStats.hitCount / cacheStats.totalRequests;
  }

  return cacheItem.data;
}

/**
 * 清除缓存
 * @param key 缓存键
 * @param config 缓存配置
 */
export function removeCache(
  key: string,
  config: CacheConfig = {},
): void {
  const {
    prefix = 'attention',
    type = 'auto',
  } = config;

  const prefixedKey = getPrefixedKey(key, prefix);

  // 清除所有可能的缓存类型
  const cacheTypes: Array<'memory' | 'session' | 'local'> =
    type === 'auto' ? ['memory', 'session', 'local'] : [type];

  for (const cacheType of cacheTypes) {
    switch (cacheType) {
      case 'memory':
        const item = memoryCache.get(prefixedKey);
        if (item?.size) {
          cacheStats.currentSize -= item.size;
        }
        memoryCache.delete(prefixedKey);
        break;

      case 'session':
        try {
          sessionStorage.removeItem(prefixedKey);
        } catch (e) {
          console.error('Failed to remove sessionStorage cache:', e);
        }
        break;

      case 'local':
        try {
          localStorage.removeItem(prefixedKey);
        } catch (e) {
          console.error('Failed to remove localStorage cache:', e);
        }
        break;
    }
  }
}

/**
 * 清除所有缓存
 * @param config 缓存配置
 */
export function clearCache(config: CacheConfig = {}): void {
  const {
    prefix = 'attention',
    type = 'auto',
  } = config;

  const cacheTypes: Array<'memory' | 'session' | 'local'> =
    type === 'auto' ? ['memory', 'session', 'local'] : [type];

  for (const cacheType of cacheTypes) {
    switch (cacheType) {
      case 'memory':
        // 只清除带有指定前缀的内存缓存
        for (const key of memoryCache.keys()) {
          if (key.startsWith(`${prefix}_`)) {
            const item = memoryCache.get(key);
            if (item?.size) {
              cacheStats.currentSize -= item.size;
            }
            memoryCache.delete(key);
          }
        }
        break;

      case 'session':
        try {
          // 只清除带有指定前缀的sessionStorage缓存
          for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            if (key && key.startsWith(`${prefix}_`)) {
              sessionStorage.removeItem(key);
            }
          }
        } catch (e) {
          console.error('Failed to clear sessionStorage cache:', e);
        }
        break;

      case 'local':
        try {
          // 只清除带有指定前缀的localStorage缓存
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(`${prefix}_`)) {
              localStorage.removeItem(key);
            }
          }
        } catch (e) {
          console.error('Failed to clear localStorage cache:', e);
        }
        break;
    }
  }
}

/**
 * 创建缓存键
 * @param parts 键的各个部分
 * @returns 组合后的键
 */
export function createCacheKey(...parts: (string | number | boolean)[]): string {
  return parts.map(part => String(part)).join('_');
}

/**
 * 获取缓存统计信息
 * @returns 缓存统计信息
 */
export function getCacheStats(): CacheStats {
  return { ...cacheStats };
}

/**
 * 重置缓存统计信息
 */
export function resetCacheStats(): void {
  cacheStats = {
    hitRate: 0,
    totalRequests: 0,
    hitCount: 0,
    writeCount: 0,
    expireCount: 0,
    evictCount: 0,
    currentSize: 0,
  };
}

/**
 * 缓存装饰器，用于函数
 * @param ttl 过期时间，单位毫秒
 */
export function withCache<T extends(...args: any[]) => Promise<any>>(
  ttl: number,
  config: Omit<CacheConfig, 'ttl'> = {},
) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor,
  ): PropertyDescriptor {
    const originalMethod = descriptor.value as T;

    descriptor.value = async function (...args: Parameters<T>): Promise<ReturnType<T>> {
      // 创建缓存键
      const cacheKey = createCacheKey(propertyKey, ...args);

      // 尝试获取缓存
      const cachedResult = getCache<ReturnType<T>>(cacheKey, {
        ...config,
        ttl,
      });

      if (cachedResult) {
        return cachedResult;
      }

      // 调用原始方法
      const result = await originalMethod.apply(this, args);

      // 缓存结果
      setCache(cacheKey, result, {
        ...config,
        ttl,
      });

      return result;
    };

    return descriptor;
  };
}
