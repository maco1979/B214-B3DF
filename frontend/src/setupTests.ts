// 测试设置文件
import { vi } from 'vitest';

// 手动mock localStorage（jsdom未自动提供时备用）
beforeAll(() => {
  // Mock localStorage
  global.localStorage = {
    getItem: vi.fn(),
    setItem: vi.fn(),
    clear: vi.fn(),
    removeItem: vi.fn(),
  } as unknown as Storage;

  // Mock ResizeObserver
  global.ResizeObserver = class ResizeObserver {
    constructor(private readonly callback: ResizeObserverCallback) {}
    observe(target: Element) {}
    unobserve(target: Element) {}
    disconnect() {}
  };
});

