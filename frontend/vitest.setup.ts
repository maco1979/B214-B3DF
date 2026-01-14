// vitest.setup.ts 完整代码 (项目根目录)
import '@testing-library/jest-dom/vitest';


// ========== 1. Mock 浏览器缺失的 ResizeObserver (解决Slider/布局组件报错) ==========
global.ResizeObserver = class ResizeObserver {
  constructor(public callback: ResizeObserverCallback) {}
  observe(target: Element, options?: ResizeObserverOptions): void {}
  unobserve(target: Element): void {}
  disconnect(): void {}
};


// ========== 2. Mock localStorage 兜底 (防止jsdom环境localStorage异常) ==========
const localStorageMock: { [key: string]: string } = {};
Object.defineProperty(global, 'localStorage', {
  value: {
    getItem: vi.fn((key: string) => localStorageMock[key] || null),
    setItem: vi.fn((key: string, value: string) => { localStorageMock[key] = value; }),
    removeItem: vi.fn((key: string) => { delete localStorageMock[key]; }),
    clear: vi.fn(() => { Object.keys(localStorageMock).forEach(key => delete localStorageMock[key]); }),
    key: vi.fn((index: number) => Object.keys(localStorageMock)[index] || null),
    length: 0,
  },
  writable: true,
});


// ========== 3. Mock window.matchMedia (UI组件高频用到，提前补全，避免后续报错) ==========
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});


// ========== 4. Mock 全局的 document.body 兜底 ==========
global.document.body = document.createElement('body');

// ========== 5. Mock window.location (解决导航测试问题) ==========
const locationMock = {
  href: 'http://localhost/',
  protocol: 'http:',
  host: 'localhost',
  hostname: 'localhost',
  port: '',
  pathname: '/',
  search: '',
  hash: '',
  assign: vi.fn((url: string) => {
    locationMock.href = url;
    if (url.startsWith('/')) {
      locationMock.pathname = url;
    } else {
      const urlObj = new URL(url);
      locationMock.pathname = urlObj.pathname;
    }
  }),
  replace: vi.fn((url: string) => {
    locationMock.href = url;
    if (url.startsWith('/')) {
      locationMock.pathname = url;
    } else {
      const urlObj = new URL(url);
      locationMock.pathname = urlObj.pathname;
    }
  }),
  reload: vi.fn(),
  toString: vi.fn(() => locationMock.href),
};

Object.defineProperty(window, 'location', {
  value: locationMock,
  writable: true,
});
