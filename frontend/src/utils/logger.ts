/**
 * 全局日志工具，添加内存地址日志过滤
 * 解决控制台中出现的 [0xc00b1d4e00 0xc00b1d4e30] 格式内存地址日志问题
 */

// 内存地址正则表达式，增强版本，匹配更多格式
const MEMORY_ADDRESS_PATTERN = /^\[\s*0x[0-9a-fA-F]+(\s+0x[0-9a-fA-F]+)*\s*\]$/;

/**
 * 检查消息是否为内存地址格式
 */
function isMemoryAddress(message: any): boolean {
  if (typeof message !== 'string') {
    return false;
  }
  return MEMORY_ADDRESS_PATTERN.test(message);
}

/**
 * 检查参数列表中是否包含内存地址
 */
function hasMemoryAddressInArgs(args: any[]): boolean {
  return args.some(arg => {
    // 检查字符串
    if (typeof arg === 'string') {
      return isMemoryAddress(arg);
    }
    // 检查对象的字符串表示
    if (typeof arg === 'object' && arg !== null) {
      const stringRepr = String(arg);
      return isMemoryAddress(stringRepr);
    }
    // 检查其他类型的字符串表示
    return isMemoryAddress(String(arg));
  });
}

/**
 * 创建增强的console代理
 */
function createEnhancedConsoleProxy() {
  // 保存原始控制台方法
  const originalConsole = {...console};
  
  // 创建代理函数生成器
  function createProxyMethod(originalMethod: Function) {
    return function(...args: any[]) {
      // 检查参数中是否包含内存地址
      if (!hasMemoryAddressInArgs(args)) {
        return originalMethod.apply(console, args);
      }
      // 包含内存地址，忽略输出
    };
  }
  
  // 创建代理控制台对象
  const consoleProxy: any = {
    log: createProxyMethod(originalConsole.log),
    warn: createProxyMethod(originalConsole.warn),
    error: createProxyMethod(originalConsole.error),
    info: createProxyMethod(originalConsole.info),
    debug: createProxyMethod(originalConsole.debug),
    dir: createProxyMethod(originalConsole.dir),
    table: createProxyMethod(originalConsole.table),
    trace: createProxyMethod(originalConsole.trace),
    group: createProxyMethod(originalConsole.group),
    groupEnd: createProxyMethod(originalConsole.groupEnd),
    groupCollapsed: createProxyMethod(originalConsole.groupCollapsed),
    // 保留其他非函数属性
    assert: originalConsole.assert,
    clear: originalConsole.clear,
    count: originalConsole.count,
    countReset: originalConsole.countReset,
    dirxml: originalConsole.dirxml,
    exception: createProxyMethod(originalConsole.exception),
    profile: originalConsole.profile,
    profileEnd: originalConsole.profileEnd,
    time: originalConsole.time,
    timeEnd: originalConsole.timeEnd,
    timeLog: originalConsole.timeLog,
    timeStamp: originalConsole.timeStamp,
  };
  
  return consoleProxy;
}

/**
 * 初始化日志过滤
 */
export function initializeLogger() {
  // 创建增强的console代理
  const consoleProxy = createEnhancedConsoleProxy();
  
  // 替换全局console对象
  Object.defineProperty(window, 'console', {
    value: consoleProxy,
    writable: true,
    enumerable: true,
    configurable: true
  });
  
  // 同时替换globalThis.console，确保在所有上下文中都生效
  Object.defineProperty(globalThis, 'console', {
    value: consoleProxy,
    writable: true,
    enumerable: true,
    configurable: true
  });
  
  // 使用原始log方法输出初始化信息
  const originalLog = (console as any).log;
  originalLog('[Logger] 全局日志过滤器已初始化，将自动过滤内存地址日志');
}

/**
 * 恢复原始控制台方法
 */
export function restoreOriginalLogger() {
  // 删除自定义console对象，让浏览器恢复默认实现
  delete (window as any).console;
  delete (globalThis as any).console;
  
  // 重新获取原始console对象
  console.log('[Logger] 已恢复原始控制台方法');
}