/**
 * WebSocket封装类，实现断线自动重连和心跳检测
 * 解决摄像头WebSocket连接断开的问题
 */

class CameraWebSocket {
  ws: WebSocket | null = null;
  url: string;
  heartbeatTimer: NodeJS.Timeout | null = null;
  reconnectTimer: NodeJS.Timeout | null = null;
  reconnectCount = 0;
  maxReconnectCount = 5; // 最大重连次数
  heartbeatInterval = 5000; // 心跳间隔5秒

  // 事件回调函数
  onMessage?: (event: MessageEvent) => void;
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: string) => void;

  constructor(url: string) {
    this.url = url;
    this.init();
  }

  /**
   * 初始化WebSocket连接
   */
  init() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = event => {
        console.log('WS连接成功 ✔️');
        this.reconnectCount = 0;
        this.startHeartbeat();
        this.onOpen?.();
      };

      this.ws.onmessage = event => {
        // 接收后端帧数据，传递给业务层
        this.onMessage?.(event);
      };

      this.ws.onclose = event => {
        if (event.code !== 1000) {
          console.log('WS连接断开 ❌，准备重连');
          this.stopHeartbeat();
          this.reconnect();
        }
        this.onClose?.(event);
      };

      this.ws.onerror = error => {
        console.error('WS连接异常 ❌', error);
        this.stopHeartbeat();
        this.onError?.(`WebSocket连接异常: ${error}`);
      };
    } catch (error) {
      console.error('WS连接初始化失败 ❌', error);
      this.onError?.(`WebSocket连接初始化失败: ${error}`);
      this.reconnect();
    }
  }

  /**
   * 心跳机制：定期发送心跳包，检测连接状态
   */
  startHeartbeat() {
    this.stopHeartbeat(); // 先清除旧的心跳计时器
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'heartbeat' }));
          console.debug('发送心跳包 💓');
        } catch (error) {
          console.error('发送心跳包失败 ❌', error);
          this.stopHeartbeat();
          this.reconnect();
        }
      }
    }, this.heartbeatInterval);
  }

  /**
   * 停止心跳
   */
  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * 断线重连：指数退避策略，避免频繁重试
   */
  reconnect() {
    if (this.reconnectCount >= this.maxReconnectCount) {
      console.error('WS重连次数耗尽 ❌，请手动刷新页面');
      this.onError?.('摄像头连接断开，无法自动重连，请检查后端服务');
      return;
    }

    this.reconnectCount++;
    const delay = 2 ** this.reconnectCount * 1000; // 1s, 2s, 4s, 8s, 16s
    console.log(`WS第${this.reconnectCount}次重连...，延迟${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      this.init();
    }, delay);
  }

  /**
   * 发送数据
   */
  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
        return true;
      } catch (error) {
        console.error('WS发送数据失败 ❌', error);
        this.onError?.(`WebSocket发送数据失败: ${error}`);
        return false;
      }
    } else {
      console.error('WS连接未建立，无法发送数据 ❌');
      this.onError?.('WebSocket连接未建立，无法发送数据');
      return false;
    }
  }

  /**
   * 关闭连接
   */
  close() {
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close(1000, '主动关闭');
      this.ws = null;
    }
  }

  /**
   * 获取当前连接状态
   */
  get readyState(): number {
    return this.ws?.readyState || WebSocket.CLOSED;
  }

  /**
   * 重新连接
   */
  reconnectNow() {
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectCount = 0;
    this.init();
  }
}

export default CameraWebSocket;

