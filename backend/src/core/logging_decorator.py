"""
日志装饰器模块
为API接口提供统一的日志记录功能
"""

import logging
import time
from functools import wraps
from fastapi import Request

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def api_logger(func):
    """API接口日志装饰器
    
    为FastAPI接口添加日志记录功能，记录请求的开始、结束和错误信息。
    
    Args:
        func: FastAPI路由函数
        
    Returns:
        wrapper: 包装后的路由函数
    """
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # 请求前日志
        start_time = time.time()
        logger.info(f"【请求开始】路径: {request.url.path} | 方法: {request.method} | IP: {request.client.host}")
        try:
            # 执行接口逻辑
            result = await func(request, *args, **kwargs)
            # 请求成功日志
            logger.info(f"【请求成功】路径: {request.url.path} | 耗时: {round(time.time()-start_time, 3)}s")
            return result
        except Exception as e:
            # 请求失败日志
            logger.error(f"【请求失败】路径: {request.url.path} | 错误: {str(e)} | 耗时: {round(time.time()-start_time, 3)}s", exc_info=True)
            raise e
    return wrapper


def ws_logger(func):
    """WebSocket日志装饰器
    
    为WebSocket连接添加日志记录功能，记录连接的建立、关闭和错误信息。
    
    Args:
        func: WebSocket路由函数
        
    Returns:
        wrapper: 包装后的路由函数
    """
    @wraps(func)
    async def wrapper(websocket, *args, **kwargs):
        # 连接建立日志
        client_host = websocket.client.host if websocket.client else "unknown"
        logger.info(f"【WS连接】客户端: {client_host} | 路径: {websocket.url.path} | 状态: 建立")
        try:
            # 执行WebSocket逻辑
            await func(websocket, *args, **kwargs)
            # 连接关闭日志
            logger.info(f"【WS关闭】客户端: {client_host} | 路径: {websocket.url.path} | 状态: 正常关闭")
        except Exception as e:
            # 连接错误日志
            logger.error(f"【WS错误】客户端: {client_host} | 路径: {websocket.url.path} | 错误: {str(e)}", exc_info=True)
    return wrapper
