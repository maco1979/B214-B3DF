"""
Hyperledger Fabric客户端实现
提供与Fabric网络交互的核心功能
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入hfc包，如果失败则使用模拟实现
try:
    import grpc
    from hfc.fabric import Client
    from hfc.fabric.user import User
    from hfc.fabric.channel.channel import Channel
    _HFC_AVAILABLE = True
except ImportError:
    logger.warning("hfc包不可用，使用模拟实现")
    _HFC_AVAILABLE = False


class FabricClient:
    """Hyperledger Fabric客户端"""
    
    def __init__(self, config_path: str = "blockchain/fabric_config.yaml"):
        self.config_path = Path(config_path)
        self.client = None
        self.channel = None
        self.user = None
        self.initialized = False
        
    async def initialize(self) -> bool:
        """初始化Fabric客户端连接"""
        try:
            if _HFC_AVAILABLE:
                # 创建Fabric客户端
                self.client = Client(net_profile=str(self.config_path))
                
                # 获取组织管理员用户
                self.user = self.client.get_user('org1.example.com', 'Admin')
                
                # 获取通道
                self.channel = self.client.get_channel('mychannel')
                
                # 检查通道是否可用
                if not await self._check_channel_health():
                    logger.error("Fabric通道健康检查失败")
                    return False
                    
                self.initialized = True
                logger.info("Fabric客户端初始化成功")
                return True
            else:
                # 模拟实现
                logger.info("使用模拟Fabric客户端")
                self.initialized = True
                return True
            
        except Exception as e:
            logger.error(f"Fabric客户端初始化失败: {e}")
            return False
    
    async def _check_channel_health(self) -> bool:
        """检查通道健康状态"""
        try:
            if _HFC_AVAILABLE:
                # 获取通道信息
                channel_info = await self.channel.get_channel_info(self.user)
                return channel_info is not None
            else:
                # 模拟实现
                return True
        except Exception as e:
            logger.error(f"通道健康检查失败: {e}")
            return False
    
    async def invoke_chaincode(self, 
                             contract_name: str,
                             function: str,
                             args: List[str]) -> Dict[str, Any]:
        """调用智能合约"""
        if not self.initialized:
            raise RuntimeError("Fabric客户端未初始化")
        
        try:
            if _HFC_AVAILABLE:
                # 调用智能合约
                response = await self.channel.chaincode_invoke(
                    requestor=self.user,
                    channel_name='mychannel',
                    peers=['peer0.org1.example.com'],
                    fcn=function,
                    args=args,
                    cc_name=contract_name,
                    wait_for_event=True
                )
                
                # 解析响应
                if response and 'tx_id' in response:
                    return {
                        'success': True,
                        'transaction_id': response['tx_id'],
                        'payload': response.get('payload', ''),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': '调用智能合约失败',
                        'timestamp': datetime.utcnow().isoformat()
                    }
            else:
                # 模拟实现
                transaction_id = f"0x{hashlib.sha256((contract_name + function + str(args)).encode()).hexdigest()[:64]}"
                return {
                    'success': True,
                    'transaction_id': transaction_id,
                    'payload': json.dumps({'message': '模拟智能合约调用'}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"调用智能合约失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def query_chaincode(self,
                            contract_name: str,
                            function: str,
                            args: List[str]) -> Dict[str, Any]:
        """查询智能合约状态"""
        if not self.initialized:
            raise RuntimeError("Fabric客户端未初始化")
        
        try:
            if _HFC_AVAILABLE:
                # 查询智能合约
                response = await self.channel.chaincode_query(
                    requestor=self.user,
                    channel_name='mychannel',
                    peers=['peer0.org1.example.com'],
                    fcn=function,
                    args=args,
                    cc_name=contract_name
                )
                
                return {
                    'success': True,
                    'data': response,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                # 模拟实现
                return {
                    'success': True,
                    'data': json.dumps({'message': '模拟智能合约查询结果', 'args': args}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"查询智能合约失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_block_info(self, block_number: int) -> Dict[str, Any]:
        """获取区块信息"""
        if not self.initialized:
            raise RuntimeError("Fabric客户端未初始化")
        
        try:
            if _HFC_AVAILABLE:
                # 获取区块信息
                block = await self.channel.get_block(block_number, self.user)
                
                return {
                    'success': True,
                    'block_number': block_number,
                    'block_hash': block.header.data_hash.hex(),
                    'previous_hash': block.header.previous_hash.hex(),
                    'transaction_count': len(block.data.data),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                # 模拟实现
                block_hash = hashlib.sha256(str(block_number).encode()).hexdigest()
                previous_hash = hashlib.sha256(str(block_number - 1).encode()).hexdigest() if block_number > 0 else "0" * 64
                return {
                    'success': True,
                    'block_number': block_number,
                    'block_hash': block_hash,
                    'previous_hash': previous_hash,
                    'transaction_count': 1,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"获取区块信息失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """关闭客户端连接"""
        self.initialized = False
        self.client = None
        self.channel = None
        self.user = None
        logger.info("Fabric客户端已关闭")