"""
分布式DCNN系统主入口
集成联邦学习、边缘计算、区块链奖励的完整系统
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from .core import DistributedDCNNSystem
from .federated_edge import FederatedEdgeManager
from .blockchain_rewards import BlockchainRewardManager


class DistributedDCNNApplication:
    """分布式DCNN应用主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # 初始化核心组件
        self.dcnn_system = DistributedDCNNSystem(config)
        self.federated_manager = FederatedEdgeManager(config)
        self.reward_manager = BlockchainRewardManager(config)
        
        self.is_running = False
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('distributed_dcnn.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def startup(self):
        """启动系统"""
        self.logger.info("启动分布式DCNN系统...")
        
        try:
            # 启动核心系统
            await self.dcnn_system.initialize()
            
            # 启动联邦学习管理器
            await self.federated_manager.start()
            
            # 启动奖励系统
            await self.reward_manager.initialize()
            
            self.is_running = True
            self.logger.info("分布式DCNN系统启动完成")
            
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭系统"""
        self.logger.info("关闭分布式DCNN系统...")
        
        self.is_running = False
        
        # 关闭组件
        await self.federated_manager.stop()
        await self.reward_manager.shutdown()
        
        self.logger.info("系统关闭完成")
    
    async def process_image_batch(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像批次数据"""
        if not self.is_running:
            raise RuntimeError("系统未运行")
        
        start_time = datetime.now()
        
        try:
            # 分布式推理
            inference_results = await self.dcnn_system.distributed_inference(image_data)
            
            # 联邦学习更新
            learning_results = await self.federated_manager.update_models(inference_results)
            
            # 计算贡献并分配奖励
            reward_results = await self.reward_manager.calculate_rewards(learning_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'inference_results': inference_results,
                'learning_updates': learning_results,
                'reward_distribution': reward_results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_running': self.is_running,
            'dcnn_status': self.dcnn_system.get_status(),
            'federated_status': self.federated_manager.get_status(),
            'reward_status': self.reward_manager.get_status(),
            'timestamp': datetime.now().isoformat()
        }


async def main():
    """主函数"""
    # 配置参数
    config = {
        'edge_nodes': [
            {'id': 'edge_1', 'location': '北京', 'compute_capacity': 'high'},
            {'id': 'edge_2', 'location': '上海', 'compute_capacity': 'medium'},
            {'id': 'edge_3', 'location': '深圳', 'compute_capacity': 'high'}
        ],
        'model_config': {
            'architecture': 'resnet50',
            'input_size': (224, 224),
            'num_classes': 1000
        },
        'federated_learning': {
            'rounds': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'blockchain': {
            'network': 'ethereum',
            'reward_token': 'PHOTON',
            'reward_pool': 1000000
        }
    }
    
    # 创建应用
    app = DistributedDCNNApplication(config)
    
    try:
        # 启动系统
        await app.startup()
        
        # 模拟处理任务
        sample_data = {
            'batch_id': 'test_batch_001',
            'images': ['image_1.jpg', 'image_2.jpg', 'image_3.jpg'],
            'metadata': {'source': 'agricultural_drone'}
        }
        
        # 处理图像批次
        result = await app.process_image_batch(sample_data)
        print("处理结果:", result)
        
        # 显示系统状态
        status = await app.get_system_status()
        print("系统状态:", status)
        
        # 保持运行（实际应用中会有事件循环）
        await asyncio.sleep(5)
        
    except KeyboardInterrupt:
        print("\n接收到中断信号...")
    finally:
        # 关闭系统
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())