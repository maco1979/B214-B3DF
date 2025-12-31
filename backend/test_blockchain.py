from src.blockchain.blockchain_manager import BlockchainManager
import asyncio

async def test():
    manager = BlockchainManager()
    await manager.initialize()
    print('Blockchain manager initialized')
    status = await manager.get_blockchain_status()
    print('Status:', status)
    
    # 测试模型注册
    result = await manager.register_ai_model("test_model_123", b"test_model_bytes", {"name": "Test Model", "version": "1.0.0"})
    print('Model registration result:', result)
    
    # 测试模型验证
    verify_result = await manager.verify_model_integrity("test_model_123", b"test_model_bytes")
    print('Model verification result:', verify_result)
    
    # 测试数据溯源
    provenance_result = await manager.record_training_data_usage("data_123", "test_model_123", {"usage": "training", "timestamp": "2023-01-01"})
    print('Data provenance result:', provenance_result)
    
    await manager.close()

if __name__ == "__main__":
    asyncio.run(test())