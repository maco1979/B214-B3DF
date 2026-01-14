#!/usr/bin/env python3
"""
测试聊天扩展统一处理功能
"""

import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, '/d/1.6/1.5/backend')
sys.path.insert(0, '/d/1.6/1.5')

def test_chat_extensions():
    """测试聊天扩展功能"""
    logger.info("开始测试聊天扩展功能...")
    
    try:
        # 导入AI模型服务
        from src.core.services.ai_model_service import aimodel_service
        
        # 1. 测试获取聊天扩展
        logger.info("1. 测试获取聊天扩展")
        extensions = aimodel_service.get_chat_extensions()
        logger.info(f"已注册的扩展: {extensions}")
        
        # 2. 测试扩展启用状态
        logger.info("2. 测试扩展启用状态")
        enabled = aimodel_service.is_chat_extensions_enabled()
        logger.info(f"扩展启用状态: {enabled}")
        
        # 3. 测试生成响应（验证扩展是否生效）
        logger.info("3. 测试生成响应（扩展生效验证）")
        response1 = aimodel_service.generate_response("你好？")
        logger.info(f"响应（扩展启用）: {response1}")
        
        # 4. 测试禁用扩展
        logger.info("4. 测试禁用扩展")
        aimodel_service.set_chat_extensions_enabled(False)
        enabled = aimodel_service.is_chat_extensions_enabled()
        logger.info(f"扩展启用状态: {enabled}")
        
        # 5. 测试禁用扩展后的响应
        logger.info("5. 测试禁用扩展后的响应")
        response2 = aimodel_service.generate_response("你好？")
        logger.info(f"响应（扩展禁用）: {response2}")
        
        # 6. 重新启用扩展
        logger.info("6. 重新启用扩展")
        aimodel_service.set_chat_extensions_enabled(True)
        enabled = aimodel_service.is_chat_extensions_enabled()
        logger.info(f"扩展启用状态: {enabled}")
        
        # 7. 测试自定义扩展
        logger.info("7. 测试自定义扩展")
        
        def custom_extension(text: str, context) -> str:
            """自定义扩展：在文本前添加前缀"""
            return f"[自定义扩展] {text}"
        
        # 注册自定义扩展
        aimodel_service.register_chat_extension("custom", custom_extension)
        extensions = aimodel_service.get_chat_extensions()
        logger.info(f"注册自定义扩展后: {extensions}")
        
        # 测试自定义扩展效果
        response3 = aimodel_service.generate_response("测试自定义扩展")
        logger.info(f"自定义扩展效果: {response3}")
        
        # 注销自定义扩展
        aimodel_service.unregister_chat_extension("custom")
        extensions = aimodel_service.get_chat_extensions()
        logger.info(f"注销自定义扩展后: {extensions}")
        
        logger.info("✅ 所有测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chat_extensions()
    sys.exit(0 if success else 1)
