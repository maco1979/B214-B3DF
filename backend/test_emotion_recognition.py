#!/usr/bin/env python3
"""
测试情感识别功能

验证Hume AI EVI情感识别服务的集成和功能
"""

import asyncio
import logging
from src.core.cognitive_architecture import cognitive_architecture
from src.core.services.hume_evi_service import hume_evi_service

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_emotion_service():
    """测试Hume AI EVI服务"""
    logger.info("测试Hume AI EVI服务...")
    
    # 测试文本
    test_texts = [
        "我很高兴今天天气这么好，心情非常愉快！",
        "我很生气，这个系统总是出问题，太让人恼火了！",
        "我感到很伤心，我的宠物狗生病了，我很担心它。",
        "我很害怕，外面的风暴太大了，窗户都在摇晃。",
        "这个新功能非常有用，感谢开发团队的努力！"
    ]
    
    for text in test_texts:
        logger.info(f"\n测试文本: {text}")
        emotions = await hume_evi_service.analyze_emotions(text)
        
        if emotions:
            logger.info(f"情绪分析结果: {emotions}")
            dominant_emotions = hume_evi_service.get_dominant_emotions(emotions, top_n=3)
            logger.info(f"主导情绪: {dominant_emotions}")
            emotional_response = hume_evi_service.generate_emotional_response(emotions, text)
            logger.info(f"情感响应: {emotional_response}")
        else:
            logger.warning("未获取到情绪分析结果")

async def test_cognitive_architecture_emotion():
    """测试认知架构中的情感处理"""
    logger.info("\n\n测试认知架构中的情感处理...")
    
    # 初始化认知架构
    await cognitive_architecture.initialize()
    
    # 测试文本
    test_texts = [
        "今天的收成太好了，我非常开心！",
        "农田里发生了病虫害，我很担心今年的产量。",
        "这个新的农业技术真的很有用，解决了我们的大问题。"
    ]
    
    for text in test_texts:
        logger.info(f"\n测试文本: {text}")
        result = cognitive_architecture.process_input(text)
        
        if "emotion_analysis" in result:
            emotion_analysis = result["emotion_analysis"]
            logger.info(f"情感分析结果: {emotion_analysis}")
        else:
            logger.warning("认知架构未返回情感分析结果")

async def test_self_assessment_with_emotion():
    """测试包含情感识别能力的自我评估"""
    logger.info("\n\n测试包含情感识别能力的自我评估...")
    
    from src.core.meta_cognitive_controller import meta_cognitive_system
    
    # 初始化元认知系统
    await meta_cognitive_system.initialize()
    
    # 执行自我评估
    meta_cognitive_system.update()
    
    # 获取自我评估摘要
    summary = meta_cognitive_system.get_status()
    logger.info(f"自我评估摘要: {summary}")
    
    # 检查是否包含情感识别能力评估
    if "capability_assessments" in summary:
        cap_assessments = summary["capability_assessments"]
        logger.info(f"能力评估: {cap_assessments}")
        
        # 查找情感识别能力评估
        emotional_cap = next((cap for cap in cap_assessments if cap["capability_name"] == "情感识别能力"), None)
        if emotional_cap:
            logger.info(f"情感识别能力评估: {emotional_cap}")
        else:
            logger.warning("未找到情感识别能力评估")

async def main():
    """主测试函数"""
    logger.info("开始情感识别功能测试...")
    
    try:
        # 测试Hume AI EVI服务
        await test_emotion_service()
        
        # 测试认知架构中的情感处理
        await test_cognitive_architecture_emotion()
        
        # 测试包含情感识别能力的自我评估
        await test_self_assessment_with_emotion()
        
        logger.info("\n\n所有测试完成！")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
