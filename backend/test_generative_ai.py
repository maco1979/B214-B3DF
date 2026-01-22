#!/usr/bin/env python3
"""
测试生成式AI功能

验证创造性内容生成功能的集成和效果
"""

import logging
from src.core.services.ai_model_service import aimodel_service
from src.core.services.creativity_service import creativity_service

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_creativity_service():
    """测试创造力服务"""
    logger.info("测试创造力服务...")
    
    # 测试生成创造性想法
    logger.info("\n1. 测试生成创造性想法:")
    test_prompts = [
        "农业病虫害防治",
        "智慧农业技术",
        "可持续农业发展"
    ]
    
    for prompt in test_prompts:
        logger.info(f"\n测试提示: {prompt}")
        ideas = creativity_service.generate_creative_ideas(prompt, num_ideas=3)
        
        if ideas:
            logger.info(f"生成的创意想法: {ideas}")
            
            # 测试增强创意
            for idea in ideas:
                enhanced_idea = creativity_service.enhance_creative_idea(idea["idea"])
                logger.info(f"增强后的创意: {enhanced_idea}")
        else:
            logger.warning("未生成创意想法")
    
    # 测试生成创造性故事
    logger.info("\n\n2. 测试生成创造性故事:")
    story_prompts = [
        "未来农业机器人",
        "智慧农场的一天",
        "农作物与AI的对话"
    ]
    
    for prompt in story_prompts:
        logger.info(f"\n测试提示: {prompt}")
        story = creativity_service.generate_creative_story(prompt, genre="科幻", length=200)
        
        if story["story"]:
            logger.info(f"生成的故事: {story}")
        else:
            logger.warning("未生成故事")

def test_ai_model_service():
    """测试AI模型服务中的生成式AI功能"""
    logger.info("\n\n测试AI模型服务中的生成式AI功能...")
    
    # 测试意图识别
    logger.info("\n1. 测试生成式AI意图识别:")
    test_inputs = [
        "生成一些农业创新的创意",
        "写一个关于未来农场的故事",
        "帮我生成一首诗歌",
        "生成一段Python代码"
    ]
    
    for input_text in test_inputs:
        logger.info(f"\n测试输入: {input_text}")
        intent = aimodel_service.recognize_intent(input_text)
        logger.info(f"识别的意图: {intent}")
    
    # 测试生成式AI响应
    logger.info("\n\n2. 测试生成式AI响应:")
    test_cases = [
        "生成一些农业病虫害防治的创意",
        "写一个关于智能农业机器人的故事"
    ]
    
    for test_case in test_cases:
        logger.info(f"\n测试输入: {test_case}")
        response = aimodel_service.generate_response({"text": test_case})
        logger.info(f"AI响应: {response}")

def main():
    """主测试函数"""
    logger.info("开始生成式AI功能测试...")
    
    try:
        # 测试创造力服务
        test_creativity_service()
        
        # 测试AI模型服务
        test_ai_model_service()
        
        logger.info("\n\n所有测试完成！")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
