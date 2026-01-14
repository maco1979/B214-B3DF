#!/usr/bin/env python3
"""
LangSmith数据集初始化脚本
用于创建数据集并添加示例
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.services.langsmith_service import langsmith_service


def init_langsmith_dataset():
    """初始化LangSmith数据集"""
    print("初始化LangSmith数据集...")
    
    # 获取LangSmith客户端
    client = langsmith_service.get_client()
    
    if not client:
        print("❌ 无法获取LangSmith客户端，请确保LANGSMITH_TRACING=true且配置了正确的API密钥")
        return
    
    try:
        # 创建数据集
        print("创建数据集...")
        dataset = client.create_dataset(
            dataset_name="ds-definite-reprocessing-31", 
            description="A sample dataset in LangSmith."
        )
        print(f"✅ 数据集创建成功，ID: {dataset.id}")
        
        # 准备示例数据
        examples = [
            {
                "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
                "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
            },
            {
                "inputs": {"question": "What is Earth's lowest point?"},
                "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
            },
        ]
        
        # 添加示例到数据集
        print("添加示例到数据集...")
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"✅ 成功添加 {len(examples)} 个示例到数据集")
        
        print("\n🎉 LangSmith数据集初始化完成！")
        print(f"数据集名称: {dataset.name}")
        print(f"数据集ID: {dataset.id}")
        print(f"示例数量: {len(examples)}")
        
    except Exception as e:
        print(f"❌ 初始化LangSmith数据集失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    init_langsmith_dataset()
