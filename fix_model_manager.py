# 修复model_manager.py中的同步方法冲突问题
import re
import os

file_path = 'd:\\1.5\\backend\\src\\core\\services\\model_manager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 定义要删除的同步方法模式
pattern = r'    def list_models\(self\) -> list:(.*?)    def update_model_metrics\(self, model_id: str, metrics: dict\):'

# 使用正则表达式删除同步方法
content = re.sub(pattern, r'    def update_model_metrics\(self, model_id: str, metrics: dict\):', content, flags=re.DOTALL)

# 保存修改后的文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('已成功删除同步的list_models方法')
